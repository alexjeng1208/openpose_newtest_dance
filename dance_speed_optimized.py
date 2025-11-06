#!/usr/bin/env python3
"""
單人舞蹈姿態高速捕捉系統 (速度優化版)
==========================================
專為單人舞蹈動作的高速、高精度骨架捕捉優化

優化重點：
- 追蹤優先策略：CSRT 追蹤 + YOLO 定期刷新（降低延遲）
- GPU 加速：CUDA + half precision (YOLO)
- 自適應閾值：根據解析度動態調整
- 手部優化：啟用手部檢測，關閉臉部以提速
- 單人流程：簡化為最快的單人處理路徑

使用方式：
1. 修改下方的 input_folder 和 output_folder
2. 確保 yolov8n.pt 或 yolov8s.pt 在同目錄
3. 運行: python dance_speed_optimized.py
"""

import os
import cv2
import numpy as np
from typing import Optional

try:
    from PIL import Image as _PILImage
except Exception:
    _PILImage = None

from controlnet_aux import DWposeDetector, OpenposeDetector

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# =============================================================================
# 速度優化配置區
# =============================================================================

class SpeedConfig:
    """速度優化配置"""

    # === YOLO 檢測策略 ===
    DETECT_EVERY_N_FRAMES = 6  # 每 N 幀才用 YOLO 檢測一次（其餘用追蹤）
    # 設定建議：
    # - 6-8: 平衡速度和穩定性（推薦）
    # - 4-5: 更穩定但稍慢
    # - 10+: 更快但可能漏失快速移動

    # === YOLO 模型選擇 ===
    YOLO_MODEL_PRIORITY = ["yolov8n.pt", "yolov8s.pt"]  # 優先使用 n（更快）
    YOLO_USE_HALF = True  # 使用 half precision (FP16) 加速

    # === 姿態檢測配置 ===
    USE_HAND_DETECTION = True   # 啟用手部檢測（舞蹈必需）
    USE_FACE_DETECTION = False  # 關閉臉部檢測以提速

    # === 圖像處理 ===
    ENABLE_CLAHE = True      # CLAHE 增強對比度
    ENABLE_SHARPEN = True    # 銳化處理
    ENABLE_HAND_ENHANCE = True  # 手部區域額外增強

    # === 輸出選項 ===
    STRICT_NO_GHOST = True   # 無殘影模式（只用當幀）
    POST_CLEAN = True        # 形態學清理

    # === 解析度設定 ===
    TARGET_LONG_SIDE = 1280     # 初始解析度
    ENHANCED_LONG_SIDE = 1536   # ROI 放大解析度
    MAX_LONG_SIDE = 1920        # 重試時最大解析度

    # === 自適應閾值參數 ===
    MIN_DENSITY_LOWER = 0.0004  # 最小密度下界
    MIN_DENSITY_UPPER = 0.0012  # 最小密度上界
    DENSITY_REF_PIXELS = 3000.0  # 參考像素數

    # === 調試 ===
    DEBUG_VERBOSE = True  # 顯示詳細信息
    SHOW_FPS = True       # 顯示處理速度

# =============================================================================
# CUDA 自動檢測
# =============================================================================

def detect_cuda_device() -> str:
    """檢測並返回可用的計算設備"""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA 可用: {gpu_name}")
            return device
    except ImportError:
        pass
    except Exception as e:
        print(f"CUDA 檢測失敗: {e}")

    print("✗ CUDA 不可用，使用 CPU")
    return "cpu"

DEVICE = detect_cuda_device()

# =============================================================================
# 路徑配置（請修改這裡）
# =============================================================================

input_folder = r"H:\202511_Calm down AI 專案\calm down"
output_folder = r"H:\202511_Calm down AI 專案\pose"

os.makedirs(output_folder, exist_ok=True)

# =============================================================================
# 模型載入
# =============================================================================

print("=" * 60)
print("正在載入 DWPose 模型...")
print("=" * 60)

detector = None
_detector_backend = ""

try:
    # 嘗試使用 CUDA 加速 DWPose
    if DEVICE == "cuda":
        try:
            detector = DWposeDetector.from_pretrained("yzd-v/DWPose", device=DEVICE)
            _detector_backend = f"DWPOSE:yzd-v/DWPose (CUDA)"
        except:
            detector = DWposeDetector.from_pretrained("yzd-v/DWPose")
            _detector_backend = "DWPOSE:yzd-v/DWPose (CPU)"
    else:
        detector = DWposeDetector.from_pretrained("yzd-v/DWPose")
        _detector_backend = "DWPOSE:yzd-v/DWPose (CPU)"
except Exception as e1:
    try:
        detector = DWposeDetector.from_pretrained()
        _detector_backend = "DWPOSE:from_pretrained()"
    except Exception as e2:
        try:
            detector = DWposeDetector()
            _detector_backend = "DWPOSE:constructor()"
        except Exception as e3:
            print(f"DWPose 載入失敗，改用 OpenPose: {type(e3).__name__}")
            detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            _detector_backend = "OPENPOSE:lllyasviel/ControlNet"

print(f"✓ 模型載入完畢: {_detector_backend}")
print("=" * 60)

# =============================================================================
# 全域變數
# =============================================================================

_yolo_person_model = None
_cv_tracker = None
_prev_bbox = None
_frame_count = 0  # 幀計數器（用於定期刷新）

# 時序穩定化變數
prev_frame_bgr = None
prev_pose_bgr = None
prev2_frame_bgr = None
prev2_pose_bgr = None
_adaptive_ema_flow = None
_adaptive_ema_hist = None

# =============================================================================
# 工具函數
# =============================================================================

def imread_unicode(path: str):
    """讀取支持 Unicode 路徑的圖像"""
    if not os.path.isfile(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_unicode(path: str, image) -> bool:
    """寫入支持 Unicode 路徑的圖像"""
    try:
        ext = os.path.splitext(path)[1]
        success, encoded = cv2.imencode(ext, image)
        if not success:
            return False
        encoded.tofile(path)
        return True
    except Exception:
        return False

def _density_threshold_for(h: int, w: int) -> float:
    """根據圖像解析度計算自適應密度閾值"""
    total_pixels = h * w
    threshold = max(
        SpeedConfig.MIN_DENSITY_LOWER,
        min(
            SpeedConfig.MIN_DENSITY_UPPER,
            SpeedConfig.DENSITY_REF_PIXELS / total_pixels
        )
    )
    if SpeedConfig.DEBUG_VERBOSE:
        print(f"  [自適應閾值] {threshold:.6f} (解析度: {w}x{h})")
    return threshold

def enhance_hand_region(image_bgr: np.ndarray, bbox: Optional[tuple[int, int, int, int]] = None) -> np.ndarray:
    """針對手部區域做額外增強"""
    if bbox is None or not SpeedConfig.ENABLE_HAND_ENHANCE:
        return image_bgr
    try:
        x1, y1, x2, y2 = bbox
        h, w = image_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return image_bgr

        hand_roi = image_bgr[y1:y2, x1:x2].copy()

        # CLAHE
        lab = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        hand_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 銳化
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 1.2
        hand_roi = cv2.filter2D(hand_roi, -1, kernel)
        hand_roi = np.clip(hand_roi, 0, 255).astype(np.uint8)

        result = image_bgr.copy()
        result[y1:y2, x1:x2] = hand_roi
        return result
    except Exception:
        return image_bgr

def preprocess_for_pose(image: np.ndarray, target_long_side: int = 1280) -> np.ndarray:
    """預處理圖像"""
    img = image

    # CLAHE
    if SpeedConfig.ENABLE_CLAHE:
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            pass

    # 縮放
    try:
        h, w = img.shape[:2]
        long_side = max(h, w)
        if target_long_side > 0 and long_side < target_long_side:
            scale = target_long_side / float(long_side)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception:
        pass

    return img

def deblur_sharpen(image_bgr: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """銳化處理"""
    if not SpeedConfig.ENABLE_SHARPEN:
        return image_bgr
    try:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * strength
        sharpened = cv2.filter2D(image_bgr, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except Exception:
        return image_bgr

def to_cv2_uint8_bgr(image) -> Optional[np.ndarray]:
    """轉換為 OpenCV 格式"""
    if image is None:
        return None

    if _PILImage is not None and isinstance(image, _PILImage.Image):
        rgb = image.convert("RGB")
        np_rgb = np.array(rgb)
        return cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255)
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return None

# =============================================================================
# YOLO 檢測（速度優化）
# =============================================================================

def _init_yolo_model():
    """初始化 YOLO 模型（優先使用輕量級模型）"""
    global _yolo_person_model
    if YOLO is None:
        return None
    if _yolo_person_model is None:
        for model_path in SpeedConfig.YOLO_MODEL_PRIORITY:
            if os.path.exists(model_path):
                try:
                    print(f"嘗試載入 YOLO 模型: {model_path}")
                    _yolo_person_model = YOLO(model_path)
                    print(f"✓ YOLO 模型載入成功: {model_path}")

                    # 設置設備
                    if DEVICE == "cuda":
                        print(f"✓ YOLO 使用 CUDA 加速")

                    break
                except Exception as e:
                    print(f"  載入 {model_path} 失敗: {e}")
                    continue

        if _yolo_person_model is None:
            print("警告: 無法載入任何 YOLO 模型，將使用全圖檢測")

    return _yolo_person_model

def detect_person_bbox(image_bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """使用 YOLO 檢測單人邊界框（支持 CUDA + half precision）"""
    model = _init_yolo_model()
    if model is None:
        return None

    h, w = image_bgr.shape[:2]
    scales = [(1280, 0.2), (960, 0.15), (640, 0.1)]

    for imgsz, conf_thresh in scales:
        try:
            # 使用 half precision 加速（如果在 CUDA 上）
            results = model.predict(
                source=image_bgr,
                imgsz=imgsz,
                conf=conf_thresh,
                classes=[0],  # person
                verbose=False,
                half=SpeedConfig.YOLO_USE_HALF and DEVICE == "cuda",
                device=DEVICE
            )

            if not results or len(results) == 0:
                continue

            boxes = getattr(results[0], 'boxes', None)
            if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            best = None
            best_area = -1.0

            for b in xyxy:
                x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2)

            if best is not None:
                if SpeedConfig.DEBUG_VERBOSE:
                    print(f"  [YOLO] imgsz={imgsz} conf={conf_thresh} bbox={best}")
                return best

        except Exception as e:
            if SpeedConfig.DEBUG_VERBOSE:
                print(f"  [YOLO] 檢測失敗 (imgsz={imgsz}): {e}")
            continue

    return None

# =============================================================================
# 追蹤功能
# =============================================================================

def _ensure_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    """確保邊界框有效"""
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)

def _expand_bbox(bbox: tuple[int, int, int, int], w: int, h: int, ratio: float) -> tuple[int, int, int, int]:
    """擴展邊界框"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * ratio
    bh = (y2 - y1) * ratio
    nx1 = int(round(cx - bw / 2.0))
    ny1 = int(round(cy - bh / 2.0))
    nx2 = int(round(cx + bw / 2.0))
    ny2 = int(round(cy + bh / 2.0))
    return _ensure_bbox(nx1, ny1, nx2, ny2, w, h) or (0, 0, w, h)

def _warp_bbox_with_flow(prev_bgr: np.ndarray, curr_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[tuple[int, int, int, int]]:
    """使用光流預測邊界框"""
    try:
        x1, y1, x2, y2 = bbox
        prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        sub = flow[y1:y2, x1:x2]
        dx = float(np.median(sub[..., 0]))
        dy = float(np.median(sub[..., 1]))
        w = curr_bgr.shape[1]
        h = curr_bgr.shape[0]
        return _ensure_bbox(int(round(x1 + dx)), int(round(y1 + dy)), int(round(x2 + dx)), int(round(y2 + dy)), w, h)
    except Exception:
        return None

def _init_or_update_tracker(frame_bgr: np.ndarray, bbox: Optional[tuple[int, int, int, int]], reset: bool = False) -> Optional[tuple[int, int, int, int]]:
    """初始化或更新追蹤器"""
    global _cv_tracker
    if reset:
        _cv_tracker = None
    try:
        if _cv_tracker is None and hasattr(cv2, 'TrackerCSRT_create') and bbox is not None:
            _cv_tracker = cv2.TrackerCSRT_create()
            _cv_tracker.init(frame_bgr, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
            if SpeedConfig.DEBUG_VERBOSE:
                print(f"  [追蹤器] CSRT 初始化 bbox={bbox}")
            return bbox
        if _cv_tracker is not None:
            ok, rect = _cv_tracker.update(frame_bgr)
            if ok and rect is not None:
                x, y, w, h = rect
                W = frame_bgr.shape[1]
                H = frame_bgr.shape[0]
                result = _ensure_bbox(int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h)), W, H)
                if SpeedConfig.DEBUG_VERBOSE and result:
                    print(f"  [追蹤器] CSRT 更新 bbox={result}")
                return result
    except Exception:
        _cv_tracker = None
    return None

# =============================================================================
# 姿態處理
# =============================================================================

def expand_crop(image_bgr: np.ndarray, bbox: tuple[int, int, int, int],
                expand_ratio: float = 1.3,
                target_long_side: int = 1280) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """擴展並裁切 ROI"""
    x1, y1, x2, y2 = bbox
    h, w = image_bgr.shape[:2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    bw2 = bw * expand_ratio
    bh2 = bh * expand_ratio
    nx1 = int(round(cx - bw2 / 2.0))
    ny1 = int(round(cy - bh2 / 2.0))
    nx2 = int(round(cx + bw2 / 2.0))
    ny2 = int(round(cy + bh2 / 2.0))
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)
    if nx2 - nx1 <= 2 or ny2 - ny1 <= 2:
        return image_bgr, (0, 0, w, h)

    crop = image_bgr[ny1:ny2, nx1:nx2]
    ch, cw = crop.shape[:2]
    long_side = max(ch, cw)
    if long_side < target_long_side:
        scale = target_long_side / float(long_side)
        new_w = int(round(cw * scale))
        new_h = int(round(ch * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return crop, (nx1, ny1, nx2, ny2)

def paste_pose(canvas_bgr: np.ndarray, pose_crop_bgr: np.ndarray, bbox: tuple[int, int, int, int],
               mask_thresh: int = 10) -> np.ndarray:
    """將骨架圖貼回 canvas"""
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return canvas_bgr

    ph, pw = pose_crop_bgr.shape[:2]
    pose_resized = cv2.resize(pose_crop_bgr, (box_w, box_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(pose_resized, cv2.COLOR_BGR2GRAY)
    mask = gray > mask_thresh
    roi = canvas_bgr[y1:y2, x1:x2]
    roi[mask] = pose_resized[mask]
    canvas_bgr[y1:y2, x1:x2] = roi
    return canvas_bgr

def clean_pose_lines(pose_bgr: np.ndarray, thresh: int = 10) -> np.ndarray:
    """清理骨架線條"""
    if not SpeedConfig.POST_CLEAN:
        return pose_bgr
    try:
        gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray > max(0, int(thresh))).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        out = np.zeros_like(pose_bgr)
        idx = opened > 0
        out[idx] = pose_bgr[idx]
        return out
    except Exception:
        return pose_bgr

def _pose_nonzero_ratio(pose_bgr: np.ndarray, thresh: int = 10) -> float:
    """計算骨架密度"""
    try:
        gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)
        mask = gray > max(0, int(thresh))
        return float(mask.mean())
    except:
        return 0.0

# =============================================================================
# 時序推論
# =============================================================================

def infer_pose_from_temporal(prev_pose_bgr: Optional[np.ndarray],
                            prev2_pose_bgr: Optional[np.ndarray],
                            prev_frame_bgr: Optional[np.ndarray],
                            prev2_frame_bgr: Optional[np.ndarray],
                            curr_frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """使用前後幀進行推論"""
    if prev_pose_bgr is None and prev2_pose_bgr is None:
        return None
    if prev_frame_bgr is None and prev2_frame_bgr is None:
        return None

    try:
        h, w = curr_frame_bgr.shape[:2]
        inferred = np.zeros((h, w, 3), dtype=np.uint8)

        if prev_pose_bgr is not None and prev_frame_bgr is not None:
            try:
                prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                map_x = grid_x + flow[..., 0].astype(np.float32)
                map_y = grid_y + flow[..., 1].astype(np.float32)
                warped_prev = cv2.remap(prev_pose_bgr, map_x, map_y, cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                inferred = warped_prev
            except Exception:
                pass

        if prev2_pose_bgr is not None and prev2_frame_bgr is not None:
            try:
                prev2_gray = cv2.cvtColor(prev2_frame_bgr, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
                flow2 = cv2.calcOpticalFlowFarneback(prev2_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                map_x2 = grid_x + flow2[..., 0].astype(np.float32)
                map_y2 = grid_y + flow2[..., 1].astype(np.float32)
                warped_prev2 = cv2.remap(prev2_pose_bgr, map_x2, map_y2, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                if prev_pose_bgr is not None:
                    inferred = cv2.addWeighted(inferred, 0.6, warped_prev2, 0.4, 0.0)
                else:
                    inferred = warped_prev2
            except Exception:
                pass

        return inferred
    except Exception:
        return None

# =============================================================================
# 主處理循環
# =============================================================================

print(f"開始處理資料夾: {input_folder}")
print(f"輸出資料夾: {output_folder}")
print("=" * 60)

if not os.path.isdir(input_folder):
    raise FileNotFoundError(f"找不到輸入資料夾: {input_folder}")

file_list = sorted(os.listdir(input_folder))
total_files = sum(1 for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))

print(f"共找到 {total_files} 個圖像文件")
print("=" * 60)

import time
processed_count = 0
total_time = 0

for filename in file_list:
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        continue

    frame_start = time.time()

    try:
        if not os.path.isfile(input_path):
            continue

        img = imread_unicode(input_path)
        if img is None:
            print(f"✗ 讀取失敗: {input_path}")
            continue

        # 預處理
        pre_img = preprocess_for_pose(img, target_long_side=SpeedConfig.TARGET_LONG_SIDE)
        pre_img = deblur_sharpen(pre_img, strength=1.5)

        # === 追蹤優先策略 ===
        _frame_count += 1
        use_yolo = (_frame_count % SpeedConfig.DETECT_EVERY_N_FRAMES == 1)  # 第一幀或每 N 幀

        if use_yolo:
            # 使用 YOLO 檢測
            yolo_bbox = detect_person_bbox(pre_img)
            if yolo_bbox is not None:
                _prev_bbox = _init_or_update_tracker(pre_img, yolo_bbox, reset=True)
                candidate_bbox = yolo_bbox
                if SpeedConfig.DEBUG_VERBOSE:
                    print(f"[{processed_count+1}/{total_files}] 幀{_frame_count} - YOLO 檢測")
            else:
                # YOLO 失敗，嘗試追蹤或光流
                tracked = _init_or_update_tracker(pre_img, None, reset=False)
                if tracked is not None:
                    candidate_bbox = tracked
                elif _prev_bbox is not None and prev_frame_bgr is not None:
                    warped = _warp_bbox_with_flow(prev_frame_bgr, pre_img, _prev_bbox)
                    candidate_bbox = warped if warped is not None else None
                else:
                    candidate_bbox = None
        else:
            # 使用追蹤（更快）
            tracked = _init_or_update_tracker(pre_img, None, reset=False)
            if tracked is not None:
                candidate_bbox = tracked
                if SpeedConfig.DEBUG_VERBOSE:
                    print(f"[{processed_count+1}/{total_files}] 幀{_frame_count} - CSRT 追蹤")
            elif _prev_bbox is not None and prev_frame_bgr is not None:
                # 追蹤失敗，使用光流
                warped = _warp_bbox_with_flow(prev_frame_bgr, pre_img, _prev_bbox)
                candidate_bbox = warped if warped is not None else None
                if SpeedConfig.DEBUG_VERBOSE and candidate_bbox:
                    print(f"[{processed_count+1}/{total_files}] 幀{_frame_count} - 光流預測")
            else:
                candidate_bbox = None

        # 全圖備援
        if candidate_bbox is None:
            candidate_bbox = (0, 0, pre_img.shape[1], pre_img.shape[0])
            if SpeedConfig.DEBUG_VERBOSE:
                print(f"[{processed_count+1}/{total_files}] 幀{_frame_count} - 全圖模式")

        # 擴展 bbox
        w0, h0 = pre_img.shape[1], pre_img.shape[0]
        candidate_bbox = _expand_bbox(candidate_bbox, w0, h0, ratio=1.3)

        # 裁切並放大
        crop_img, placed_bbox = expand_crop(pre_img, candidate_bbox, expand_ratio=1.0,
                                           target_long_side=SpeedConfig.ENHANCED_LONG_SIDE)

        # 手部增強
        h_roi = crop_img.shape[0]
        w_roi = crop_img.shape[1]
        hand_bbox = (int(w_roi * 0.2), int(h_roi * 0.1), int(w_roi * 0.8), int(h_roi * 0.6))
        crop_img = enhance_hand_region(crop_img, hand_bbox)

        # 執行 DWPose
        try:
            processed_img = detector(crop_img,
                                    detect_hand=SpeedConfig.USE_HAND_DETECTION,
                                    detect_face=SpeedConfig.USE_FACE_DETECTION)
        except Exception:
            try:
                processed_img = detector(crop_img)
            except Exception as e:
                print(f"✗ DWPose 失敗: {filename} - {e}")
                continue

        processed_cv = to_cv2_uint8_bgr(processed_img)
        if processed_cv is None:
            print(f"✗ 轉換失敗: {filename}")
            continue

        # 貼回並清理
        canvas = np.zeros_like(pre_img)
        composed = paste_pose(canvas, processed_cv, placed_bbox, mask_thresh=10)
        composed = clean_pose_lines(composed, thresh=10)

        # 檢查密度並重試
        min_density = _density_threshold_for(composed.shape[0], composed.shape[1])
        density = _pose_nonzero_ratio(composed, thresh=10)

        # 時序推論補洞
        if density < min_density:
            inferred_pose = infer_pose_from_temporal(
                prev_pose_bgr, prev2_pose_bgr,
                prev_frame_bgr, prev2_frame_bgr,
                pre_img
            )
            if inferred_pose is not None:
                inferred_density = _pose_nonzero_ratio(inferred_pose, thresh=10)
                if inferred_density >= min_density:
                    composed = inferred_pose
                    if SpeedConfig.DEBUG_VERBOSE:
                        print(f"  [時序推論] 密度={inferred_density:.6f}")

        # 重試機制
        density = _pose_nonzero_ratio(composed, thresh=10)
        if density < min_density:
            # 更大的 ROI
            bigger_bbox = _expand_bbox(candidate_bbox, w0, h0, ratio=1.6)
            crop2, placed2 = expand_crop(pre_img, bigger_bbox, expand_ratio=1.0,
                                        target_long_side=SpeedConfig.MAX_LONG_SIDE)
            h2_roi = crop2.shape[0]
            w2_roi = crop2.shape[1]
            hand2_bbox = (int(w2_roi * 0.2), int(h2_roi * 0.1), int(w2_roi * 0.8), int(h2_roi * 0.6))
            crop2 = enhance_hand_region(crop2, hand2_bbox)

            try:
                processed2 = detector(crop2,
                                     detect_hand=SpeedConfig.USE_HAND_DETECTION,
                                     detect_face=SpeedConfig.USE_FACE_DETECTION)
            except:
                try:
                    processed2 = detector(crop2)
                except:
                    processed2 = None

            proc2 = to_cv2_uint8_bgr(processed2)
            if proc2 is not None:
                canvas2 = np.zeros_like(pre_img)
                composed2 = paste_pose(canvas2, proc2, placed2, mask_thresh=10)
                if _pose_nonzero_ratio(composed2, thresh=10) >= density:
                    composed = composed2
                    placed_bbox = placed2
                    candidate_bbox = bigger_bbox
                    if SpeedConfig.DEBUG_VERBOSE:
                        print(f"  [重試] 使用更大 ROI")

            # 全圖重試
            density = _pose_nonzero_ratio(composed, thresh=10)
            if density < min_density:
                h_full = pre_img.shape[0]
                w_full = pre_img.shape[1]
                hand_full_bbox = (int(w_full * 0.2), int(h_full * 0.1), int(w_full * 0.8), int(h_full * 0.6))
                pre_img_enhanced = enhance_hand_region(pre_img, hand_full_bbox)

                try:
                    processed3 = detector(pre_img_enhanced,
                                         detect_hand=SpeedConfig.USE_HAND_DETECTION,
                                         detect_face=SpeedConfig.USE_FACE_DETECTION)
                except:
                    try:
                        processed3 = detector(pre_img_enhanced)
                    except:
                        processed3 = None

                if processed3 is not None:
                    proc3 = to_cv2_uint8_bgr(processed3)
                    if proc3 is not None:
                        composed = proc3
                        placed_bbox = (0, 0, w0, h0)
                        if SpeedConfig.DEBUG_VERBOSE:
                            print(f"  [重試] 使用全圖")

        _prev_bbox = placed_bbox

        # 無殘影輸出
        if SpeedConfig.STRICT_NO_GHOST:
            stabilized_cv = composed
            prev2_frame_bgr = prev_frame_bgr
            prev2_pose_bgr = prev_pose_bgr
            prev_frame_bgr = pre_img
            prev_pose_bgr = composed
        else:
            # 可選：時序穩定化（目前停用）
            stabilized_cv = composed
            prev2_frame_bgr = prev_frame_bgr
            prev2_pose_bgr = prev_pose_bgr
            prev_frame_bgr = pre_img
            prev_pose_bgr = composed

        # 保存
        if not imwrite_unicode(output_path, stabilized_cv):
            print(f"✗ 寫入失敗: {output_path}")
            continue

        processed_count += 1
        frame_time = time.time() - frame_start
        total_time += frame_time

        if SpeedConfig.SHOW_FPS:
            fps = 1.0 / frame_time if frame_time > 0 else 0
            avg_fps = processed_count / total_time if total_time > 0 else 0
            print(f"✓ [{processed_count}/{total_files}] {filename} | {fps:.1f} FPS | 平均: {avg_fps:.1f} FPS")
        else:
            print(f"✓ [{processed_count}/{total_files}] {filename}")

    except Exception as e:
        print(f"✗ 處理失敗 {filename}: {e}")

print("=" * 60)
print(f"處理完畢！成功: {processed_count}/{total_files}")
if total_time > 0:
    avg_fps = processed_count / total_time
    print(f"平均速度: {avg_fps:.2f} FPS ({total_time/processed_count:.2f}s/幀)")
print("=" * 60)
