#!/usr/bin/env python3
"""
單人舞蹈姿態穩定捕捉系統 (三遍處理版)
=========================================
專為解決抽動、跑偏問題的序列級穩定性優化

三遍處理流程：
1. 第一遍：高精度檢測，記錄所有幀的元數據（bbox、密度等）
2. 第二遍：平滑 bbox 序列後重新檢測 + 輕量時序穩定
3. 第三遍：一致性修補（光流對齊補洞）

特點：
- 序列級 bbox 平滑（雙向 EMA）
- 時序穩定化（光流對齊混合）
- 一致性修補（補洞填充）
- GPU 加速支持
"""

import os
import sys
import cv2
import numpy as np
import json
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict

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
# 配置區
# =============================================================================

class StableConfig:
    """穩定性優化配置"""

    # === 路徑 ===
    INPUT_FOLDER = r"H:\202511_Calm down AI 專案\calm down"
    OUTPUT_FOLDER = r"H:\202511_Calm down AI 專案\pose"
    TEMP_FOLDER = r"H:\202511_Calm down AI 專案\temp"  # 存放中間結果

    # === YOLO 檢測策略 ===
    DETECT_EVERY_N_FRAMES = 6  # 每 N 幀 YOLO 刷新
    YOLO_MODEL_PRIORITY = ["yolov8n.pt", "yolov8s.pt"]
    YOLO_USE_HALF = True

    # === 姿態檢測 ===
    USE_HAND_DETECTION = True
    USE_FACE_DETECTION = False

    # === 圖像處理 ===
    ENABLE_CLAHE = True
    ENABLE_SHARPEN = True
    ENABLE_HAND_ENHANCE = True

    # === 第一遍：高精度檢測 ===
    PASS1_STRICT_NO_GHOST = True  # 第一遍無殘影
    PASS1_TARGET_LONG_SIDE = 1280
    PASS1_ENHANCED_LONG_SIDE = 1536

    # === 第二遍：平滑 + 穩定 ===
    PASS2_BBOX_SMOOTH_ALPHA = 0.3  # bbox 平滑強度 (0.1-0.5)
    PASS2_ENABLE_TEMPORAL = True    # 啟用時序穩定
    PASS2_TEMPORAL_WEIGHT = 0.2     # 時序混合權重 (0.1-0.3)
    PASS2_TARGET_LONG_SIDE = 1536   # 第二遍解析度

    # === 第三遍：一致性修補 ===
    PASS3_ENABLE_REPAIR = True      # 啟用修補
    PASS3_REPAIR_RADIUS = 2         # 修補半徑（前後幀數）
    PASS3_MIN_HOLE_SIZE = 50        # 最小補洞面積（像素）

    # === 解析度 ===
    MAX_LONG_SIDE = 1920

    # === 自適應閾值 ===
    MIN_DENSITY_LOWER = 0.0004
    MIN_DENSITY_UPPER = 0.0012
    DENSITY_REF_PIXELS = 3000.0

    # === 調試 ===
    DEBUG_VERBOSE = True
    SAVE_METADATA = True  # 保存元數據到 JSON
    SHOW_PROGRESS = True

# =============================================================================
# 元數據結構
# =============================================================================

@dataclass
class FrameMetadata:
    """幀元數據"""
    filename: str
    frame_index: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    density: float
    width: int
    height: int
    detection_method: str  # "yolo", "csrt", "flow", "full"

# =============================================================================
# CUDA 檢測
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
    except:
        pass
    print("✗ CUDA 不可用，使用 CPU")
    return "cpu"

DEVICE = detect_cuda_device()

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
    """自適應密度閾值"""
    total_pixels = h * w
    threshold = max(
        StableConfig.MIN_DENSITY_LOWER,
        min(
            StableConfig.MIN_DENSITY_UPPER,
            StableConfig.DENSITY_REF_PIXELS / total_pixels
        )
    )
    return threshold

def _pose_nonzero_ratio(pose_bgr: np.ndarray, thresh: int = 10) -> float:
    """計算骨架密度"""
    try:
        gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)
        mask = gray > max(0, int(thresh))
        return float(mask.mean())
    except:
        return 0.0

# =============================================================================
# 平滑函數
# =============================================================================

def _ema_smooth(values: List[float], alpha: float = 0.3) -> List[float]:
    """指數移動平均平滑（單向）"""
    if not values:
        return []

    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed

def _bidirectional_ema_smooth(values: List[float], alpha: float = 0.3) -> List[float]:
    """雙向 EMA 平滑"""
    if len(values) <= 1:
        return values

    # 前向平滑
    forward = _ema_smooth(values, alpha)

    # 後向平滑
    backward = _ema_smooth(values[::-1], alpha)[::-1]

    # 平均
    result = [(f + b) / 2.0 for f, b in zip(forward, backward)]
    return result

def smooth_bbox_sequence(bboxes: List[Tuple[int, int, int, int]],
                         alpha: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """平滑整個 bbox 序列"""
    if not bboxes:
        return []

    # 分別平滑 x1, y1, x2, y2
    x1_list = [b[0] for b in bboxes]
    y1_list = [b[1] for b in bboxes]
    x2_list = [b[2] for b in bboxes]
    y2_list = [b[3] for b in bboxes]

    x1_smooth = _bidirectional_ema_smooth(x1_list, alpha)
    y1_smooth = _bidirectional_ema_smooth(y1_list, alpha)
    x2_smooth = _bidirectional_ema_smooth(x2_list, alpha)
    y2_smooth = _bidirectional_ema_smooth(y2_list, alpha)

    # 組合回 bbox
    smoothed_bboxes = [
        (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        for x1, y1, x2, y2 in zip(x1_smooth, y1_smooth, x2_smooth, y2_smooth)
    ]

    return smoothed_bboxes

# =============================================================================
# 載入模型（與之前相同的代碼）
# =============================================================================

print("=" * 60)
print("正在載入模型...")
print("=" * 60)

# DWPose 載入
detector = None
_detector_backend = ""

try:
    # 方法 1: 嘗試使用 easy_dwpose
    try:
        from easy_dwpose import DWposeDetector as EasyDWpose
        if DEVICE == "cuda":
            detector = EasyDWpose(device=DEVICE)
            _detector_backend = "DWPOSE:easy_dwpose (CUDA)"
        else:
            detector = EasyDWpose(device="cpu")
            _detector_backend = "DWPOSE:easy_dwpose (CPU)"
    except ImportError:
        # 方法 2: 使用 controlnet_aux 直接初始化
        detector = DWposeDetector()
        _detector_backend = "DWPOSE:controlnet_aux"
except Exception as e1:
    try:
        # 方法 3: 嘗試 OpenPose 作為替代
        print(f"DWPose 載入失敗，改用 OpenPose: {type(e1).__name__}")
        try:
            detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            _detector_backend = "OPENPOSE:lllyasviel/Annotators"
        except:
            detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            _detector_backend = "OPENPOSE:lllyasviel/ControlNet"
    except Exception as e2:
        print(f"✗ 所有模型載入失敗: {type(e2).__name__}")
        raise

print(f"✓ 模型載入完畢: {_detector_backend}")

# YOLO 載入（延遲初始化）
_yolo_person_model = None

def _init_yolo_model():
    """初始化 YOLO 模型"""
    global _yolo_person_model
    if YOLO is None:
        return None
    if _yolo_person_model is None:
        for model_path in StableConfig.YOLO_MODEL_PRIORITY:
            if os.path.exists(model_path):
                try:
                    print(f"載入 YOLO: {model_path}")
                    _yolo_person_model = YOLO(model_path)
                    print(f"✓ YOLO 載入成功")
                    break
                except Exception as e:
                    print(f"  載入失敗: {e}")
                    continue
    return _yolo_person_model

# =============================================================================
# 圖像處理函數（與之前相同）
# =============================================================================

def enhance_hand_region(image_bgr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """手部區域增強"""
    if bbox is None or not StableConfig.ENABLE_HAND_ENHANCE:
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
        lab = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        hand_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
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
    if StableConfig.ENABLE_CLAHE:
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except:
            pass

    try:
        h, w = img.shape[:2]
        long_side = max(h, w)
        if target_long_side > 0 and long_side < target_long_side:
            scale = target_long_side / float(long_side)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except:
        pass

    return img

def deblur_sharpen(image_bgr: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """銳化"""
    if not StableConfig.ENABLE_SHARPEN:
        return image_bgr
    try:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * strength
        sharpened = cv2.filter2D(image_bgr, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except:
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
# YOLO 和追蹤（與之前相同）
# =============================================================================

def detect_person_bbox(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """YOLO 檢測"""
    model = _init_yolo_model()
    if model is None:
        return None
    h, w = image_bgr.shape[:2]
    scales = [(1280, 0.2), (960, 0.15), (640, 0.1)]
    for imgsz, conf_thresh in scales:
        try:
            results = model.predict(
                source=image_bgr,
                imgsz=imgsz,
                conf=conf_thresh,
                classes=[0],
                verbose=False,
                half=StableConfig.YOLO_USE_HALF and DEVICE == "cuda",
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
                return best
        except Exception:
            continue
    return None

_cv_tracker = None
_prev_bbox = None

def _ensure_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)

def _expand_bbox(bbox: Tuple[int, int, int, int], w: int, h: int, ratio: float) -> Tuple[int, int, int, int]:
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

def _init_or_update_tracker(frame_bgr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], reset: bool = False) -> Optional[Tuple[int, int, int, int]]:
    global _cv_tracker
    if reset:
        _cv_tracker = None
    try:
        if _cv_tracker is None and hasattr(cv2, 'TrackerCSRT_create') and bbox is not None:
            _cv_tracker = cv2.TrackerCSRT_create()
            _cv_tracker.init(frame_bgr, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
            return bbox
        if _cv_tracker is not None:
            ok, rect = _cv_tracker.update(frame_bgr)
            if ok and rect is not None:
                x, y, w, h = rect
                W = frame_bgr.shape[1]
                H = frame_bgr.shape[0]
                return _ensure_bbox(int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h)), W, H)
    except:
        _cv_tracker = None
    return None

def _warp_bbox_with_flow(prev_bgr: np.ndarray, curr_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
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
    except:
        return None

# =============================================================================
# 姿態處理
# =============================================================================

def expand_crop(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int],
                expand_ratio: float = 1.3, target_long_side: int = 1280) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
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

def paste_pose(canvas_bgr: np.ndarray, pose_crop_bgr: np.ndarray, bbox: Tuple[int, int, int, int],
               mask_thresh: int = 10) -> np.ndarray:
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
    try:
        gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray > max(0, int(thresh))).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        out = np.zeros_like(pose_bgr)
        idx = opened > 0
        out[idx] = pose_bgr[idx]
        return out
    except:
        return pose_bgr

# =============================================================================
# 光流對齊函數
# =============================================================================

def warp_pose_with_flow(source_frame: np.ndarray, target_frame: np.ndarray,
                        source_pose: np.ndarray) -> Optional[np.ndarray]:
    """使用光流將 source_pose 對齊到 target_frame"""
    try:
        src_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(src_gray, tgt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        h, w = target_frame.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + flow[..., 0].astype(np.float32)
        map_y = grid_y + flow[..., 1].astype(np.float32)

        warped = cv2.remap(source_pose, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped
    except:
        return None

# =============================================================================
# 三遍處理主流程
# =============================================================================

def pass1_detect_all_frames(file_list: List[str]) -> List[FrameMetadata]:
    """第一遍：高精度檢測所有幀，記錄元數據"""
    print("\n" + "=" * 60)
    print("第一遍：高精度檢測")
    print("=" * 60)

    metadata_list = []
    frame_count = 0
    _cv_tracker = None
    _prev_bbox = None
    prev_frame_bgr = None

    temp_poses_folder = os.path.join(StableConfig.TEMP_FOLDER, "pass1_poses")
    temp_frames_folder = os.path.join(StableConfig.TEMP_FOLDER, "pass1_frames")
    os.makedirs(temp_poses_folder, exist_ok=True)
    os.makedirs(temp_frames_folder, exist_ok=True)

    for idx, filename in enumerate(file_list):
        input_path = os.path.join(StableConfig.INPUT_FOLDER, filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        frame_count += 1

        # 讀取並預處理
        img = imread_unicode(input_path)
        if img is None:
            continue

        pre_img = preprocess_for_pose(img, target_long_side=StableConfig.PASS1_TARGET_LONG_SIDE)
        pre_img = deblur_sharpen(pre_img, strength=1.5)

        # ROI 檢測（追蹤優先）
        use_yolo = (frame_count % StableConfig.DETECT_EVERY_N_FRAMES == 1)
        detection_method = "full"

        if use_yolo:
            yolo_bbox = detect_person_bbox(pre_img)
            if yolo_bbox is not None:
                _prev_bbox = _init_or_update_tracker(pre_img, yolo_bbox, reset=True)
                candidate_bbox = yolo_bbox
                detection_method = "yolo"
            else:
                tracked = _init_or_update_tracker(pre_img, None, reset=False)
                if tracked is not None:
                    candidate_bbox = tracked
                    detection_method = "csrt"
                elif _prev_bbox is not None and prev_frame_bgr is not None:
                    warped = _warp_bbox_with_flow(prev_frame_bgr, pre_img, _prev_bbox)
                    candidate_bbox = warped if warped is not None else None
                    detection_method = "flow" if warped else "full"
                else:
                    candidate_bbox = None
        else:
            tracked = _init_or_update_tracker(pre_img, None, reset=False)
            if tracked is not None:
                candidate_bbox = tracked
                detection_method = "csrt"
            elif _prev_bbox is not None and prev_frame_bgr is not None:
                warped = _warp_bbox_with_flow(prev_frame_bgr, pre_img, _prev_bbox)
                candidate_bbox = warped if warped is not None else None
                detection_method = "flow" if warped else "full"
            else:
                candidate_bbox = None

        if candidate_bbox is None:
            candidate_bbox = (0, 0, pre_img.shape[1], pre_img.shape[0])
            detection_method = "full"

        # 擴展並裁切
        w0, h0 = pre_img.shape[1], pre_img.shape[0]
        candidate_bbox = _expand_bbox(candidate_bbox, w0, h0, ratio=1.3)
        crop_img, placed_bbox = expand_crop(pre_img, candidate_bbox, expand_ratio=1.0,
                                           target_long_side=StableConfig.PASS1_ENHANCED_LONG_SIDE)

        # 手部增強
        h_roi, w_roi = crop_img.shape[:2]
        hand_bbox = (int(w_roi * 0.2), int(h_roi * 0.1), int(w_roi * 0.8), int(h_roi * 0.6))
        crop_img = enhance_hand_region(crop_img, hand_bbox)

        # DWPose 檢測
        try:
            processed_img = detector(crop_img,
                                    detect_hand=StableConfig.USE_HAND_DETECTION,
                                    detect_face=StableConfig.USE_FACE_DETECTION)
        except:
            try:
                processed_img = detector(crop_img)
            except:
                continue

        processed_cv = to_cv2_uint8_bgr(processed_img)
        if processed_cv is None:
            continue

        # 貼回並清理
        canvas = np.zeros_like(pre_img)
        composed = paste_pose(canvas, processed_cv, placed_bbox, mask_thresh=10)
        composed = clean_pose_lines(composed, thresh=10)

        # 計算密度
        density = _pose_nonzero_ratio(composed, thresh=10)

        # 保存
        pose_path = os.path.join(temp_poses_folder, filename)
        frame_path = os.path.join(temp_frames_folder, filename)
        imwrite_unicode(pose_path, composed)
        imwrite_unicode(frame_path, pre_img)

        # 記錄元數據
        metadata = FrameMetadata(
            filename=filename,
            frame_index=idx,
            bbox=placed_bbox,
            density=density,
            width=w0,
            height=h0,
            detection_method=detection_method
        )
        metadata_list.append(metadata)

        _prev_bbox = placed_bbox
        prev_frame_bgr = pre_img

        if StableConfig.SHOW_PROGRESS:
            print(f"  [{idx+1}/{len(file_list)}] {filename} | 密度={density:.6f} | 方法={detection_method}")

    # 保存元數據
    if StableConfig.SAVE_METADATA:
        metadata_path = os.path.join(StableConfig.TEMP_FOLDER, "pass1_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(m) for m in metadata_list], f, indent=2, ensure_ascii=False)
        print(f"\n✓ 元數據已保存: {metadata_path}")

    return metadata_list

def pass2_smooth_and_redetect(metadata_list: List[FrameMetadata]) -> List[FrameMetadata]:
    """第二遍：平滑 bbox 後重新檢測 + 時序穩定"""
    print("\n" + "=" * 60)
    print("第二遍：bbox 平滑 + 重新檢測")
    print("=" * 60)

    # 平滑 bbox 序列
    bboxes = [m.bbox for m in metadata_list]
    smoothed_bboxes = smooth_bbox_sequence(bboxes, alpha=StableConfig.PASS2_BBOX_SMOOTH_ALPHA)

    print(f"✓ bbox 序列已平滑（alpha={StableConfig.PASS2_BBOX_SMOOTH_ALPHA}）")

    # 重新檢測
    temp_poses_folder = os.path.join(StableConfig.TEMP_FOLDER, "pass2_poses")
    temp_frames_folder = os.path.join(StableConfig.TEMP_FOLDER, "pass1_frames")
    os.makedirs(temp_poses_folder, exist_ok=True)

    new_metadata_list = []
    prev_pose = None
    prev_frame = None

    for idx, (metadata, smooth_bbox) in enumerate(zip(metadata_list, smoothed_bboxes)):
        filename = metadata.filename
        frame_path = os.path.join(temp_frames_folder, filename)

        # 讀取預處理後的幀
        pre_img = imread_unicode(frame_path)
        if pre_img is None:
            continue

        # 使用平滑後的 bbox
        crop_img, placed_bbox = expand_crop(pre_img, smooth_bbox, expand_ratio=1.0,
                                           target_long_side=StableConfig.PASS2_TARGET_LONG_SIDE)

        # 手部增強
        h_roi, w_roi = crop_img.shape[:2]
        hand_bbox = (int(w_roi * 0.2), int(h_roi * 0.1), int(w_roi * 0.8), int(h_roi * 0.6))
        crop_img = enhance_hand_region(crop_img, hand_bbox)

        # DWPose 檢測
        try:
            processed_img = detector(crop_img,
                                    detect_hand=StableConfig.USE_HAND_DETECTION,
                                    detect_face=StableConfig.USE_FACE_DETECTION)
        except:
            try:
                processed_img = detector(crop_img)
            except:
                continue

        processed_cv = to_cv2_uint8_bgr(processed_img)
        if processed_cv is None:
            continue

        # 貼回
        canvas = np.zeros_like(pre_img)
        composed = paste_pose(canvas, processed_cv, placed_bbox, mask_thresh=10)
        composed = clean_pose_lines(composed, thresh=10)

        # 時序穩定（輕量）
        if StableConfig.PASS2_ENABLE_TEMPORAL and prev_pose is not None and prev_frame is not None:
            try:
                warped_prev = warp_pose_with_flow(prev_frame, pre_img, prev_pose)
                if warped_prev is not None:
                    # 輕量混合
                    temporal_weight = StableConfig.PASS2_TEMPORAL_WEIGHT
                    composed = cv2.addWeighted(composed, 1.0 - temporal_weight,
                                             warped_prev, temporal_weight, 0.0)
            except:
                pass

        # 保存
        pose_path = os.path.join(temp_poses_folder, filename)
        imwrite_unicode(pose_path, composed)

        # 更新元數據
        density = _pose_nonzero_ratio(composed, thresh=10)
        new_metadata = FrameMetadata(
            filename=filename,
            frame_index=idx,
            bbox=placed_bbox,
            density=density,
            width=metadata.width,
            height=metadata.height,
            detection_method="smooth_bbox"
        )
        new_metadata_list.append(new_metadata)

        prev_pose = composed
        prev_frame = pre_img

        if StableConfig.SHOW_PROGRESS:
            print(f"  [{idx+1}/{len(metadata_list)}] {filename} | 密度={density:.6f}")

    return new_metadata_list

def pass3_consistency_repair(metadata_list: List[FrameMetadata]):
    """第三遍：一致性修補（補洞）"""
    print("\n" + "=" * 60)
    print("第三遍：一致性修補")
    print("=" * 60)

    if not StableConfig.PASS3_ENABLE_REPAIR:
        print("跳過（PASS3_ENABLE_REPAIR=False）")
        return

    temp_poses_folder = os.path.join(StableConfig.TEMP_FOLDER, "pass2_poses")
    temp_frames_folder = os.path.join(StableConfig.TEMP_FOLDER, "pass1_frames")
    output_folder = StableConfig.OUTPUT_FOLDER

    radius = StableConfig.PASS3_REPAIR_RADIUS
    min_hole_size = StableConfig.PASS3_MIN_HOLE_SIZE

    for idx, metadata in enumerate(metadata_list):
        filename = metadata.filename
        pose_path = os.path.join(temp_poses_folder, filename)
        frame_path = os.path.join(temp_frames_folder, filename)

        curr_pose = imread_unicode(pose_path)
        curr_frame = imread_unicode(frame_path)

        if curr_pose is None or curr_frame is None:
            continue

        # 找到洞（黑色區域）
        gray = cv2.cvtColor(curr_pose, cv2.COLOR_BGR2GRAY)
        holes_mask = gray <= 10  # 黑色區域

        # 過濾小洞
        holes_mask_uint8 = holes_mask.astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(holes_mask_uint8)

        large_holes = np.zeros_like(holes_mask_uint8)
        for i in range(1, num_labels):  # 跳過背景
            if stats[i, cv2.CC_STAT_AREA] >= min_hole_size:
                large_holes[labels == i] = 255

        if np.sum(large_holes) == 0:
            # 沒有大洞，直接複製
            output_path = os.path.join(output_folder, filename)
            imwrite_unicode(output_path, curr_pose)
            if StableConfig.SHOW_PROGRESS:
                print(f"  [{idx+1}/{len(metadata_list)}] {filename} | 無需修補")
            continue

        # 收集相鄰幀進行補洞
        repaired = curr_pose.copy()

        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue

            neighbor_idx = idx + offset
            if neighbor_idx < 0 or neighbor_idx >= len(metadata_list):
                continue

            neighbor_filename = metadata_list[neighbor_idx].filename
            neighbor_pose_path = os.path.join(temp_poses_folder, neighbor_filename)
            neighbor_frame_path = os.path.join(temp_frames_folder, neighbor_filename)

            neighbor_pose = imread_unicode(neighbor_pose_path)
            neighbor_frame = imread_unicode(neighbor_frame_path)

            if neighbor_pose is None or neighbor_frame is None:
                continue

            # 光流對齊
            warped = warp_pose_with_flow(neighbor_frame, curr_frame, neighbor_pose)
            if warped is None:
                continue

            # 在洞的位置填充
            repaired[large_holes > 0] = warped[large_holes > 0]

        # 保存
        output_path = os.path.join(output_folder, filename)
        imwrite_unicode(output_path, repaired)

        if StableConfig.SHOW_PROGRESS:
            holes_pct = np.sum(large_holes > 0) / large_holes.size * 100
            print(f"  [{idx+1}/{len(metadata_list)}] {filename} | 修補了 {holes_pct:.2f}% 的洞")

# =============================================================================
# 主程序
# =============================================================================

def main():
    print("=" * 60)
    print("單人舞蹈姿態穩定捕捉系統 (三遍處理版)")
    print("=" * 60)

    # 創建臨時和輸出資料夾
    os.makedirs(StableConfig.INPUT_FOLDER, exist_ok=True)
    os.makedirs(StableConfig.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(StableConfig.TEMP_FOLDER, exist_ok=True)

    # 獲取文件列表
    if not os.path.isdir(StableConfig.INPUT_FOLDER):
        print(f"✗ 找不到輸入資料夾: {StableConfig.INPUT_FOLDER}")
        return

    file_list = sorted(os.listdir(StableConfig.INPUT_FOLDER))
    file_list = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    if not file_list:
        print("✗ 沒有找到圖像文件")
        return

    print(f"共找到 {len(file_list)} 個圖像文件")

    # 三遍處理
    metadata_pass1 = pass1_detect_all_frames(file_list)
    metadata_pass2 = pass2_smooth_and_redetect(metadata_pass1)
    pass3_consistency_repair(metadata_pass2)

    print("\n" + "=" * 60)
    print("✓ 全部處理完畢！")
    print("=" * 60)
    print(f"輸出資料夾: {StableConfig.OUTPUT_FOLDER}")
    print(f"臨時資料夾: {StableConfig.TEMP_FOLDER}")

if __name__ == "__main__":
    main()
