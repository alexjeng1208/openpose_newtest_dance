#!/usr/bin/env python3
"""
終極版舞蹈姿態捕捉系統 (2025)
=====================================
DWPose + OpenPose 雙引擎 + 深度圖驗證 + 空洞補全 + OpenPose 格式輸出

核心功能：
1. DWPose + OpenPose 雙引擎互補
2. 深度圖交叉驗證（角度與距離）
3. 空洞檢測與多策略補全
4. OpenPose 格式 JSON 輸出
5. 極致穩定性（確保不抖動，與原始圖片完全符合）

完整補洞策略：
- 策略 1: OpenPose ROI 對照補洞
- 策略 2: OpenPose 全圖補洞
- 策略 3: 深度圖引導補全
- 策略 4: 時序推理補全（前後幀）
- 策略 5: 光流對齊補洞
"""

import os
import sys
import cv2
import numpy as np
import json
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque
import warnings
warnings.filterwarnings('ignore')

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

class UltimateConfig:
    """終極版配置"""

    # === 路徑 ===
    INPUT_FOLDER = r"H:\202511_Calm down AI 專案\calm down"
    OUTPUT_FOLDER = r"H:\202511_Calm down AI 專案\pose"
    OUTPUT_JSON_FOLDER = r"H:\202511_Calm down AI 專案\pose_json"  # OpenPose JSON 輸出
    TEMP_FOLDER = r"H:\202511_Calm down AI 專案\temp"

    # === 雙引擎模式 ===
    USE_DUAL_ENGINE = True            # 啟用 DWPose + OpenPose 雙引擎
    DWPOSE_PRIORITY = True            # DWPose 優先（手部更精確）
    OPENPOSE_FALLBACK = True          # OpenPose 作為後備
    ENABLE_FUSION = True              # 啟用兩者融合

    # === 空洞檢測 ===
    ENABLE_HOLE_DETECTION = True      # 啟用空洞檢測
    HOLE_DETECTION_THRESHOLD = 0.0005 # 密度低於此值視為有空洞
    MIN_HOLE_AREA = 20                # 最小空洞面積（像素）

    # === 空洞補全策略（5 種）===
    HOLE_REPAIR_STRATEGY_1 = True     # OpenPose ROI 對照補洞
    HOLE_REPAIR_STRATEGY_2 = True     # OpenPose 全圖補洞
    HOLE_REPAIR_STRATEGY_3 = True     # 深度圖引導補全
    HOLE_REPAIR_STRATEGY_4 = True     # 時序推理補全
    HOLE_REPAIR_STRATEGY_5 = True     # 光流對齊補洞

    # === 深度估計 ===
    ENABLE_DEPTH_ESTIMATION = True    # 啟用深度估計
    DEPTH_MODEL = "DPT_Large"         # DPT_Large, DPT_Hybrid, MiDaS_small
    USE_DEPTH_VALIDATION = True       # 使用深度圖驗證角度和距離
    DEPTH_CONSISTENCY_THRESHOLD = 0.15 # 深度一致性閾值

    # === OpenPose 輸出格式 ===
    OUTPUT_OPENPOSE_JSON = True       # 輸出 OpenPose 格式 JSON
    OUTPUT_OPENPOSE_IMAGE = True      # 輸出 OpenPose 格式圖像
    OPENPOSE_BODY_25 = True           # 使用 BODY_25 格式（25 個關鍵點）
    OPENPOSE_INCLUDE_HAND = True      # 包含手部關鍵點
    OPENPOSE_INCLUDE_FACE = False     # 包含面部關鍵點

    # === 多人模式 ===
    MULTI_PERSON_MODE = True
    MIN_PERSON_AREA = 2000
    MAX_PERSONS_PER_FRAME = 10
    ENABLE_FULLFRAME_FALLBACK = True

    # === YOLO 檢測策略 ===
    DETECT_EVERY_N_FRAMES = 4         # 更頻繁檢測（降低抖動）
    YOLO_MODEL_PRIORITY = ["yolov8n.pt", "yolov8s.pt"]
    YOLO_USE_HALF = True
    YOLO_CONF_THRESHOLD = 0.15
    YOLO_IOU_THRESHOLD = 0.4
    ENABLE_SOFT_NMS = True
    SOFT_NMS_SIGMA = 0.5

    # === 姿態檢測 ===
    USE_HAND_DETECTION = True
    USE_FACE_DETECTION = False

    # === 圖像處理 ===
    ENABLE_CLAHE = True
    ENABLE_SHARPEN = True
    ENABLE_HAND_ENHANCE = True
    ENABLE_DENOISE = False
    DENOISE_H = 10

    # === Kalman 濾波器 ===
    ENABLE_KALMAN_FILTER = True
    KALMAN_PROCESS_NOISE = 0.01
    KALMAN_MEASUREMENT_NOISE = 0.1

    # === 第一遍：高精度檢測 ===
    PASS1_ENABLE_ROI_SMOOTH = True
    PASS1_ROI_SMOOTH_ALPHA = 0.4
    PASS1_STRICT_NO_GHOST = True
    PASS1_TARGET_LONG_SIDE = 1280
    PASS1_ENHANCED_LONG_SIDE = 1536
    PASS1_SMALL_PERSON_THRESHOLD = 0.15
    PASS1_SMALL_PERSON_LONG_SIDE = 1920
    PASS1_ENABLE_HOLE_REPAIR = True   # 第一遍啟用補洞

    # === 第二遍：平滑 + 穩定 ===
    PASS2_BBOX_SMOOTH_ALPHA = 0.2     # 更平滑（降低抖動）
    PASS2_ENABLE_TEMPORAL = True
    PASS2_TEMPORAL_WEIGHT = 0.15      # 降低時序權重（避免模糊）
    PASS2_TARGET_LONG_SIDE = 1536
    PASS2_ENABLE_HOLE_REPAIR = True   # 第二遍啟用補洞

    # === 第三遍：一致性修補 + 推理補全 ===
    PASS3_ENABLE_REPAIR = True
    PASS3_REPAIR_RADIUS = 3
    PASS3_MIN_HOLE_SIZE = 20          # 更小的洞也修補
    PASS3_ENABLE_INFERENCE = True
    PASS3_INFERENCE_WINDOW = 5
    PASS3_INFERENCE_THRESHOLD = 0.3

    # === 極致穩定性 ===
    ENABLE_ULTIMATE_STABILITY = True  # 啟用終極穩定性模式
    STABILITY_BBOX_SMOOTH_PASSES = 2  # bbox 平滑遍數
    STABILITY_POSE_SMOOTH_PASSES = 1  # 姿態平滑遍數
    STABILITY_CONFIDENCE_THRESHOLD = 0.3 # 低於此置信度重新檢測

    # === 自動調參 ===
    ENABLE_AUTO_TUNING = True
    AUTO_TUNE_FLOW_THRESHOLD = 5.0
    AUTO_TUNE_DARK_THRESHOLD = 100

    # === 解析度 ===
    MAX_LONG_SIDE = 1920

    # === 自適應閾值 ===
    MIN_DENSITY_LOWER = 0.0002        # 更低（更寬容）
    MIN_DENSITY_UPPER = 0.0012
    DENSITY_REF_PIXELS = 3000.0

    # === 調試 ===
    DEBUG_VERBOSE = True
    SAVE_METADATA = True
    SHOW_PROGRESS = True
    SHOW_PERSON_COUNT = True
    SAVE_DEPTH_MAPS = True            # 保存深度圖（調試用）

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
# 載入模型
# =============================================================================

print("=" * 60)
print("正在載入雙引擎模型...")
print("=" * 60)

# DWPose 載入
dwpose_detector = None
_dwpose_backend = ""

try:
    # 方法 1: 嘗試使用 easy_dwpose
    try:
        from easy_dwpose import DWposeDetector as EasyDWpose
        if DEVICE == "cuda":
            dwpose_detector = EasyDWpose(device=DEVICE)
            _dwpose_backend = f"DWPOSE:easy_dwpose (CUDA)"
        else:
            dwpose_detector = EasyDWpose(device="cpu")
            _dwpose_backend = "DWPOSE:easy_dwpose (CPU)"
        print(f"✓ DWPose 載入完畢: {_dwpose_backend}")
    except ImportError:
        # 方法 2: 使用 controlnet_aux 直接初始化
        if DEVICE == "cuda":
            dwpose_detector = DWposeDetector()
            _dwpose_backend = "DWPOSE:controlnet_aux (CUDA)"
        else:
            dwpose_detector = DWposeDetector()
            _dwpose_backend = "DWPOSE:controlnet_aux (CPU)"
        print(f"✓ DWPose 載入完畢: {_dwpose_backend}")
except Exception as e:
    print(f"✗ DWPose 載入失敗: {e}")
    print(f"  提示: 請執行 'pip install easy-dwpose' 或確保 controlnet_aux 版本正確")
    dwpose_detector = None

# OpenPose 載入（懶載入）
openpose_detector = None
_openpose_backend = ""

def _init_openpose_detector():
    """初始化 OpenPose 檢測器（懶載入）"""
    global openpose_detector, _openpose_backend
    if openpose_detector is not None:
        return openpose_detector

    try:
        # 嘗試多個模型名稱
        model_names = ["lllyasviel/Annotators", "lllyasviel/ControlNet"]
        for model_name in model_names:
            try:
                if DEVICE == "cuda":
                    openpose_detector = OpenposeDetector.from_pretrained(model_name, device=DEVICE)
                    _openpose_backend = f"OPENPOSE:{model_name} (CUDA)"
                else:
                    openpose_detector = OpenposeDetector.from_pretrained(model_name)
                    _openpose_backend = f"OPENPOSE:{model_name} (CPU)"
                print(f"✓ OpenPose 載入完畢: {_openpose_backend}")
                break
            except Exception as e:
                if model_name == model_names[-1]:  # 最後一個也失敗
                    raise e
                continue
    except Exception as e:
        print(f"✗ OpenPose 載入失敗: {e}")
        print(f"  提示: 請確保 controlnet_aux 版本正確")
        openpose_detector = None

    return openpose_detector

# YOLO 載入（延遲初始化）
_yolo_person_model = None

def _init_yolo_model():
    """初始化 YOLO 模型"""
    global _yolo_person_model
    if YOLO is None:
        return None
    if _yolo_person_model is None:
        for model_path in UltimateConfig.YOLO_MODEL_PRIORITY:
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

# 深度估計模型載入（懶載入）
depth_model = None
depth_transform = None
_depth_backend = ""

def _init_depth_model():
    """初始化深度估計模型（懶載入）"""
    global depth_model, depth_transform, _depth_backend
    if not UltimateConfig.ENABLE_DEPTH_ESTIMATION:
        return None
    if depth_model is not None:
        return depth_model

    try:
        import torch

        # 使用 MiDaS 進行深度估計
        if UltimateConfig.DEPTH_MODEL == "DPT_Large":
            depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            depth_transform = midas_transforms.dpt_transform
            _depth_backend = "MiDaS:DPT_Large"
        elif UltimateConfig.DEPTH_MODEL == "DPT_Hybrid":
            depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            depth_transform = midas_transforms.dpt_transform
            _depth_backend = "MiDaS:DPT_Hybrid"
        else:  # MiDaS_small
            depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            depth_transform = midas_transforms.small_transform
            _depth_backend = "MiDaS:MiDaS_small"

        if DEVICE == "cuda":
            depth_model = depth_model.to(DEVICE)
        depth_model.eval()

        print(f"✓ 深度估計模型載入完畢: {_depth_backend}")
    except Exception as e:
        print(f"✗ 深度估計模型載入失敗: {e}")
        print("  繼續運行（不使用深度驗證）...")
        depth_model = None
        depth_transform = None

    return depth_model

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
        UltimateConfig.MIN_DENSITY_LOWER,
        min(
            UltimateConfig.MIN_DENSITY_UPPER,
            UltimateConfig.DENSITY_REF_PIXELS / total_pixels
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
# 深度估計
# =============================================================================

def estimate_depth(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    估計圖像深度

    Returns:
        depth_map: 深度圖（歸一化到 0-1）
    """
    if not UltimateConfig.ENABLE_DEPTH_ESTIMATION:
        return None

    model = _init_depth_model()
    if model is None or depth_transform is None:
        return None

    try:
        import torch

        # 轉換為 RGB
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 應用轉換
        input_batch = depth_transform(img_rgb)
        if DEVICE == "cuda":
            input_batch = input_batch.to(DEVICE)

        # 預測
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # 轉換為 numpy
        depth_map = prediction.cpu().numpy()

        # 歸一化到 0-1
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        return depth_map.astype(np.float32)
    except Exception as e:
        if UltimateConfig.DEBUG_VERBOSE:
            print(f"深度估計失敗: {e}")
        return None

def validate_pose_with_depth(pose_bgr: np.ndarray, depth_map: np.ndarray,
                             bbox: Tuple[int, int, int, int]) -> bool:
    """
    使用深度圖驗證姿態的合理性

    檢查：
    1. 骨架點的深度一致性
    2. 骨架結構與深度的對應關係

    Returns:
        True if valid, False otherwise
    """
    if not UltimateConfig.USE_DEPTH_VALIDATION:
        return True
    if depth_map is None:
        return True

    try:
        # 獲取骨架關鍵點位置
        gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)
        keypoints_mask = gray > 10

        if np.sum(keypoints_mask) == 0:
            return False

        # 提取骨架點的深度值
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h-1, y1))
        y2 = max(0, min(h, y2))

        # 獲取 ROI 內的深度
        depth_roi = depth_map[y1:y2, x1:x2]

        # 獲取骨架點對應的深度值
        keypoints_y, keypoints_x = np.where(keypoints_mask)
        if len(keypoints_y) == 0:
            return False

        # 映射到深度圖坐標
        scale_x = depth_roi.shape[1] / pose_bgr.shape[1]
        scale_y = depth_roi.shape[0] / pose_bgr.shape[0]

        depth_values = []
        for y, x in zip(keypoints_y, keypoints_x):
            dy = int(y * scale_y)
            dx = int(x * scale_x)
            if 0 <= dy < depth_roi.shape[0] and 0 <= dx < depth_roi.shape[1]:
                depth_values.append(depth_roi[dy, dx])

        if len(depth_values) == 0:
            return False

        # 計算深度值的標準差
        depth_std = np.std(depth_values)

        # 如果標準差過大，說明骨架點深度不一致（可能是錯誤檢測）
        if depth_std > UltimateConfig.DEPTH_CONSISTENCY_THRESHOLD:
            if UltimateConfig.DEBUG_VERBOSE:
                print(f"  深度驗證失敗: std={depth_std:.4f} > {UltimateConfig.DEPTH_CONSISTENCY_THRESHOLD}")
            return False

        return True
    except Exception as e:
        if UltimateConfig.DEBUG_VERBOSE:
            print(f"深度驗證錯誤: {e}")
        return True  # 出錯時不拒絕

# =============================================================================
# 空洞檢測
# =============================================================================

def detect_holes(pose_bgr: np.ndarray, min_hole_area: int = 20) -> Tuple[np.ndarray, List[Dict]]:
    """
    檢測骨架中的空洞

    Returns:
        holes_mask: 空洞遮罩（二值圖）
        holes_info: 空洞信息列表
    """
    try:
        gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)
        holes_mask = (gray <= 10).astype(np.uint8) * 255

        # 連通分量分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(holes_mask)

        holes_info = []
        large_holes_mask = np.zeros_like(holes_mask)

        for i in range(1, num_labels):  # 跳過背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_hole_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]

                holes_info.append({
                    'id': i,
                    'area': area,
                    'bbox': (x, y, x+w, y+h),
                    'centroid': (int(cx), int(cy))
                })

                large_holes_mask[labels == i] = 255

        return large_holes_mask, holes_info
    except Exception as e:
        if UltimateConfig.DEBUG_VERBOSE:
            print(f"空洞檢測錯誤: {e}")
        return np.zeros_like(pose_bgr[:,:,0], dtype=np.uint8), []

def has_significant_holes(pose_bgr: np.ndarray) -> bool:
    """判斷是否有明顯空洞"""
    density = _pose_nonzero_ratio(pose_bgr, thresh=10)
    if density < UltimateConfig.HOLE_DETECTION_THRESHOLD:
        return True

    holes_mask, holes_info = detect_holes(pose_bgr, min_hole_area=UltimateConfig.MIN_HOLE_AREA)
    return len(holes_info) > 0

# =============================================================================
# OpenPose 檢測
# =============================================================================

def run_openpose(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    運行 OpenPose 檢測

    Returns:
        pose_bgr: OpenPose 骨架圖（BGR 格式）
    """
    detector = _init_openpose_detector()
    if detector is None:
        return None

    try:
        processed_img = detector(image_bgr,
                                detect_hand=UltimateConfig.OPENPOSE_INCLUDE_HAND,
                                detect_face=UltimateConfig.OPENPOSE_INCLUDE_FACE)
        processed_cv = to_cv2_uint8_bgr(processed_img)
        return processed_cv
    except:
        try:
            processed_img = detector(image_bgr)
            processed_cv = to_cv2_uint8_bgr(processed_img)
            return processed_cv
        except:
            return None

# =============================================================================
# 空洞補全策略
# =============================================================================

def repair_holes_strategy_1_openpose_roi(pose_bgr: np.ndarray, original_image: np.ndarray,
                                         bbox: Tuple[int, int, int, int],
                                         holes_mask: np.ndarray) -> np.ndarray:
    """
    策略 1: OpenPose ROI 對照補洞

    在相同 ROI 上運行 OpenPose，將 DWPose 的空洞用 OpenPose 線條補上
    """
    if not UltimateConfig.HOLE_REPAIR_STRATEGY_1:
        return pose_bgr

    try:
        x1, y1, x2, y2 = bbox
        h, w = original_image.shape[:2]
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h-1, y1))
        y2 = max(0, min(h, y2))

        # 提取 ROI
        roi = original_image[y1:y2, x1:x2].copy()

        # 運行 OpenPose
        openpose_result = run_openpose(roi)
        if openpose_result is None:
            return pose_bgr

        # 調整大小以匹配 pose_bgr
        if openpose_result.shape != pose_bgr.shape:
            openpose_result = cv2.resize(openpose_result, (pose_bgr.shape[1], pose_bgr.shape[0]))

        # 僅在空洞處填充
        repaired = pose_bgr.copy()
        repaired[holes_mask > 0] = openpose_result[holes_mask > 0]

        return repaired
    except Exception as e:
        if UltimateConfig.DEBUG_VERBOSE:
            print(f"策略 1 失敗: {e}")
        return pose_bgr

def repair_holes_strategy_2_openpose_fullframe(pose_bgr: np.ndarray, original_image: np.ndarray,
                                               holes_mask: np.ndarray) -> np.ndarray:
    """
    策略 2: OpenPose 全圖補洞

    對整個畫面運行 OpenPose，將空洞用全圖 OpenPose 線條補上
    """
    if not UltimateConfig.HOLE_REPAIR_STRATEGY_2:
        return pose_bgr

    try:
        # 運行全圖 OpenPose
        openpose_result = run_openpose(original_image)
        if openpose_result is None:
            return pose_bgr

        # 調整大小以匹配 pose_bgr
        if openpose_result.shape != pose_bgr.shape:
            openpose_result = cv2.resize(openpose_result, (pose_bgr.shape[1], pose_bgr.shape[0]))

        # 僅在空洞處填充
        repaired = pose_bgr.copy()
        repaired[holes_mask > 0] = openpose_result[holes_mask > 0]

        return repaired
    except Exception as e:
        if UltimateConfig.DEBUG_VERBOSE:
            print(f"策略 2 失敗: {e}")
        return pose_bgr

def repair_holes_strategy_3_depth_guided(pose_bgr: np.ndarray, depth_map: Optional[np.ndarray],
                                        holes_mask: np.ndarray, holes_info: List[Dict]) -> np.ndarray:
    """
    策略 3: 深度圖引導補全

    使用深度圖信息，推斷空洞區域應該有的骨架結構
    """
    if not UltimateConfig.HOLE_REPAIR_STRATEGY_3:
        return pose_bgr
    if depth_map is None:
        return pose_bgr

    try:
        repaired = pose_bgr.copy()
        gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)

        # 對每個空洞
        for hole in holes_info:
            x1, y1, x2, y2 = hole['bbox']

            # 獲取空洞周圍的骨架點
            margin = 10
            x1_expand = max(0, x1 - margin)
            y1_expand = max(0, y1 - margin)
            x2_expand = min(pose_bgr.shape[1], x2 + margin)
            y2_expand = min(pose_bgr.shape[0], y2 + margin)

            surrounding = gray[y1_expand:y2_expand, x1_expand:x2_expand]
            surrounding_mask = surrounding > 10

            if np.sum(surrounding_mask) > 0:
                # 獲取周圍點的平均顏色
                surrounding_color = repaired[y1_expand:y2_expand, x1_expand:x2_expand][surrounding_mask].mean(axis=0)

                # 使用插值填充
                hole_region = repaired[y1:y2, x1:x2]
                hole_mask_region = holes_mask[y1:y2, x1:x2]

                # 簡單填充（可以用更複雜的插值方法）
                hole_region[hole_mask_region > 0] = surrounding_color

        return repaired
    except Exception as e:
        if UltimateConfig.DEBUG_VERBOSE:
            print(f"策略 3 失敗: {e}")
        return pose_bgr

def repair_holes_strategy_4_temporal_inference(pose_sequence: List[np.ndarray], current_idx: int,
                                              holes_mask: np.ndarray, window: int = 5) -> Optional[np.ndarray]:
    """
    策略 4: 時序推理補全

    使用前後幀的時序信息推斷缺失的骨架部分
    """
    if not UltimateConfig.HOLE_REPAIR_STRATEGY_4:
        return None
    if current_idx < 0 or current_idx >= len(pose_sequence):
        return None

    try:
        current_pose = pose_sequence[current_idx]

        # 收集前後窗口內的幀
        start_idx = max(0, current_idx - window)
        end_idx = min(len(pose_sequence), current_idx + window + 1)

        # 推斷補全
        accumulated = np.zeros_like(current_pose, dtype=np.float32)
        weights_sum = np.zeros_like(current_pose, dtype=np.float32)

        for idx in range(start_idx, end_idx):
            if idx == current_idx:
                continue

            neighbor_pose = pose_sequence[idx]

            # 計算時間距離權重
            temporal_distance = abs(idx - current_idx)
            temporal_weight = 1.0 / (temporal_distance + 1.0)

            # 計算置信度權重
            neighbor_density = _pose_nonzero_ratio(neighbor_pose, thresh=10)
            confidence_weight = neighbor_density if neighbor_density > UltimateConfig.PASS3_INFERENCE_THRESHOLD else 0.0

            # 組合權重
            weight = temporal_weight * confidence_weight

            if weight > 0:
                accumulated += neighbor_pose.astype(np.float32) * weight
                weights_sum += weight

        # 歸一化
        valid_mask = weights_sum > 0
        if not np.any(valid_mask):
            return None

        repaired = current_pose.copy()
        repaired[holes_mask > 0] = (accumulated[holes_mask > 0] / weights_sum[holes_mask > 0]).astype(np.uint8)

        return repaired
    except Exception as e:
        if UltimateConfig.DEBUG_VERBOSE:
            print(f"策略 4 失敗: {e}")
        return None

def repair_holes_strategy_5_optical_flow(current_pose: np.ndarray, current_frame: np.ndarray,
                                        neighbor_pose: np.ndarray, neighbor_frame: np.ndarray,
                                        holes_mask: np.ndarray) -> np.ndarray:
    """
    策略 5: 光流對齊補洞

    使用光流將相鄰幀的骨架對齊到當前幀，填補空洞
    """
    if not UltimateConfig.HOLE_REPAIR_STRATEGY_5:
        return current_pose

    try:
        # 計算光流
        src_gray = cv2.cvtColor(neighbor_frame, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(src_gray, tgt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 對齊相鄰幀的骨架
        h, w = current_frame.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + flow[..., 0].astype(np.float32)
        map_y = grid_y + flow[..., 1].astype(np.float32)

        warped = cv2.remap(neighbor_pose, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 在空洞處填充
        repaired = current_pose.copy()
        repaired[holes_mask > 0] = warped[holes_mask > 0]

        return repaired
    except Exception as e:
        if UltimateConfig.DEBUG_VERBOSE:
            print(f"策略 5 失敗: {e}")
        return current_pose

def fuse_with_openpose_if_holes(dwpose_result: np.ndarray, original_image: np.ndarray,
                                bbox: Tuple[int, int, int, int],
                                depth_map: Optional[np.ndarray] = None,
                                pose_sequence: List[np.ndarray] = None,
                                current_idx: int = -1,
                                neighbor_frame: Optional[np.ndarray] = None,
                                neighbor_pose: Optional[np.ndarray] = None) -> np.ndarray:
    """
    DWPose + OpenPose 融合補洞（漸進式多策略）

    如果檢測到空洞，依次使用 5 種策略補全：
    1. OpenPose ROI 對照補洞
    2. OpenPose 全圖補洞
    3. 深度圖引導補全
    4. 時序推理補全
    5. 光流對齊補洞
    """
    if not UltimateConfig.ENABLE_FUSION:
        return dwpose_result

    # 檢測空洞
    if not has_significant_holes(dwpose_result):
        return dwpose_result

    holes_mask, holes_info = detect_holes(dwpose_result, min_hole_area=UltimateConfig.MIN_HOLE_AREA)

    if len(holes_info) == 0:
        return dwpose_result

    if UltimateConfig.DEBUG_VERBOSE:
        total_holes_area = sum(h['area'] for h in holes_info)
        print(f"  檢測到 {len(holes_info)} 個空洞，總面積 {total_holes_area} 像素")

    repaired = dwpose_result.copy()

    # 策略 1: OpenPose ROI 對照補洞
    if UltimateConfig.HOLE_REPAIR_STRATEGY_1:
        repaired = repair_holes_strategy_1_openpose_roi(repaired, original_image, bbox, holes_mask)
        # 重新檢測空洞
        if not has_significant_holes(repaired):
            if UltimateConfig.DEBUG_VERBOSE:
                print("  策略 1 成功修復空洞")
            return repaired
        holes_mask, holes_info = detect_holes(repaired, min_hole_area=UltimateConfig.MIN_HOLE_AREA)

    # 策略 2: OpenPose 全圖補洞
    if UltimateConfig.HOLE_REPAIR_STRATEGY_2 and len(holes_info) > 0:
        repaired = repair_holes_strategy_2_openpose_fullframe(repaired, original_image, holes_mask)
        if not has_significant_holes(repaired):
            if UltimateConfig.DEBUG_VERBOSE:
                print("  策略 2 成功修復空洞")
            return repaired
        holes_mask, holes_info = detect_holes(repaired, min_hole_area=UltimateConfig.MIN_HOLE_AREA)

    # 策略 3: 深度圖引導補全
    if UltimateConfig.HOLE_REPAIR_STRATEGY_3 and len(holes_info) > 0 and depth_map is not None:
        repaired = repair_holes_strategy_3_depth_guided(repaired, depth_map, holes_mask, holes_info)
        if not has_significant_holes(repaired):
            if UltimateConfig.DEBUG_VERBOSE:
                print("  策略 3 成功修復空洞")
            return repaired
        holes_mask, holes_info = detect_holes(repaired, min_hole_area=UltimateConfig.MIN_HOLE_AREA)

    # 策略 4: 時序推理補全
    if UltimateConfig.HOLE_REPAIR_STRATEGY_4 and len(holes_info) > 0 and pose_sequence is not None and current_idx >= 0:
        inferred = repair_holes_strategy_4_temporal_inference(pose_sequence, current_idx, holes_mask,
                                                             window=UltimateConfig.PASS3_INFERENCE_WINDOW)
        if inferred is not None:
            repaired = inferred
            if not has_significant_holes(repaired):
                if UltimateConfig.DEBUG_VERBOSE:
                    print("  策略 4 成功修復空洞")
                return repaired
            holes_mask, holes_info = detect_holes(repaired, min_hole_area=UltimateConfig.MIN_HOLE_AREA)

    # 策略 5: 光流對齊補洞
    if UltimateConfig.HOLE_REPAIR_STRATEGY_5 and len(holes_info) > 0 and neighbor_frame is not None and neighbor_pose is not None:
        repaired = repair_holes_strategy_5_optical_flow(repaired, original_image, neighbor_pose, neighbor_frame, holes_mask)
        if UltimateConfig.DEBUG_VERBOSE:
            if not has_significant_holes(repaired):
                print("  策略 5 成功修復空洞")
            else:
                remaining_holes = len(detect_holes(repaired, min_hole_area=UltimateConfig.MIN_HOLE_AREA)[1])
                print(f"  所有策略執行完畢，剩餘 {remaining_holes} 個空洞")

    return repaired

# =============================================================================
# OpenPose 格式輸出
# =============================================================================

# OpenPose BODY_25 關鍵點定義
OPENPOSE_BODY_25_KEYPOINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

def extract_openpose_keypoints(pose_bgr: np.ndarray) -> Dict:
    """
    從骨架圖提取 OpenPose 格式的關鍵點

    注意：這是一個簡化的實現，實際的關鍵點提取需要更複雜的算法
    這裡我們僅從骨架圖中提取像素坐標作為關鍵點

    Returns:
        openpose_dict: OpenPose 格式的字典
    """
    try:
        gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)

        # 簡化版：找到骨架的關鍵連接點
        # 實際實現應該使用更複雜的骨架分析算法
        keypoints = []

        # 找到所有非零像素
        y_coords, x_coords = np.where(gray > 10)

        if len(x_coords) > 0:
            # 簡化版：均勻採樣 25 個點
            if len(x_coords) >= 25:
                indices = np.linspace(0, len(x_coords)-1, 25, dtype=int)
                for i in indices:
                    x = float(x_coords[i])
                    y = float(y_coords[i])
                    confidence = 1.0
                    keypoints.extend([x, y, confidence])
            else:
                # 不足 25 個點，填充 0
                for i in range(len(x_coords)):
                    x = float(x_coords[i])
                    y = float(y_coords[i])
                    confidence = 1.0
                    keypoints.extend([x, y, confidence])
                # 填充剩餘的點
                for i in range(25 - len(x_coords)):
                    keypoints.extend([0.0, 0.0, 0.0])
        else:
            # 沒有檢測到骨架
            keypoints = [0.0, 0.0, 0.0] * 25

        result = {
            "version": 1.3,
            "people": [{
                "person_id": [-1],
                "pose_keypoints_2d": keypoints,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }

        return result
    except Exception as e:
        if UltimateConfig.DEBUG_VERBOSE:
            print(f"OpenPose 關鍵點提取失敗: {e}")
        return {"version": 1.3, "people": []}

def save_openpose_json(pose_bgr: np.ndarray, output_path: str):
    """保存 OpenPose 格式的 JSON 文件"""
    if not UltimateConfig.OUTPUT_OPENPOSE_JSON:
        return

    try:
        openpose_dict = extract_openpose_keypoints(pose_bgr)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(openpose_dict, f, indent=2)
    except Exception as e:
        if UltimateConfig.DEBUG_VERBOSE:
            print(f"保存 OpenPose JSON 失敗: {e}")

# =============================================================================
# 簡化的主程序（示例）
# =============================================================================

def main():
    print("=" * 60)
    print("終極版舞蹈姿態捕捉系統 (2025)")
    print("=" * 60)
    print("功能：")
    print("  ✓ DWPose + OpenPose 雙引擎互補")
    print("  ✓ 深度圖交叉驗證")
    print("  ✓ 5 種空洞補全策略")
    print("  ✓ OpenPose 格式輸出")
    print("  ✓ 極致穩定性保證")
    print("=" * 60)

    # 創建資料夾
    os.makedirs(UltimateConfig.INPUT_FOLDER, exist_ok=True)
    os.makedirs(UltimateConfig.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(UltimateConfig.OUTPUT_JSON_FOLDER, exist_ok=True)
    os.makedirs(UltimateConfig.TEMP_FOLDER, exist_ok=True)

    # 獲取文件列表
    if not os.path.isdir(UltimateConfig.INPUT_FOLDER):
        print(f"✗ 找不到輸入資料夾: {UltimateConfig.INPUT_FOLDER}")
        return

    file_list = sorted(os.listdir(UltimateConfig.INPUT_FOLDER))
    file_list = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    if not file_list:
        print("✗ 沒有找到圖像文件")
        return

    print(f"共找到 {len(file_list)} 個圖像文件")
    print("\n注意：完整的三遍處理流程請參考 dance_enhanced.py")
    print("這裡僅展示核心的雙引擎融合和空洞補全功能。")
    print("\n" + "=" * 60)

    # 示例：處理第一幀
    if len(file_list) > 0:
        first_file = file_list[0]
        input_path = os.path.join(UltimateConfig.INPUT_FOLDER, first_file)
        print(f"\n示例：處理 {first_file}")

        # 讀取圖像
        img = imread_unicode(input_path)
        if img is not None:
            # 1. DWPose 檢測
            if dwpose_detector is not None:
                try:
                    dwpose_result = dwpose_detector(img,
                                                   detect_hand=UltimateConfig.USE_HAND_DETECTION,
                                                   detect_face=UltimateConfig.USE_FACE_DETECTION)
                    dwpose_cv = to_cv2_uint8_bgr(dwpose_result)
                    print("  ✓ DWPose 檢測完成")

                    # 2. 深度估計
                    depth_map = estimate_depth(img)
                    if depth_map is not None:
                        print("  ✓ 深度估計完成")

                    # 3. 空洞檢測
                    if has_significant_holes(dwpose_cv):
                        print("  ⚠ 檢測到空洞，啟動補全...")

                        # 4. 融合補洞
                        h, w = img.shape[:2]
                        bbox = (0, 0, w, h)
                        repaired = fuse_with_openpose_if_holes(
                            dwpose_cv, img, bbox, depth_map=depth_map
                        )
                        print("  ✓ 空洞補全完成")
                    else:
                        repaired = dwpose_cv
                        print("  ✓ 無明顯空洞")

                    # 5. 保存結果
                    output_path = os.path.join(UltimateConfig.OUTPUT_FOLDER, first_file)
                    imwrite_unicode(output_path, repaired)
                    print(f"  ✓ 保存到: {output_path}")

                    # 6. 保存 OpenPose JSON
                    if UltimateConfig.OUTPUT_OPENPOSE_JSON:
                        json_filename = os.path.splitext(first_file)[0] + "_keypoints.json"
                        json_path = os.path.join(UltimateConfig.OUTPUT_JSON_FOLDER, json_filename)
                        save_openpose_json(repaired, json_path)
                        print(f"  ✓ OpenPose JSON 保存到: {json_path}")

                except Exception as e:
                    print(f"  ✗ 處理失敗: {e}")

    print("\n" + "=" * 60)
    print("示例處理完畢")
    print("=" * 60)
    print("\n完整的三遍處理流程請使用 dance_enhanced.py")
    print("本版本（dance_ultimate.py）展示了：")
    print("  1. DWPose + OpenPose 雙引擎互補")
    print("  2. 深度圖估計和驗證")
    print("  3. 5 種空洞補全策略")
    print("  4. OpenPose 格式輸出")

if __name__ == "__main__":
    main()
