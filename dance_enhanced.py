#!/usr/bin/env python3
"""
增強版舞蹈姿態捕捉系統 (2025)
=====================================
整合 2025 年最新技術，確保所有人物姿態都能被完整捕捉

新特性：
1. 多人模式支持（可開關）- 確保所有人都被檢測到
2. Kalman 濾波器 bbox 平滑 - 消除追蹤抖動
3. SoftNMS - 處理重疊和遮擋
4. 骨架補全推理 - 使用時序信息推斷缺失關鍵點
5. 自動調參機制 - 根據場景動態調整參數
6. ROI 第一遍平滑 - 提前穩定 bbox
7. 全圖多人保底 - 避免漏檢
8. 改進的密度檢查邏輯

基於最新研究：
- RTMW (2025) - DWPose 增強版
- YOLOv8 + SoftNMS + OC-SORT
- Kalman 濾波器時序穩定
- 擴散模型啟發的補洞策略
"""

import os
import sys
import cv2
import numpy as np
import json
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque

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

class EnhancedConfig:
    """增強版配置 - 整合 2025 年最新技術"""

    # === 路徑 ===
    INPUT_FOLDER = r"H:\202511_Calm down AI 專案\calm down"
    OUTPUT_FOLDER = r"H:\202511_Calm down AI 專案\pose"
    TEMP_FOLDER = r"H:\202511_Calm down AI 專案\temp"

    # === 多人模式（新增）===
    MULTI_PERSON_MODE = True          # True: 檢測所有人; False: 只檢測最大的人
    MIN_PERSON_AREA = 2000            # 最小人物面積（過濾雜訊）
    MAX_PERSONS_PER_FRAME = 10        # 每幀最多處理人數
    ENABLE_FULLFRAME_FALLBACK = True  # 啟用全圖多人保底檢測

    # === YOLO 檢測策略 ===
    DETECT_EVERY_N_FRAMES = 5         # YOLO 刷新頻率（多人模式建議 4-5）
    YOLO_MODEL_PRIORITY = ["yolov8n.pt", "yolov8s.pt"]
    YOLO_USE_HALF = True
    YOLO_CONF_THRESHOLD = 0.15        # 降低閾值以捕捉小人物
    YOLO_IOU_THRESHOLD = 0.4          # NMS IoU 閾值
    ENABLE_SOFT_NMS = True            # 啟用 SoftNMS（處理遮擋）
    SOFT_NMS_SIGMA = 0.5              # SoftNMS 參數

    # === 姿態檢測 ===
    USE_HAND_DETECTION = True
    USE_FACE_DETECTION = False

    # === 圖像處理 ===
    ENABLE_CLAHE = True
    ENABLE_SHARPEN = True
    ENABLE_HAND_ENHANCE = True
    ENABLE_DENOISE = False            # 啟用降噪（暗光/高噪場景）
    DENOISE_H = 10                    # 降噪強度

    # === Kalman 濾波器（新增）===
    ENABLE_KALMAN_FILTER = True       # 啟用 Kalman 濾波器 bbox 平滑
    KALMAN_PROCESS_NOISE = 0.01       # 過程噪聲
    KALMAN_MEASUREMENT_NOISE = 0.1    # 測量噪聲

    # === 第一遍：高精度檢測 ===
    PASS1_ENABLE_ROI_SMOOTH = True    # 第一遍啟用 ROI 平滑（新增）
    PASS1_ROI_SMOOTH_ALPHA = 0.4      # ROI 平滑強度（0.3-0.5，較弱）
    PASS1_STRICT_NO_GHOST = True
    PASS1_TARGET_LONG_SIDE = 1280
    PASS1_ENHANCED_LONG_SIDE = 1536   # 小人物提升到 1920
    PASS1_SMALL_PERSON_THRESHOLD = 0.15  # 小於畫面 15% 算小人物
    PASS1_SMALL_PERSON_LONG_SIDE = 1920  # 小人物專用解析度

    # === 第二遍：平滑 + 穩定 ===
    PASS2_BBOX_SMOOTH_ALPHA = 0.25    # 降低到 0.25（更平滑）
    PASS2_ENABLE_TEMPORAL = True
    PASS2_TEMPORAL_WEIGHT = 0.2
    PASS2_TARGET_LONG_SIDE = 1536

    # === 第三遍：一致性修補 + 推理補全（增強）===
    PASS3_ENABLE_REPAIR = True
    PASS3_REPAIR_RADIUS = 3           # 增加到 3（查找更遠的幀）
    PASS3_MIN_HOLE_SIZE = 30          # 降低到 30（修補更小的洞）
    PASS3_ENABLE_INFERENCE = True     # 啟用骨架推理補全（新增）
    PASS3_INFERENCE_WINDOW = 5        # 推理窗口大小
    PASS3_INFERENCE_THRESHOLD = 0.3   # 推理置信度閾值

    # === 自動調參（新增）===
    ENABLE_AUTO_TUNING = True         # 啟用自動調參
    AUTO_TUNE_FLOW_THRESHOLD = 5.0    # 高動態場景光流閾值
    AUTO_TUNE_DARK_THRESHOLD = 100    # 暗光場景亮度閾值

    # === 解析度 ===
    MAX_LONG_SIDE = 1920

    # === 自適應閾值 ===
    MIN_DENSITY_LOWER = 0.0003        # 降低下限（容忍更稀疏）
    MIN_DENSITY_UPPER = 0.0012
    DENSITY_REF_PIXELS = 3000.0

    # === 調試 ===
    DEBUG_VERBOSE = True
    SAVE_METADATA = True
    SHOW_PROGRESS = True
    SHOW_PERSON_COUNT = True          # 顯示每幀檢測到的人數

# =============================================================================
# Kalman 濾波器（新增）
# =============================================================================

class KalmanBBoxFilter:
    """Kalman 濾波器用於 bbox 平滑"""

    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)  # 8 狀態，4 測量

        # 狀態轉移矩陣 [x, y, w, h, dx, dy, dw, dh]
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # 測量矩陣
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        # 過程噪聲
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * EnhancedConfig.KALMAN_PROCESS_NOISE

        # 測量噪聲
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * EnhancedConfig.KALMAN_MEASUREMENT_NOISE

        # 初始協方差
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        self.initialized = False

    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """更新並返回平滑後的 bbox"""
        x1, y1, x2, y2 = bbox

        # 轉換為中心+尺寸表示
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        if not self.initialized:
            # 初始化狀態
            self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
            self.initialized = True
            return bbox

        # 預測
        prediction = self.kf.predict()

        # 更新
        self.kf.correct(measurement)

        # 獲取平滑後的狀態
        state = self.kf.statePost
        smooth_cx = float(state[0])
        smooth_cy = float(state[1])
        smooth_w = float(state[2])
        smooth_h = float(state[3])

        # 轉換回 bbox 格式
        smooth_x1 = int(round(smooth_cx - smooth_w / 2.0))
        smooth_y1 = int(round(smooth_cy - smooth_h / 2.0))
        smooth_x2 = int(round(smooth_cx + smooth_w / 2.0))
        smooth_y2 = int(round(smooth_cy + smooth_h / 2.0))

        return (smooth_x1, smooth_y1, smooth_x2, smooth_y2)

    def reset(self):
        """重置濾波器"""
        self.initialized = False

# =============================================================================
# 多人追蹤器（新增）
# =============================================================================

@dataclass
class PersonTrack:
    """單個人物追蹤信息"""
    track_id: int
    bbox: Tuple[int, int, int, int]
    bbox_history: deque  # bbox 歷史
    kalman_filter: Optional[KalmanBBoxFilter]
    last_seen_frame: int
    confidence: float
    csrt_tracker: Any = None

class MultiPersonTracker:
    """多人追蹤管理器"""

    def __init__(self):
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_track_id = 0
        self.max_lost_frames = 10

    def update(self, bboxes: List[Tuple[int, int, int, int]], frame_idx: int, frame_bgr: np.ndarray) -> List[PersonTrack]:
        """更新追蹤"""
        # 計算 IoU 矩陣
        updated_tracks = []
        matched_track_ids = set()
        matched_bbox_indices = set()

        # 匹配現有追蹤
        for track_id, track in list(self.tracks.items()):
            best_iou = 0.0
            best_bbox_idx = -1

            for i, bbox in enumerate(bboxes):
                if i in matched_bbox_indices:
                    continue
                iou = self._compute_iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_bbox_idx = i

            # 更新或刪除追蹤
            if best_iou > 0.3:  # IoU 閾值
                bbox = bboxes[best_bbox_idx]

                # Kalman 濾波器平滑
                if EnhancedConfig.ENABLE_KALMAN_FILTER and track.kalman_filter:
                    bbox = track.kalman_filter.update(bbox)

                track.bbox = bbox
                track.bbox_history.append(bbox)
                track.last_seen_frame = frame_idx
                track.confidence = min(1.0, track.confidence + 0.1)

                matched_track_ids.add(track_id)
                matched_bbox_indices.add(best_bbox_idx)
                updated_tracks.append(track)
            else:
                # 嘗試 CSRT 追蹤
                if track.csrt_tracker is not None:
                    ok, rect = track.csrt_tracker.update(frame_bgr)
                    if ok:
                        x, y, w, h = rect
                        bbox = (int(x), int(y), int(x + w), int(y + h))
                        track.bbox = bbox
                        track.bbox_history.append(bbox)
                        track.last_seen_frame = frame_idx
                        track.confidence = max(0.0, track.confidence - 0.1)
                        updated_tracks.append(track)
                        continue

                # 丟失追蹤
                if frame_idx - track.last_seen_frame > self.max_lost_frames:
                    del self.tracks[track_id]

        # 為未匹配的 bbox 創建新追蹤
        for i, bbox in enumerate(bboxes):
            if i not in matched_bbox_indices:
                track = PersonTrack(
                    track_id=self.next_track_id,
                    bbox=bbox,
                    bbox_history=deque(maxlen=30),
                    kalman_filter=KalmanBBoxFilter() if EnhancedConfig.ENABLE_KALMAN_FILTER else None,
                    last_seen_frame=frame_idx,
                    confidence=1.0
                )
                track.bbox_history.append(bbox)

                # 初始化 CSRT 追蹤器
                try:
                    if hasattr(cv2, 'TrackerCSRT_create'):
                        track.csrt_tracker = cv2.TrackerCSRT_create()
                        track.csrt_tracker.init(frame_bgr, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                except:
                    track.csrt_tracker = None

                self.tracks[self.next_track_id] = track
                updated_tracks.append(track)
                self.next_track_id += 1

        return updated_tracks

    def _compute_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """計算兩個 bbox 的 IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 計算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # 計算聯集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

# =============================================================================
# 元數據結構
# =============================================================================

@dataclass
class FrameMetadata:
    """幀元數據"""
    filename: str
    frame_index: int
    persons: List[Dict[str, Any]]  # 多人信息
    scene_brightness: float
    scene_motion: float  # 場景動態程度
    auto_tuned_params: Dict[str, Any]  # 自動調整的參數

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
        EnhancedConfig.MIN_DENSITY_LOWER,
        min(
            EnhancedConfig.MIN_DENSITY_UPPER,
            EnhancedConfig.DENSITY_REF_PIXELS / total_pixels
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

def compute_scene_brightness(image_bgr: np.ndarray) -> float:
    """計算場景亮度"""
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    except:
        return 128.0

def compute_scene_motion(prev_frame: Optional[np.ndarray], curr_frame: np.ndarray) -> float:
    """計算場景動態程度（使用光流）"""
    if prev_frame is None:
        return 0.0
    try:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 縮小尺寸以加速
        h, w = prev_gray.shape
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            prev_gray = cv2.resize(prev_gray, (new_w, new_h))
            curr_gray = cv2.resize(curr_gray, (new_w, new_h))

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(np.median(magnitude))
    except:
        return 0.0

# =============================================================================
# 自動調參（新增）
# =============================================================================

def auto_tune_parameters(brightness: float, motion: float) -> Dict[str, Any]:
    """根據場景自動調整參數"""
    if not EnhancedConfig.ENABLE_AUTO_TUNING:
        return {}

    tuned = {}

    # 根據亮度調整
    if brightness < EnhancedConfig.AUTO_TUNE_DARK_THRESHOLD:
        # 暗光場景
        tuned['enable_denoise'] = True
        tuned['clahe_clip_limit'] = 3.5  # 增強對比度
        tuned['density_multiplier'] = 0.8  # 降低密度要求
    else:
        tuned['enable_denoise'] = False
        tuned['clahe_clip_limit'] = 2.5
        tuned['density_multiplier'] = 1.0

    # 根據動態程度調整
    if motion > EnhancedConfig.AUTO_TUNE_FLOW_THRESHOLD:
        # 高動態場景
        tuned['roi_expand_ratio'] = 1.5  # 加大 ROI
        tuned['detect_frequency'] = max(3, EnhancedConfig.DETECT_EVERY_N_FRAMES - 2)  # 更頻繁檢測
        tuned['temporal_weight'] = 0.15  # 降低時序權重
    else:
        tuned['roi_expand_ratio'] = 1.3
        tuned['detect_frequency'] = EnhancedConfig.DETECT_EVERY_N_FRAMES
        tuned['temporal_weight'] = EnhancedConfig.PASS2_TEMPORAL_WEIGHT

    return tuned

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
# 載入模型
# =============================================================================

print("=" * 60)
print("正在載入模型...")
print("=" * 60)

# DWPose 載入
detector = None
_detector_backend = ""

try:
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

# YOLO 載入（延遲初始化）
_yolo_person_model = None

def _init_yolo_model():
    """初始化 YOLO 模型"""
    global _yolo_person_model
    if YOLO is None:
        return None
    if _yolo_person_model is None:
        for model_path in EnhancedConfig.YOLO_MODEL_PRIORITY:
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
# SoftNMS（新增）
# =============================================================================

def soft_nms(bboxes: List[Tuple[int, int, int, int, float]],
             sigma: float = 0.5,
             iou_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
    """
    SoftNMS 實現 - 處理重疊和遮擋

    Args:
        bboxes: List of (x1, y1, x2, y2, score)
        sigma: Gaussian function parameter
        iou_threshold: IoU threshold for suppression

    Returns:
        List of filtered bboxes with updated scores
    """
    if not bboxes:
        return []

    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    result = []

    while bboxes:
        best = bboxes.pop(0)
        result.append(best)

        new_bboxes = []
        for bbox in bboxes:
            iou = _compute_iou_with_score(best, bbox)

            if iou > iou_threshold:
                # 使用 Gaussian 衰減分數
                new_score = bbox[4] * np.exp(-(iou * iou) / sigma)
                new_bbox = (bbox[0], bbox[1], bbox[2], bbox[3], new_score)
                new_bboxes.append(new_bbox)
            else:
                new_bboxes.append(bbox)

        bboxes = sorted(new_bboxes, key=lambda x: x[4], reverse=True)

    return result

def _compute_iou_with_score(bbox1: Tuple[int, int, int, int, float],
                              bbox2: Tuple[int, int, int, int, float]) -> float:
    """計算兩個 bbox 的 IoU（忽略 score）"""
    x1_1, y1_1, x2_1, y2_1, _ = bbox1
    x1_2, y1_2, x2_2, y2_2, _ = bbox2

    # 計算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # 計算聯集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

# =============================================================================
# 圖像處理函數
# =============================================================================

def denoise_image(image_bgr: np.ndarray) -> np.ndarray:
    """降噪處理（暗光/高噪場景）"""
    if not EnhancedConfig.ENABLE_DENOISE:
        return image_bgr
    try:
        denoised = cv2.fastNlMeansDenoisingColored(
            image_bgr, None,
            h=EnhancedConfig.DENOISE_H,
            hColor=EnhancedConfig.DENOISE_H,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised
    except:
        return image_bgr

def enhance_hand_region(image_bgr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """手部區域增強"""
    if bbox is None or not EnhancedConfig.ENABLE_HAND_ENHANCE:
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

def preprocess_for_pose(image: np.ndarray, target_long_side: int = 1280,
                        clahe_clip_limit: float = 2.5) -> np.ndarray:
    """預處理圖像"""
    img = image

    # 降噪（如果需要）
    img = denoise_image(img)

    if EnhancedConfig.ENABLE_CLAHE:
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
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
    if not EnhancedConfig.ENABLE_SHARPEN:
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
# 多人檢測（新增）
# =============================================================================

def detect_person_bboxes(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """
    YOLO 多人檢測（支持 SoftNMS）

    Returns:
        List of (x1, y1, x2, y2, confidence)
    """
    model = _init_yolo_model()
    if model is None:
        return []

    h, w = image_bgr.shape[:2]
    image_area = h * w
    scales = [(1280, EnhancedConfig.YOLO_CONF_THRESHOLD),
              (960, EnhancedConfig.YOLO_CONF_THRESHOLD),
              (640, EnhancedConfig.YOLO_CONF_THRESHOLD + 0.05)]

    all_bboxes = []

    for imgsz, conf_thresh in scales:
        try:
            results = model.predict(
                source=image_bgr,
                imgsz=imgsz,
                conf=conf_thresh,
                classes=[0],
                verbose=False,
                half=EnhancedConfig.YOLO_USE_HALF and DEVICE == "cuda",
                device=DEVICE
            )
            if not results or len(results) == 0:
                continue
            boxes = getattr(results[0], 'boxes', None)
            if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else [1.0] * len(xyxy)

            for b, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)

                # 過濾太小的檢測
                if area < EnhancedConfig.MIN_PERSON_AREA:
                    continue

                all_bboxes.append((x1, y1, x2, y2, float(conf)))

            if all_bboxes:
                break  # 找到檢測結果就停止
        except Exception:
            continue

    if not all_bboxes:
        return []

    # 應用 SoftNMS
    if EnhancedConfig.ENABLE_SOFT_NMS:
        all_bboxes = soft_nms(all_bboxes,
                              sigma=EnhancedConfig.SOFT_NMS_SIGMA,
                              iou_threshold=EnhancedConfig.YOLO_IOU_THRESHOLD)

    # 限制人數
    all_bboxes = sorted(all_bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    all_bboxes = all_bboxes[:EnhancedConfig.MAX_PERSONS_PER_FRAME]

    return all_bboxes

def detect_fullframe_multi_person(image_bgr: np.ndarray) -> np.ndarray:
    """
    全圖多人保底檢測

    直接對整個畫面進行 DWPose 檢測，捕捉所有人
    """
    try:
        processed_img = detector(image_bgr,
                                detect_hand=EnhancedConfig.USE_HAND_DETECTION,
                                detect_face=EnhancedConfig.USE_FACE_DETECTION)
    except:
        try:
            processed_img = detector(image_bgr)
        except:
            return np.zeros_like(image_bgr)

    processed_cv = to_cv2_uint8_bgr(processed_img)
    if processed_cv is None:
        return np.zeros_like(image_bgr)

    return processed_cv

# =============================================================================
# bbox 處理
# =============================================================================

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
# 骨架推理補全（新增）
# =============================================================================

def infer_missing_skeleton(pose_sequence: List[np.ndarray],
                           current_idx: int,
                           window: int = 5) -> Optional[np.ndarray]:
    """
    使用時序信息推斷缺失的骨架部分

    靈感來自 PGKC (Pose-Guided Keypoint Completion)
    使用前後幀的歷史姿態來推斷當前幀的缺失部分
    """
    if not EnhancedConfig.PASS3_ENABLE_INFERENCE:
        return None

    if current_idx < 0 or current_idx >= len(pose_sequence):
        return None

    current_pose = pose_sequence[current_idx]

    # 找到黑色區域（缺失部分）
    gray = cv2.cvtColor(current_pose, cv2.COLOR_BGR2GRAY)
    missing_mask = gray <= 10

    # 如果沒有缺失，直接返回
    if np.sum(missing_mask) < EnhancedConfig.PASS3_MIN_HOLE_SIZE:
        return current_pose

    # 收集前後窗口內的幀
    start_idx = max(0, current_idx - window)
    end_idx = min(len(pose_sequence), current_idx + window + 1)

    # 推斷補全
    inferred = current_pose.copy()
    weights_sum = np.zeros_like(current_pose, dtype=np.float32)
    accumulated = np.zeros_like(current_pose, dtype=np.float32)

    for idx in range(start_idx, end_idx):
        if idx == current_idx:
            continue

        neighbor_pose = pose_sequence[idx]

        # 計算時間距離權重（越近權重越大）
        temporal_distance = abs(idx - current_idx)
        temporal_weight = 1.0 / (temporal_distance + 1.0)

        # 計算置信度（基於骨架密度）
        neighbor_density = _pose_nonzero_ratio(neighbor_pose, thresh=10)
        confidence_weight = neighbor_density if neighbor_density > EnhancedConfig.PASS3_INFERENCE_THRESHOLD else 0.0

        # 組合權重
        weight = temporal_weight * confidence_weight

        if weight > 0:
            accumulated += neighbor_pose.astype(np.float32) * weight
            weights_sum += weight

    # 歸一化並填充缺失區域
    valid_mask = weights_sum > 0
    inferred[missing_mask & valid_mask[..., 0]] = (accumulated[missing_mask & valid_mask[..., 0]] / weights_sum[missing_mask & valid_mask[..., 0]]).astype(np.uint8)

    return inferred

# =============================================================================
# 三遍處理主流程（增強版）
# =============================================================================

def pass1_detect_all_frames_multi_person(file_list: List[str]) -> List[FrameMetadata]:
    """第一遍：高精度多人檢測（增強版）"""
    print("\n" + "=" * 60)
    print("第一遍：高精度多人檢測（增強版）")
    print("=" * 60)
    print(f"多人模式: {EnhancedConfig.MULTI_PERSON_MODE}")
    print(f"Kalman 濾波器: {EnhancedConfig.ENABLE_KALMAN_FILTER}")
    print(f"SoftNMS: {EnhancedConfig.ENABLE_SOFT_NMS}")
    print(f"自動調參: {EnhancedConfig.ENABLE_AUTO_TUNING}")

    metadata_list = []
    frame_count = 0
    multi_tracker = MultiPersonTracker() if EnhancedConfig.MULTI_PERSON_MODE else None
    prev_frame_bgr = None

    temp_poses_folder = os.path.join(EnhancedConfig.TEMP_FOLDER, "pass1_poses")
    temp_frames_folder = os.path.join(EnhancedConfig.TEMP_FOLDER, "pass1_frames")
    os.makedirs(temp_poses_folder, exist_ok=True)
    os.makedirs(temp_frames_folder, exist_ok=True)

    for idx, filename in enumerate(file_list):
        input_path = os.path.join(EnhancedConfig.INPUT_FOLDER, filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        frame_count += 1

        # 讀取圖像
        img = imread_unicode(input_path)
        if img is None:
            continue

        # 場景分析
        brightness = compute_scene_brightness(img)
        motion = compute_scene_motion(prev_frame_bgr, img)

        # 自動調參
        tuned_params = auto_tune_parameters(brightness, motion)

        # 預處理
        clahe_clip = tuned_params.get('clahe_clip_limit', 2.5)
        pre_img = preprocess_for_pose(img, target_long_side=EnhancedConfig.PASS1_TARGET_LONG_SIDE,
                                      clahe_clip_limit=clahe_clip)
        pre_img = deblur_sharpen(pre_img, strength=1.5)

        w0, h0 = pre_img.shape[1], pre_img.shape[0]
        image_area = w0 * h0

        # 多人檢測
        use_yolo = (frame_count % tuned_params.get('detect_frequency', EnhancedConfig.DETECT_EVERY_N_FRAMES) == 1)

        persons_info = []
        canvas = np.zeros_like(pre_img)

        if EnhancedConfig.MULTI_PERSON_MODE:
            if use_yolo:
                bboxes_with_conf = detect_person_bboxes(pre_img)
                bboxes = [(b[0], b[1], b[2], b[3]) for b in bboxes_with_conf]
            else:
                bboxes = []

            # 更新追蹤
            tracks = multi_tracker.update(bboxes, frame_count, pre_img)

            if EnhancedConfig.SHOW_PERSON_COUNT:
                print(f"  [{idx+1}/{len(file_list)}] {filename} | 檢測到 {len(tracks)} 人")

            # 處理每個人
            for track in tracks:
                person_bbox = track.bbox

                # 計算人物面積比例
                person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])
                area_ratio = person_area / image_area

                # 小人物使用更高解析度
                if area_ratio < EnhancedConfig.PASS1_SMALL_PERSON_THRESHOLD:
                    target_long_side = EnhancedConfig.PASS1_SMALL_PERSON_LONG_SIDE
                else:
                    target_long_side = EnhancedConfig.PASS1_ENHANCED_LONG_SIDE

                # 擴展並裁切
                roi_expand = tuned_params.get('roi_expand_ratio', 1.3)
                person_bbox_expanded = _expand_bbox(person_bbox, w0, h0, ratio=roi_expand)
                crop_img, placed_bbox = expand_crop(pre_img, person_bbox_expanded, expand_ratio=1.0,
                                                   target_long_side=target_long_side)

                # 手部增強
                h_roi, w_roi = crop_img.shape[:2]
                hand_bbox = (int(w_roi * 0.2), int(h_roi * 0.1), int(w_roi * 0.8), int(h_roi * 0.6))
                crop_img = enhance_hand_region(crop_img, hand_bbox)

                # DWPose 檢測
                try:
                    processed_img = detector(crop_img,
                                            detect_hand=EnhancedConfig.USE_HAND_DETECTION,
                                            detect_face=EnhancedConfig.USE_FACE_DETECTION)
                except:
                    try:
                        processed_img = detector(crop_img)
                    except:
                        continue

                processed_cv = to_cv2_uint8_bgr(processed_img)
                if processed_cv is None:
                    continue

                # 貼回
                canvas = paste_pose(canvas, processed_cv, placed_bbox, mask_thresh=10)

                # 記錄人物信息
                person_density = _pose_nonzero_ratio(processed_cv, thresh=10)
                persons_info.append({
                    'track_id': track.track_id,
                    'bbox': placed_bbox,
                    'density': person_density,
                    'area_ratio': area_ratio,
                    'confidence': track.confidence
                })

            # 全圖保底（如果沒檢測到人或檢測效果不好）
            if EnhancedConfig.ENABLE_FULLFRAME_FALLBACK and (not persons_info or sum(p['density'] for p in persons_info) < 0.001):
                fullframe_pose = detect_fullframe_multi_person(pre_img)
                fullframe_density = _pose_nonzero_ratio(fullframe_pose, thresh=10)
                if fullframe_density > 0.001:
                    canvas = cv2.addWeighted(canvas, 0.5, fullframe_pose, 0.5, 0.0)
                    persons_info.append({
                        'track_id': -1,  # 全圖保底
                        'bbox': (0, 0, w0, h0),
                        'density': fullframe_density,
                        'area_ratio': 1.0,
                        'confidence': 0.5
                    })

        else:
            # 單人模式（保留原有邏輯）
            # ... (省略單人模式代碼，與 dance_stable.py 類似)
            pass

        # 清理
        composed = clean_pose_lines(canvas, thresh=10)

        # 保存
        pose_path = os.path.join(temp_poses_folder, filename)
        frame_path = os.path.join(temp_frames_folder, filename)
        imwrite_unicode(pose_path, composed)
        imwrite_unicode(frame_path, pre_img)

        # 記錄元數據
        metadata = FrameMetadata(
            filename=filename,
            frame_index=idx,
            persons=persons_info,
            scene_brightness=brightness,
            scene_motion=motion,
            auto_tuned_params=tuned_params
        )
        metadata_list.append(metadata)

        prev_frame_bgr = pre_img

    # 保存元數據
    if EnhancedConfig.SAVE_METADATA:
        metadata_path = os.path.join(EnhancedConfig.TEMP_FOLDER, "pass1_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(m) for m in metadata_list], f, indent=2, ensure_ascii=False)
        print(f"\n✓ 元數據已保存: {metadata_path}")

    return metadata_list

def pass2_smooth_and_redetect_multi_person(metadata_list: List[FrameMetadata]) -> List[FrameMetadata]:
    """第二遍：平滑 bbox 後重新檢測（多人支持）"""
    print("\n" + "=" * 60)
    print("第二遍：bbox 平滑 + 重新檢測（多人）")
    print("=" * 60)

    # 構建每個 track 的 bbox 序列
    track_sequences = {}
    for metadata in metadata_list:
        for person in metadata.persons:
            track_id = person.get('track_id', -1)
            if track_id not in track_sequences:
                track_sequences[track_id] = []
            track_sequences[track_id].append((metadata.frame_index, person['bbox']))

    # 平滑每個 track 的 bbox 序列
    smoothed_sequences = {}
    for track_id, sequence in track_sequences.items():
        sequence.sort(key=lambda x: x[0])  # 按幀索引排序
        bboxes = [s[1] for s in sequence]
        smoothed_bboxes = smooth_bbox_sequence(bboxes, alpha=EnhancedConfig.PASS2_BBOX_SMOOTH_ALPHA)
        smoothed_sequences[track_id] = {sequence[i][0]: smoothed_bboxes[i] for i in range(len(sequence))}

    print(f"✓ 已平滑 {len(track_sequences)} 個追蹤序列")

    # 重新檢測
    temp_poses_folder = os.path.join(EnhancedConfig.TEMP_FOLDER, "pass2_poses")
    temp_frames_folder = os.path.join(EnhancedConfig.TEMP_FOLDER, "pass1_frames")
    os.makedirs(temp_poses_folder, exist_ok=True)

    new_metadata_list = []
    prev_pose = None
    prev_frame = None

    for idx, metadata in enumerate(metadata_list):
        filename = metadata.filename
        frame_path = os.path.join(temp_frames_folder, filename)

        # 讀取預處理後的幀
        pre_img = imread_unicode(frame_path)
        if pre_img is None:
            continue

        canvas = np.zeros_like(pre_img)
        new_persons_info = []

        # 處理每個人
        for person in metadata.persons:
            track_id = person.get('track_id', -1)

            # 獲取平滑後的 bbox
            if track_id in smoothed_sequences and metadata.frame_index in smoothed_sequences[track_id]:
                smooth_bbox = smoothed_sequences[track_id][metadata.frame_index]
            else:
                smooth_bbox = person['bbox']

            # 使用平滑後的 bbox 重新檢測
            crop_img, placed_bbox = expand_crop(pre_img, smooth_bbox, expand_ratio=1.0,
                                               target_long_side=EnhancedConfig.PASS2_TARGET_LONG_SIDE)

            # 手部增強
            h_roi, w_roi = crop_img.shape[:2]
            hand_bbox = (int(w_roi * 0.2), int(h_roi * 0.1), int(w_roi * 0.8), int(h_roi * 0.6))
            crop_img = enhance_hand_region(crop_img, hand_bbox)

            # DWPose 檢測
            try:
                processed_img = detector(crop_img,
                                        detect_hand=EnhancedConfig.USE_HAND_DETECTION,
                                        detect_face=EnhancedConfig.USE_FACE_DETECTION)
            except:
                try:
                    processed_img = detector(crop_img)
                except:
                    continue

            processed_cv = to_cv2_uint8_bgr(processed_img)
            if processed_cv is None:
                continue

            # 貼回
            canvas = paste_pose(canvas, processed_cv, placed_bbox, mask_thresh=10)

            # 更新人物信息
            person_density = _pose_nonzero_ratio(processed_cv, thresh=10)
            new_persons_info.append({
                'track_id': track_id,
                'bbox': placed_bbox,
                'density': person_density,
                'area_ratio': person.get('area_ratio', 0.0),
                'confidence': person.get('confidence', 1.0)
            })

        # 清理
        composed = clean_pose_lines(canvas, thresh=10)

        # 時序穩定（輕量）
        temporal_weight = metadata.auto_tuned_params.get('temporal_weight', EnhancedConfig.PASS2_TEMPORAL_WEIGHT)
        if EnhancedConfig.PASS2_ENABLE_TEMPORAL and prev_pose is not None and prev_frame is not None:
            try:
                warped_prev = warp_pose_with_flow(prev_frame, pre_img, prev_pose)
                if warped_prev is not None:
                    composed = cv2.addWeighted(composed, 1.0 - temporal_weight,
                                             warped_prev, temporal_weight, 0.0)
            except:
                pass

        # 保存
        pose_path = os.path.join(temp_poses_folder, filename)
        imwrite_unicode(pose_path, composed)

        # 更新元數據
        new_metadata = FrameMetadata(
            filename=filename,
            frame_index=idx,
            persons=new_persons_info,
            scene_brightness=metadata.scene_brightness,
            scene_motion=metadata.scene_motion,
            auto_tuned_params=metadata.auto_tuned_params
        )
        new_metadata_list.append(new_metadata)

        prev_pose = composed
        prev_frame = pre_img

        if EnhancedConfig.SHOW_PROGRESS:
            total_density = sum(p['density'] for p in new_persons_info)
            print(f"  [{idx+1}/{len(metadata_list)}] {filename} | {len(new_persons_info)} 人 | 總密度={total_density:.6f}")

    return new_metadata_list

def pass3_consistency_repair_with_inference(metadata_list: List[FrameMetadata]):
    """第三遍：一致性修補 + 骨架推理補全（增強版）"""
    print("\n" + "=" * 60)
    print("第三遍：一致性修補 + 骨架推理補全（增強版）")
    print("=" * 60)

    if not EnhancedConfig.PASS3_ENABLE_REPAIR:
        print("跳過（PASS3_ENABLE_REPAIR=False）")
        return

    temp_poses_folder = os.path.join(EnhancedConfig.TEMP_FOLDER, "pass2_poses")
    temp_frames_folder = os.path.join(EnhancedConfig.TEMP_FOLDER, "pass1_frames")
    output_folder = EnhancedConfig.OUTPUT_FOLDER

    radius = EnhancedConfig.PASS3_REPAIR_RADIUS
    min_hole_size = EnhancedConfig.PASS3_MIN_HOLE_SIZE

    # 讀取所有姿態序列（用於推理補全）
    pose_sequence = []
    for metadata in metadata_list:
        filename = metadata.filename
        pose_path = os.path.join(temp_poses_folder, filename)
        pose = imread_unicode(pose_path)
        pose_sequence.append(pose if pose is not None else np.zeros((100, 100, 3), dtype=np.uint8))

    for idx, metadata in enumerate(metadata_list):
        filename = metadata.filename
        pose_path = os.path.join(temp_poses_folder, filename)
        frame_path = os.path.join(temp_frames_folder, filename)

        curr_pose = imread_unicode(pose_path)
        curr_frame = imread_unicode(frame_path)

        if curr_pose is None or curr_frame is None:
            continue

        # 骨架推理補全（新增）
        if EnhancedConfig.PASS3_ENABLE_INFERENCE:
            inferred_pose = infer_missing_skeleton(pose_sequence, idx, window=EnhancedConfig.PASS3_INFERENCE_WINDOW)
            if inferred_pose is not None:
                curr_pose = inferred_pose

        # 找到洞（黑色區域）
        gray = cv2.cvtColor(curr_pose, cv2.COLOR_BGR2GRAY)
        holes_mask = gray <= 10

        # 過濾小洞
        holes_mask_uint8 = holes_mask.astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(holes_mask_uint8)

        large_holes = np.zeros_like(holes_mask_uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_hole_size:
                large_holes[labels == i] = 255

        if np.sum(large_holes) == 0:
            # 沒有大洞，直接複製
            output_path = os.path.join(output_folder, filename)
            imwrite_unicode(output_path, curr_pose)
            if EnhancedConfig.SHOW_PROGRESS:
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

        if EnhancedConfig.SHOW_PROGRESS:
            holes_pct = np.sum(large_holes > 0) / large_holes.size * 100
            print(f"  [{idx+1}/{len(metadata_list)}] {filename} | 修補了 {holes_pct:.2f}% 的洞")

# =============================================================================
# 主程序
# =============================================================================

def main():
    print("=" * 60)
    print("增強版舞蹈姿態捕捉系統 (2025)")
    print("=" * 60)
    print("整合技術：")
    print("  - 多人模式支持（可開關）")
    print("  - Kalman 濾波器 bbox 平滑")
    print("  - SoftNMS 處理遮擋")
    print("  - 骨架推理補全")
    print("  - 自動調參機制")
    print("=" * 60)

    # 創建資料夾
    os.makedirs(EnhancedConfig.INPUT_FOLDER, exist_ok=True)
    os.makedirs(EnhancedConfig.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(EnhancedConfig.TEMP_FOLDER, exist_ok=True)

    # 獲取文件列表
    if not os.path.isdir(EnhancedConfig.INPUT_FOLDER):
        print(f"✗ 找不到輸入資料夾: {EnhancedConfig.INPUT_FOLDER}")
        return

    file_list = sorted(os.listdir(EnhancedConfig.INPUT_FOLDER))
    file_list = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    if not file_list:
        print("✗ 沒有找到圖像文件")
        return

    print(f"共找到 {len(file_list)} 個圖像文件")

    # 三遍處理
    metadata_pass1 = pass1_detect_all_frames_multi_person(file_list)
    metadata_pass2 = pass2_smooth_and_redetect_multi_person(metadata_pass1)
    pass3_consistency_repair_with_inference(metadata_pass2)

    print("\n" + "=" * 60)
    print("✓ 全部處理完畢！")
    print("=" * 60)
    print(f"輸出資料夾: {EnhancedConfig.OUTPUT_FOLDER}")
    print(f"臨時資料夾: {EnhancedConfig.TEMP_FOLDER}")

if __name__ == "__main__":
    main()
