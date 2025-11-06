#!/usr/bin/env python3
"""
多人姿態檢測增強系統
========================
高級視頻序列幀姿態偵測系統，專門優化多人場景檢測。

主要功能:
- 多人姿態偵測 (支持所有人物，不只最大框)
- DWPose 原生多人檢測模式
- 手部細節強化
- 人物追蹤 (YOLO + CSRT + 光流)
- 時序穩定化
- GPU 加速支持

研究基礎 (2024-2025):
- DWPose: SOTA whole-body pose estimation (ICCV 2023)
- RTMO: Real-time one-stage multi-person pose estimation
- YOLO11 Pose: Latest production standard for 2025
"""

import os
import sys
import cv2
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

from controlnet_aux import DWposeDetector, OpenposeDetector

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# =============================================================================
# 常量定義
# =============================================================================

class Constants:
    """全局常量定義"""

    # 圖像處理相關
    DEFAULT_TARGET_LONG_SIDE = 1280
    ENHANCED_TARGET_LONG_SIDE = 1536
    MAX_TARGET_LONG_SIDE = 1920
    ULTRA_TARGET_LONG_SIDE = 2048  # 超高解析度模式

    # CLAHE 參數
    CLAHE_CLIP_LIMIT = 2.5
    CLAHE_TILE_GRID_SIZE = (8, 8)
    HAND_CLAHE_CLIP_LIMIT = 3.0
    HAND_CLAHE_TILE_GRID_SIZE = (4, 4)

    # 銳化參數
    DEFAULT_SHARPEN_STRENGTH = 1.5
    HAND_SHARPEN_STRENGTH = 1.2

    # YOLO 檢測參數 - 多人優化
    YOLO_SCALES = [(1280, 0.2), (960, 0.15), (640, 0.1)]  # (解析度, 置信度)
    YOLO_PERSON_CLASS = 0
    YOLO_MAX_DET = 1000  # 增加最大檢測數以支持多人場景
    YOLO_IOU_THRESH = 0.45  # 降低 IOU 閾值避免過度 NMS
    YOLO_CONF_THRESH = 0.1  # 最低置信度，用於小人物檢測

    # 多人檢測參數
    MIN_PERSON_AREA = 400  # 最小人物面積（像素），過濾噪聲
    MIN_PERSON_RATIO = 0.3  # 最小寬高比
    MAX_PERSON_RATIO = 5.0  # 最大寬高比

    # ROI 擴展參數
    DEFAULT_EXPAND_RATIO = 1.3
    RETRY_EXPAND_RATIO = 1.6

    # 骨架檢測參數
    MASK_THRESH = 10
    POST_CLEAN_KERNEL_SIZE = (3, 3)
    MIN_POSE_DENSITY_BASE = 0.0008  # 基礎密度閾值

    # 自適應密度閾值參數
    MIN_DENSITY_LOWER_BOUND = 0.0004  # 最小密度下界
    MIN_DENSITY_UPPER_BOUND = 0.0012  # 最小密度上界
    DENSITY_REFERENCE_PIXELS = 3000.0  # 參考像素數

    # 手部區域估計 (相對於人物 bbox)
    HAND_REGION_X_START = 0.2
    HAND_REGION_X_END = 0.8
    HAND_REGION_Y_START = 0.1
    HAND_REGION_Y_END = 0.6

    # 光流參數
    FLOW_PYR_SCALE = 0.5
    FLOW_LEVELS = 3
    FLOW_WINSIZE = 15
    FLOW_ITERATIONS = 3
    FLOW_POLY_N = 5
    FLOW_POLY_SIGMA = 1.2

    # 自適應參數 EMA
    EMA_ALPHA = 0.2

    # 圖像文件擴展名
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')


# =============================================================================
# 配置類
# =============================================================================

@dataclass
class Config:
    """處理配置"""

    # 路徑配置
    input_folder: str = ""
    output_folder: str = ""

    # 模型配置
    detector_type: str = "dwpose"  # dwpose 或 openpose
    yolo_model: str = "yolov8s.pt"
    device: str = "auto"  # auto, cpu, cuda, cuda:0 等

    # 多人檢測選項 *** 新增 ***
    multi_person_mode: bool = True  # True: 檢測所有人; False: 只檢測最大框
    fullimage_multiperson: bool = False  # True: 使用全圖 DWPose 多人檢測（更快）; False: 逐人 ROI 檢測（更精確）
    min_person_area: int = Constants.MIN_PERSON_AREA  # 最小人物面積閾值
    max_persons: int = 50  # 最多處理的人數（避免過度處理）

    # 處理選項
    strict_no_ghost: bool = True
    post_clean: bool = True
    enable_clahe: bool = True
    enable_hand_enhancement: bool = True
    enable_tracking: bool = True

    # 調試選項
    debug_verbose: bool = False

    # 處理參數
    target_long_side: int = Constants.DEFAULT_TARGET_LONG_SIDE
    max_long_side: int = Constants.MAX_TARGET_LONG_SIDE  # 最大解析度上限
    expand_ratio: float = Constants.DEFAULT_EXPAND_RATIO

    # YOLO 多人檢測參數 *** 新增 ***
    yolo_max_det: int = Constants.YOLO_MAX_DET
    yolo_iou_thresh: float = Constants.YOLO_IOU_THRESH
    yolo_conf_thresh: float = Constants.YOLO_CONF_THRESH

    # 光流穩定化參數
    max_prev_weight: Optional[float] = None
    flow_mag_thresh: Optional[float] = None
    cut_mag_thresh: Optional[float] = None
    cut_hist_corr_thresh: Optional[float] = None
    mask_thresh: Optional[int] = None
    prev_fill_weight: Optional[float] = None

    def validate(self) -> bool:
        """驗證配置"""
        if not self.input_folder:
            logging.error("輸入資料夾路徑不能為空")
            return False
        if not os.path.isdir(self.input_folder):
            logging.error(f"輸入資料夾不存在: {self.input_folder}")
            return False
        if not self.output_folder:
            logging.error("輸出資料夾路徑不能為空")
            return False

        # 驗證設備配置
        if self.device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    logging.warning("請求 CUDA 但 CUDA 不可用，將使用 CPU")
                    self.device = "cpu"
            except ImportError:
                logging.warning("PyTorch 未安裝，無法使用 CUDA")
                self.device = "cpu"

        return True

    def get_device(self) -> str:
        """獲取實際使用的設備"""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


# =============================================================================
# 工具函數
# =============================================================================

class ImageIO:
    """處理 Unicode 路徑的圖像讀寫工具"""

    @staticmethod
    def imread(path: str) -> Optional[np.ndarray]:
        """
        讀取圖像，支援 Unicode 路徑

        Args:
            path: 圖像文件路徑

        Returns:
            BGR 格式的圖像數組，失敗返回 None
        """
        if not os.path.isfile(path):
            return None
        try:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"讀取圖像失敗 {path}: {e}")
            return None

    @staticmethod
    def imwrite(path: str, image: np.ndarray) -> bool:
        """
        寫入圖像，支援 Unicode 路徑

        Args:
            path: 輸出文件路徑
            image: 要寫入的圖像數組

        Returns:
            成功返回 True，失敗返回 False
        """
        try:
            ext = os.path.splitext(path)[1]
            success, encoded = cv2.imencode(ext, image)
            if not success:
                return False
            encoded.tofile(path)
            return True
        except Exception as e:
            logging.error(f"寫入圖像失敗 {path}: {e}")
            return False


class ImageProcessor:
    """圖像處理工具集"""

    @staticmethod
    def enhance_contrast_clahe(image_bgr: np.ndarray,
                                clip_limit: float = Constants.CLAHE_CLIP_LIMIT,
                                tile_grid_size: Tuple[int, int] = Constants.CLAHE_TILE_GRID_SIZE) -> np.ndarray:
        """
        使用 CLAHE 增強對比度

        Args:
            image_bgr: BGR 格式圖像
            clip_limit: CLAHE 限制
            tile_grid_size: 網格大小

        Returns:
            增強後的圖像
        """
        try:
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logging.warning(f"CLAHE 增強失敗: {e}")
            return image_bgr

    @staticmethod
    def sharpen(image_bgr: np.ndarray, strength: float = Constants.DEFAULT_SHARPEN_STRENGTH) -> np.ndarray:
        """
        銳化圖像

        Args:
            image_bgr: BGR 格式圖像
            strength: 銳化強度

        Returns:
            銳化後的圖像
        """
        try:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * strength
            sharpened = cv2.filter2D(image_bgr, -1, kernel)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        except Exception as e:
            logging.warning(f"銳化失敗: {e}")
            return image_bgr

    @staticmethod
    def resize_to_target(image: np.ndarray, target_long_side: int) -> np.ndarray:
        """
        將圖像縮放到目標長邊尺寸

        Args:
            image: 輸入圖像
            target_long_side: 目標長邊像素數

        Returns:
            縮放後的圖像
        """
        try:
            h, w = image.shape[:2]
            long_side = max(h, w)
            if target_long_side > 0 and long_side < target_long_side:
                scale = target_long_side / float(long_side)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            return image
        except Exception as e:
            logging.warning(f"圖像縮放失敗: {e}")
            return image

    @staticmethod
    def enhance_hand_region(image_bgr: np.ndarray,
                           bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        針對手部區域做額外增強（對比度提升 + 銳化）

        Args:
            image_bgr: BGR 格式圖像
            bbox: 手部區域的邊界框 (x1, y1, x2, y2)

        Returns:
            增強後的圖像
        """
        if bbox is None:
            return image_bgr

        try:
            x1, y1, x2, y2 = bbox
            h, w = image_bgr.shape[:2]

            # 確保坐標在圖像範圍內
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))

            if x2 <= x1 or y2 <= y1:
                return image_bgr

            # 提取手部區域
            hand_roi = image_bgr[y1:y2, x1:x2].copy()

            # CLAHE 增強對比度
            hand_roi = ImageProcessor.enhance_contrast_clahe(
                hand_roi,
                clip_limit=Constants.HAND_CLAHE_CLIP_LIMIT,
                tile_grid_size=Constants.HAND_CLAHE_TILE_GRID_SIZE
            )

            # 銳化
            hand_roi = ImageProcessor.sharpen(hand_roi, strength=Constants.HAND_SHARPEN_STRENGTH)

            # 貼回原圖
            result = image_bgr.copy()
            result[y1:y2, x1:x2] = hand_roi
            return result
        except Exception as e:
            logging.warning(f"手部區域增強失敗: {e}")
            return image_bgr

    @staticmethod
    def preprocess_for_pose(image: np.ndarray,
                           enable_clahe: bool = True,
                           target_long_side: int = Constants.DEFAULT_TARGET_LONG_SIDE) -> np.ndarray:
        """
        姿態檢測前的預處理

        Args:
            image: 輸入圖像
            enable_clahe: 是否啟用 CLAHE
            target_long_side: 目標長邊尺寸

        Returns:
            預處理後的圖像
        """
        img = image

        if enable_clahe:
            img = ImageProcessor.enhance_contrast_clahe(img)

        img = ImageProcessor.resize_to_target(img, target_long_side)

        return img

    @staticmethod
    def to_cv2_uint8_bgr(image) -> Optional[np.ndarray]:
        """
        將檢測器輸出統一轉為 OpenCV 可編碼的 uint8 BGR 圖像

        Args:
            image: PIL.Image 或 numpy array

        Returns:
            BGR 格式的 uint8 圖像，失敗返回 None
        """
        if image is None:
            return None

        # PIL -> numpy
        if PILImage is not None and isinstance(image, PILImage.Image):
            rgb = image.convert("RGB")
            np_rgb = np.array(rgb)
            return cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

        # numpy array
        if isinstance(image, np.ndarray):
            arr = image

            # 轉換為 uint8
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255)
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)

            # 處理不同維度
            if arr.ndim == 2:
                return arr
            if arr.ndim == 3 and arr.shape[2] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        return None


class BBoxUtils:
    """邊界框處理工具"""

    @staticmethod
    def ensure_bbox(x1: int, y1: int, x2: int, y2: int,
                    w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        """
        確保邊界框在圖像範圍內且有效

        Args:
            x1, y1, x2, y2: 邊界框坐標
            w, h: 圖像寬高

        Returns:
            有效的邊界框或 None
        """
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        return (x1, y1, x2, y2)

    @staticmethod
    def expand_bbox(bbox: Tuple[int, int, int, int],
                   w: int, h: int,
                   ratio: float) -> Tuple[int, int, int, int]:
        """
        擴展邊界框

        Args:
            bbox: 原始邊界框 (x1, y1, x2, y2)
            w, h: 圖像寬高
            ratio: 擴展比例

        Returns:
            擴展後的邊界框
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * ratio
        bh = (y2 - y1) * ratio

        nx1 = int(round(cx - bw / 2.0))
        ny1 = int(round(cy - bh / 2.0))
        nx2 = int(round(cx + bw / 2.0))
        ny2 = int(round(cy + bh / 2.0))

        result = BBoxUtils.ensure_bbox(nx1, ny1, nx2, ny2, w, h)
        return result if result else (0, 0, w, h)

    @staticmethod
    def crop_and_resize(image_bgr: np.ndarray,
                       bbox: Tuple[int, int, int, int],
                       expand_ratio: float = Constants.DEFAULT_EXPAND_RATIO,
                       target_long_side: int = Constants.DEFAULT_TARGET_LONG_SIDE
                       ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        對邊界框做擴張裁切並放大

        Args:
            image_bgr: 輸入圖像
            bbox: 邊界框
            expand_ratio: 擴展比例
            target_long_side: 目標長邊尺寸

        Returns:
            (裁切並放大後的圖像, 擴展後的邊界框)
        """
        x1, y1, x2, y2 = bbox
        h, w = image_bgr.shape[:2]

        # 計算擴展後的邊界框
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * expand_ratio
        bh = (y2 - y1) * expand_ratio

        nx1 = int(round(cx - bw / 2.0))
        ny1 = int(round(cy - bh / 2.0))
        nx2 = int(round(cx + bw / 2.0))
        ny2 = int(round(cy + bh / 2.0))

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(w, nx2)
        ny2 = min(h, ny2)

        if nx2 - nx1 <= 2 or ny2 - ny1 <= 2:
            return image_bgr, (0, 0, w, h)

        # 裁切
        crop = image_bgr[ny1:ny2, nx1:nx2]

        # 放大
        ch, cw = crop.shape[:2]
        long_side = max(ch, cw)
        if long_side < target_long_side:
            scale = target_long_side / float(long_side)
            new_w = int(round(cw * scale))
            new_h = int(round(ch * scale))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return crop, (nx1, ny1, nx2, ny2)

    @staticmethod
    def get_hand_bbox(image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        根據圖像尺寸估計手部區域

        Args:
            image_shape: (height, width)

        Returns:
            手部區域的邊界框
        """
        h, w = image_shape
        x1 = int(w * Constants.HAND_REGION_X_START)
        x2 = int(w * Constants.HAND_REGION_X_END)
        y1 = int(h * Constants.HAND_REGION_Y_START)
        y2 = int(h * Constants.HAND_REGION_Y_END)
        return (x1, y1, x2, y2)


# =============================================================================
# 姿態處理類
# =============================================================================

class PoseProcessor:
    """姿態圖後處理"""

    @staticmethod
    def paste_pose(canvas_bgr: np.ndarray,
                   pose_crop_bgr: np.ndarray,
                   bbox: Tuple[int, int, int, int],
                   mask_thresh: int = Constants.MASK_THRESH) -> np.ndarray:
        """
        將裁切後的骨架圖貼回全圖黑底畫布

        Args:
            canvas_bgr: 黑底畫布
            pose_crop_bgr: 裁切的骨架圖
            bbox: 目標位置
            mask_thresh: 遮罩閾值

        Returns:
            合成後的圖像
        """
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            return canvas_bgr

        # 將骨架圖縮回 bbox 尺寸
        pose_resized = cv2.resize(pose_crop_bgr, (box_w, box_h), interpolation=cv2.INTER_AREA)

        # 創建遮罩
        gray = cv2.cvtColor(pose_resized, cv2.COLOR_BGR2GRAY)
        mask = gray > mask_thresh

        # 貼回
        roi = canvas_bgr[y1:y2, x1:x2]
        roi[mask] = pose_resized[mask]
        canvas_bgr[y1:y2, x1:x2] = roi

        return canvas_bgr

    @staticmethod
    def clean_pose_lines(pose_bgr: np.ndarray,
                        thresh: int = Constants.MASK_THRESH,
                        enable: bool = True) -> np.ndarray:
        """
        輕量清理：去除微弱殘點、確保線條銳利

        Args:
            pose_bgr: 骨架圖
            thresh: 閾值
            enable: 是否啟用清理

        Returns:
            清理後的圖像
        """
        if not enable:
            return pose_bgr

        try:
            gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)
            mask = (gray > max(0, int(thresh))).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, Constants.POST_CLEAN_KERNEL_SIZE)
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # 只保留開運算後的線條位置顏色
            out = np.zeros_like(pose_bgr)
            idx = opened > 0
            out[idx] = pose_bgr[idx]

            return out
        except Exception as e:
            logging.warning(f"骨架清理失敗: {e}")
            return pose_bgr

    @staticmethod
    def calculate_density(pose_bgr: np.ndarray,
                         thresh: int = Constants.MASK_THRESH) -> float:
        """
        計算骨架圖的密度（非零像素比例）

        Args:
            pose_bgr: 骨架圖
            thresh: 閾值

        Returns:
            密度值 (0-1)
        """
        try:
            gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)
            mask = gray > max(0, int(thresh))
            return float(mask.mean())
        except Exception as e:
            logging.warning(f"密度計算失敗: {e}")
            return 0.0


# =============================================================================
# 人物檢測和追蹤
# =============================================================================

class PersonDetector:
    """人物檢測器（使用 YOLO）"""

    def __init__(self, model_path: str = "yolov8s.pt"):
        """
        初始化檢測器

        Args:
            model_path: YOLO 模型路徑
        """
        self.model = None
        self.model_path = model_path
        self._init_model()

    def _init_model(self):
        """初始化 YOLO 模型"""
        if YOLO is None:
            logging.warning("YOLO 未安裝，人物檢測功能將不可用")
            return

        try:
            self.model = YOLO(self.model_path)
            logging.info(f"YOLO 模型載入成功: {self.model_path}")
        except Exception as e:
            logging.error(f"YOLO 模型載入失敗: {e}")
            self.model = None

    def detect(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        檢測圖像中的人物，返回最大的邊界框

        Args:
            image_bgr: BGR 格式圖像

        Returns:
            人物邊界框 (x1, y1, x2, y2) 或 None
        """
        if self.model is None:
            return None

        h, w = image_bgr.shape[:2]

        # 多尺度嘗試
        for imgsz, conf_thresh in Constants.YOLO_SCALES:
            try:
                results = self.model.predict(
                    source=image_bgr,
                    imgsz=imgsz,
                    conf=conf_thresh,
                    classes=[Constants.YOLO_PERSON_CLASS],
                    verbose=False
                )

                if not results or len(results) == 0:
                    continue

                boxes = getattr(results[0], 'boxes', None)
                if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                    continue

                xyxy = boxes.xyxy.cpu().numpy()

                # 找最大的邊界框
                best = None
                best_area = -1.0

                for b in xyxy:
                    x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
                    bbox = BBoxUtils.ensure_bbox(x1, y1, x2, y2, w, h)
                    if bbox is None:
                        continue

                    x1, y1, x2, y2 = bbox
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best = bbox

                if best is not None:
                    logging.debug(f"YOLO 檢測成功: imgsz={imgsz} conf={conf_thresh} bbox={best}")
                    return best

            except Exception as e:
                logging.warning(f"YOLO 檢測失敗 (imgsz={imgsz}): {e}")
                continue

        return None


class PersonTracker:
    """人物追蹤器（使用 CSRT 和光流）"""

    def __init__(self):
        """初始化追蹤器"""
        self.tracker = None
        self.prev_bbox = None

    def reset(self):
        """重置追蹤器"""
        self.tracker = None
        self.prev_bbox = None

    def init_tracker(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        初始化追蹤器

        Args:
            frame_bgr: 當前幀
            bbox: 初始邊界框

        Returns:
            成功返回 True
        """
        try:
            if not hasattr(cv2, 'TrackerCSRT_create'):
                logging.warning("CSRT 追蹤器不可用")
                return False

            self.tracker = cv2.TrackerCSRT_create()
            x1, y1, x2, y2 = bbox
            self.tracker.init(frame_bgr, (x1, y1, x2 - x1, y2 - y1))
            self.prev_bbox = bbox
            logging.debug(f"CSRT 追蹤器初始化: {bbox}")
            return True
        except Exception as e:
            logging.warning(f"追蹤器初始化失敗: {e}")
            self.tracker = None
            return False

    def update(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        更新追蹤器

        Args:
            frame_bgr: 當前幀

        Returns:
            更新後的邊界框或 None
        """
        if self.tracker is None:
            return None

        try:
            ok, rect = self.tracker.update(frame_bgr)
            if ok and rect is not None:
                x, y, w, h = rect
                H, W = frame_bgr.shape[:2]
                bbox = BBoxUtils.ensure_bbox(
                    int(round(x)), int(round(y)),
                    int(round(x + w)), int(round(y + h)),
                    W, H
                )
                if bbox:
                    self.prev_bbox = bbox
                    logging.debug(f"CSRT 追蹤更新: {bbox}")
                return bbox
        except Exception as e:
            logging.warning(f"追蹤器更新失敗: {e}")
            self.tracker = None

        return None

    def predict_with_flow(self, prev_frame: np.ndarray,
                         curr_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        使用光流預測邊界框位置

        Args:
            prev_frame: 前一幀
            curr_frame: 當前幀

        Returns:
            預測的邊界框或 None
        """
        if self.prev_bbox is None:
            return None

        try:
            x1, y1, x2, y2 = self.prev_bbox

            # 計算光流
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                Constants.FLOW_PYR_SCALE,
                Constants.FLOW_LEVELS,
                Constants.FLOW_WINSIZE,
                Constants.FLOW_ITERATIONS,
                Constants.FLOW_POLY_N,
                Constants.FLOW_POLY_SIGMA,
                0
            )

            # 計算邊界框內的中值位移
            sub = flow[y1:y2, x1:x2]
            dx = float(np.median(sub[..., 0]))
            dy = float(np.median(sub[..., 1]))

            # 應用位移
            h, w = curr_frame.shape[:2]
            new_bbox = BBoxUtils.ensure_bbox(
                int(round(x1 + dx)), int(round(y1 + dy)),
                int(round(x2 + dx)), int(round(y2 + dy)),
                w, h
            )

            if new_bbox:
                logging.debug(f"光流預測: {new_bbox} (位移: dx={dx:.1f}, dy={dy:.1f})")
            return new_bbox
        except Exception as e:
            logging.warning(f"光流預測失敗: {e}")
            return None


# =============================================================================
# 時序穩定化
# =============================================================================

class TemporalStabilizer:
    """時序穩定化處理器"""

    def __init__(self, config: Config):
        """
        初始化穩定器

        Args:
            config: 配置對象
        """
        self.config = config
        self.prev_frame = None
        self.prev_pose = None
        self.prev2_frame = None
        self.prev2_pose = None
        self.adaptive_ema_flow = None
        self.adaptive_ema_hist = None

    def reset(self):
        """重置狀態"""
        self.prev_frame = None
        self.prev_pose = None
        self.prev2_frame = None
        self.prev2_pose = None
        self.adaptive_ema_flow = None
        self.adaptive_ema_hist = None

    def infer_from_temporal(self, curr_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        使用前後幀的骨架進行推論（光流對齊）

        Args:
            curr_frame: 當前幀

        Returns:
            推論的骨架圖或 None
        """
        if self.prev_pose is None and self.prev2_pose is None:
            return None
        if self.prev_frame is None and self.prev2_frame is None:
            return None

        try:
            h, w = curr_frame.shape[:2]
            inferred = np.zeros((h, w, 3), dtype=np.uint8)

            # 優先使用前一幀
            if self.prev_pose is not None and self.prev_frame is not None:
                warped = self._warp_with_flow(self.prev_frame, curr_frame, self.prev_pose)
                if warped is not None:
                    inferred = warped

            # 如果有更前一幀，進行插值
            if self.prev2_pose is not None and self.prev2_frame is not None:
                warped2 = self._warp_with_flow(self.prev2_frame, curr_frame, self.prev2_pose)
                if warped2 is not None:
                    if self.prev_pose is not None:
                        inferred = cv2.addWeighted(inferred, 0.6, warped2, 0.4, 0.0)
                    else:
                        inferred = warped2

            return inferred
        except Exception as e:
            logging.warning(f"時序推論失敗: {e}")
            return None

    def _warp_with_flow(self, prev_frame: np.ndarray,
                       curr_frame: np.ndarray,
                       prev_pose: np.ndarray) -> Optional[np.ndarray]:
        """使用光流對齊骨架"""
        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                Constants.FLOW_PYR_SCALE,
                Constants.FLOW_LEVELS,
                Constants.FLOW_WINSIZE,
                Constants.FLOW_ITERATIONS,
                Constants.FLOW_POLY_N,
                Constants.FLOW_POLY_SIGMA,
                0
            )

            h, w = curr_frame.shape[:2]
            grid_x, grid_y = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32)
            )
            map_x = grid_x + flow[..., 0].astype(np.float32)
            map_y = grid_y + flow[..., 1].astype(np.float32)

            warped = cv2.remap(
                prev_pose, map_x, map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            return warped
        except Exception as e:
            logging.warning(f"光流對齊失敗: {e}")
            return None

    def stabilize(self, curr_frame: np.ndarray,
                 curr_pose: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        使用稠密光流進行時序穩定化

        Args:
            curr_frame: 當前幀
            curr_pose: 當前幀的骨架圖

        Returns:
            (穩定化後的骨架圖, 是否檢測到剪輯)
        """
        # 嚴格無殘影模式：直接返回當前幀
        if self.config.strict_no_ghost:
            self._update_history(curr_frame, curr_pose)
            return curr_pose, False

        # 首幀
        if self.prev_frame is None or self.prev_pose is None:
            self._update_history(curr_frame, curr_pose)
            return curr_pose, False

        try:
            # 計算直方圖相關性
            hist_corr = self._calculate_hist_correlation(self.prev_frame, curr_frame)

            # 計算光流
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                Constants.FLOW_PYR_SCALE,
                Constants.FLOW_LEVELS,
                Constants.FLOW_WINSIZE,
                Constants.FLOW_ITERATIONS,
                Constants.FLOW_POLY_N,
                Constants.FLOW_POLY_SIGMA,
                0
            )

            mag = np.linalg.norm(flow, axis=2).mean()

            # 更新自適應參數
            self._update_adaptive_params(mag, hist_corr)

            # 獲取閾值
            thresholds = self._get_adaptive_thresholds()

            # 檢測剪輯
            if (hist_corr < thresholds['cut_hist_corr'] or
                mag >= thresholds['cut_mag']):
                logging.info("檢測到剪輯或巨大位移，重置時序")
                self._update_history(curr_frame, curr_pose)
                return curr_pose, True

            # 對齊前一幀骨架
            h, w = curr_frame.shape[:2]
            grid_x, grid_y = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32)
            )
            map_x = grid_x + flow[..., 0].astype(np.float32)
            map_y = grid_y + flow[..., 1].astype(np.float32)

            warped_prev_pose = cv2.remap(
                self.prev_pose, map_x, map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # 計算混合權重
            prev_w = max(0.0, min(
                thresholds['max_prev_weight'],
                1.0 - (mag / max(thresholds['flow_mag'], 1e-6))
            ))
            curr_w = 1.0 - prev_w

            # 創建當前骨架遮罩
            gray_curr = cv2.cvtColor(curr_pose, cv2.COLOR_BGR2GRAY)
            mask_curr = gray_curr > thresholds['mask_thresh']

            # 基底混合
            base = cv2.addWeighted(curr_pose, curr_w, warped_prev_pose, prev_w, 0.0)

            # 強制使用當前線條像素
            stabilized = base.copy()
            if mask_curr.any():
                stabilized[mask_curr] = curr_pose[mask_curr]

            # 背景填補
            if 0.0 < thresholds['prev_fill_weight'] < prev_w:
                not_line = ~mask_curr
                if not_line.any():
                    blended_bg = cv2.addWeighted(
                        curr_pose, 1.0 - thresholds['prev_fill_weight'],
                        warped_prev_pose, thresholds['prev_fill_weight'],
                        0.0
                    )
                    stabilized[not_line] = blended_bg[not_line]

            self._update_history(curr_frame, stabilized)
            return stabilized, False

        except Exception as e:
            logging.warning(f"時序穩定化失敗: {e}")
            self._update_history(curr_frame, curr_pose)
            return curr_pose, False

    def _update_history(self, curr_frame: np.ndarray, curr_pose: np.ndarray):
        """更新歷史幀"""
        self.prev2_frame = self.prev_frame
        self.prev2_pose = self.prev_pose
        self.prev_frame = curr_frame
        self.prev_pose = curr_pose

    def _calculate_hist_correlation(self, prev_frame: np.ndarray,
                                   curr_frame: np.ndarray) -> float:
        """計算 HSV 直方圖相關係數"""
        try:
            prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
            curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)

            hist_prev = cv2.calcHist([prev_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            hist_curr = cv2.calcHist([curr_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])

            cv2.normalize(hist_prev, hist_prev)
            cv2.normalize(hist_curr, hist_curr)

            corr = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
            return float(max(0.0, min(1.0, corr)))
        except Exception as e:
            logging.warning(f"直方圖相關性計算失敗: {e}")
            return 1.0

    def _update_adaptive_params(self, mag: float, hist_corr: float):
        """更新自適應參數（指數滑動平均）"""
        alpha = Constants.EMA_ALPHA

        if self.adaptive_ema_flow is None:
            self.adaptive_ema_flow = float(mag)
        else:
            self.adaptive_ema_flow = (1 - alpha) * self.adaptive_ema_flow + alpha * float(mag)

        if self.adaptive_ema_hist is None:
            self.adaptive_ema_hist = float(hist_corr)
        else:
            self.adaptive_ema_hist = (1 - alpha) * self.adaptive_ema_hist + alpha * float(hist_corr)

    def _get_adaptive_thresholds(self) -> dict:
        """獲取自適應閾值"""
        if self.adaptive_ema_flow is None:
            ema_flow = 5.0
        else:
            ema_flow = self.adaptive_ema_flow

        # 計算自適應閾值
        auto_flow_mag_thresh = max(6.0, min(18.0, 2.2 * ema_flow))
        auto_cut_mag_thresh = max(15.0, min(40.0, 2.8 * auto_flow_mag_thresh))
        auto_cut_hist_corr_thresh = 0.2 if ema_flow < 10.0 else 0.25
        auto_max_prev_weight = 0.5 - min(0.25, max(0.0, (ema_flow - 2.0) / (18.0 - 2.0) * 0.25))
        auto_mask_thresh = 16 if ema_flow < 6.0 else (24 if ema_flow < 12.0 else 32)
        auto_prev_fill_weight = 0.35 if ema_flow < 8.0 else (0.25 if ema_flow < 12.0 else 0.15)

        # 使用配置值或自動值
        return {
            'flow_mag': self.config.flow_mag_thresh or auto_flow_mag_thresh,
            'cut_mag': self.config.cut_mag_thresh or auto_cut_mag_thresh,
            'cut_hist_corr': self.config.cut_hist_corr_thresh or auto_cut_hist_corr_thresh,
            'max_prev_weight': self.config.max_prev_weight or auto_max_prev_weight,
            'mask_thresh': self.config.mask_thresh or auto_mask_thresh,
            'prev_fill_weight': self.config.prev_fill_weight or auto_prev_fill_weight,
        }


# =============================================================================
# 主處理器
# =============================================================================

class PoseDetectionPipeline:
    """姿態檢測處理流程"""

    def __init__(self, config: Config):
        """
        初始化處理流程

        Args:
            config: 配置對象
        """
        self.config = config
        self.detector = None
        self.person_detector = None
        self.person_tracker = None
        self.stabilizer = None

        self._init_components()

    def _init_components(self):
        """初始化各個組件"""
        # 載入姿態檢測器
        self._init_pose_detector()

        # 初始化人物檢測器
        if self.config.enable_tracking:
            self.person_detector = PersonDetector(self.config.yolo_model)
            self.person_tracker = PersonTracker()

        # 初始化穩定器
        self.stabilizer = TemporalStabilizer(self.config)

    def _init_pose_detector(self):
        """初始化姿態檢測器"""
        logging.info("正在載入姿態檢測模型...")

        backend = ""
        try:
            self.detector = DWposeDetector.from_pretrained("yzd-v/DWPose")
            backend = "DWPose (yzd-v/DWPose)"
        except Exception as e1:
            logging.warning(f"無法載入 DWPose (yzd-v/DWPose): {e1}")
            try:
                self.detector = DWposeDetector.from_pretrained()
                backend = "DWPose (from_pretrained)"
            except Exception as e2:
                logging.warning(f"無法載入 DWPose (from_pretrained): {e2}")
                try:
                    self.detector = DWposeDetector()
                    backend = "DWPose (default)"
                except Exception as e3:
                    logging.warning(f"DWPose 載入失敗，改用 OpenPose: {e3}")
                    try:
                        self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                        backend = "OpenPose (lllyasviel/ControlNet)"
                    except Exception as e4:
                        logging.error(f"所有姿態檢測器載入失敗: {e4}")
                        raise RuntimeError("無法載入任何姿態檢測器")

        logging.info(f"姿態檢測模型載入完畢: {backend}")

    def process_folder(self):
        """處理整個資料夾"""
        if not self.config.validate():
            return False

        # 確保輸出資料夾存在
        os.makedirs(self.config.output_folder, exist_ok=True)

        # 獲取文件列表
        file_list = sorted(os.listdir(self.config.input_folder))
        total = sum(1 for f in file_list if f.lower().endswith(Constants.IMAGE_EXTENSIONS))

        logging.info(f"開始處理資料夾: {self.config.input_folder}")
        logging.info(f"輸出資料夾: {self.config.output_folder}")
        logging.info(f"共找到 {total} 個圖像文件")

        processed = 0
        failed = 0

        for filename in file_list:
            if not filename.lower().endswith(Constants.IMAGE_EXTENSIONS):
                continue

            input_path = os.path.join(self.config.input_folder, filename)
            output_path = os.path.join(self.config.output_folder, filename)

            try:
                success = self.process_image(input_path, output_path)
                if success:
                    processed += 1
                    logging.info(f"[{processed}/{total}] 已處理: {filename}")
                else:
                    failed += 1
                    logging.warning(f"[{processed + failed}/{total}] 處理失敗: {filename}")
            except Exception as e:
                failed += 1
                logging.error(f"處理 {filename} 時發生錯誤: {e}")

        logging.info(f"處理完畢！成功: {processed}, 失敗: {failed}")
        return True

    def process_image(self, input_path: str, output_path: str) -> bool:
        """
        處理單張圖像

        Args:
            input_path: 輸入圖像路徑
            output_path: 輸出圖像路徑

        Returns:
            成功返回 True
        """
        # 讀取圖像
        img = ImageIO.imread(input_path)
        if img is None:
            logging.error(f"讀取失敗: {input_path}")
            return False

        # 預處理
        pre_img = self._preprocess_image(img)

        # 檢測人物 ROI
        roi_bbox = self._detect_roi(pre_img)

        # 裁切並放大 ROI
        crop_img, placed_bbox = BBoxUtils.crop_and_resize(
            pre_img, roi_bbox,
            expand_ratio=1.0,
            target_long_side=Constants.ENHANCED_TARGET_LONG_SIDE
        )

        # 手部增強
        if self.config.enable_hand_enhancement:
            hand_bbox = BBoxUtils.get_hand_bbox(crop_img.shape[:2])
            crop_img = ImageProcessor.enhance_hand_region(crop_img, hand_bbox)

        # 執行姿態檢測
        pose_result = self._detect_pose(crop_img)
        if pose_result is None:
            logging.warning(f"姿態檢測失敗: {input_path}")
            return False

        # 貼回全圖
        canvas = np.zeros_like(pre_img)
        composed = PoseProcessor.paste_pose(canvas, pose_result, placed_bbox)
        composed = PoseProcessor.clean_pose_lines(composed, enable=self.config.post_clean)

        # 檢查密度並重試
        composed = self._retry_if_empty(pre_img, composed, roi_bbox)

        # 時序穩定化
        stabilized, is_cut = self.stabilizer.stabilize(pre_img, composed)

        if is_cut:
            if self.config.enable_tracking and self.person_tracker:
                self.person_tracker.reset()

        # 保存結果
        success = ImageIO.imwrite(output_path, stabilized)
        if not success:
            logging.error(f"寫入失敗: {output_path}")

        return success

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """預處理圖像"""
        pre_img = ImageProcessor.preprocess_for_pose(
            image,
            enable_clahe=self.config.enable_clahe,
            target_long_side=self.config.target_long_side
        )
        pre_img = ImageProcessor.sharpen(pre_img, Constants.DEFAULT_SHARPEN_STRENGTH)
        return pre_img

    def _detect_roi(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """檢測人物 ROI"""
        h, w = image.shape[:2]

        # 嘗試 YOLO 檢測
        if self.config.enable_tracking and self.person_detector:
            yolo_bbox = self.person_detector.detect(image)
            if yolo_bbox is not None:
                if self.person_tracker:
                    self.person_tracker.init_tracker(image, yolo_bbox)
                logging.debug(f"[ROI] source=YOLO bbox={yolo_bbox}")
                return BBoxUtils.expand_bbox(yolo_bbox, w, h, self.config.expand_ratio)

        # 嘗試追蹤器
        if self.config.enable_tracking and self.person_tracker:
            tracked = self.person_tracker.update(image)
            if tracked is not None:
                logging.debug(f"[ROI] source=CSRT bbox={tracked}")
                return BBoxUtils.expand_bbox(tracked, w, h, self.config.expand_ratio)

            # 嘗試光流預測
            if (self.stabilizer.prev_frame is not None and
                self.person_tracker.prev_bbox is not None):
                predicted = self.person_tracker.predict_with_flow(
                    self.stabilizer.prev_frame, image
                )
                if predicted is not None:
                    logging.debug(f"[ROI] source=FLOW bbox={predicted}")
                    return BBoxUtils.expand_bbox(predicted, w, h, self.config.expand_ratio)

        # 使用全圖
        full_bbox = (0, 0, w, h)
        logging.debug(f"[ROI] source=FULL_IMAGE bbox={full_bbox}")
        return full_bbox

    def _detect_pose(self, image: np.ndarray) -> Optional[np.ndarray]:
        """執行姿態檢測"""
        try:
            processed_img = self.detector(image, detect_hand=True, detect_face=True)
        except TypeError:
            # 版本不支援參數
            try:
                processed_img = self.detector(image)
            except Exception as e:
                logging.error(f"姿態檢測失敗: {e}")
                return None

        return ImageProcessor.to_cv2_uint8_bgr(processed_img)

    def _retry_if_empty(self, pre_img: np.ndarray,
                       composed: np.ndarray,
                       roi_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """如果骨架為空，進行重試"""
        density = PoseProcessor.calculate_density(composed)

        # 嘗試從時序推論
        if density < Constants.MIN_POSE_DENSITY:
            inferred = self.stabilizer.infer_from_temporal(pre_img)
            if inferred is not None:
                inferred_density = PoseProcessor.calculate_density(inferred)
                if inferred_density >= Constants.MIN_POSE_DENSITY:
                    logging.debug(f"使用時序推論，密度={inferred_density:.6f}")
                    return inferred

        # 重試：更大的 ROI
        density = PoseProcessor.calculate_density(composed)
        if density < Constants.MIN_POSE_DENSITY:
            h, w = pre_img.shape[:2]
            bigger_bbox = BBoxUtils.expand_bbox(roi_bbox, w, h, Constants.RETRY_EXPAND_RATIO)

            crop2, placed2 = BBoxUtils.crop_and_resize(
                pre_img, bigger_bbox,
                expand_ratio=1.0,
                target_long_side=Constants.MAX_TARGET_LONG_SIDE
            )

            if self.config.enable_hand_enhancement:
                hand_bbox2 = BBoxUtils.get_hand_bbox(crop2.shape[:2])
                crop2 = ImageProcessor.enhance_hand_region(crop2, hand_bbox2)

            pose2 = self._detect_pose(crop2)
            if pose2 is not None:
                canvas2 = np.zeros_like(pre_img)
                composed2 = PoseProcessor.paste_pose(canvas2, pose2, placed2)
                if PoseProcessor.calculate_density(composed2) >= density:
                    composed = composed2

        # 全圖重試
        if PoseProcessor.calculate_density(composed) < Constants.MIN_POSE_DENSITY:
            if self.config.enable_hand_enhancement:
                hand_bbox_full = BBoxUtils.get_hand_bbox(pre_img.shape[:2])
                pre_img_enhanced = ImageProcessor.enhance_hand_region(pre_img, hand_bbox_full)
            else:
                pre_img_enhanced = pre_img

            pose3 = self._detect_pose(pre_img_enhanced)
            if pose3 is not None:
                composed = pose3

        return composed


# =============================================================================
# 命令行界面
# =============================================================================

def setup_logging(verbose: bool = False):
    """設置日誌"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> Config:
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description='優化的姿態檢測系統',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s -i input_folder -o output_folder
  %(prog)s -i input_folder -o output_folder --no-ghost --verbose
  %(prog)s -i input_folder -o output_folder --no-tracking
        """
    )

    # 必需參數
    parser.add_argument('-i', '--input', required=True,
                       help='輸入資料夾路徑（包含圖像序列）')
    parser.add_argument('-o', '--output', required=True,
                       help='輸出資料夾路徑（儲存骨架圖）')

    # 可選參數
    parser.add_argument('--yolo-model', default='yolov8s.pt',
                       help='YOLO 模型路徑 (默認: yolov8s.pt)')
    parser.add_argument('--no-ghost', action='store_true',
                       help='啟用嚴格無殘影模式')
    parser.add_argument('--no-clean', action='store_true',
                       help='禁用形態學清理')
    parser.add_argument('--no-clahe', action='store_true',
                       help='禁用 CLAHE 增強')
    parser.add_argument('--no-hand-enhancement', action='store_true',
                       help='禁用手部增強')
    parser.add_argument('--no-tracking', action='store_true',
                       help='禁用人物追蹤')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='顯示詳細調試信息')

    args = parser.parse_args()

    # 創建配置
    config = Config(
        input_folder=args.input,
        output_folder=args.output,
        yolo_model=args.yolo_model,
        strict_no_ghost=args.no_ghost,
        post_clean=not args.no_clean,
        enable_clahe=not args.no_clahe,
        enable_hand_enhancement=not args.no_hand_enhancement,
        enable_tracking=not args.no_tracking,
        debug_verbose=args.verbose
    )

    return config


def main():
    """主函數"""
    # 解析參數
    config = parse_arguments()

    # 設置日誌
    setup_logging(config.debug_verbose)

    try:
        # 創建處理流程
        pipeline = PoseDetectionPipeline(config)

        # 處理資料夾
        success = pipeline.process_folder()

        if success:
            logging.info("全部處理完畢！")
            return 0
        else:
            logging.error("處理失敗")
            return 1

    except KeyboardInterrupt:
        logging.info("用戶中斷處理")
        return 130
    except Exception as e:
        logging.error(f"發生錯誤: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
