# 多人姿態檢測增強計劃

## 研究基礎 (基於 2024-2025 最新技術)

### DWPose 優勢
- **SOTA 性能**: ICCV 2023, COCO-WholeBody benchmark
- **多人檢測**: 原生支持多人場景（通過 YOLOX）
- **全身追蹤**: 優於 OpenPose，特別是手部細節
- **兩階段蒸餾**: 提升整體性能和效率

### YOLO 最新技術
- **YOLO11 Pose**: 2025 年生產標準
- **RTMO**: 實時一階段多人姿態估計
- **關鍵參數**:
  - `max_det=1000` (預設 300 太少)
  - `iou=0.45` (降低避免過度 NMS)
  - `conf=0.1` (檢測小人物)

## 當前問題

### 1. 單人檢測限制
```python
# 原始代碼只取最大框
best = None
best_area = -1.0
for b in xyxy:
    area = (x2 - x1) * (y2 - y1)
    if area > best_area:  # 只保留最大的
        best_area = area
        best = bbox
```

**影響**: 多人場景只處理一個人，其他人漏檢

### 2. 固定密度閾值
```python
MIN_POSE_DENSITY = 0.0008  # 固定值
```

**問題**: 超高解析度時過於嚴格，低解析度時過於寬鬆

### 3. 無 GPU 優化
- 未使用設備選擇
- 無批處理支持
- 無混合精度訓練

## 改進方案

### 方案 A: 逐人 ROI 檢測（精確模式）

**流程**:
1. YOLO 檢測所有人框（返回 List[bbox]）
2. 對每個人框:
   - 擴展並放大
   - 手部增強
   - DWPose 檢測
3. 合成所有骨架到同一張 canvas

**優點**:
- 每個人都有獨立的高解析度處理
- 手部細節最佳
- 適合少量人物（<10人）

**缺點**:
- 計算量大
- 人多時慢

**適用場景**: 舞蹈、體育、少量人物高質量需求

### 方案 B: 全圖多人檢測（快速模式）

**流程**:
1. 直接對全圖運行 DWPose
2. DWPose 自動檢測所有人
3. 直接輸出結果

**優點**:
- 單次推論完成
- 速度快
- 適合大量人物

**缺點**:
- 解析度限制
- 手部細節可能不如 ROI 模式

**適用場景**: 人群、監控、實時處理

### 方案 C: 混合模式（推薦）

**流程**:
1. 如果人數 <= 6: 使用方案 A（逐人 ROI）
2. 如果人數 > 6: 使用方案 B（全圖檢測）
3. 自動根據人數和解析度選擇

**依據**: 研究表明兩階段推理在人數不超過 6 人時速度和準確度都優於 bottom-up

## 具體實現

### 1. 多人檢測函數

```python
def detect_all_persons(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """檢測所有人物邊界框"""
    if self.model is None:
        return []

    all_bboxes = []
    h, w = image_bgr.shape[:2]

    for imgsz, conf_thresh in Constants.YOLO_SCALES:
        try:
            results = self.model.predict(
                source=image_bgr,
                imgsz=imgsz,
                conf=min(conf_thresh, self.config.yolo_conf_thresh),
                iou=self.config.yolo_iou_thresh,
                max_det=self.config.yolo_max_det,
                classes=[Constants.YOLO_PERSON_CLASS],
                verbose=False
            )

            if results and len(results) > 0:
                boxes = getattr(results[0], 'boxes', None)
                if boxes is not None and boxes.xyxy is not None:
                    xyxy = boxes.xyxy.cpu().numpy()

                    for b in xyxy:
                        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
                        bbox = BBoxUtils.ensure_bbox(x1, y1, x2, y2, w, h)
                        if bbox is None:
                            continue

                        x1, y1, x2, y2 = bbox
                        area = (x2 - x1) * (y2 - y1)

                        # 過濾過小或不合理的框
                        if area < self.config.min_person_area:
                            continue

                        ratio = (x2 - x1) / max(1, (y2 - y1))
                        if ratio < Constants.MIN_PERSON_RATIO or ratio > Constants.MAX_PERSON_RATIO:
                            continue

                        all_bboxes.append(bbox)

                    if all_bboxes:
                        # 成功檢測到人，不再嘗試其他尺度
                        break

        except Exception as e:
            logging.warning(f"YOLO 檢測失敗 (imgsz={imgsz}): {e}")
            continue

    # 按面積排序，大的在前
    all_bboxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

    # 限制最大人數
    if len(all_bboxes) > self.config.max_persons:
        logging.warning(f"檢測到 {len(all_bboxes)} 人，僅處理前 {self.config.max_persons} 人")
        all_bboxes = all_bboxes[:self.config.max_persons]

    return all_bboxes
```

### 2. 自適應密度閾值

```python
def calculate_adaptive_density_threshold(h: int, w: int) -> float:
    """根據圖像解析度計算自適應密度閾值"""
    total_pixels = h * w
    threshold = max(
        Constants.MIN_DENSITY_LOWER_BOUND,
        min(
            Constants.MIN_DENSITY_UPPER_BOUND,
            Constants.DENSITY_REFERENCE_PIXELS / total_pixels
        )
    )
    logging.debug(f"自適應密度閾值: {threshold:.6f} (解析度: {w}x{h})")
    return threshold
```

### 3. 多人合成流程

```python
def process_multiperson(self, pre_img: np.ndarray, bboxes: List[Tuple]) -> np.ndarray:
    """處理多人並合成到同一個 canvas"""
    canvas = np.zeros_like(pre_img)
    processed_count = 0

    for i, bbox in enumerate(bboxes):
        logging.debug(f"處理第 {i+1}/{len(bboxes)} 人，bbox={bbox}")

        try:
            # 裁切並放大
            crop_img, placed_bbox = BBoxUtils.crop_and_resize(
                pre_img, bbox,
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
                continue

            # 貼回 canvas（累加模式）
            canvas = PoseProcessor.paste_pose(canvas, pose_result, placed_bbox)
            processed_count += 1

        except Exception as e:
            logging.warning(f"處理第 {i+1} 人失敗: {e}")
            continue

    logging.info(f"成功處理 {processed_count}/{len(bboxes)} 人")
    return canvas
```

### 4. GPU 優化

```python
def _init_pose_detector(self):
    """初始化姿態檢測器（支持 GPU）"""
    logging.info("正在載入姿態檢測模型...")

    device = self.config.get_device()
    logging.info(f"使用設備: {device}")

    try:
        # 嘗試使用設備參數
        self.detector = DWposeDetector.from_pretrained(
            "yzd-v/DWPose",
            device=device
        )
        backend = f"DWPose (yzd-v/DWPose) on {device}"
    except Exception:
        # 備援方案
        self.detector = DWposeDetector.from_pretrained("yzd-v/DWPose")
        backend = "DWPose (yzd-v/DWPose)"

    logging.info(f"姿態檢測模型載入完畢: {backend}")
```

## 命令行參數

```bash
# 多人模式（逐人 ROI，默認）
python dance_multiperson.py -i input -o output --multi-person

# 全圖多人檢測（快速模式）
python dance_multiperson.py -i input -o output --multi-person --fullimage

# 單人模式（向後兼容）
python dance_multiperson.py -i input -o output --single-person

# GPU 加速
python dance_multiperson.py -i input -o output --device cuda

# 調整參數
python dance_multiperson.py -i input -o output \
    --multi-person \
    --max-persons 20 \
    --min-area 500 \
    --max-det 1000 \
    --device cuda:0
```

## 性能預期

### 計算量對比（假設 10 人場景）

| 模式 | 處理次數 | 相對速度 | 質量 |
|------|---------|---------|------|
| 原始（單人） | 1 | 1x | ⭐⭐⭐⭐⭐ (僅1人) |
| 逐人 ROI | 10 | 8-10x | ⭐⭐⭐⭐⭐ (所有人) |
| 全圖多人 | 1 | 1.2x | ⭐⭐⭐⭐ (所有人) |

### 建議配置

- **舞蹈/體育 (1-6人)**: 逐人 ROI + GPU + 手部增強
- **人群/監控 (7+人)**: 全圖多人 + GPU
- **混合場景**: 自動模式（根據人數切換）

## 測試場景

### 必測場景
1. ✅ 單人（確保向後兼容）
2. ✅ 2-3 人近景
3. ✅ 5-10 人中景
4. ✅ 20+ 人遠景/人群
5. ✅ 遮擋場景
6. ✅ 側面/背面
7. ✅ 小人物（<50px）
8. ✅ 混合尺度

### 驗證指標
- 人物檢出率: >95%
- 骨架完整度: >90%
- 手部準確度: >85%
- 處理速度: <2s/幀 (GPU)

---

**下一步**: 實施上述改進並進行測試
**預期效果**: 確保所有人物姿態都能被準確捕捉