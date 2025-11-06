# 增強版舞蹈姿態捕捉系統 (2025) - 完整指南

## 🌟 概述

**dance_enhanced.py** 是基於 2025 年最新研究成果的全面增強版本，整合多項前沿技術，確保**所有人物姿態都能被完整捕捉並推論成完整的骨架**。

---

## 🎯 核心目標

### 1. **確保所有人物都能被捕捉到**
- ✅ 多人模式支持（可開關）
- ✅ SoftNMS 處理重疊和遮擋
- ✅ 全圖多人保底檢測
- ✅ 小人物專用高解析度處理
- ✅ 動態人數追蹤（支持 1-10 人）

### 2. **推論成完整的骨架**
- ✅ 骨架推理補全（PGKC 啟發）
- ✅ 時序信息推斷缺失關鍵點
- ✅ 光流對齊補洞
- ✅ 多幀加權推理

### 3. **極致穩定性**
- ✅ Kalman 濾波器 bbox 平滑
- ✅ 三遍處理流程
- ✅ 自動調參機制
- ✅ 場景自適應

---

## 🔬 整合的 2025 年最新技術

### 1. **YOLO v8 + SoftNMS + OC-SORT**
基於 2024-2025 年的最新研究成果：

**SoftNMS（Soft Non-Maximum Suppression）**
- **來源**：Multi-Object Pedestrian Tracking Using Improved YOLOv8 and OC-SORT (2024)
- **效果**：處理重疊邊界框，提升 mAP@0.5:0.95 達 10.17%
- **原理**：使用 Gaussian 衰減分數，而非直接抑制
- **公式**：`new_score = score * exp(-(IoU^2) / sigma)`

**多尺度檢測**
- 1280 / 960 / 640 三個尺度
- 針對不同距離的人物優化
- 降低 confidence 閾值至 0.15（捕捉小人物）

**面積過濾**
- 最小人物面積：2000 像素（過濾雜訊）
- 小人物自動提升解析度到 1920
- 面積比例 < 15% 算小人物

---

### 2. **Kalman 濾波器時序平滑**
基於 2025 年 IEEE 研究：Kalman filter 在眼部追蹤中表現最佳（MSE 0.49, RMSE 0.70）

**實作細節**：
```python
狀態向量 (8 維): [x, y, w, h, dx, dy, dw, dh]
  - x, y, w, h: bbox 位置和尺寸
  - dx, dy, dw, dh: 速度

測量向量 (4 維): [x, y, w, h]

過程噪聲: 0.01 (可調)
測量噪聲: 0.1 (可調)
```

**優點**：
- 消除 YOLO 檢測抖動
- 適應動態場景
- 預測遮擋後的位置

---

### 3. **多人追蹤器（MultiPersonTracker）**
整合 ByteTrack 和 OC-SORT 的概念：

**追蹤策略**：
1. **IoU 匹配**：優先使用 IoU > 0.3 匹配現有追蹤
2. **Kalman 預測**：使用 Kalman 濾波器平滑 bbox
3. **CSRT 後備**：YOLO 丟失時使用 OpenCV CSRT 追蹤
4. **ID 管理**：自動分配和回收 track_id
5. **丟失容忍**：最多容忍 10 幀丟失

**追蹤信息**：
```python
@dataclass
class PersonTrack:
    track_id: int               # 唯一標識
    bbox: Tuple[int, int, int, int]
    bbox_history: deque         # bbox 歷史（用於平滑）
    kalman_filter: KalmanBBoxFilter
    last_seen_frame: int        # 最後看到的幀
    confidence: float           # 追蹤置信度
    csrt_tracker: Any           # CSRT 追蹤器
```

---

### 4. **骨架推理補全（PGKC 啟發）**
基於 2025 年 arXiv 論文：Pose Estimation of Space Targets Using KC-WLS

**Pose-Guided Keypoint Completion (PGKC)**：
- 使用歷史姿態推斷當前幀的缺失關鍵點
- 時序窗口：前後各 5 幀（可調）
- 置信度閾值：0.3（低於此值不參與推理）

**實作原理**：
```python
def infer_missing_skeleton(pose_sequence, current_idx, window=5):
    # 1. 檢測缺失區域（黑色區域）
    missing_mask = gray <= 10

    # 2. 收集前後窗口內的幀
    for idx in range(current_idx - window, current_idx + window):
        # 3. 計算時間距離權重
        temporal_weight = 1.0 / (temporal_distance + 1.0)

        # 4. 計算置信度權重（基於骨架密度）
        confidence_weight = neighbor_density

        # 5. 加權累積
        weight = temporal_weight * confidence_weight
        accumulated += neighbor_pose * weight
        weights_sum += weight

    # 6. 歸一化並填充缺失區域
    inferred[missing_mask] = (accumulated / weights_sum)[missing_mask]
```

**效果**：
- 推斷完全缺失的肢體
- 補全部分遮擋的關鍵點
- 保持時序一致性

---

### 5. **自動調參機制**
根據場景特性動態調整參數：

**場景分析**：
1. **亮度檢測**：`brightness = mean(gray)`
2. **動態檢測**：`motion = median(optical_flow_magnitude)`

**調參邏輯**：

| 場景 | 條件 | 調整參數 |
|------|------|----------|
| **暗光** | brightness < 100 | • 啟用降噪<br>• CLAHE clip_limit 3.5<br>• 降低密度要求 0.8x |
| **正常** | brightness >= 100 | • 關閉降噪<br>• CLAHE clip_limit 2.5<br>• 標準密度要求 |
| **高動態** | motion > 5.0 | • ROI 擴張 1.5x<br>• YOLO 頻率 +2<br>• 時序權重 0.15 |
| **低動態** | motion <= 5.0 | • ROI 擴張 1.3x<br>• YOLO 標準頻率<br>• 時序權重 0.2 |

**優點**：
- 無需手動調參
- 適應不同場景
- 優化性能和質量

---

### 6. **全圖多人保底檢測**
避免漏檢的最後防線：

**觸發條件**：
1. 沒有檢測到任何人
2. 所有人的密度總和 < 0.001（檢測效果極差）

**處理方式**：
```python
if ENABLE_FULLFRAME_FALLBACK and (no_persons or low_density):
    # 直接對整個畫面進行 DWPose 檢測
    fullframe_pose = detect_fullframe_multi_person(frame)

    # 與現有檢測結果混合（權重 0.5）
    canvas = cv2.addWeighted(canvas, 0.5, fullframe_pose, 0.5, 0.0)
```

**效果**：
- 捕捉被 YOLO 漏掉的人物
- 處理極端遮擋情況
- 保證至少有基本的骨架輸出

---

## 📋 配置參數詳解

### 多人模式配置

```python
# === 多人模式（核心）===
MULTI_PERSON_MODE = True          # True: 檢測所有人; False: 只檢測最大的人
MIN_PERSON_AREA = 2000            # 最小人物面積（過濾雜訊）
MAX_PERSONS_PER_FRAME = 10        # 每幀最多處理人數
ENABLE_FULLFRAME_FALLBACK = True  # 啟用全圖多人保底檢測
```

**調整建議**：
- **舞台表演**：`MULTI_PERSON_MODE = True`, `MAX_PERSONS_PER_FRAME = 10`
- **獨舞**：`MULTI_PERSON_MODE = False`（更快）
- **群舞**：`MULTI_PERSON_MODE = True`, `MAX_PERSONS_PER_FRAME = 20`

---

### YOLO 檢測策略

```python
# === YOLO 檢測策略 ===
DETECT_EVERY_N_FRAMES = 5         # YOLO 刷新頻率（多人模式建議 4-5）
YOLO_CONF_THRESHOLD = 0.15        # 降低閾值以捕捉小人物
YOLO_IOU_THRESHOLD = 0.4          # NMS IoU 閾值
ENABLE_SOFT_NMS = True            # 啟用 SoftNMS（處理遮擋）
SOFT_NMS_SIGMA = 0.5              # SoftNMS 參數
```

**SoftNMS Sigma 調整**：
| Sigma | 效果 | 適用場景 |
|-------|------|----------|
| 0.3 | 較強抑制 | 人物間距較大 |
| 0.5 | 平衡（推薦）| 一般場景 |
| 0.7 | 較弱抑制 | 密集人群、嚴重遮擋 |

---

### Kalman 濾波器

```python
# === Kalman 濾波器 ===
ENABLE_KALMAN_FILTER = True       # 啟用 Kalman 濾波器 bbox 平滑
KALMAN_PROCESS_NOISE = 0.01       # 過程噪聲（預測不確定性）
KALMAN_MEASUREMENT_NOISE = 0.1    # 測量噪聲（觀測不確定性）
```

**調整建議**：
| 場景 | PROCESS_NOISE | MEASUREMENT_NOISE | 說明 |
|------|---------------|-------------------|------|
| 穩定鏡頭 | 0.01 | 0.1 | 標準配置 |
| 晃動鏡頭 | 0.05 | 0.2 | 增加不確定性 |
| 高速運動 | 0.1 | 0.1 | 更快響應 |

---

### 第一遍：ROI 平滑（新增）

```python
# === 第一遍：高精度檢測 ===
PASS1_ENABLE_ROI_SMOOTH = True    # 第一遍啟用 ROI 平滑
PASS1_ROI_SMOOTH_ALPHA = 0.4      # ROI 平滑強度（0.3-0.5，較弱）
PASS1_SMALL_PERSON_THRESHOLD = 0.15  # 小於畫面 15% 算小人物
PASS1_SMALL_PERSON_LONG_SIDE = 1920  # 小人物專用解析度
```

**小人物處理**：
- 自動檢測人物面積比例
- 小人物（< 15% 畫面）使用 1920 解析度
- 大人物使用 1536 解析度
- 確保遠處人物也能清晰捕捉

---

### 第二遍：bbox 平滑

```python
# === 第二遍：平滑 + 穩定 ===
PASS2_BBOX_SMOOTH_ALPHA = 0.25    # 降低到 0.25（更平滑）
PASS2_ENABLE_TEMPORAL = True
PASS2_TEMPORAL_WEIGHT = 0.2
```

**對比 dance_stable.py**：
| 版本 | PASS2_BBOX_SMOOTH_ALPHA | 說明 |
|------|------------------------|------|
| dance_stable.py | 0.3 | 單人模式 |
| **dance_enhanced.py** | **0.25** | **多人模式，需要更平滑** |

---

### 第三遍：推理補全（增強）

```python
# === 第三遍：一致性修補 + 推理補全 ===
PASS3_ENABLE_REPAIR = True
PASS3_REPAIR_RADIUS = 3           # 增加到 3（查找更遠的幀）
PASS3_MIN_HOLE_SIZE = 30          # 降低到 30（修補更小的洞）
PASS3_ENABLE_INFERENCE = True     # 啟用骨架推理補全（新增）
PASS3_INFERENCE_WINDOW = 5        # 推理窗口大小
PASS3_INFERENCE_THRESHOLD = 0.3   # 推理置信度閾值
```

**推理補全 vs 光流補洞**：
| 方法 | 原理 | 優點 | 缺點 |
|------|------|------|------|
| **推理補全**（新） | 加權平均多幀姿態 | 保持姿態結構、時序一致 | 需要計算量 |
| **光流補洞**（舊） | 對齊並填充 | 保持細節 | 可能不連貫 |
| **兩者結合** | 先推理再補洞 | **最佳效果** | 處理時間較長 |

---

### 自動調參

```python
# === 自動調參 ===
ENABLE_AUTO_TUNING = True         # 啟用自動調參
AUTO_TUNE_FLOW_THRESHOLD = 5.0    # 高動態場景光流閾值
AUTO_TUNE_DARK_THRESHOLD = 100    # 暗光場景亮度閾值
```

**自動調參的參數**：
1. `enable_denoise` - 是否降噪
2. `clahe_clip_limit` - CLAHE 對比度增強強度
3. `density_multiplier` - 密度要求乘數
4. `roi_expand_ratio` - ROI 擴張比例
5. `detect_frequency` - YOLO 檢測頻率
6. `temporal_weight` - 時序混合權重

---

## 🚀 快速開始

### 1. 準備環境

```bash
# 安裝依賴
pip install -r requirements.txt

# GPU 支持（推薦）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 準備 YOLO 模型

下載並放置在同目錄：
- `yolov8n.pt`（推薦，更快）或
- `yolov8s.pt`（更穩定）

### 3. 修改配置

編輯 `dance_enhanced.py`，修改路徑（第 58-60 行）：

```python
class EnhancedConfig:
    INPUT_FOLDER = r"你的輸入資料夾路徑"
    OUTPUT_FOLDER = r"你的輸出資料夾路徑"
    TEMP_FOLDER = r"你的臨時資料夾路徑"
```

### 4. 選擇模式

**多人模式**（推薦）：
```python
MULTI_PERSON_MODE = True  # 檢測所有人
```

**單人模式**（更快）：
```python
MULTI_PERSON_MODE = False  # 只檢測最大的人
```

### 5. 運行

```bash
python dance_enhanced.py
```

---

## 📊 處理流程詳解

### 第一遍：高精度多人檢測

```
1. 場景分析
   └─> 計算亮度和動態程度
   └─> 自動調整參數

2. YOLO 多人檢測（每 N 幀）
   └─> 多尺度檢測（1280/960/640）
   └─> SoftNMS 處理重疊
   └─> 面積過濾（> 2000 px）
   └─> 限制人數（<= 10）

3. 多人追蹤更新
   └─> IoU 匹配現有追蹤
   └─> Kalman 濾波器平滑 bbox
   └─> CSRT 追蹤補充（YOLO 失效時）
   └─> 自動分配 track_id

4. 逐人處理
   └─> 判斷是否為小人物（< 15% 畫面）
   └─> 小人物 → 1920 解析度
   └─> 大人物 → 1536 解析度
   └─> 擴展 ROI（1.3x 或動態調整）
   └─> 手部區域增強
   └─> DWPose 檢測
   └─> 貼回 canvas

5. 全圖保底（如果需要）
   └─> 檢測效果差 → 全圖 DWPose
   └─> 與現有結果混合（50%）

6. 保存
   └─> temp/pass1_poses/ - 初步檢測結果
   └─> temp/pass1_frames/ - 預處理幀
   └─> temp/pass1_metadata.json - 元數據
```

**輸出元數據示例**：
```json
{
  "filename": "frame_0001.jpg",
  "frame_index": 0,
  "persons": [
    {
      "track_id": 0,
      "bbox": [450, 120, 890, 960],
      "density": 0.0234,
      "area_ratio": 0.45,
      "confidence": 1.0
    },
    {
      "track_id": 1,
      "bbox": [1100, 200, 1400, 950],
      "density": 0.0189,
      "area_ratio": 0.28,
      "confidence": 0.95
    }
  ],
  "scene_brightness": 128.5,
  "scene_motion": 3.2,
  "auto_tuned_params": {
    "enable_denoise": false,
    "clahe_clip_limit": 2.5,
    "density_multiplier": 1.0,
    "roi_expand_ratio": 1.3,
    "detect_frequency": 5,
    "temporal_weight": 0.2
  }
}
```

---

### 第二遍：bbox 平滑 + 重新檢測

```
1. 構建追蹤序列
   └─> 按 track_id 分組所有 bbox
   └─> 每個人一個獨立序列

2. 雙向 EMA 平滑
   └─> 對每個追蹤序列的 bbox 平滑
   └─> alpha = 0.25（比單人模式更平滑）
   └─> 前向平滑 + 後向平滑 → 平均

3. 逐人重新檢測
   └─> 使用平滑後的 bbox
   └─> 擴展並裁切 ROI
   └─> 手部增強
   └─> DWPose 檢測
   └─> 貼回 canvas

4. 時序穩定化（輕量）
   └─> 光流對齊前一幀
   └─> 混合當前幀（80%）+ 對齊前幀（20%）
   └─> 權重可根據場景動態調整

5. 保存
   └─> temp/pass2_poses/ - 平滑後結果
```

**效果**：
- 每個人的 bbox 獨立平滑，互不干擾
- 消除追蹤抖動
- 時序更連貫

---

### 第三遍：推理補全 + 修補

```
1. 加載姿態序列
   └─> 讀取所有第二遍的結果
   └─> 構建完整序列（用於推理）

2. 骨架推理補全（新增）
   └─> 檢測缺失區域（黑色 <= 10）
   └─> 收集前後 5 幀（窗口可調）
   └─> 計算時間距離權重
   └─> 計算置信度權重（基於密度）
   └─> 加權平均推斷缺失部分

3. 光流補洞
   └─> 檢測大洞（面積 >= 30 px）
   └─> 查找前後 3 幀
   └─> 光流對齊相鄰幀
   └─> 在洞的位置填充

4. 保存最終結果
   └─> OUTPUT_FOLDER/ - 完整骨架
```

**推理補全示例**：
```
幀 100 有缺失的左手

1. 收集前後幀：
   - 幀 95, 96, 97, 98, 99 (前)
   - 幀 101, 102, 103, 104, 105 (後)

2. 計算權重：
   - 幀 99: temporal_weight = 1/(1+1) = 0.5
   - 幀 98: temporal_weight = 1/(2+1) = 0.33
   - 幀 101: temporal_weight = 1/(1+1) = 0.5
   - ...

3. 加權累積：
   inferred_hand = Σ(neighbor_hand * weight) / Σ(weight)

4. 填充缺失區域
```

---

## 📈 性能對比

### 與其他版本的對比

| 特性 | dance_speed | dance_stable | **dance_enhanced** |
|------|------------|--------------|-------------------|
| **速度** | ⚡⚡⚡⚡ | ⚡⚡ | ⚡⚡ |
| **穩定性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **多人支持** | 單人 | 單人 | **✅ 1-10 人** |
| **骨架補全** | 無 | 補洞 | **✅ 推理+補洞** |
| **自動調參** | 無 | 無 | **✅ 場景自適應** |
| **遮擋處理** | 一般 | 良好 | **✅ SoftNMS** |
| **小人物** | 一般 | 良好 | **✅ 專用高解析度** |
| **bbox 平滑** | EMA | 雙向 EMA | **✅ Kalman 濾波器** |

---

### 處理時間估算

假設 100 幀序列，2 人場景，GPU 環境：

| 階段 | 時間 | 說明 |
|------|------|------|
| 第一遍 | ~20-30 秒 | 多人檢測 + 追蹤 |
| 第二遍 | ~25-35 秒 | 平滑 bbox + 重檢測 |
| 第三遍 | ~10-15 秒 | 推理補全 + 補洞 |
| **總計** | ~55-80 秒 | 相當於 ~1.5-2 FPS |

**影響因素**：
- 人數：線性增長（2 人 vs 1 人約 1.8x）
- 解析度：高解析度小人物較慢
- GPU：CUDA 加速約 3-5x

---

## 🔧 調試和監控

### 啟用詳細輸出

```python
DEBUG_VERBOSE = True              # 詳細日誌
SAVE_METADATA = True              # 保存元數據
SHOW_PROGRESS = True              # 顯示處理進度
SHOW_PERSON_COUNT = True          # 顯示每幀檢測到的人數
```

### 輸出示例

```
第一遍：高精度多人檢測（增強版）
============================================================
多人模式: True
Kalman 濾波器: True
SoftNMS: True
自動調參: True

  [1/100] frame_0001.jpg | 檢測到 2 人
  [2/100] frame_0002.jpg | 檢測到 2 人
  ...

✓ 元數據已保存: temp/pass1_metadata.json

第二遍：bbox 平滑 + 重新檢測（多人）
============================================================
✓ 已平滑 2 個追蹤序列

  [1/100] frame_0001.jpg | 2 人 | 總密度=0.045678
  [2/100] frame_0002.jpg | 2 人 | 總密度=0.046123
  ...

第三遍：一致性修補 + 骨架推理補全（增強版）
============================================================

  [1/100] frame_0001.jpg | 修補了 2.34% 的洞
  [2/100] frame_0002.jpg | 無需修補
  ...

============================================================
✓ 全部處理完畢！
============================================================
輸出資料夾: H:\202511_Calm down AI 專案\pose
臨時資料夾: H:\202511_Calm down AI 專案\temp
```

---

## 🐛 故障排除

### 問題 1：檢測不到某些人

**可能原因**：
1. 人物太小（面積 < 2000 px）
2. YOLO confidence 閾值太高
3. 被 SoftNMS 抑制

**解決方案**：
```python
# 降低面積閾值
MIN_PERSON_AREA = 1000  # 從 2000 降到 1000

# 降低 confidence 閾值
YOLO_CONF_THRESHOLD = 0.1  # 從 0.15 降到 0.1

# 調整 SoftNMS
SOFT_NMS_SIGMA = 0.7  # 從 0.5 增到 0.7（較弱抑制）

# 確保全圖保底開啟
ENABLE_FULLFRAME_FALLBACK = True
```

---

### 問題 2：骨架不完整

**可能原因**：
1. 推理置信度閾值太高
2. 推理窗口太小
3. 補洞半徑太小

**解決方案**：
```python
# 降低推理閾值
PASS3_INFERENCE_THRESHOLD = 0.2  # 從 0.3 降到 0.2

# 增加推理窗口
PASS3_INFERENCE_WINDOW = 7  # 從 5 增到 7

# 增加補洞半徑
PASS3_REPAIR_RADIUS = 5  # 從 3 增到 5

# 降低最小洞大小
PASS3_MIN_HOLE_SIZE = 20  # 從 30 降到 20
```

---

### 問題 3：仍有抖動

**可能原因**：
1. Kalman 濾波器參數不當
2. bbox 平滑不足
3. 時序權重太低

**解決方案**：
```python
# 調整 Kalman 濾波器
KALMAN_MEASUREMENT_NOISE = 0.2  # 增加測量噪聲（更平滑）

# 增強 bbox 平滑
PASS2_BBOX_SMOOTH_ALPHA = 0.2  # 從 0.25 降到 0.2

# 增強時序穩定
PASS2_TEMPORAL_WEIGHT = 0.3  # 從 0.2 增到 0.3
```

---

### 問題 4：處理速度太慢

**解決方案**：

**1. 限制人數**
```python
MAX_PERSONS_PER_FRAME = 5  # 從 10 降到 5
```

**2. 降低檢測頻率**
```python
DETECT_EVERY_N_FRAMES = 8  # 從 5 增到 8
```

**3. 降低解析度**
```python
PASS1_ENHANCED_LONG_SIDE = 1280  # 從 1536 降到 1280
PASS1_SMALL_PERSON_LONG_SIDE = 1536  # 從 1920 降到 1536
PASS2_TARGET_LONG_SIDE = 1280  # 從 1536 降到 1280
```

**4. 關閉推理補全**（如果不需要）
```python
PASS3_ENABLE_INFERENCE = False
```

**5. 單人模式**（如果只有一個人）
```python
MULTI_PERSON_MODE = False
```

---

## 🎯 最佳實踐

### 1. 根據場景選擇模式

| 場景 | 配置 |
|------|------|
| **獨舞** | `MULTI_PERSON_MODE = False` |
| **雙人舞** | `MULTI_PERSON_MODE = True`, `MAX_PERSONS = 2` |
| **群舞（<10人）** | `MULTI_PERSON_MODE = True`, `MAX_PERSONS = 10` |
| **大型演出（>10人）** | `MULTI_PERSON_MODE = True`, `MAX_PERSONS = 20` |

### 2. 調參優先級

**遇到問題時的調參順序**：

1. **漏檢人物** → 調整 YOLO 參數
2. **骨架不完整** → 調整推理補全參數
3. **抖動** → 調整 Kalman 和平滑參數
4. **速度慢** → 降低解析度和人數限制

### 3. 序列圖命名

```
✅ 正確：
frame_0001.jpg
frame_0002.jpg
frame_0003.jpg

❌ 錯誤：
frame1.jpg
frame2.jpg
frame10.jpg  # 排序會亂
```

### 4. 查看中間結果

處理完成後，檢查臨時資料夾：
```
temp/
  ├── pass1_poses/        # 第一遍：多人檢測
  ├── pass1_frames/       # 預處理後的幀
  ├── pass2_poses/        # 第二遍：平滑後
  └── pass1_metadata.json # 完整元數據
```

**分析元數據**：
```python
import json

with open('temp/pass1_metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 統計每幀的人數
person_counts = [len(frame['persons']) for frame in metadata]
print(f"平均人數: {sum(person_counts) / len(person_counts):.2f}")
print(f"最多人數: {max(person_counts)}")
print(f"最少人數: {min(person_counts)}")

# 統計場景亮度
brightnesses = [frame['scene_brightness'] for frame in metadata]
print(f"平均亮度: {sum(brightnesses) / len(brightnesses):.2f}")

# 統計場景動態
motions = [frame['scene_motion'] for frame in metadata]
print(f"平均動態: {sum(motions) / len(motions):.2f}")
```

---

## 📚 技術細節

### SoftNMS 數學原理

傳統 NMS：
```python
if IoU(bbox_i, bbox_max) > threshold:
    score_i = 0  # 直接抑制
```

SoftNMS：
```python
if IoU(bbox_i, bbox_max) > threshold:
    score_i = score_i * exp(-(IoU^2) / sigma)  # Gaussian 衰減
```

**優點**：
- 不完全抑制，保留被遮擋的人物
- 分數衰減程度與重疊程度成正比
- 適合多人密集場景

---

### Kalman 濾波器狀態方程

**預測步驟**：
```
X_k|k-1 = F * X_k-1|k-1
P_k|k-1 = F * P_k-1|k-1 * F^T + Q
```

**更新步驟**：
```
K_k = P_k|k-1 * H^T * (H * P_k|k-1 * H^T + R)^-1
X_k|k = X_k|k-1 + K_k * (Z_k - H * X_k|k-1)
P_k|k = (I - K_k * H) * P_k|k-1
```

其中：
- `X`: 狀態向量 [x, y, w, h, dx, dy, dw, dh]
- `Z`: 測量向量 [x, y, w, h]
- `F`: 狀態轉移矩陣
- `H`: 測量矩陣
- `Q`: 過程噪聲協方差
- `R`: 測量噪聲協方差
- `K`: Kalman 增益

---

### 推理補全加權公式

```python
# 時間距離權重
w_temporal = 1 / (|idx - current_idx| + 1)

# 置信度權重（基於密度）
w_confidence = density  # if density > threshold else 0

# 組合權重
w = w_temporal * w_confidence

# 加權累積
accumulated = Σ(neighbor_pose * w)
weights_sum = Σ(w)

# 歸一化
inferred = accumulated / weights_sum
```

**示例**：
```
當前幀：100
窗口：5

幀 95: w_temporal = 1/6 = 0.167, density = 0.8 → w = 0.133
幀 96: w_temporal = 1/5 = 0.200, density = 0.9 → w = 0.180
幀 97: w_temporal = 1/4 = 0.250, density = 0.85 → w = 0.213
幀 98: w_temporal = 1/3 = 0.333, density = 0.95 → w = 0.317
幀 99: w_temporal = 1/2 = 0.500, density = 1.0 → w = 0.500

幀 101: w_temporal = 1/2 = 0.500, density = 0.9 → w = 0.450
幀 102: w_temporal = 1/3 = 0.333, density = 0.85 → w = 0.283
幀 103: w_temporal = 1/4 = 0.250, density = 0.8 → w = 0.200
幀 104: w_temporal = 1/5 = 0.200, density = 0.75 → w = 0.150
幀 105: w_temporal = 1/6 = 0.167, density = 0.7 → w = 0.117

總權重和：2.543
推斷結果 = Σ(pose * w) / 2.543
```

---

## 🌐 基於的研究文獻

### 核心技術來源

1. **DWPose & RTMW**
   - Yang et al., "Effective Whole-body Pose Estimation with Two-stages Distillation", ICCV 2023
   - arXiv:2407.08634, "RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation"

2. **YOLOv8 + SoftNMS**
   - "Multi-Object Pedestrian Tracking Using Improved YOLOv8 and OC-SORT", Sensors 2023
   - mAP@0.5:0.95 提升 10.17%

3. **Kalman 濾波器**
   - "Comparative Analysis of Filtering Techniques in Eye Landmark Tracking", IEEE 2025
   - MSE: 0.49, RMSE: 0.70（優於 Savitzky-Golay）

4. **PGKC（骨架推理補全）**
   - "Pose estimation of space targets based on keypoint completion and weighted least squares optimization", arXiv 2025
   - Di2Pose: diffusion-based occlusion handling

5. **Cycle Skeleton Structure**
   - "Human pose estimation in crowded scenes using Keypoint Likelihood Variance Reduction", ScienceDirect 2024
   - 處理嚴重遮擋的鍵點分配

---

**版本**: 2.0 Enhanced
**更新日期**: 2025-11-06
**基於**: dance_stable.py v1.0
**整合技術**: 2024-2025 年最新研究成果
**作者**: Claude Code
**授權**: 與原項目相同
