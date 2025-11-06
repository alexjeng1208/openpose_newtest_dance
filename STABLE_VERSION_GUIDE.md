# 單人舞蹈姿態穩定捕捉系統 - 使用指南

## 🎯 解決的問題

**dance_stable.py** 專門解決姿態檢測中的抽動和跑偏問題，通過三遍處理流程提供序列級的穩定性優化。

### 問題現象
- ✗ 骨架檢測有抽動（jittering）
- ✗ 追蹤不穩定，人物框會跑偏
- ✗ 幀與幀之間的骨架位置不連貫
- ✗ 部分幀出現骨架缺失（空洞）

### 解決方案
- ✓ 三遍處理流程，充分利用序列上下文信息
- ✓ 雙向 EMA 平滑 bbox 序列
- ✓ 時序穩定化（光流對齊混合）
- ✓ 一致性修補（從相鄰幀補洞）

---

## 🔄 三遍處理流程

### 第一遍：高精度檢測

**目標**：檢測所有幀，記錄完整的元數據

**流程**：
```
對於每一幀：
  1. 讀取並預處理圖像
  2. YOLO + CSRT 追蹤檢測人物框
  3. 擴展並裁切 ROI
  4. 手部區域增強
  5. DWPose 檢測姿態
  6. 貼回 canvas 並清理
  7. 計算密度並記錄元數據
  8. 保存到 temp/pass1_poses/
```

**輸出**：
- `temp/pass1_poses/` - 初步檢測的骨架圖
- `temp/pass1_frames/` - 預處理後的原始幀
- `temp/pass1_metadata.json` - 所有幀的元數據（bbox、密度、檢測方法等）

**特點**：
- 使用追蹤優先策略（YOLO 每 N 幀刷新）
- 記錄每幀的檢測信息，為後續平滑做準備
- 無殘影，每幀獨立檢測

---

### 第二遍：bbox 平滑 + 重新檢測

**目標**：消除 bbox 抖動，提高姿態檢測的穩定性

**流程**：
```
1. 從元數據中提取所有 bbox 序列
2. 對 bbox 坐標進行雙向 EMA 平滑
   - 前向平滑：x_smooth[i] = alpha * x[i] + (1-alpha) * x_smooth[i-1]
   - 後向平滑：逆向再平滑一次
   - 最終結果：取前向和後向的平均
3. 使用平滑後的 bbox 重新檢測姿態
4. 可選：輕量時序穩定化
   - 使用光流將前一幀的姿態對齊到當前幀
   - 混合當前幀和對齊後的前一幀：
     result = current * (1-w) + warped_prev * w
```

**輸出**：
- `temp/pass2_poses/` - 平滑後重新檢測的骨架圖

**參數調優**：
```python
PASS2_BBOX_SMOOTH_ALPHA = 0.3    # bbox 平滑強度 (0.1-0.5)
                                 # 越小越平滑，但可能滯後
                                 # 推薦: 0.3（平衡）

PASS2_ENABLE_TEMPORAL = True     # 啟用時序穩定
PASS2_TEMPORAL_WEIGHT = 0.2      # 時序混合權重 (0.1-0.3)
                                 # 越大越平滑，但可能模糊
                                 # 推薦: 0.2（輕量）
```

**效果**：
- 消除 bbox 抖動導致的姿態跳變
- 時序上更連貫
- 減少幀間抽動

---

### 第三遍：一致性修補

**目標**：填補骨架中的空洞，保證完整性

**流程**：
```
對於每一幀：
  1. 讀取第二遍的骨架圖
  2. 檢測黑色區域（骨架缺失）
  3. 過濾小洞（面積 < MIN_HOLE_SIZE）
  4. 對於每個大洞：
     - 在前後 N 幀範圍內查找相鄰幀
     - 使用光流將相鄰幀的骨架對齊到當前幀
     - 在洞的位置填充對齊後的骨架
  5. 保存最終結果到輸出資料夾
```

**輸出**：
- 最終的穩定骨架圖（輸出到配置的 OUTPUT_FOLDER）

**參數調優**：
```python
PASS3_ENABLE_REPAIR = True       # 啟用修補
PASS3_REPAIR_RADIUS = 2          # 修補半徑（前後幀數）
                                 # 範圍: 1-5
                                 # 推薦: 2（平衡速度和效果）

PASS3_MIN_HOLE_SIZE = 50         # 最小補洞面積（像素）
                                 # 太小的洞不修補（避免過度處理）
                                 # 推薦: 50
```

**效果**：
- 填補骨架缺失
- 保證骨架完整性
- 時序一致性更強

---

## ⚙️ 配置參數

### 路徑配置

```python
INPUT_FOLDER = r"H:\202511_Calm down AI 專案\calm down"   # 輸入序列圖資料夾
OUTPUT_FOLDER = r"H:\202511_Calm down AI 專案\pose"       # 最終輸出資料夾
TEMP_FOLDER = r"H:\202511_Calm down AI 專案\temp"         # 中間結果（可刪除）
```

**重要**：修改為你的實際路徑！

---

### 檢測策略

```python
DETECT_EVERY_N_FRAMES = 6        # YOLO 刷新頻率
                                 # 4-5: 更準確，適合快速移動
                                 # 6-8: 平衡（推薦）
                                 # 10+: 更快，適合慢動作
```

---

### 解析度配置

```python
# 第一遍
PASS1_TARGET_LONG_SIDE = 1280    # 初始解析度
PASS1_ENHANCED_LONG_SIDE = 1536  # ROI 放大解析度

# 第二遍
PASS2_TARGET_LONG_SIDE = 1536    # 平滑後重檢測解析度

# 全局
MAX_LONG_SIDE = 1920             # 最大解析度限制
```

**建議**：
- 低階顯卡：1280 / 1280 / 1536
- 中階顯卡：1280 / 1536 / 1920（默認）
- 高階顯卡：1536 / 1920 / 2048

---

### 穩定性參數（關鍵）

#### bbox 平滑強度
```python
PASS2_BBOX_SMOOTH_ALPHA = 0.3
```
| 值 | 效果 | 適用場景 |
|----|------|----------|
| 0.1-0.2 | 極度平滑，可能滯後 | 靜態或緩慢移動 |
| 0.3 | 平衡（推薦）| 一般舞蹈 |
| 0.4-0.5 | 較少平滑，響應快 | 快速、激烈舞蹈 |

#### 時序混合權重
```python
PASS2_TEMPORAL_WEIGHT = 0.2
```
| 值 | 效果 | 適用場景 |
|----|------|----------|
| 0.1 | 輕量穩定 | 檢測已經比較穩定 |
| 0.2 | 平衡（推薦）| 一般抽動情況 |
| 0.3 | 強力穩定，可能模糊 | 嚴重抽動 |

#### 修補半徑
```python
PASS3_REPAIR_RADIUS = 2
```
| 值 | 效果 | 速度 |
|----|------|------|
| 1 | 僅查找相鄰 1 幀 | 快 |
| 2 | 查找前後 2 幀（推薦）| 中 |
| 3-5 | 查找更遠的幀 | 慢 |

---

## 🚀 快速開始

### 1. 準備環境

```bash
# 安裝依賴
pip install -r requirements.txt

# 如果有 GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 準備 YOLO 模型

下載並放置在同目錄：
- `yolov8n.pt`（推薦，更快）或
- `yolov8s.pt`（更穩定）

### 3. 修改配置

編輯 `dance_stable.py`，修改路徑：

```python
class StableConfig:
    INPUT_FOLDER = r"你的輸入資料夾路徑"
    OUTPUT_FOLDER = r"你的輸出資料夾路徑"
    TEMP_FOLDER = r"你的臨時資料夾路徑"
```

### 4. 運行

```bash
python dance_stable.py
```

---

## 📊 處理流程時間估算

假設 100 幀序列，GPU 環境：

| 階段 | 時間 | 說明 |
|------|------|------|
| 第一遍 | ~10-15 秒 | 初始檢測 + 元數據記錄 |
| 第二遍 | ~15-20 秒 | 平滑 bbox + 重新檢測 |
| 第三遍 | ~5-10 秒 | 補洞（取決於空洞數量）|
| **總計** | ~30-45 秒 | 相當於 ~2-3 FPS |

**注意**：
- 第一遍和第二遍都需要完整的 DWPose 檢測，所以時間較長
- 第三遍只是圖像處理，速度較快
- 總時間約為單遍處理的 2-3 倍，但穩定性大幅提升

---

## 🔧 調試和監控

### 啟用詳細進度

```python
DEBUG_VERBOSE = True    # 詳細日誌
SHOW_PROGRESS = True    # 顯示處理進度
```

### 查看中間結果

處理完成後，檢查臨時資料夾：
```
temp/
  ├── pass1_poses/        # 第一遍檢測結果
  ├── pass1_frames/       # 預處理後的幀
  ├── pass2_poses/        # 第二遍平滑後結果
  └── pass1_metadata.json # 元數據
```

### 元數據示例

```json
[
  {
    "filename": "frame_0001.jpg",
    "frame_index": 0,
    "bbox": [450, 120, 890, 960],
    "density": 0.0234,
    "width": 1920,
    "height": 1080,
    "detection_method": "yolo"
  },
  ...
]
```

---

## 🐛 故障排除

### 問題：處理速度太慢

**檢查清單**：
1. ✅ 是否使用 GPU？查看啟動日誌
2. ✅ DETECT_EVERY_N_FRAMES 是否過小？建議 >= 6
3. ✅ 解析度是否過高？降低 PASS1/PASS2_TARGET_LONG_SIDE

**加速配置**：
```python
DETECT_EVERY_N_FRAMES = 8           # 增加到 8
PASS1_TARGET_LONG_SIDE = 960        # 降低解析度
PASS2_TARGET_LONG_SIDE = 1280
PASS3_ENABLE_REPAIR = False         # 如果沒有空洞，可關閉
```

---

### 問題：仍有抖動

**解決方案 1：增強 bbox 平滑**
```python
PASS2_BBOX_SMOOTH_ALPHA = 0.2       # 從 0.3 降到 0.2（更平滑）
```

**解決方案 2：增強時序穩定**
```python
PASS2_ENABLE_TEMPORAL = True
PASS2_TEMPORAL_WEIGHT = 0.3         # 從 0.2 增到 0.3（更強混合）
```

**解決方案 3：更頻繁檢測**
```python
DETECT_EVERY_N_FRAMES = 4           # 從 6 降到 4（更頻繁 YOLO）
```

---

### 問題：骨架有殘影或模糊

**原因**：時序混合權重過高

**解決**：
```python
PASS2_TEMPORAL_WEIGHT = 0.1         # 降低到 0.1
# 或者
PASS2_ENABLE_TEMPORAL = False       # 直接關閉時序穩定
```

---

### 問題：補洞效果不佳

**調整修補參數**：
```python
PASS3_REPAIR_RADIUS = 3             # 增加到 3（查找更遠的幀）
PASS3_MIN_HOLE_SIZE = 30            # 降低到 30（修補更小的洞）
```

---

## 📈 與其他版本對比

| 版本 | 速度 | 穩定性 | 適用場景 |
|------|------|--------|----------|
| `dance_speed_optimized.py` | ⚡⚡⚡⚡ | ⭐⭐⭐ | 速度優先，可接受輕微抖動 |
| `dance_stable.py` | ⚡⚡ | ⭐⭐⭐⭐⭐ | **穩定性優先，不能接受抖動**（本版本）|

**選擇建議**：
- 如果你的應用對穩定性要求極高（如專業舞蹈分析），使用 `dance_stable.py`
- 如果你需要準實時處理，可以接受輕微抖動，使用 `dance_speed_optimized.py`

---

## 🎯 最佳實踐

### 1. 序列圖命名

確保文件按順序命名：
```
frame_0001.jpg
frame_0002.jpg
frame_0003.jpg
...
```

### 2. 參數調優流程

```
步驟 1：使用默認參數運行
步驟 2：檢查輸出，識別問題
步驟 3：根據問題調整參數
  - 抖動嚴重 → 降低 PASS2_BBOX_SMOOTH_ALPHA
  - 仍有跳變 → 增加 PASS2_TEMPORAL_WEIGHT
  - 有空洞 → 增加 PASS3_REPAIR_RADIUS
步驟 4：重新運行並驗證
```

### 3. 批量處理

處理多個資料夾時，使用腳本：

```bash
#!/bin/bash
folders=("dance1" "dance2" "dance3")
for folder in "${folders[@]}"; do
    # 修改配置中的路徑
    sed -i "s|INPUT_FOLDER = .*|INPUT_FOLDER = r\"$folder\"|" dance_stable.py
    python dance_stable.py
done
```

### 4. 臨時文件管理

處理完成後，如果確認結果正常，可刪除臨時資料夾以節省空間：
```bash
rm -rf temp/
```

如需調試或重新處理，保留 `temp/pass1_metadata.json`，可跳過第一遍處理。

---

## 🔬 技術細節

### 雙向 EMA 平滑原理

```python
# 前向平滑
forward[0] = values[0]
forward[i] = alpha * values[i] + (1-alpha) * forward[i-1]

# 後向平滑
backward[n-1] = values[n-1]
backward[i] = alpha * values[i] + (1-alpha) * backward[i+1]

# 最終結果
result[i] = (forward[i] + backward[i]) / 2
```

**優點**：
- 消除因果性偏差（前向平滑會滯後）
- 保持序列端點的準確性
- 更平滑的結果

---

### 光流對齊原理

```python
# 計算從 source 到 target 的光流
flow = calcOpticalFlowFarneback(source_gray, target_gray)

# 構建映射網格
map_x = x + flow[:,:,0]
map_y = y + flow[:,:,1]

# 對齊 source_pose
warped = remap(source_pose, map_x, map_y)
```

**用途**：
1. **時序穩定**：將前一幀的姿態對齊到當前幀，混合以減少抖動
2. **補洞**：從相鄰幀對齊骨架，填補當前幀的空洞

---

## 📝 輸出說明

### 輸出格式

- **純骨架圖**：黑底 + 彩色骨架線條
- **無殘影**：每幀獨立（第一遍），或輕量混合（第二遍）
- **格式**：與輸入相同（PNG/JPG/JPEG/WEBP）

### 骨架組成

- **身體**：18 個關鍵點
- **手部**：每手 21 個關鍵點（如啟用）
- **臉部**：68 個關鍵點（如啟用，默認關閉）

---

**版本**: 1.0
**更新日期**: 2025-11-06
**作者**: Claude Code
**授權**: 與原項目相同
