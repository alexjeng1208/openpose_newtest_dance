# DWPose 載入修復說明 (2025-01-07)

## 問題描述

之前的版本使用了已棄用的 `DWposeDetector.from_pretrained()` API，導致以下錯誤：

```
✗ DWPose 載入失敗: type object 'DWposeDetector' has no attribute 'from_pretrained'
```

## 修復內容

### 1. 更新了以下文件的模型載入代碼：

- `dance_ultimate.py` (lines 203-226)
- `dance_enhanced.py` (lines 556-582)
- `dance_stable.py` (lines 244-270)

### 2. 新的載入策略（多重回退）：

**方法 1 - easy_dwpose (推薦)**
```python
from easy_dwpose import DWposeDetector as EasyDWpose
detector = EasyDWpose(device=device)
```

優點：
- ONNX 模型，速度快
- 不需要 MMDetection/MMCV/MMPose 等重型依賴
- 自動下載模型

**方法 2 - controlnet_aux 直接初始化**
```python
from controlnet_aux import DWposeDetector
detector = DWposeDetector()
```

優點：
- 使用 controlnet_aux 內建實現
- 無需額外安裝

**方法 3 - OpenPose 作為替代**
```python
from controlnet_aux import OpenposeDetector
detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
```

優點：
- 當 DWPose 無法載入時的穩定替代方案
- 同樣提供良好的姿態檢測

### 3. OpenPose 模型名稱修正

修正前：
```python
OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
```

修正後（優先嘗試）：
```python
OpenposeDetector.from_pretrained("lllyasviel/Annotators")
```

## 安裝指南

### 基本安裝

```bash
# 安裝核心依賴
pip install -r requirements.txt
```

### 推薦：安裝 easy-dwpose

```bash
# 安裝 easy-dwpose 以獲得最佳 DWPose 支持
pip install easy-dwpose
```

### 完整安裝（包含 PyTorch）

```bash
# 如果還沒有安裝 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# 或
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# 安裝所有依賴
pip install -r requirements.txt
pip install easy-dwpose
```

## 測試修復

執行以下命令測試模型載入：

```bash
python3 test_model_loading.py
```

或手動測試：

```bash
python3 -c "
from controlnet_aux import DWposeDetector
detector = DWposeDetector()
print('✓ DWPose 載入成功！')
"
```

## 驗證成功標誌

當您看到以下任一訊息時，表示載入成功：

```
✓ DWPose 載入完畢: DWPOSE:easy_dwpose (CUDA)
```
或
```
✓ DWPose 載入完畢: DWPOSE:controlnet_aux
```
或（作為替代）
```
DWPose 載入失敗，改用 OpenPose: ...
✓ 模型載入完畢: OPENPOSE:lllyasviel/Annotators
```

## 技術細節

### controlnet_aux API 變更

舊版 controlnet_aux 使用 `from_pretrained()` 方法：
```python
# ❌ 已棄用
detector = DWposeDetector.from_pretrained("yzd-v/DWPose")
```

新版 controlnet_aux 使用直接初始化：
```python
# ✓ 正確
detector = DWposeDetector()
```

### easy-dwpose 優勢

1. **輕量級**: 使用 ONNX 模型，不需要 PyTorch 訓練框架
2. **快速**: ONNX Runtime 優化推理速度
3. **簡單**: 自動處理模型下載和配置
4. **穩定**: 專門為 ControlNet 應用優化

## 故障排除

### 問題 1: "No module named 'easy_dwpose'"

**解決方案**:
```bash
pip install easy-dwpose
```

### 問題 2: "No module named 'controlnet_aux'"

**解決方案**:
```bash
pip install controlnet-aux
```

### 問題 3: 所有模型都載入失敗

**解決方案**:
1. 檢查網絡連接（模型需要從 HuggingFace 下載）
2. 檢查 PyTorch 安裝：
   ```bash
   python3 -c "import torch; print(torch.__version__)"
   ```
3. 嘗試手動安裝模型：
   ```bash
   python3 -c "
   from huggingface_hub import hf_hub_download
   hf_hub_download(repo_id='lllyasviel/Annotators', filename='body_pose_model.pth')
   "
   ```

### 問題 4: CUDA 記憶體不足

**解決方案**: 修改配置使用 CPU
```python
# 在文件開頭找到 DEVICE 設定
DEVICE = "cpu"  # 改為 CPU
```

## 參考資源

- [controlnet_aux GitHub](https://github.com/huggingface/controlnet_aux)
- [easy-dwpose PyPI](https://pypi.org/project/easy-dwpose/)
- [controlnet-dwpose GitHub](https://github.com/kapong/controlnet_dwpose)

## 更新日誌

**2025-01-07**:
- 修復 DWposeDetector.from_pretrained() API 錯誤
- 添加 easy-dwpose 支持
- 更新 OpenPose 模型名稱
- 添加多重回退機制
- 更新文檔和安裝指南
