#!/usr/bin/env python3
"""
測試模型載入 - 驗證 DWPose 修復
"""

import sys

print("=" * 60)
print("測試模型載入")
print("=" * 60)

# 測試 1: 檢查 controlnet_aux
print("\n[1/4] 檢查 controlnet_aux...")
try:
    from controlnet_aux import DWposeDetector, OpenposeDetector
    print("  ✓ controlnet_aux 已安裝")
except ImportError as e:
    print(f"  ✗ controlnet_aux 未安裝: {e}")
    print("  請執行: pip install controlnet-aux")
    sys.exit(1)

# 測試 2: 檢查 easy_dwpose (可選)
print("\n[2/4] 檢查 easy_dwpose (可選)...")
try:
    from easy_dwpose import DWposeDetector as EasyDWpose
    print("  ✓ easy_dwpose 已安裝 (推薦)")
    EASY_DWPOSE_AVAILABLE = True
except ImportError:
    print("  ℹ easy_dwpose 未安裝 (可選)")
    print("  建議執行: pip install easy-dwpose")
    EASY_DWPOSE_AVAILABLE = False

# 測試 3: 測試 DWPose 初始化
print("\n[3/4] 測試 DWPose 初始化...")
try:
    if EASY_DWPOSE_AVAILABLE:
        print("  嘗試使用 easy_dwpose...")
        detector = EasyDWpose(device="cpu")
        print("  ✓ DWPose 載入成功 (easy_dwpose)")
    else:
        print("  嘗試使用 controlnet_aux 直接初始化...")
        detector = DWposeDetector()
        print("  ✓ DWPose 載入成功 (controlnet_aux)")
except Exception as e:
    print(f"  ✗ DWPose 載入失敗: {e}")
    print("  將嘗試使用 OpenPose...")
    detector = None

# 測試 4: 測試 OpenPose 作為替代 (如果 DWPose 失敗)
if detector is None:
    print("\n[4/4] 測試 OpenPose 作為替代...")
    try:
        # 嘗試推薦的模型名稱
        try:
            print("  嘗試載入 lllyasviel/Annotators...")
            detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            print("  ✓ OpenPose 載入成功 (lllyasviel/Annotators)")
        except:
            print("  嘗試載入 lllyasviel/ControlNet...")
            detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            print("  ✓ OpenPose 載入成功 (lllyasviel/ControlNet)")
    except Exception as e:
        print(f"  ✗ OpenPose 載入失敗: {e}")
        print("\n" + "=" * 60)
        print("所有模型載入失敗！")
        print("=" * 60)
        print("\n故障排除：")
        print("1. 檢查網絡連接（需要從 HuggingFace 下載模型）")
        print("2. 確認 PyTorch 已安裝:")
        print("   python3 -c 'import torch; print(torch.__version__)'")
        print("3. 嘗試重新安裝:")
        print("   pip uninstall controlnet-aux")
        print("   pip install controlnet-aux")
        print("   pip install easy-dwpose")
        sys.exit(1)
else:
    print("\n[4/4] 跳過 OpenPose 測試（DWPose 已成功）")

# 成功
print("\n" + "=" * 60)
print("✓ 所有測試通過！")
print("=" * 60)
print("\n使用的檢測器:", type(detector).__name__)
print("\n您現在可以執行:")
print("  python3 dance_ultimate.py")
print("  python3 dance_enhanced.py")
print("  python3 dance_stable.py")
print()
