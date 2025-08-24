# DINOv3 Wrapper 使用指南

## 概述

DINOv3 Wrapper 是基於 DINOv2 架構開發的下一代視覺特徵提取工具，提供了增強的密集預測能力和改進的自監督學習特性。本指南將幫助您快速上手使用 DINOv3 Wrapper。

## 主要特性

### 🚀 DINOv3 的改進
- **增強的密集預測能力**：改進的 patch-level 特徵提取
- **更好的空間一致性**：優化的局部特徵表示
- **改進的自監督學習**：更強的視覺表示能力
- **與 DINOv2 的相容性**：保持相同的 API 介面

### 🔧 技術特點
- 支持多種模型尺寸 (ViT-S, ViT-B, ViT-L, ViT-G)
- 可選的增強功能模式
- 完整的向後相容性
- 高效的批次處理支持

## 安裝和設置

### 前置需求
```bash
# 確保已安裝基本依賴
pip install torch torchvision numpy pillow matplotlib scikit-learn
```

### 導入模組
```python
from castle.models.dinov3_wrapper import DINOv3Wrapper
import numpy as np
```

## 快速開始

### 基本用法
```python
# 初始化 DINOv3 Wrapper
wrapper = DINOv3Wrapper(
    model_type='dinov3_vitb14',  # 可選: dinov3_vits14, dinov3_vitl14, dinov3_vitg14
    device='cpu',                # 或 'cuda', 'mps'
    enable_enhanced_features=True # 啟用 DINOv3 增強功能
)

# 準備測試數據
image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
roi_mask = np.zeros((480, 640), dtype=np.uint8)
roi_mask[120:360, 160:480] = 255  # 定義感興趣區域

# 提取特徵
features = wrapper.extract_features(image, roi_mask)
print(f"特徵形狀: {features.shape}")  # (768,) for ViT-B
```

### 批次處理
```python
# 準備多個影像和遮罩
images = [image1, image2, image3]
masks = [mask1, mask2, mask3]

# 批次提取特徵
batch_features = wrapper.extract_batch_features(images, masks)
print(f"批次特徵形狀: {len(batch_features)} x {batch_features[0].shape}")
```

## 詳細功能

### 1. 模型配置選項

```python
# 基本 DINOv3 模型
wrapper_basic = DINOv3Wrapper(
    model_type='dinov3_vitb14',
    device='cpu',
    enable_enhanced_features=False  # 使用標準模式
)

# 增強 DINOv3 模型
wrapper_enhanced = DINOv3Wrapper(
    model_type='dinov3_vitb14',
    device='cpu',
    enable_enhanced_features=True,  # 啟用增強功能
    use_fp16=True,                  # 使用半精度 (僅 GPU)
    batch_size=8                    # 批次大小
)
```

### 2. 不同的特徵提取方式

```python
# 方法 1: 完整的 ROI 特徵提取
roi_features = wrapper.extract_features(image, roi_mask)

# 方法 2: 獲取原始 patch 特徵
patch_features = wrapper.get_patch_features(image)
print(f"Patch 特徵形狀: {patch_features.shape}")  # (37, 37, 768)

# 方法 3: 手動 ROI 特徵計算
roi_features_manual = wrapper._extract_roi_features(patch_features, roi_mask)
```

### 3. 遮罩格式支持

```python
# 支持多種遮罩格式
mask_uint8 = np.zeros((480, 640), dtype=np.uint8)  # 0-255
mask_bool = np.zeros((480, 640), dtype=bool)       # True/False
mask_float = np.zeros((480, 640), dtype=np.float32)  # 0.0-1.0

# 所有格式都可以直接使用
features_uint8 = wrapper.extract_features(image, mask_uint8)
features_bool = wrapper.extract_features(image, mask_bool)
features_float = wrapper.extract_features(image, mask_float)
```

### 4. Debug 和診斷

```python
# 使用 debug 功能檢查遮罩處理
debug_info = wrapper.debug_mask_processing(roi_mask)

print("Debug 信息:")
for key, value in debug_info.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
    else:
        print(f"  {key}: {value}")
```

## 性能優化

### 1. 設備選擇
```python
# 自動選擇最佳設備
wrapper_auto = DINOv3Wrapper(device=None)  # 自動選擇

# 手動指定設備
wrapper_gpu = DINOv3Wrapper(device='cuda')   # GPU
wrapper_mps = DINOv3Wrapper(device='mps')    # Apple Silicon
wrapper_cpu = DINOv3Wrapper(device='cpu')    # CPU
```

### 2. 記憶體管理
```python
# 定期清除快取
wrapper.clear_cache()

# 使用半精度節省記憶體 (僅 GPU)
wrapper_fp16 = DINOv3Wrapper(use_fp16=True, device='cuda')
```

### 3. 批次處理優化
```python
# 調整批次大小以平衡速度和記憶體使用
wrapper_large_batch = DINOv3Wrapper(batch_size=32)  # 大批次
wrapper_small_batch = DINOv3Wrapper(batch_size=4)   # 小批次
```

## 與 DINOv2 的對比

### 特徵相容性
```python
from castle.models.dinov2_wrapper import DINOv2Wrapper
from sklearn.metrics.pairwise import cosine_similarity

# 初始化兩個 wrapper
dinov3_wrapper = DINOv3Wrapper(model_type='dinov3_vitb14')
dinov2_wrapper = DINOv2Wrapper(model_type='dinov2_vitb14_reg')

# 提取特徵
v3_features = dinov3_wrapper.extract_features(image, mask)
v2_features = dinov2_wrapper.extract_features(image, mask)

# 計算相似性
similarity = cosine_similarity([v3_features], [v2_features])[0, 0]
print(f"DINOv3 vs DINOv2 相似性: {similarity:.4f}")  # 通常 > 0.95
```

### 性能比較
```python
import time

# 測試處理速度
def benchmark_model(wrapper, image, mask, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.time()
        wrapper.extract_features(image, mask)
        times.append(time.time() - start)
    return np.mean(times)

v3_time = benchmark_model(dinov3_wrapper, image, mask)
v2_time = benchmark_model(dinov2_wrapper, image, mask)

print(f"DINOv3 平均時間: {v3_time:.3f}s")
print(f"DINOv2 平均時間: {v2_time:.3f}s")
print(f"速度比率: {v2_time/v3_time:.2f}x")
```

## 測試和驗證

### 運行基本測試
```bash
# 設置環境變數
export PYTHONPATH=/path/to/castle-ai

# 運行基本功能測試
python -c "from tests.test_dinov3_wrapper import run_manual_test; run_manual_test()"

# 運行完整綜合測試
python tests/test_dinov3_manual_demo.py
```

### 測試結果解釋
```python
# 檢查測試結果
import json
with open('tmp/dinov3_manual_test_*/test_summary.json', 'r') as f:
    results = json.load(f)

print("測試總結:")
for section, data in results.items():
    print(f"\n{section}:")
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"  {key}: {value}")
```

## 故障排除

### 常見問題

#### 1. 模組導入錯誤
```bash
# 確保 PYTHONPATH 設置正確
export PYTHONPATH=/path/to/castle-ai:$PYTHONPATH
```

#### 2. CUDA 記憶體不足
```python
# 使用更小的批次大小或半精度
wrapper = DINOv3Wrapper(
    device='cuda',
    batch_size=4,      # 減少批次大小
    use_fp16=True      # 使用半精度
)
```

#### 3. 模型載入緩慢
```python
# 模型會自動快取，首次載入較慢是正常的
# 可以預先下載模型到本地
wrapper = DINOv3Wrapper(checkpoint_path='/path/to/local/model.pth')
```

#### 4. 特徵數值異常
```python
# 檢查輸入數據格式
print(f"影像形狀: {image.shape}")      # 應該是 (H, W, 3)
print(f"影像數據類型: {image.dtype}")   # 應該是 uint8
print(f"影像數值範圍: [{image.min()}, {image.max()}]")  # 應該是 [0, 255]

print(f"遮罩形狀: {mask.shape}")       # 應該是 (H, W)
print(f"遮罩數據類型: {mask.dtype}")    # uint8, bool, 或 float32
print(f"遮罩數值範圍: [{mask.min()}, {mask.max()}]")
```

## 最佳實踐

### 1. 效率優化
- 使用批次處理處理多個影像
- 在 GPU 上使用半精度以節省記憶體
- 定期清除快取以防止記憶體洩漏

### 2. 準確性優化
- 使用適當的 ROI 遮罩大小 (不要太小或太大)
- 確保輸入影像品質良好
- 啟用增強功能以獲得更好的特徵表示

### 3. 調試技巧
- 使用 `debug_mask_processing()` 檢查遮罩處理
- 比較 DINOv3 和 DINOv2 的結果以驗證相容性
- 監控處理時間以發現性能瓶頸

## 示例應用

### 影片特徵提取
```python
from castle.utils.video_io import VideoIO

# 載入影片
video_io = VideoIO('path/to/video.mp4')

# 提取所有幀的特徵
features_list = []
for i in range(video_io.frame_count):
    frame = video_io.get_frame(i)
    if frame is not None:
        # 使用動態 ROI 或固定 ROI
        roi_mask = create_roi_mask(frame)  # 自定義函數
        features = wrapper.extract_features(frame, roi_mask)
        features_list.append(features)

# 保存結果
np.savez('video_features.npz', features=np.array(features_list))
```

### 批次影像分析
```python
from pathlib import Path

# 載入批次影像
image_dir = Path('path/to/images')
images = []
masks = []

for img_path in image_dir.glob('*.jpg'):
    image = np.array(Image.open(img_path))
    mask = create_mask_for_image(image)  # 自定義函數
    images.append(image)
    masks.append(mask)

# 批次處理
batch_features = wrapper.extract_batch_features(images, masks)

# 分析結果
print(f"處理了 {len(batch_features)} 個影像")
print(f"平均特徵範數: {np.mean([np.linalg.norm(f) for f in batch_features]):.3f}")
```

## 結論

DINOv3 Wrapper 提供了強大且靈活的視覺特徵提取能力，適合各種計算機視覺任務。通過適當的配置和優化，您可以獲得高品質的特徵表示，同時保持與現有 DINOv2 工作流程的相容性。

如需更多詳細信息，請參考：
- `castle/models/dinov3_wrapper.py` - 源代碼
- `tests/test_dinov3_wrapper.py` - 單元測試
- `tests/test_dinov3_manual_demo.py` - 綜合測試示例
- `tmp/dinov3_manual_test_*/` - 測試結果和可視化
