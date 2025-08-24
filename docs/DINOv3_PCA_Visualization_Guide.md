# DINOv3 Patch Latent PCA 可視化指南

## 概述

DINOv3 Patch Latent PCA 可視化是一個強大的 debug 工具，可以幫助開發者理解和分析 DINOv3 模型產生的 patch-level 特徵。通過主成分分析 (PCA)，我們可以將高維的 patch 特徵（768 維）降維到低維空間（主要是前 3 個主成分），並以直觀的方式可視化出來。

## 主要功能

### 🔍 PCA 分析功能
- **降維分析**: 將 37×37×768 的 patch 特徵降維到可視化的低維空間
- **解釋方差分析**: 計算每個主成分的解釋方差比例
- **多影像對比**: 同時分析多個影像的 patch 特徵分布
- **空間分布可視化**: 展示每個影像內 patch 的空間特徵分布

### 🎨 可視化輸出
1. **3D 散點圖**: PC1, PC2, PC3 的三維分布
2. **2D 投影圖**: PC1 vs PC2, PC1 vs PC3, PC2 vs PC3
3. **解釋方差圖**: 各主成分的解釋方差比例
4. **累積方差圖**: 累積解釋方差曲線
5. **空間分布圖**: 每個影像的 patch 空間分布 (R=PC1, G=PC2, B=PC3)

## 測試結果解讀

### 典型的 PCA 結果

根據最新的測試結果：

```json
{
  "PC1 解釋方差": 0.1887,  // 18.87%
  "PC2 解釋方差": 0.1374,  // 13.74%
  "PC3 解釋方差": 0.1010,  // 10.10%
  "前3PC累積解釋方差": 0.4270  // 42.70%
}
```

### 結果解讀

#### 1. **解釋方差分析**
- **PC1 (18.87%)**: 主要特徵變異，通常與主要的視覺模式相關
- **PC2 (13.74%)**: 次要特徵變異，可能與紋理或邊緣相關
- **PC3 (10.10%)**: 第三重要的變異，通常與局部細節相關
- **累積 42.70%**: 前三個主成分能解釋約 43% 的總變異

#### 2. **數值範圍分析**
```json
{
  "PC1": [-12.69, 12.54],   // 較大的數值範圍，表示強變異
  "PC2": [-14.88, 14.35],   // 更大的範圍，可能捕捉重要特徵
  "PC3": [-15.87, 12.29]    // 最大範圍，但解釋方差較小
}
```

#### 3. **空間分布特徵**
- **RGB 映射**: R=PC1, G=PC2, B=PC3
- **顏色變化**: 反映不同 patch 的特徵差異
- **空間一致性**: 相鄰 patch 的顏色相似性表示特徵連續性

## 如何使用

### 1. 運行 PCA 分析

```bash
# 執行完整測試（包含 PCA 分析）
cd /path/to/castle-ai
PYTHONPATH=/path/to/castle-ai python tests/test_dinov3_manual_demo.py
```

### 2. 獨立運行 PCA 測試

```python
from tests.test_dinov3_manual_demo import test_patch_latent_pca_visualization
from pathlib import Path

# 設置輸出目錄
output_dir = Path("tmp/pca_analysis")
output_dir.mkdir(exist_ok=True)

# 運行 PCA 分析
pca_features, explained_variance = test_patch_latent_pca_visualization(output_dir)

print(f"PCA 特徵形狀: {pca_features.shape}")
print(f"PC1-3 解釋方差: {explained_variance[:3]}")
```

### 3. 自定義 PCA 分析

```python
from castle.models.dinov3_wrapper import DINOv3Wrapper
from sklearn.decomposition import PCA
import numpy as np

# 初始化 wrapper
wrapper = DINOv3Wrapper(
    model_type='dinov3_vitb14',
    enable_enhanced_features=True
)

# 提取 patch features
images = [your_image_list]  # 您的影像列表
all_patches = []

for image in images:
    patch_features = wrapper.get_patch_features(image)  # (37, 37, 768)
    all_patches.append(patch_features.reshape(-1, 768))  # (37*37, 768)

# 合併所有 patches
combined_patches = np.vstack(all_patches)  # (N*37*37, 768)

# 執行 PCA
pca = PCA(n_components=10)
pca_features = pca.fit_transform(combined_patches)

# 分析結果
explained_variance = pca.explained_variance_ratio_
print(f"前3個PC解釋方差: {explained_variance[:3]}")
print(f"累積解釋方差: {np.cumsum(explained_variance[:3])[-1]:.3f}")
```

## 生成的檔案說明

### 輸出檔案結構

```
tmp/dinov3_manual_test_TIMESTAMP/patch_latent_pca/
├── patch_latent_pca_results.npz           # 原始 PCA 數據
├── patch_latent_pca_visualization.png     # 主要 PCA 可視化圖表
└── patch_pca_spatial_distribution.png     # 空間分布可視化
```

### 檔案內容

#### 1. `patch_latent_pca_results.npz`
```python
# 載入數據
data = np.load('patch_latent_pca_results.npz')

# 可用的數據:
pca_features = data['pca_features']           # (N, 10) PCA 特徵
explained_variance = data['explained_variance_ratio']  # (10,) 解釋方差
image_labels = data['image_labels']           # (N,) 影像標籤
pca_components = data['pca_components']       # (3, 768) 前3個主成分
metadata = data['metadata'].item()           # 元數據字典
```

#### 2. `patch_latent_pca_visualization.png`
包含 6 個子圖：
- 3D 散點圖 (PC1, PC2, PC3)
- PC1 vs PC2 散點圖
- PC1 vs PC3 散點圖  
- PC2 vs PC3 散點圖
- 解釋方差條形圖
- 累積解釋方差曲線

#### 3. `patch_pca_spatial_distribution.png`
每個影像的 patch 空間分布，使用 RGB 顏色編碼：
- R 通道 = PC1 正規化值
- G 通道 = PC2 正規化值
- B 通道 = PC3 正規化值

## Debug 應用案例

### 1. 特徵分布異常檢測

```python
# 檢查是否有異常的 patch 特徵
outliers = np.abs(pca_features) > 3 * np.std(pca_features, axis=0)
outlier_patches = np.any(outliers, axis=1)

print(f"異常 patch 數量: {np.sum(outlier_patches)}")
print(f"異常比例: {np.mean(outlier_patches):.2%}")
```

### 2. 影像間特徵相似性分析

```python
from sklearn.metrics.pairwise import cosine_similarity

# 計算每個影像的平均 patch 特徵
image_means = []
for i in range(n_images):
    mask = np.array(image_labels) == f'Image_{i+1}'
    image_mean = pca_features[mask].mean(axis=0)
    image_means.append(image_mean)

# 計算影像間相似性
similarity_matrix = cosine_similarity(image_means)
print("影像間相似性矩陣:")
print(similarity_matrix)
```

### 3. 空間連續性檢查

```python
# 檢查相鄰 patch 的特徵連續性
def check_spatial_continuity(patch_features_37x37):
    # 計算水平和垂直相鄰 patch 的相似性
    h_similarity = []
    v_similarity = []
    
    for i in range(36):  # 0-35
        for j in range(36):  # 0-35
            # 水平相鄰
            sim_h = cosine_similarity([patch_features_37x37[i, j]], 
                                    [patch_features_37x37[i, j+1]])[0, 0]
            h_similarity.append(sim_h)
            
            # 垂直相鄰
            sim_v = cosine_similarity([patch_features_37x37[i, j]], 
                                    [patch_features_37x37[i+1, j]])[0, 0]
            v_similarity.append(sim_v)
    
    return np.mean(h_similarity), np.mean(v_similarity)

# 對每個影像檢查空間連續性
for i, patch_features in enumerate(all_patch_features):
    h_cont, v_cont = check_spatial_continuity(patch_features)
    print(f"影像 {i+1} - 水平連續性: {h_cont:.3f}, 垂直連續性: {v_cont:.3f}")
```

## 最佳實踐

### 1. 影像選擇
- **多樣性**: 選擇具有不同視覺特徵的影像
- **對比度**: 包含高對比度和低對比度的影像
- **紋理變化**: 涵蓋不同紋理模式的影像

### 2. PCA 參數調整
```python
# 自適應主成分數量
n_samples, n_features = combined_patches.shape
n_components = min(50, n_samples, n_features)  # 避免過擬合

pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(combined_patches)

# 選擇重要的主成分（累積解釋方差 > 90%）
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
important_components = np.argmax(cumulative_variance > 0.9) + 1
```

### 3. 可視化優化
```python
# 使用更好的顏色映射
import matplotlib.pyplot as plt

# 為不同影像使用不同的顏色和標記
colors = plt.cm.Set3(np.linspace(0, 1, n_images))
markers = ['o', 's', '^', 'v', 'd', 'p', 'h', '*']

for i in range(n_images):
    mask = np.array(image_labels) == f'Image_{i+1}'
    plt.scatter(pca_features[mask, 0], pca_features[mask, 1], 
               c=[colors[i]], marker=markers[i % len(markers)], 
               alpha=0.7, s=30, label=f'Image {i+1}')
```

## 故障排除

### 常見問題

#### 1. PCA 解釋方差過低
```python
# 檢查數據預處理
print(f"特徵數據範圍: [{combined_patches.min():.3f}, {combined_patches.max():.3f}]")
print(f"特徵數據標準差: {combined_patches.std():.3f}")

# 嘗試標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_patches = scaler.fit_transform(combined_patches)
```

#### 2. 記憶體不足
```python
# 使用增量 PCA 處理大量數據
from sklearn.decomposition import IncrementalPCA

n_batches = 10
inc_pca = IncrementalPCA(n_components=10)

for i in range(n_batches):
    batch_data = combined_patches[i::n_batches]
    inc_pca.partial_fit(batch_data)

pca_features = inc_pca.transform(combined_patches)
```

#### 3. 可視化效果不佳
```python
# 使用 t-SNE 進行非線性降維對比
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(combined_patches[:1000])  # 取樣分析

# 比較 PCA 和 t-SNE 的結果
```

## 結論

DINOv3 Patch Latent PCA 可視化是一個強大的分析工具，可以幫助開發者：

1. **理解特徵分布**: 通過 PCA 降維看到高維特徵的主要變化模式
2. **檢測異常**: 識別異常的 patch 特徵或不合理的特徵分布
3. **驗證模型**: 確認 DINOv3 模型產生合理的空間一致性特徵
4. **優化性能**: 基於特徵分析結果調整模型參數或訓練策略

通過定期運行 PCA 分析，開發者可以持續監控模型的特徵提取品質，並及時發現潛在問題。
