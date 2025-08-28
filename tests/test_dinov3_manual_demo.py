"""
DINOv3 手動演示測試

這個腳本展示 DINOv3 wrapper 的完整功能，包括：
- 基本特徵提取
- 增強功能演示
- 與 DINOv2 的對比
- 結果生成到 tmp 目錄供開發者檢驗
"""

import pytest
import numpy as np
import torch
import time
import os
from pathlib import Path
import tempfile
import json
import logging

# CPU 環境下預先禁用 xformers 以避免兼容性問題
if not torch.cuda.is_available() and 'XFORMERS_DISABLED' not in os.environ:
    os.environ['XFORMERS_DISABLED'] = '1'
    print("Test: Disabled xformers for CPU compatibility")

# 檢查可選套件
try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from castle.models.dinov3_wrapper import DINOv3Wrapper

# 嘗試導入 DINOv2 進行對比
try:
    from castle.models.dinov2_wrapper import DINOv2Wrapper
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip 條件
requires_sklearn = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="需要 sklearn 套件"
)

requires_matplotlib = pytest.mark.skipif(
    not MPL_AVAILABLE,
    reason="需要 matplotlib 套件"
)


# Pytest fixtures
@pytest.fixture
def output_dir():
    """測試輸出目錄 fixture"""
    project_root = Path(__file__).parent.parent
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    
    temp_dir = tempfile.mkdtemp(prefix="dinov3_test_", dir=str(tmp_dir))
    output_dir = Path(temp_dir)
    print(f"✅ 測試輸出目錄: {output_dir}")
    return output_dir


def create_synthetic_data():
    """創建合成測試數據"""
    print("🎨 創建合成測試數據...")
    
    # 創建多個測試影像
    images = []
    masks = []
    
    # 影像 1: 隨機噪聲
    img1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    mask1 = np.zeros((480, 640), dtype=np.uint8)
    mask1[120:360, 160:480] = 255  # 中央矩形
    images.append(img1)
    masks.append(mask1)
    
    # 影像 2: 漸變模式
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(480):
        for j in range(640):
            img2[i, j] = [i * 255 // 480, j * 255 // 640, (i + j) * 255 // (480 + 640)]
    mask2 = np.zeros((480, 640), dtype=np.uint8)
    # 圓形遮罩
    center_y, center_x = 240, 320
    y, x = np.ogrid[:480, :640]
    mask2[(x - center_x) ** 2 + (y - center_y) ** 2 <= 100 ** 2] = 255
    images.append(img2)
    masks.append(mask2)
    
    # 影像 3: 棋盤模式
    img3 = np.zeros((480, 640, 3), dtype=np.uint8)
    block_size = 40
    for i in range(0, 480, block_size):
        for j in range(0, 640, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                img3[i:i+block_size, j:j+block_size] = [255, 255, 255]
            else:
                img3[i:i+block_size, j:j+block_size] = [0, 0, 0]
    mask3 = np.zeros((480, 640), dtype=np.uint8)
    mask3[240:480, 320:640] = 255  # 右下角
    images.append(img3)
    masks.append(mask3)
    
    print(f"   ✅ 創建了 {len(images)} 個測試影像")
    return images, masks


@pytest.mark.slow
def test_dinov3_basic_functionality(output_dir):
    """測試 DINOv3 基本功能"""
    print("\n🔬 測試 DINOv3 基本功能...")
    
    # 創建輸出子目錄
    basic_dir = output_dir / "basic_functionality"
    basic_dir.mkdir(exist_ok=True)
    
    # 初始化 wrapper
    wrapper = DINOv3Wrapper(
        model_type='dinov3_vitb14',
        device='cpu',
        enable_enhanced_features=True
    )
    
    # 創建測試數據
    images, masks = create_synthetic_data()
    
    # 提取特徵
    print("   🔍 提取特徵...")
    features = []
    times = []
    
    for i, (image, mask) in enumerate(zip(images, masks)):
        start_time = time.time()
        feature = wrapper.extract_features(image, mask)
        elapsed = time.time() - start_time
        
        features.append(feature)
        times.append(elapsed)
        
        print(f"      影像 {i+1}: 特徵形狀 {feature.shape}, 時間 {elapsed:.3f}s")
    
    features_array = np.array(features)
    
    # 保存基本結果
    basic_results = {
        'features': features_array,
        'processing_times': np.array(times),
        'metadata': {
            'model_type': 'dinov3_vitb14',
            'enhanced_features': True,
            'n_images': len(images),
            'feature_dim': wrapper.embed_dim,
            'avg_time': np.mean(times)
        }
    }
    
    np.savez(basic_dir / "dinov3_basic_features.npz", **basic_results)
    
    # 創建特徵可視化
    if MPL_AVAILABLE:
        create_feature_visualization(features_array, basic_dir)
    
    print(f"   ✅ 基本功能測試完成，結果保存至 {basic_dir}")
    
    # 簡單的斷言來驗證結果
    assert features_array.shape[0] == len(images)  # 應該有對應數量的特徵
    assert features_array.shape[1] == 768  # DINOv3 ViT-B 的嵌入維度
    assert not np.any(np.isnan(features_array))  # 不應該有 NaN
    assert not np.any(np.isinf(features_array))  # 不應該有無限值
    assert len(times) == len(images)  # 處理時間記錄應該對應
    
    return features_array, times


@pytest.mark.slow
def test_enhanced_vs_standard(output_dir):
    """測試增強功能 vs 標準功能"""
    print("\n⚡ 測試增強功能 vs 標準功能...")
    
    # 創建輸出子目錄
    enhanced_dir = output_dir / "enhanced_vs_standard"
    enhanced_dir.mkdir(exist_ok=True)
    
    # 創建測試影像
    test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    test_mask[120:360, 160:480] = 255
    
    # 初始化兩個 wrapper
    wrapper_enhanced = DINOv3Wrapper(
        model_type='dinov3_vitb14',
        device='cpu',
        enable_enhanced_features=True
    )
    
    wrapper_standard = DINOv3Wrapper(
        model_type='dinov3_vitb14',
        device='cpu',
        enable_enhanced_features=False
    )
    
    # 提取特徵
    print("   🔍 提取增強特徵...")
    start_time = time.time()
    enhanced_features = wrapper_enhanced.extract_features(test_image, test_mask)
    enhanced_time = time.time() - start_time
    
    print("   🔍 提取標準特徵...")
    start_time = time.time()
    standard_features = wrapper_standard.extract_features(test_image, test_mask)
    standard_time = time.time() - start_time
    
    # 分析差異
    feature_diff = np.abs(enhanced_features - standard_features)
    if SKLEARN_AVAILABLE:
        feature_corr = cosine_similarity([enhanced_features], [standard_features])[0, 0]
    else:
        feature_corr = np.corrcoef(enhanced_features, standard_features)[0, 1]
    
    print(f"   📊 增強 vs 標準:")
    print(f"      特徵差異 (平均): {feature_diff.mean():.6f}")
    print(f"      特徵差異 (最大): {feature_diff.max():.6f}")
    print(f"      特徵相關性: {feature_corr:.6f}")
    print(f"      增強時間: {enhanced_time:.3f}s")
    print(f"      標準時間: {standard_time:.3f}s")
    
    # 獲取 debug 信息
    debug_enhanced = wrapper_enhanced.debug_mask_processing(test_mask)
    debug_standard = wrapper_standard.debug_mask_processing(test_mask)
    
    # 保存結果
    enhanced_results = {
        'enhanced_features': enhanced_features,
        'standard_features': standard_features,
        'feature_diff': feature_diff,
        'feature_correlation': feature_corr,
        'enhanced_time': enhanced_time,
        'standard_time': standard_time,
        'debug_enhanced': debug_enhanced,
        'debug_standard': debug_standard,
        'metadata': {
            'model_type': 'dinov3_vitb14',
            'test_image_shape': test_image.shape,
            'test_mask_shape': test_mask.shape
        }
    }
    
    np.savez(enhanced_dir / "enhanced_vs_standard.npz", **enhanced_results)
    
    if MPL_AVAILABLE:
        create_comparison_visualization(enhanced_features, standard_features, enhanced_dir)
    
    print(f"   ✅ 增強功能測試完成，結果保存至 {enhanced_dir}")
    
    # 簡單的斷言來驗證結果
    assert enhanced_features.shape == standard_features.shape  # 形狀應該相同
    assert enhanced_features.shape[0] == 768  # DINOv3 ViT-B 的嵌入維度
    assert not np.any(np.isnan(enhanced_features))  # 不應該有 NaN
    assert not np.any(np.isnan(standard_features))  # 不應該有 NaN
    assert feature_corr > 0.95  # 相關性應該很高但不完全相同
    assert feature_corr < 1.0   # 不應該完全相同
    
    return enhanced_features, standard_features


@pytest.mark.slow
@requires_sklearn
@requires_matplotlib
def test_patch_latent_pca_visualization(output_dir):
    """測試 patch latent PCA 可視化"""
    if not SKLEARN_AVAILABLE:
        pytest.skip("sklearn 不可用")
        
    if not MPL_AVAILABLE:
        pytest.skip("matplotlib 不可用")
        
    print("\n🔬 測試 Patch Latent PCA 可視化...")
    
    # 創建輸出子目錄
    pca_dir = output_dir / "patch_latent_pca"
    pca_dir.mkdir(exist_ok=True)
    
    # 初始化 wrapper
    wrapper = DINOv3Wrapper(
        model_type='dinov3_vitb14',
        device='cpu',
        enable_enhanced_features=True
    )
    
    # 創建測試數據 - 更多樣化的影像用於更好的 PCA 分析
    images, masks = create_diverse_test_images()
    
    print(f"   📊 分析 {len(images)} 個影像的 patch latent...")
    
    # 提取所有影像的 patch features
    all_patch_features = []
    image_labels = []
    
    for i, image in enumerate(images):
        patch_features = wrapper.get_patch_features(image)  # (37, 37, 768)
        all_patch_features.append(patch_features)
        image_labels.extend([f'Image_{i+1}'] * (37 * 37))  # 每個 patch 都標記來源影像
        
        print(f"      影像 {i+1}: patch 特徵形狀 {patch_features.shape}")
    
    # 將所有 patch features 合併為 2D array
    # Shape: (n_images * 37 * 37, 768)
    combined_patches = np.vstack([pf.reshape(-1, pf.shape[-1]) for pf in all_patch_features])
    
    print(f"   🔍 合併後的 patch 特徵形狀: {combined_patches.shape}")
    print(f"   📈 執行 PCA 降維...")
    
    # 執行 PCA
    pca = PCA(n_components=min(10, combined_patches.shape[1]))  # 計算前10個主成分
    pca_features = pca.fit_transform(combined_patches)
    
    # 計算解釋方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"   📊 PCA 分析結果:")
    print(f"      PC1 解釋方差: {explained_variance_ratio[0]:.4f} ({explained_variance_ratio[0]*100:.2f}%)")
    print(f"      PC2 解釋方差: {explained_variance_ratio[1]:.4f} ({explained_variance_ratio[1]*100:.2f}%)")
    print(f"      PC3 解釋方差: {explained_variance_ratio[2]:.4f} ({explained_variance_ratio[2]*100:.2f}%)")
    print(f"      前3個PC累積解釋方差: {cumulative_variance[2]:.4f} ({cumulative_variance[2]*100:.2f}%)")
    
    # 創建可視化
    create_patch_pca_visualization(
        pca_features, image_labels, explained_variance_ratio, 
        cumulative_variance, pca_dir, len(images)
    )
    
    # 分析每個影像的 patch 分布
    analyze_per_image_patch_distribution(
        all_patch_features, pca, pca_dir
    )
    
    # 保存 PCA 結果
    pca_results = {
        'pca_features': pca_features,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'image_labels': image_labels,
        'pca_components': pca.components_[:3],  # 保存前3個主成分
        'metadata': {
            'n_images': len(images),
            'n_patches_per_image': 37 * 37,
            'total_patches': combined_patches.shape[0],
            'original_dim': combined_patches.shape[1],
            'pc1_variance': float(explained_variance_ratio[0]),
            'pc2_variance': float(explained_variance_ratio[1]),
            'pc3_variance': float(explained_variance_ratio[2]),
            'top3_cumulative_variance': float(cumulative_variance[2])
        }
    }
    
    np.savez(pca_dir / "patch_latent_pca_results.npz", **pca_results)
    
    print(f"   ✅ Patch Latent PCA 分析完成，結果保存至 {pca_dir}")
    
    # 簡單的斷言來驗證結果
    assert pca_features.shape[0] == 6845  # 5個影像 × 37×37
    assert pca_features.shape[1] == 10   # 10個主成分
    assert len(explained_variance_ratio) == 10
    assert 0 < explained_variance_ratio[0] < 1  # PC1 應該有合理的解釋方差
    
    return pca_features, explained_variance_ratio


def create_diverse_test_images():
    """創建更多樣化的測試影像用於 PCA 分析"""
    print("   🎨 創建多樣化測試影像...")
    
    images = []
    masks = []
    
    # 影像 1: 隨機噪聲
    img1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    mask1 = np.zeros((480, 640), dtype=np.uint8)
    mask1[120:360, 160:480] = 255
    images.append(img1)
    masks.append(mask1)
    
    # 影像 2: 水平漸變
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    for j in range(640):
        img2[:, j] = [j * 255 // 640, 128, 255 - j * 255 // 640]
    mask2 = np.zeros((480, 640), dtype=np.uint8)
    mask2[:, 320:] = 255  # 右半邊
    images.append(img2)
    masks.append(mask2)
    
    # 影像 3: 垂直漸變
    img3 = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(480):
        img3[i, :] = [255 - i * 255 // 480, i * 255 // 480, 128]
    mask3 = np.zeros((480, 640), dtype=np.uint8)
    mask3[240:, :] = 255  # 下半邊
    images.append(img3)
    masks.append(mask3)
    
    # 影像 4: 棋盤模式
    img4 = np.zeros((480, 640, 3), dtype=np.uint8)
    block_size = 40
    for i in range(0, 480, block_size):
        for j in range(0, 640, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                img4[i:i+block_size, j:j+block_size] = [255, 255, 255]
            else:
                img4[i:i+block_size, j:j+block_size] = [0, 0, 0]
    mask4 = np.ones((480, 640), dtype=np.uint8) * 255  # 全域
    images.append(img4)
    masks.append(mask4)
    
    # 影像 5: 同心圓模式
    img5 = np.zeros((480, 640, 3), dtype=np.uint8)
    center_y, center_x = 240, 320
    y, x = np.ogrid[:480, :640]
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.max(distances)
    for i in range(480):
        for j in range(640):
            dist_norm = distances[i, j] / max_distance
            intensity = int(255 * (np.sin(dist_norm * 10) + 1) / 2)
            img5[i, j] = [intensity, 255 - intensity, 128]
    mask5 = np.zeros((480, 640), dtype=np.uint8)
    mask5[(x - center_x) ** 2 + (y - center_y) ** 2 <= 150 ** 2] = 255  # 圓形
    images.append(img5)
    masks.append(mask5)
    
    print(f"      ✅ 創建了 {len(images)} 個多樣化測試影像")
    return images, masks


def create_patch_pca_visualization(pca_features, image_labels, explained_variance_ratio, 
                                 cumulative_variance, output_dir, n_images):
    """創建 patch PCA 可視化"""
    if not MPL_AVAILABLE:
        print("      ⚠️ matplotlib 不可用，跳過可視化")
        return
        
    try:
        # 準備顏色映射
        colors = plt.cm.tab10(np.linspace(0, 1, n_images))
        
        # 創建大型圖表
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 3D 散點圖 (PC1, PC2, PC3)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        # 為每個影像分別繪製
        for i in range(n_images):
            mask = np.array(image_labels) == f'Image_{i+1}'
            if np.any(mask):
                ax1.scatter(pca_features[mask, 0], pca_features[mask, 1], pca_features[mask, 2], 
                           c=[colors[i]], label=f'Image {i+1}', alpha=0.6, s=20)
        
        ax1.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
        ax1.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)')
        ax1.set_title('3D PCA: Patch Latent Features')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. PC1 vs PC2
        ax2 = fig.add_subplot(2, 3, 2)
        for i in range(n_images):
            mask = np.array(image_labels) == f'Image_{i+1}'
            if np.any(mask):
                ax2.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                           c=[colors[i]], label=f'Image {i+1}', alpha=0.6, s=15)
        
        ax2.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
        ax2.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
        ax2.set_title('PC1 vs PC2')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. PC1 vs PC3
        ax3 = fig.add_subplot(2, 3, 3)
        for i in range(n_images):
            mask = np.array(image_labels) == f'Image_{i+1}'
            if np.any(mask):
                ax3.scatter(pca_features[mask, 0], pca_features[mask, 2], 
                           c=[colors[i]], label=f'Image {i+1}', alpha=0.6, s=15)
        
        ax3.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
        ax3.set_ylabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)')
        ax3.set_title('PC1 vs PC3')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. PC2 vs PC3
        ax4 = fig.add_subplot(2, 3, 4)
        for i in range(n_images):
            mask = np.array(image_labels) == f'Image_{i+1}'
            if np.any(mask):
                ax4.scatter(pca_features[mask, 1], pca_features[mask, 2], 
                           c=[colors[i]], label=f'Image {i+1}', alpha=0.6, s=15)
        
        ax4.set_xlabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
        ax4.set_ylabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)')
        ax4.set_title('PC2 vs PC3')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. 解釋方差比例
        ax5 = fig.add_subplot(2, 3, 5)
        pc_indices = range(1, len(explained_variance_ratio[:10]) + 1)
        bars = ax5.bar(pc_indices, explained_variance_ratio[:10] * 100, alpha=0.7, color='skyblue')
        ax5.set_xlabel('Principal Component')
        ax5.set_ylabel('Explained Variance (%)')
        ax5.set_title('Explained Variance by PC')
        ax5.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 6. 累積解釋方差
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(pc_indices, cumulative_variance[:10] * 100, 'o-', color='red', linewidth=2, markersize=6)
        ax6.set_xlabel('Principal Component')
        ax6.set_ylabel('Cumulative Explained Variance (%)')
        ax6.set_title('Cumulative Explained Variance')
        ax6.grid(True, alpha=0.3)
        
        # 添加重要的累積方差標記
        for i in [2, 4, 9]:  # PC3, PC5, PC10
            if i < len(cumulative_variance):
                ax6.axhline(y=cumulative_variance[i] * 100, color='gray', linestyle='--', alpha=0.5)
                ax6.text(1, cumulative_variance[i] * 100 + 1, 
                        f'PC{i+1}: {cumulative_variance[i]*100:.1f}%', fontsize=8)
        
        plt.suptitle('DINOv3 Patch Latent PCA Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存圖表
        plot_path = output_dir / "patch_latent_pca_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 PCA 可視化已保存: {plot_path}")
        
    except Exception as e:
        print(f"      ⚠️ PCA 可視化創建失敗: {e}")


def analyze_per_image_patch_distribution(all_patch_features, pca, output_dir):
    """分析每個影像的 patch 分布"""
    if not MPL_AVAILABLE:
        return
        
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, patch_features in enumerate(all_patch_features):
            if i >= len(axes):
                break
                
            # 將 patch features 重塑並進行 PCA 轉換
            patches_2d = patch_features.reshape(-1, patch_features.shape[-1])  # (37*37, 768)
            patches_pca = pca.transform(patches_2d)  # (37*37, n_components)
            
            # 重塑回空間結構
            pc1_spatial = patches_pca[:, 0].reshape(37, 37)
            pc2_spatial = patches_pca[:, 1].reshape(37, 37)
            pc3_spatial = patches_pca[:, 2].reshape(37, 37)
            
            # 創建 RGB 圖像：R=PC1, G=PC2, B=PC3
            # 先正規化到 [0, 1]
            pc1_norm = (pc1_spatial - pc1_spatial.min()) / (pc1_spatial.max() - pc1_spatial.min())
            pc2_norm = (pc2_spatial - pc2_spatial.min()) / (pc2_spatial.max() - pc2_spatial.min())
            pc3_norm = (pc3_spatial - pc3_spatial.min()) / (pc3_spatial.max() - pc3_spatial.min())
            
            rgb_image = np.stack([pc1_norm, pc2_norm, pc3_norm], axis=-1)
            
            axes[i].imshow(rgb_image)
            axes[i].set_title(f'Image {i+1}\nPatch PCA (R=PC1, G=PC2, B=PC3)')
            axes[i].set_xlabel('Patch X')
            axes[i].set_ylabel('Patch Y')
        
        # 隱藏多餘的子圖
        for j in range(len(all_patch_features), len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Per-Image Patch PCA Spatial Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存圖表
        spatial_plot_path = output_dir / "patch_pca_spatial_distribution.png"
        plt.savefig(spatial_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 空間分布可視化已保存: {spatial_plot_path}")
        
    except Exception as e:
        print(f"      ⚠️ 空間分布可視化創建失敗: {e}")


@pytest.mark.slow
@requires_sklearn
@requires_matplotlib
def test_dinov3_vs_dinov2_comparison(output_dir):
    """測試 DINOv3 vs DINOv2 對比"""
    if not DINOV2_AVAILABLE:
        pytest.skip("DINOv2 不可用")
        
    print("\n🆚 測試 DINOv3 vs DINOv2 對比...")
    
    # 創建輸出子目錄
    comparison_dir = output_dir / "dinov3_vs_dinov2"
    comparison_dir.mkdir(exist_ok=True)
    
    # 創建測試數據
    images, masks = create_synthetic_data()
    
    # 初始化兩個 wrapper
    dinov3_wrapper = DINOv3Wrapper(
        model_type='dinov3_vitb14',
        device='cpu',
        enable_enhanced_features=True
    )
    
    dinov2_wrapper = DINOv2Wrapper(
        model_type='dinov2_vitb14_reg',
        device='cpu'
    )
    
    # 提取特徵
    print("   🔍 提取 DINOv3 特徵...")
    dinov3_features = []
    dinov3_times = []
    
    for i, (image, mask) in enumerate(zip(images, masks)):
        start_time = time.time()
        feature = dinov3_wrapper.extract_features(image, mask)
        elapsed = time.time() - start_time
        
        dinov3_features.append(feature)
        dinov3_times.append(elapsed)
    
    print("   🔍 提取 DINOv2 特徵...")
    dinov2_features = []
    dinov2_times = []
    
    for i, (image, mask) in enumerate(zip(images, masks)):
        start_time = time.time()
        feature = dinov2_wrapper.extract_features(image, mask)
        elapsed = time.time() - start_time
        
        dinov2_features.append(feature)
        dinov2_times.append(elapsed)
    
    dinov3_features = np.array(dinov3_features)
    dinov2_features = np.array(dinov2_features)
    
    # 計算相關性
    correlations = []
    if SKLEARN_AVAILABLE:
        for i in range(len(dinov3_features)):
            corr = cosine_similarity([dinov3_features[i]], [dinov2_features[i]])[0, 0]
            correlations.append(corr)
    else:
        for i in range(len(dinov3_features)):
            corr = np.corrcoef(dinov3_features[i], dinov2_features[i])[0, 1]
            correlations.append(corr)
    
    avg_correlation = np.mean(correlations)
    
    print(f"   📊 DINOv3 vs DINOv2:")
    print(f"      平均相關性: {avg_correlation:.6f}")
    print(f"      DINOv3 平均時間: {np.mean(dinov3_times):.3f}s")
    print(f"      DINOv2 平均時間: {np.mean(dinov2_times):.3f}s")
    print(f"      速度比率: {np.mean(dinov2_times)/np.mean(dinov3_times):.2f}x")
    
    # 保存結果
    comparison_results = {
        'dinov3_features': dinov3_features,
        'dinov2_features': dinov2_features,
        'correlations': np.array(correlations),
        'dinov3_times': np.array(dinov3_times),
        'dinov2_times': np.array(dinov2_times),
        'metadata': {
            'avg_correlation': avg_correlation,
            'dinov3_avg_time': np.mean(dinov3_times),
            'dinov2_avg_time': np.mean(dinov2_times),
            'n_images': len(images)
        }
    }
    
    np.savez(comparison_dir / "dinov3_vs_dinov2_comparison.npz", **comparison_results)
    
    if MPL_AVAILABLE:
        create_model_comparison_visualization(dinov3_features, dinov2_features, correlations, comparison_dir)
    
    print(f"   ✅ 對比測試完成，結果保存至 {comparison_dir}")
    
    # 簡單的斷言來驗證結果
    assert dinov3_features.shape == dinov2_features.shape  # 形狀應該相同
    assert dinov3_features.shape[1] == 768  # 嵌入維度
    assert len(correlations) == len(images)  # 相關性數量應該對應
    assert avg_correlation > 0.9  # 平均相關性應該很高（向後相容性）
    assert not np.any(np.isnan(dinov3_features))  # 不應該有 NaN
    assert not np.any(np.isnan(dinov2_features))  # 不應該有 NaN
    
    return dinov3_features, dinov2_features


def create_feature_visualization(features, output_dir):
    """創建特徵可視化"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('DINOv3 Feature Analysis', fontsize=16, fontweight='bold')
        
        # 1. 特徵分布
        axes[0, 0].hist(features.flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Feature Value Distribution')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 特徵範數
        norms = np.linalg.norm(features, axis=1)
        axes[0, 1].plot(norms, marker='o', color='red')
        axes[0, 1].set_title('Feature Norm per Image')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('L2 Norm')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 特徵相似性矩陣
        if SKLEARN_AVAILABLE:
            similarity_matrix = cosine_similarity(features)
        else:
            similarity_matrix = np.corrcoef(features)
        
        im = axes[1, 0].imshow(similarity_matrix, cmap='viridis')
        axes[1, 0].set_title('Feature Similarity Matrix')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Image Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. 特徵統計
        means = features.mean(axis=1)
        stds = features.std(axis=1)
        
        axes[1, 1].scatter(means, stds, alpha=0.7, color='green')
        axes[1, 1].set_title('Feature Statistics')
        axes[1, 1].set_xlabel('Mean Feature Value')
        axes[1, 1].set_ylabel('Standard Deviation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 特徵可視化已保存")
        
    except Exception as e:
        print(f"      ⚠️ 特徵可視化失敗: {e}")


def create_comparison_visualization(enhanced_features, standard_features, output_dir):
    """創建增強功能對比可視化"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Enhanced vs Standard Features', fontsize=16, fontweight='bold')
        
        # 1. 特徵分布對比
        axes[0].hist(enhanced_features, alpha=0.7, label='Enhanced', bins=50, color='blue')
        axes[0].hist(standard_features, alpha=0.7, label='Standard', bins=50, color='red')
        axes[0].set_title('Feature Distribution')
        axes[0].set_xlabel('Feature Value')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 特徵差異
        diff = np.abs(enhanced_features - standard_features)
        axes[1].hist(diff, bins=50, alpha=0.7, color='green')
        axes[1].set_title('Feature Difference')
        axes[1].set_xlabel('Absolute Difference')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 散點圖對比
        axes[2].scatter(standard_features, enhanced_features, alpha=0.5, color='purple')
        axes[2].plot([standard_features.min(), standard_features.max()], 
                    [standard_features.min(), standard_features.max()], 'k--', alpha=0.5)
        axes[2].set_title('Enhanced vs Standard')
        axes[2].set_xlabel('Standard Features')
        axes[2].set_ylabel('Enhanced Features')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "enhanced_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 對比可視化已保存")
        
    except Exception as e:
        print(f"      ⚠️ 對比可視化失敗: {e}")


def create_model_comparison_visualization(dinov3_features, dinov2_features, correlations, output_dir):
    """創建模型對比可視化"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('DINOv3 vs DINOv2 Comparison', fontsize=16, fontweight='bold')
        
        # 1. 特徵分布對比
        axes[0, 0].hist(dinov3_features.flatten(), alpha=0.7, label='DINOv3', bins=50, color='blue')
        axes[0, 0].hist(dinov2_features.flatten(), alpha=0.7, label='DINOv2', bins=50, color='red')
        axes[0, 0].set_title('Feature Distribution')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 特徵範數對比
        v3_norms = np.linalg.norm(dinov3_features, axis=1)
        v2_norms = np.linalg.norm(dinov2_features, axis=1)
        
        axes[0, 1].plot(v3_norms, marker='o', label='DINOv3', color='blue')
        axes[0, 1].plot(v2_norms, marker='s', label='DINOv2', color='red')
        axes[0, 1].set_title('Feature Norm Comparison')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('L2 Norm')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 相關性
        axes[1, 0].bar(range(len(correlations)), correlations, color='green', alpha=0.7)
        axes[1, 0].set_title('Feature Correlations')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 散點圖對比
        axes[1, 1].scatter(v2_norms, v3_norms, alpha=0.7, color='purple')
        axes[1, 1].plot([v2_norms.min(), v2_norms.max()], 
                       [v2_norms.min(), v2_norms.max()], 'k--', alpha=0.5)
        axes[1, 1].set_title('Norm Comparison')
        axes[1, 1].set_xlabel('DINOv2 Norm')
        axes[1, 1].set_ylabel('DINOv3 Norm')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 模型對比可視化已保存")
        
    except Exception as e:
        print(f"      ⚠️ 模型對比可視化失敗: {e}")


def create_summary_report(output_dir, basic_features, enhanced_features, standard_features, 
                         dinov3_features, dinov2_features, pca_features=None, pca_explained_variance=None):
    """創建總結報告"""
    print("\n📋 創建總結報告...")
    
    summary = {
        "DINOv3 測試總結": {
            "測試時間": time.strftime("%Y-%m-%d %H:%M:%S"),
            "測試環境": {
                "設備": "cpu",
                "模型類型": "dinov3_vitb14",
                "增強功能": "啟用"
            }
        },
        "基本功能測試": {
            "測試影像數量": len(basic_features),
            "特徵維度": basic_features.shape[1],
            "特徵範圍": [float(basic_features.min()), float(basic_features.max())],
            "特徵平均範數": float(np.linalg.norm(basic_features, axis=1).mean())
        },
        "增強功能測試": {
            "特徵差異_平均": float(np.abs(enhanced_features - standard_features).mean()),
            "特徵差異_最大": float(np.abs(enhanced_features - standard_features).max()),
            "特徵相關性": float(np.corrcoef(enhanced_features, standard_features)[0, 1]) if not SKLEARN_AVAILABLE 
                       else float(cosine_similarity([enhanced_features], [standard_features])[0, 0])
        }
    }
    
    # 添加 PCA 分析結果
    if pca_features is not None and pca_explained_variance is not None:
        summary["Patch Latent PCA 分析"] = {
            "總 patch 數量": len(pca_features),
            "PCA 降維後形狀": list(pca_features.shape),
            "PC1 解釋方差": float(pca_explained_variance[0]),
            "PC2 解釋方差": float(pca_explained_variance[1]),
            "PC3 解釋方差": float(pca_explained_variance[2]),
            "前3PC累積解釋方差": float(np.sum(pca_explained_variance[:3])),
            "PC1-3 數值範圍": {
                "PC1": [float(pca_features[:, 0].min()), float(pca_features[:, 0].max())],
                "PC2": [float(pca_features[:, 1].min()), float(pca_features[:, 1].max())],
                "PC3": [float(pca_features[:, 2].min()), float(pca_features[:, 2].max())]
            }
        }
    
    if dinov3_features is not None and dinov2_features is not None:
        if SKLEARN_AVAILABLE:
            correlations = [cosine_similarity([v3], [v2])[0, 0] 
                          for v3, v2 in zip(dinov3_features, dinov2_features)]
        else:
            correlations = [np.corrcoef(v3, v2)[0, 1] 
                          for v3, v2 in zip(dinov3_features, dinov2_features)]
        
        summary["模型對比測試"] = {
            "平均相關性": float(np.mean(correlations)),
            "相關性範圍": [float(np.min(correlations)), float(np.max(correlations))],
            "DINOv3_特徵範圍": [float(dinov3_features.min()), float(dinov3_features.max())],
            "DINOv2_特徵範圍": [float(dinov2_features.min()), float(dinov2_features.max())]
        }
    
    # 保存報告
    with open(output_dir / "test_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印報告
    print("📊 測試總結:")
    for section, data in summary.items():
        print(f"\n{section}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {data}")
    
    print(f"\n💾 詳細報告已保存至: {output_dir / 'test_summary.json'}")


def main():
    """主測試函數"""
    print("🚀 開始 DINOv3 綜合測試...")
    
    # 創建輸出目錄
    project_root = Path(__file__).parent.parent
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    
    test_dir = tmp_dir / f"dinov3_manual_test_{int(time.time())}"
    test_dir.mkdir(exist_ok=True)
    
    print(f"📁 測試輸出目錄: {test_dir}")
    
    try:
        # 1. 基本功能測試
        basic_features, _ = test_dinov3_basic_functionality(test_dir)
        
        # 2. 增強功能測試
        enhanced_features, standard_features = test_enhanced_vs_standard(test_dir)
        
        # 3. Patch Latent PCA 可視化測試
        pca_features, pca_explained_variance = test_patch_latent_pca_visualization(test_dir)
        
        # 4. 模型對比測試
        dinov3_features, dinov2_features = test_dinov3_vs_dinov2_comparison(test_dir)
        
        # 5. 創建總結報告
        create_summary_report(test_dir, basic_features, enhanced_features, standard_features,
                             dinov3_features, dinov2_features, pca_features, pca_explained_variance)
        
        print(f"\n🎉 所有測試完成!")
        print(f"📁 結果保存在: {test_dir}")
        print(f"📋 查看 test_summary.json 獲取詳細結果")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
