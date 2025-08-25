"""
DINOv3 Wrapper 測試模組

測試 DINOv3Wrapper 類的各種功能，包括：
- 基本初始化和配置
- 特徵提取功能  
- 與 video_io 和 SAM 模組的整合測試
- 實際影片特徵提取測試並與 DINOv2 進行對比
- DINOv3 增強功能驗證
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import logging
from unittest.mock import patch, MagicMock
import json
import os
import gc
import time

# 檢查可選套件
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 從 castle 模組導入需要測試的類別
from castle.models.dinov3_wrapper import (
    DINOv3Model, DINOv3Transforms, DINOv3FeatureExtractor, 
    DINOv3Wrapper, MODEL_CONFIGS, DINOV3_RESOLUTION, 
    DINOV3_PATCH_SIZE, DINOV3_PATCH_LEN, DEFAULT_DEVICE,
    DINOV3_ENHANCED_FEATURES
)

# 同時導入 DINOv2 進行對比測試
try:
    from castle.models.dinov2_wrapper import DINOv2Wrapper
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False

# 嘗試導入額外的模組進行整合測試
try:
    from castle.utils.video_io import VideoIO
    VIDEO_IO_AVAILABLE = True
except ImportError:
    VIDEO_IO_AVAILABLE = False

try:
    from castle.models.sam_wrapper import SAMWrapper, ModelSize
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

try:
    from castle.models.deaot_wrapper import DeAOTWrapper, ModelType as DeAOTModelType
    DEAOT_AVAILABLE = True
except ImportError:
    DEAOT_AVAILABLE = False

# 設置日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 環境變數來控制測試行為
IS_CI = os.environ.get('CI', 'false').lower() == 'true'
SKIP_HEAVY_TESTS = os.environ.get('SKIP_HEAVY_TESTS', 'false').lower() == 'true'

# Skip 條件
skip_on_ci = pytest.mark.skipif(
    IS_CI and SKIP_HEAVY_TESTS,
    reason="在 CI 環境中跳過重量級測試"
)

requires_sklearn = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="需要 sklearn 套件"
)

requires_matplotlib = pytest.mark.skipif(
    not MPL_AVAILABLE,
    reason="需要 matplotlib 套件"
)

requires_dinov2 = pytest.mark.skipif(
    not DINOV2_AVAILABLE,
    reason="需要 DINOv2 模組進行對比"
)


class TestDINOv3ModelConfigs:
    """測試 DINOv3 模型配置"""
    
    def test_model_configs_exist(self):
        """測試模型配置是否存在"""
        assert isinstance(MODEL_CONFIGS, dict)
        assert len(MODEL_CONFIGS) > 0
        
        # 檢查 DINOv3 特定的配置
        dinov3_models = [k for k in MODEL_CONFIGS.keys() if k.startswith('dinov3')]
        assert len(dinov3_models) > 0, "應該包含 DINOv3 模型配置"
        
    def test_model_config_structure(self):
        """測試模型配置結構"""
        for model_name, config in MODEL_CONFIGS.items():
            if model_name.startswith('dinov3'):
                assert 'embed_dim' in config
                assert 'version' in config
                assert config['version'] in ['v3', 'v3_enhanced']
                assert isinstance(config['embed_dim'], int)
                assert config['embed_dim'] > 0
                
    def test_dinov3_constants(self):
        """測試 DINOv3 常數"""
        assert DINOV3_RESOLUTION == 518
        assert DINOV3_PATCH_SIZE == 14
        assert DINOV3_PATCH_LEN == 37
        assert isinstance(DINOV3_ENHANCED_FEATURES, dict)


class TestDINOv3ModelWithMocks:
    """使用 Mock 的 DINOv3Model 測試"""
    
    @pytest.fixture
    def mock_torch_hub(self):
        """Mock torch.hub.load"""
        with patch('torch.hub.load') as mock_load:
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.half.return_value = mock_model
            mock_load.return_value = mock_model
            yield mock_load, mock_model
            
    def test_model_initialization(self, mock_torch_hub):
        """測試模型初始化"""
        mock_load, mock_model = mock_torch_hub
        
        model = DINOv3Model(
            model_type='dinov3_vitb14',
            device='cpu',
            use_fp16=False
        )
        
        assert model.model_type == 'dinov3_vitb14'
        assert model.device == 'cpu'
        assert model.embed_dim == 768
        assert model.version == 'v3'
        assert not model.use_fp16
        
    def test_model_initialization_with_enhanced_features(self, mock_torch_hub):
        """測試啟用增強功能的模型初始化"""
        mock_load, mock_model = mock_torch_hub
        
        model = DINOv3Model(
            model_type='dinov3_vitb14',
            device='cpu',
            enable_enhanced_features=True
        )
        
        assert model.enable_enhanced_features
        
    def test_device_fallback(self, mock_torch_hub):
        """測試設備回退邏輯"""
        mock_load, mock_model = mock_torch_hub
        
        # 測試 CUDA 不可用時的回退
        with patch('torch.cuda.is_available', return_value=False):
            model = DINOv3Model(device='cuda')
            assert model.device == 'cpu'
            
    def test_invalid_model_type(self, mock_torch_hub):
        """測試無效模型類型"""
        with pytest.raises(ValueError, match="Unknown model type"):
            DINOv3Model(model_type='invalid_model')
            
    def test_forward_method(self, mock_torch_hub):
        """測試前向傳播方法"""
        mock_load, mock_model = mock_torch_hub
        
        # 設置 mock 返回值
        mock_features = {'x_norm_patchtokens': torch.randn(1, 37*37, 768)}
        mock_model.forward_features.return_value = mock_features
        
        model = DINOv3Model(device='cpu')
        
        # 測試前向傳播
        input_tensor = torch.randn(1, 3, 518, 518)
        result = model.forward(input_tensor)
        
        assert isinstance(result, dict)
        mock_model.forward_features.assert_called_once()


class TestDINOv3Transforms:
    """測試 DINOv3Transforms 類"""
    
    @pytest.fixture
    def transforms(self):
        """創建轉換器實例"""
        return DINOv3Transforms()
        
    @pytest.fixture
    def enhanced_transforms(self):
        """創建增強轉換器實例"""
        return DINOv3Transforms(enhanced_normalization=True)
        
    @pytest.fixture
    def sample_image(self):
        """創建測試影像"""
        return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
    @pytest.fixture
    def sample_mask(self):
        """創建測試遮罩"""
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        return mask
        
    def test_transform_image(self, transforms, sample_image):
        """測試影像轉換"""
        result = transforms.transform_image(sample_image)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, DINOV3_RESOLUTION, DINOV3_RESOLUTION)
        assert result.dtype == torch.float32
        
    def test_enhanced_transform_image(self, enhanced_transforms, sample_image):
        """測試增強影像轉換"""
        result = enhanced_transforms.transform_image(sample_image)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, DINOV3_RESOLUTION, DINOV3_RESOLUTION)
        
    def test_transform_mask_uint8(self, transforms, sample_mask):
        """測試 uint8 遮罩轉換"""
        result = transforms.transform_mask(sample_mask)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, DINOV3_RESOLUTION, DINOV3_RESOLUTION)
        assert 0 <= result.min() <= result.max() <= 1
        
    def test_transform_mask_bool(self, transforms):
        """測試布爾遮罩轉換"""
        bool_mask = np.zeros((224, 224), dtype=bool)
        bool_mask[50:150, 50:150] = True
        
        result = transforms.transform_mask(bool_mask)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, DINOV3_RESOLUTION, DINOV3_RESOLUTION)
        assert 0 <= result.min() <= result.max() <= 1
        
    def test_transform_mask_float(self, transforms):
        """測試浮點遮罩轉換"""
        float_mask = np.zeros((224, 224), dtype=np.float32)
        float_mask[50:150, 50:150] = 1.0
        
        result = transforms.transform_mask(float_mask)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, DINOV3_RESOLUTION, DINOV3_RESOLUTION)
        
    def test_batch_transform_images(self, transforms, sample_image):
        """測試批次影像轉換"""
        images = [sample_image, sample_image, sample_image]
        result = transforms.batch_transform_images(images)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3, DINOV3_RESOLUTION, DINOV3_RESOLUTION)
        
    def test_batch_transform_masks(self, transforms, sample_mask):
        """測試批次遮罩轉換"""
        masks = [sample_mask, sample_mask, sample_mask]
        result = transforms.batch_transform_masks(masks)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 1, DINOV3_RESOLUTION, DINOV3_RESOLUTION)
        
    def test_invalid_mask_input(self, transforms):
        """測試無效遮罩輸入"""
        # 測試非 numpy array
        with pytest.raises(ValueError, match="mask 必須是 numpy array"):
            transforms.transform_mask([1, 2, 3])
            
        # 測試非 2D array
        with pytest.raises(ValueError, match="mask 必須是 2D array"):
            transforms.transform_mask(np.array([[[1, 2], [3, 4]]]))


class TestDINOv3WrapperWithMocks:
    """使用 Mock 的 DINOv3Wrapper 測試"""
    
    @pytest.fixture
    def mock_dinov3_wrapper(self):
        """Mock DINOv3Wrapper"""
        with patch('castle.models.dinov3_wrapper.DINOv3Model') as mock_model_class, \
             patch('castle.models.dinov3_wrapper.torch.hub.load'):
            
            # 設置 mock 模型實例的屬性
            mock_model_instance = MagicMock()
            mock_model_instance.embed_dim = 768
            mock_model_instance.device = 'cpu'
            mock_model_instance.use_fp16 = False
            mock_model_instance.enable_enhanced_features = True
            mock_model_instance.model_type = 'dinov3_vitb14'
            mock_model_instance.version = 'v3'
            
            # 讓 mock 類返回配置好的實例
            mock_model_class.return_value = mock_model_instance
            
            wrapper = DINOv3Wrapper(
                model_type='dinov3_vitb14',
                device='cpu',
                batch_size=8,
                use_fp16=False
            )
            yield wrapper
            
    def test_wrapper_initialization(self, mock_dinov3_wrapper):
        """測試 wrapper 初始化"""
        wrapper = mock_dinov3_wrapper
        
        assert wrapper.embed_dim == 768
        assert wrapper.batch_size == 8
        assert wrapper.device == 'cpu'
        assert not wrapper.use_fp16
        
    def test_wrapper_enhanced_features(self):
        """測試增強功能初始化"""
        with patch('castle.models.dinov3_wrapper.DINOv3Model') as mock_model_class, \
             patch('castle.models.dinov3_wrapper.torch.hub.load'):
            
            # 設置 mock 模型實例的屬性
            mock_model_instance = MagicMock()
            mock_model_instance.embed_dim = 768
            mock_model_instance.device = 'cpu'
            mock_model_instance.use_fp16 = False
            mock_model_instance.enable_enhanced_features = True
            mock_model_instance.model_type = 'dinov3_vitb14'
            mock_model_instance.version = 'v3'
            
            # 讓 mock 類返回配置好的實例
            mock_model_class.return_value = mock_model_instance
            
            wrapper = DINOv3Wrapper(
                model_type='dinov3_vitb14',
                device='cpu',
                enable_enhanced_features=True
            )
            
            assert wrapper.enable_enhanced_features
            
    def test_clear_cache(self, mock_dinov3_wrapper):
        """測試清除快取"""
        wrapper = mock_dinov3_wrapper
        
        # 清除快取不應該引發錯誤
        wrapper.clear_cache()


@pytest.mark.integration
@pytest.mark.skipif(not VIDEO_IO_AVAILABLE, reason="需要 VideoIO 模組")
class TestDINOv3WrapperWithVideo:
    """DINOv3 Wrapper 與影片整合測試"""
    
    @pytest.fixture
    def video_path(self):
        """測試影片路徑"""
        project_root = Path(__file__).parent.parent
        video_path = project_root / "notebooks" / "open_field_videos" / "oft_1min.mp4"
        return video_path
    
    @pytest.fixture
    def ground_truth_path(self):
        """DINOv2 真實數據檔案路徑（用於對比）"""
        project_root = Path(__file__).parent.parent
        gt_path = project_root / "notebooks" / "open_field_videos" / "openfield-1min-raw_ROI_1_latent.npz"
        return gt_path
    
    @pytest.fixture
    def output_dir(self):
        """測試輸出目錄"""
        project_root = Path(__file__).parent.parent
        tmp_dir = project_root / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        
        temp_dir = tempfile.mkdtemp(prefix="dinov3_test_", dir=str(tmp_dir))
        output_dir = Path(temp_dir)
        return output_dir
    
    def test_video_file_exists(self, video_path):
        """確認測試影片檔案存在"""
        assert video_path.exists(), f"測試影片檔案不存在: {video_path}"
    
    def test_ground_truth_file_exists(self, ground_truth_path):
        """確認 DINOv2 真實數據檔案存在"""
        assert ground_truth_path.exists(), f"DINOv2 真實數據檔案不存在: {ground_truth_path}"
        
        # 載入並檢查數據結構
        data = np.load(ground_truth_path)
        assert 'latent' in data, "真實數據檔案缺少 'latent' key"
        
        latent = data['latent']
        print(f"DINOv2 真實數據形狀: {latent.shape}")
        print(f"DINOv2 真實數據類型: {latent.dtype}")
        
        # 應該是 (time, embed_dim) 形狀
        assert len(latent.shape) == 2, "Latent 數據應該是 2D array"
        assert latent.shape[1] == 768, "Embed dimension 應該是 768 (vitb14)"
    
    @pytest.mark.slow
    @skip_on_ci
    def test_extract_features_from_video(self, video_path, output_dir):
        """從影片提取 DINOv3 特徵測試（不使用 ROI）"""
        if not video_path.exists():
            pytest.skip(f"測試影片檔案不存在: {video_path}")
            
        print(f"\n🎬 開始 DINOv3 影片特徵提取測試")
        print(f"   影片路徑: {video_path}")
        print(f"   輸出目錄: {output_dir}")
        
        # 初始化 DINOv3 wrapper
        wrapper = DINOv3Wrapper(
            model_type='dinov3_vitb14',
            device='cpu',  # 使用 CPU 確保測試穩定性
            batch_size=4,
            enable_enhanced_features=True
        )
        
        # 初始化影片讀取器
        video_reader = VideoIO.load_video(str(video_path))
        
        # 載入影片幀
        frames = []
        frame_indices = []
        
        print("📖 載入影片幀...")
        # 只處理前30幀進行測試
        for i in range(min(30, video_reader.frame_count)):
            frame = video_reader.get_frame(i)
            if frame is not None:
                frames.append(frame)
                frame_indices.append(i)
        
        print(f"   載入了 {len(frames)} 幀")
        
        # 創建簡單的 ROI 遮罩（中央區域）
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            roi_mask[h//4:3*h//4, w//4:3*w//4] = 255
            
            print(f"   ROI 遮罩形狀: {roi_mask.shape}")
            print(f"   ROI 遮罩範圍: [{roi_mask.min()}, {roi_mask.max()}]")
        
        # 提取特徵
        print("🔍 提取 DINOv3 特徵...")
        extracted_features = []
        processing_times = []
        
        start_time = time.time()
        
        for i, frame in enumerate(frames):
            frame_start = time.time()
            
            # 提取特徵
            feature = wrapper.extract_features(frame, roi_mask)
            extracted_features.append(feature)
            
            frame_time = time.time() - frame_start
            processing_times.append(frame_time)
            
            if (i + 1) % 10 == 0:
                print(f"   處理了 {i + 1}/{len(frames)} 幀 "
                      f"(平均: {np.mean(processing_times):.3f}s/幀)")
        
        total_time = time.time() - start_time
        
        # 轉換為 numpy array
        features_array = np.array(extracted_features)
        
        print(f"✅ 特徵提取完成!")
        print(f"   特徵形狀: {features_array.shape}")
        print(f"   特徵類型: {features_array.dtype}")
        print(f"   特徵範圍: [{features_array.min():.6f}, {features_array.max():.6f}]")
        print(f"   總處理時間: {total_time:.2f}s")
        print(f"   平均處理時間: {np.mean(processing_times):.3f}s/幀")
        
        # 保存結果
        output_file = output_dir / "dinov3_extracted_features.npz"
        np.savez(
            output_file,
            latent=features_array,
            frame_indices=np.array(frame_indices),
            processing_times=np.array(processing_times),
            metadata={
                'model_type': 'dinov3_vitb14',
                'enhanced_features': True,
                'video_path': str(video_path),
                'total_frames': len(frames),
                'embed_dim': wrapper.embed_dim,
                'total_time': total_time,
                'avg_time_per_frame': np.mean(processing_times)
            }
        )
        
        print(f"💾 結果已保存至: {output_file}")
        
        # 基本驗證
        assert features_array.shape[0] == len(frames)
        assert features_array.shape[1] == wrapper.embed_dim
        assert not np.any(np.isnan(features_array))
        assert not np.any(np.isinf(features_array))
        
        # 清理
        wrapper.clear_cache()
        
    @pytest.mark.slow
    @requires_dinov2
    @requires_sklearn
    @requires_matplotlib
    @skip_on_ci
    def test_compare_dinov3_vs_dinov2(self, video_path, ground_truth_path, output_dir):
        """DINOv3 vs DINOv2 特徵對比測試"""
        if not video_path.exists():
            pytest.skip(f"測試影片檔案不存在: {video_path}")
        if not ground_truth_path.exists():
            pytest.skip(f"DINOv2 真實數據檔案不存在: {ground_truth_path}")
            
        print(f"\n🆚 開始 DINOv3 vs DINOv2 對比測試")
        print(f"   影片路徑: {video_path}")
        print(f"   DINOv2 GT 路徑: {ground_truth_path}")
        print(f"   輸出目錄: {output_dir}")
        
        # 載入 DINOv2 真實數據
        gt_data = np.load(ground_truth_path)
        dinov2_features = gt_data['latent']
        
        print(f"📊 DINOv2 真實數據: {dinov2_features.shape}")
        
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
        
        # 初始化影片讀取器
        video_reader = VideoIO.load_video(str(video_path))
        
        # 載入影片幀（與真實數據對應的幀數）
        frames = []
        n_frames = min(dinov2_features.shape[0], 20)  # 限制測試幀數
        
        print(f"📖 載入 {n_frames} 幀進行對比...")
        for i in range(n_frames):
            frame = video_reader.get_frame(i)
            if frame is not None:
                frames.append(frame)
        
        # 創建 ROI 遮罩
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            roi_mask[h//4:3*h//4, w//4:3*w//4] = 255
        
        # 提取 DINOv3 特徵
        print("🔍 提取 DINOv3 特徵...")
        dinov3_features = []
        dinov3_times = []
        
        for i, frame in enumerate(frames):
            start_time = time.time()
            feature = dinov3_wrapper.extract_features(frame, roi_mask)
            dinov3_features.append(feature)
            dinov3_times.append(time.time() - start_time)
            
        dinov3_features = np.array(dinov3_features)
        
        # 提取 DINOv2 特徵進行對比
        print("🔍 提取 DINOv2 特徵...")
        dinov2_test_features = []
        dinov2_times = []
        
        for i, frame in enumerate(frames):
            start_time = time.time()
            feature = dinov2_wrapper.extract_features(frame, roi_mask)
            dinov2_test_features.append(feature)
            dinov2_times.append(time.time() - start_time)
            
        dinov2_test_features = np.array(dinov2_test_features)
        
        print(f"✅ 特徵提取完成!")
        print(f"   DINOv3 特徵: {dinov3_features.shape}")
        print(f"   DINOv2 特徵: {dinov2_test_features.shape}")
        print(f"   DINOv2 GT 特徵: {dinov2_features[:n_frames].shape}")
        
        # 計算相關性
        correlations = {}
        
        # DINOv3 vs DINOv2 (新提取)
        v3_v2_corr = []
        for i in range(min(len(dinov3_features), len(dinov2_test_features))):
            corr = cosine_similarity([dinov3_features[i]], [dinov2_test_features[i]])[0, 0]
            v3_v2_corr.append(corr)
        correlations['dinov3_vs_dinov2_test'] = np.mean(v3_v2_corr)
        
        # DINOv3 vs DINOv2 GT
        v3_gt_corr = []
        for i in range(min(len(dinov3_features), n_frames)):
            corr = cosine_similarity([dinov3_features[i]], [dinov2_features[i]])[0, 0]
            v3_gt_corr.append(corr)
        correlations['dinov3_vs_dinov2_gt'] = np.mean(v3_gt_corr)
        
        # DINOv2 test vs GT
        v2_gt_corr = []
        for i in range(min(len(dinov2_test_features), n_frames)):
            corr = cosine_similarity([dinov2_test_features[i]], [dinov2_features[i]])[0, 0]
            v2_gt_corr.append(corr)
        correlations['dinov2_test_vs_gt'] = np.mean(v2_gt_corr)
        
        print(f"📈 相關性分析:")
        for key, value in correlations.items():
            print(f"   {key}: {value:.6f}")
        
        # 性能比較
        performance = {
            'dinov3_avg_time': np.mean(dinov3_times),
            'dinov2_avg_time': np.mean(dinov2_times),
            'dinov3_total_time': np.sum(dinov3_times),
            'dinov2_total_time': np.sum(dinov2_times)
        }
        
        print(f"⚡ 性能比較:")
        print(f"   DINOv3 平均時間: {performance['dinov3_avg_time']:.3f}s/幀")
        print(f"   DINOv2 平均時間: {performance['dinov2_avg_time']:.3f}s/幀")
        print(f"   速度比率: {performance['dinov2_avg_time']/performance['dinov3_avg_time']:.2f}x")
        
        # 創建可視化
        self._create_comparison_visualization(
            dinov3_features, dinov2_test_features, dinov2_features[:n_frames],
            correlations, performance, output_dir
        )
        
        # 保存結果
        comparison_file = output_dir / "dinov3_vs_dinov2_comparison.npz"
        np.savez(
            comparison_file,
            dinov3_features=dinov3_features,
            dinov2_test_features=dinov2_test_features,
            dinov2_gt_features=dinov2_features[:n_frames],
            correlations=correlations,
            performance=performance,
            metadata={
                'n_frames': n_frames,
                'dinov3_model': 'dinov3_vitb14',
                'dinov2_model': 'dinov2_vitb14_reg',
                'enhanced_features': True
            }
        )
        
        print(f"💾 對比結果已保存至: {comparison_file}")
        
        # 驗證
        assert correlations['dinov3_vs_dinov2_test'] > 0.9, "DINOv3 和 DINOv2 相關性應該 > 0.9 (高度相容)"
        # 注意：test vs GT 相關性可能較低，因為 ROI 遮罩和生成條件可能不同
        assert correlations['dinov2_test_vs_gt'] > 0.1, "DINOv2 測試與 GT 應該有基本相關性"
        
        # 清理
        dinov3_wrapper.clear_cache()
        dinov2_wrapper.clear_cache()
    
    @requires_matplotlib
    def _create_comparison_visualization(self, dinov3_features, dinov2_features, gt_features,
                                       correlations, performance, output_dir):
        """創建對比可視化"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('DINOv3 vs DINOv2 Feature Comparison', fontsize=16, fontweight='bold')
            
            # 1. 特徵維度分布對比
            axes[0, 0].hist(dinov3_features.mean(axis=0), alpha=0.7, label='DINOv3', bins=50, color='blue')
            axes[0, 0].hist(dinov2_features.mean(axis=0), alpha=0.7, label='DINOv2', bins=50, color='red')
            axes[0, 0].set_xlabel('Feature Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Feature Distribution Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 時間序列特徵範數對比
            dinov3_norms = np.linalg.norm(dinov3_features, axis=1)
            dinov2_norms = np.linalg.norm(dinov2_features, axis=1)
            gt_norms = np.linalg.norm(gt_features, axis=1)
            
            axes[0, 1].plot(dinov3_norms, label='DINOv3', marker='o', color='blue', alpha=0.7)
            axes[0, 1].plot(dinov2_norms, label='DINOv2 (Test)', marker='s', color='red', alpha=0.7)
            axes[0, 1].plot(gt_norms, label='DINOv2 (GT)', marker='^', color='green', alpha=0.7)
            axes[0, 1].set_xlabel('Frame Index')
            axes[0, 1].set_ylabel('Feature Norm')
            axes[0, 1].set_title('Feature Norm Over Time')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 相關性條形圖
            corr_names = list(correlations.keys())
            corr_values = list(correlations.values())
            colors = ['blue', 'red', 'green']
            
            bars = axes[1, 0].bar(range(len(corr_names)), corr_values, color=colors[:len(corr_names)])
            axes[1, 0].set_xlabel('Comparison Type')
            axes[1, 0].set_ylabel('Cosine Similarity')
            axes[1, 0].set_title('Feature Correlation Analysis')
            axes[1, 0].set_xticks(range(len(corr_names)))
            axes[1, 0].set_xticklabels([name.replace('_', '\n') for name in corr_names], fontsize=8)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加數值標籤
            for bar, value in zip(bars, corr_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 4. 性能比較
            perf_names = ['DINOv3', 'DINOv2']
            perf_values = [performance['dinov3_avg_time'], performance['dinov2_avg_time']]
            colors_perf = ['blue', 'red']
            
            bars_perf = axes[1, 1].bar(perf_names, perf_values, color=colors_perf)
            axes[1, 1].set_ylabel('Average Time (seconds/frame)')
            axes[1, 1].set_title('Processing Speed Comparison')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加數值標籤
            for bar, value in zip(bars_perf, perf_values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                              f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # 保存圖表
            plot_path = output_dir / "dinov3_vs_dinov2_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 對比可視化已保存至: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ 可視化創建失敗: {e}")


@pytest.mark.integration  
@pytest.mark.slow
@pytest.mark.skipif(not SAM_AVAILABLE, reason="需要 SAM 模組")
@pytest.mark.skipif(not VIDEO_IO_AVAILABLE, reason="需要 VideoIO 模組")
@requires_sklearn
@skip_on_ci
class TestDINOv3EnhancedFeatures:
    """DINOv3 增強功能專項測試"""
    
    @pytest.fixture
    def output_dir(self):
        """測試輸出目錄"""
        project_root = Path(__file__).parent.parent
        tmp_dir = project_root / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        
        temp_dir = tempfile.mkdtemp(prefix="dinov3_enhanced_test_", dir=str(tmp_dir))
        output_dir = Path(temp_dir)
        print(f"DINOv3 增強功能測試輸出目錄: {output_dir}")
        return output_dir
    
    def test_enhanced_vs_standard_features(self, output_dir):
        """測試增強功能 vs 標準功能"""
        print(f"\n🔬 開始 DINOv3 增強功能對比測試")
        
        # 創建測試影像
        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 創建測試 ROI
        roi_mask = np.zeros((480, 640), dtype=np.uint8)
        roi_mask[120:360, 160:480] = 255
        
        # 初始化兩個 wrapper：一個啟用增強功能，一個不啟用
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
        
        print("🔍 提取增強特徵...")
        start_time = time.time()
        enhanced_features = wrapper_enhanced.extract_features(test_image, roi_mask)
        enhanced_time = time.time() - start_time
        
        print("🔍 提取標準特徵...")
        start_time = time.time()
        standard_features = wrapper_standard.extract_features(test_image, roi_mask)
        standard_time = time.time() - start_time
        
        print(f"✅ 特徵提取完成!")
        print(f"   增強特徵形狀: {enhanced_features.shape}")
        print(f"   標準特徵形狀: {standard_features.shape}")
        print(f"   增強特徵時間: {enhanced_time:.3f}s")
        print(f"   標準特徵時間: {standard_time:.3f}s")
        
        # 計算特徵差異
        feature_diff = np.abs(enhanced_features - standard_features)
        feature_corr = cosine_similarity([enhanced_features], [standard_features])[0, 0]
        
        print(f"📊 特徵分析:")
        print(f"   特徵差異 (平均): {feature_diff.mean():.6f}")
        print(f"   特徵差異 (最大): {feature_diff.max():.6f}")
        print(f"   特徵相關性: {feature_corr:.6f}")
        
        # 測試 debug 功能
        print("\n🐛 測試 debug 功能...")
        debug_enhanced = wrapper_enhanced.debug_mask_processing(roi_mask)
        debug_standard = wrapper_standard.debug_mask_processing(roi_mask)
        
        # 保存結果
        results = {
            'enhanced_features': enhanced_features,
            'standard_features': standard_features,
            'feature_diff': feature_diff,
            'feature_correlation': feature_corr,
            'enhanced_time': enhanced_time,
            'standard_time': standard_time,
            'debug_enhanced': debug_enhanced,
            'debug_standard': debug_standard
        }
        
        result_file = output_dir / "dinov3_enhanced_vs_standard.npz"
        np.savez(result_file, **results)
        
        print(f"💾 結果已保存至: {result_file}")
        
        # 驗證
        assert enhanced_features.shape == standard_features.shape
        assert feature_corr > 0.8, "增強和標準特徵應該高度相關"
        assert not np.array_equal(enhanced_features, standard_features), "增強特徵應該與標準特徵有所不同"
        
        # 清理
        wrapper_enhanced.clear_cache()
        wrapper_standard.clear_cache()


def run_manual_test():
    """手動測試函數，可直接執行進行開發測試"""
    print("🚀 執行 DINOv3 手動測試...")
    
    try:
        # 測試基本初始化
        print("\n1. 測試基本初始化...")
        wrapper = DINOv3Wrapper(
            model_type='dinov3_vitb14',
            device='cpu',
            enable_enhanced_features=True
        )
        print(f"   ✅ DINOv3Wrapper 初始化成功")
        print(f"   模型類型: {wrapper.model.model_type}")
        print(f"   嵌入維度: {wrapper.embed_dim}")
        print(f"   增強功能: {wrapper.enable_enhanced_features}")
        
        # 測試特徵提取
        print("\n2. 測試特徵提取...")
        test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        roi_mask = np.ones((224, 224), dtype=np.uint8) * 255
        
        features = wrapper.extract_features(test_image, roi_mask)
        print(f"   ✅ 特徵提取成功")
        print(f"   特徵形狀: {features.shape}")
        print(f"   特徵範圍: [{features.min():.6f}, {features.max():.6f}]")
        
        # 測試 debug 功能
        print("\n3. 測試 debug 功能...")
        debug_info = wrapper.debug_mask_processing(roi_mask)
        print(f"   ✅ Debug 功能正常")
        
        print("\n🎉 所有手動測試通過!")
        
    except Exception as e:
        print(f"\n❌ 手動測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_manual_test()
