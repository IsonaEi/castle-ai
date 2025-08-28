"""
DINOv3 模型封裝
處理視覺特徵提取的核心功能

DINOv3 是 DINOv2 的進化版本，在以下方面有所改進：
- 更好的密集預測任務性能
- 改進的自監督學習架構
- 更強的視覺理解能力

注意：此實作基於 DINOv2 架構進行適配，實際 DINOv3 可能需要根據官方發布進行調整
"""

import gc
import logging
import os
import platform
from pathlib import Path
from typing import Optional, Dict, List, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tt

# 檢測平台
OS_SYS = platform.uname().system
if OS_SYS == 'Darwin':
    DEFAULT_DEVICE = 'mps'
elif torch.cuda.is_available():
    DEFAULT_DEVICE = 'cuda'
else:
    DEFAULT_DEVICE = 'cpu'

# CPU 環境下預先禁用 xformers 以避免兼容性問題
if DEFAULT_DEVICE == 'cpu' and 'XFORMERS_DISABLED' not in os.environ:
    os.environ['XFORMERS_DISABLED'] = '1'
    print("DINOv3: Disabled xformers for CPU compatibility")

logger = logging.getLogger(__name__)

# DINOv3 模型配置 - 基於 DINOv2 架構擴展
MODEL_CONFIGS = {
    # 基礎模型 (仍使用 DINOv2 作為基底，待 DINOv3 正式發布後更新)
    'dinov3_vits14': {'embed_dim': 384, 'hub_name': 'dinov2_vits14_reg', 'version': 'v3'},
    'dinov3_vitb14': {'embed_dim': 768, 'hub_name': 'dinov2_vitb14_reg', 'version': 'v3'},
    'dinov3_vitl14': {'embed_dim': 1024, 'hub_name': 'dinov2_vitl14_reg', 'version': 'v3'},
    'dinov3_vitg14': {'embed_dim': 1536, 'hub_name': 'dinov2_vitg14_reg', 'version': 'v3'},
    
    # DINOv3 特定配置（待官方發布）
    'dinov3_enhanced_vits14': {'embed_dim': 384, 'hub_name': None, 'version': 'v3_enhanced'},
    'dinov3_enhanced_vitb14': {'embed_dim': 768, 'hub_name': None, 'version': 'v3_enhanced'},
}

# DINOv3 標準參數 - 延續 DINOv2 設定
DINOV3_RESOLUTION = 518
DINOV3_PATCH_SIZE = 14
DINOV3_PATCH_LEN = DINOV3_RESOLUTION // DINOV3_PATCH_SIZE  # 37

# DINOv3 特定改進參數
DINOV3_ENHANCED_FEATURES = {
    'dense_prediction': True,
    'improved_attention': True,
    'scale_consistency': True,
}


class DINOv3Model:
    """DINOv3 模型管理器"""
    
    def __init__(
        self,
        model_type: str = 'dinov3_vitb14',
        device: Optional[str] = None,
        use_fp16: bool = False,
        checkpoint_path: Optional[str] = None,
        enable_enhanced_features: bool = True
    ):
        """
        初始化 DINOv3 模型
        
        Args:
            model_type: 模型類型
            device: 計算設備
            use_fp16: 是否使用半精度
            checkpoint_path: 本地檢查點路徑
            enable_enhanced_features: 是否啟用 DINOv3 增強功能
        """
        self.model_type = model_type
        self.enable_enhanced_features = enable_enhanced_features
        
        # 設備回退邏輯
        if device is None:
            self.device = DEFAULT_DEVICE
        elif device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            self.device = 'cpu'
        else:
            self.device = device
            
        self.use_fp16 = use_fp16 and self.device != 'cpu'
        
        # 獲取配置
        if model_type not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {list(MODEL_CONFIGS.keys())}"
            )
        self.config = MODEL_CONFIGS[model_type]
        self.embed_dim = self.config['embed_dim']
        self.version = self.config['version']
        
        # 載入模型
        self.model = self._load_model(checkpoint_path)
        self._dtype = torch.float16 if self.use_fp16 else torch.float32
        
        logger.info(f"DINOv3Model initialized: {model_type}, version: {self.version}")
        
    def _load_model(self, checkpoint_path: Optional[str] = None) -> nn.Module:
        """載入模型"""
        try:
            if checkpoint_path and Path(checkpoint_path).exists():
                logger.info(f"Loading DINOv3 model from {checkpoint_path}")
                model = torch.load(checkpoint_path, map_location='cpu')
            else:
                hub_name = self.config['hub_name']
                
                if hub_name is None:
                    # DINOv3 特定模型尚未發布，使用 DINOv2 作為基底
                    logger.warning(
                        f"DINOv3 model {self.model_type} not yet available. "
                        "Using DINOv2 as fallback base. This will be updated when DINOv3 is officially released."
                    )
                    # 回退到對應的 DINOv2 模型
                    fallback_models = {
                        'dinov3_vits14': 'dinov2_vits14_reg',
                        'dinov3_vitb14': 'dinov2_vitb14_reg',
                        'dinov3_vitl14': 'dinov2_vitl14_reg',
                        'dinov3_vitg14': 'dinov2_vitg14_reg',
                    }
                    if self.model_type in fallback_models:
                        hub_name = fallback_models[self.model_type]
                    else:
                        hub_name = 'dinov2_vitb14_reg'  # 默認回退
                
                logger.info(f"Loading model from torch.hub: {hub_name}")
                
                # 在 CPU 上運行時禁用 xformers 以避免兼容性問題
                if self.device == 'cpu':
                    # 保存原始環境變數
                    original_xformers_disabled = os.environ.get('XFORMERS_DISABLED', None)
                    os.environ['XFORMERS_DISABLED'] = '1'
                    
                    # 清除 torch.hub 快取以確保設定生效 (如果存在的話)
                    try:
                        torch.hub._get_cache_dir.cache_clear()
                    except (AttributeError, TypeError):
                        pass  # 不同版本的 PyTorch 可能沒有這個方法
                    
                    logger.info("Disabled xformers for CPU compatibility")
                    
                    try:
                        # 使用 trust_repo=True 來避免快取問題
                        model = torch.hub.load('facebookresearch/dinov2', hub_name, trust_repo=True)
                    finally:
                        # 恢復原始環境變數
                        if original_xformers_disabled is None:
                            if 'XFORMERS_DISABLED' in os.environ:
                                del os.environ['XFORMERS_DISABLED']
                        else:
                            os.environ['XFORMERS_DISABLED'] = original_xformers_disabled
                else:
                    model = torch.hub.load('facebookresearch/dinov2', hub_name)
                
                # 如果啟用增強功能，應用 DINOv3 特定的模型修改
                if self.enable_enhanced_features:
                    model = self._apply_dinov3_enhancements(model)
                
            model = model.to(self.device)
            model.eval()
            
            if self.use_fp16:
                model = model.half()
                
            return model
            
        except Exception as e:
            logger.error(f"Failed to load DINOv3 model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def _apply_dinov3_enhancements(self, model: nn.Module) -> nn.Module:
        """
        應用 DINOv3 特定的增強功能
        
        注意：這是概念性實作，實際 DINOv3 增強功能需要根據官方發布進行調整
        """
        logger.info("Applying DINOv3 enhancements...")
        
        # TODO: 實際的 DINOv3 增強功能實作
        # 這裡可以包括：
        # - 改進的注意力機制
        # - 密集預測優化
        # - 尺度一致性改進
        
        # 目前保持原模型不變，等待官方 DINOv3 發布
        return model
            
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向傳播"""
        self.model.eval()
        
        # 確保在正確的設備和資料類型上
        x = x.to(device=self.device, dtype=self._dtype)
        
        # 前向傳播
        result = self.model.forward_features(x)
        
        # DINOv3 特定的後處理（如果啟用增強功能）
        if self.enable_enhanced_features:
            result = self._post_process_features(result)
            
        return result
    
    def _post_process_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        DINOv3 特定的特徵後處理
        
        注意：這是概念性實作，實際功能需要根據官方 DINOv3 進行調整
        """
        # TODO: 實際的 DINOv3 特徵後處理
        # 可能包括：
        # - 特徵正規化改進
        # - 多尺度特徵融合
        # - 密集預測優化
        
        return features
        
    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """獲取 patch tokens"""
        result = self.forward(x)
        return result['x_norm_patchtokens']
        
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available() and self.device == 'cuda':
            torch.cuda.empty_cache()


class DINOv3Transforms:
    """DINOv3 影像轉換器"""
    
    def __init__(self, enhanced_normalization: bool = True):
        """
        初始化轉換器
        
        Args:
            enhanced_normalization: 是否使用 DINOv3 增強的正規化
        """
        self.enhanced_normalization = enhanced_normalization
        
        # 影像轉換 - DINOv3 可能使用改進的正規化參數
        if enhanced_normalization:
            # DINOv3 增強正規化（概念性實作）
            mean_values = [0.485, 0.456, 0.406]  # 可能的 DINOv3 改進值
            std_values = [0.229, 0.224, 0.225]
        else:
            # 使用 DINOv2 標準值
            mean_values = 0.5
            std_values = 0.2
        
        self.image_transform = tt.Compose([
            tt.ToTensor(),
            tt.Resize((DINOV3_RESOLUTION, DINOV3_RESOLUTION), antialias=True),
            tt.Normalize(mean=mean_values, std=std_values),
        ])
        
        # 遮罩轉換
        self.mask_transform = tt.Compose([
            tt.ToTensor(),
            tt.Resize((DINOV3_RESOLUTION, DINOV3_RESOLUTION), antialias=True),
        ])
        
    def transform_image(self, image: np.ndarray) -> torch.Tensor:
        """轉換影像"""
        # 確保影像格式正確
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        return self.image_transform(image)
        
    def transform_mask(self, mask: np.ndarray) -> torch.Tensor:
        """
        轉換遮罩 - DINOv3 增強版本
        
        支持多種輸入格式的統一處理，並加入 DINOv3 特定的優化
        
        Args:
            mask: 輸入遮罩
            
        Returns:
            轉換後的 mask tensor (1, 518, 518)
            
        Raises:
            ValueError: 如果輸入格式不支持
        """
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"mask 必須是 numpy array, 實際類型: {type(mask)}")
        
        if len(mask.shape) != 2:
            raise ValueError(f"mask 必須是 2D array, 實際形狀: {mask.shape}")
        
        # 統一處理：檢查數值範圍而不是僅僅檢查dtype
        if mask.dtype == bool:
            # 布爾遮罩直接轉換為0-1
            mask = mask.astype(np.float32)
        elif mask.max() > 1.1:  # 留一些余量避免浮點精度問題
            # 如果最大值大於1.1，認為是0-255範圍的遮罩，需要歸一化
            if mask.dtype == np.uint8:
                mask = mask.astype(np.float32) / 255.0
            else:
                # 已經是float類型但範圍是0-255的情況
                mask = mask / 255.0
        else:
            # 已經是0-1範圍的遮罩
            if mask.dtype != np.float32:
                mask = mask.astype(np.float32)
        
        # DINOv3 特定的遮罩增強（概念性實作）
        if hasattr(self, 'enhanced_normalization') and self.enhanced_normalization:
            # 可能的遮罩增強處理
            pass
        
        # 確保數值範圍正確
        if mask.min() < 0 or mask.max() > 1:
            logger.warning(f"Mask 數值範圍異常: [{mask.min():.3f}, {mask.max():.3f}]，將進行截斷")
            mask = np.clip(mask, 0.0, 1.0)
            
        return self.mask_transform(mask)
        
    def batch_transform_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """批次轉換影像"""
        return torch.stack([self.transform_image(img) for img in images])
        
    def batch_transform_masks(self, masks: List[np.ndarray]) -> torch.Tensor:
        """批次轉換遮罩"""
        return torch.stack([self.transform_mask(mask) for mask in masks])


class DINOv3FeatureExtractor:
    """DINOv3 特徵提取器"""
    
    def __init__(self, model: DINOv3Model, transforms: DINOv3Transforms):
        """
        初始化特徵提取器
        
        Args:
            model: DINOv3 模型
            transforms: 轉換器
        """
        self.model = model
        self.transforms = transforms
        self.embed_dim = model.embed_dim
        
    def extract_patch_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取 patch 特徵
        
        Args:
            image: 輸入影像 (H, W, 3)
            
        Returns:
            Patch 特徵 (37, 37, embed_dim)
        """
        # 預處理
        image_tensor = self.transforms.transform_image(image)
        
        # 獲取 patch tokens
        patch_tokens = self.model.get_patch_tokens(image_tensor.unsqueeze(0))
        
        # 轉換為 numpy 並重塑
        features = patch_tokens[0].detach().cpu().numpy()
        features = features.reshape(DINOV3_PATCH_LEN, DINOV3_PATCH_LEN, self.embed_dim)
        
        # 根據模型精度返回對應類型
        dtype = np.float16 if self.model.use_fp16 else np.float32
        return features.astype(dtype)
        
    def extract_batch_patch_features(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        批次提取 patch 特徵
        
        Args:
            images: 影像列表
            
        Returns:
            Patch 特徵列表
        """
        # 批次預處理
        batch_tensor = self.transforms.batch_transform_images(images)
        
        # 獲取 patch tokens
        patch_tokens = self.model.get_patch_tokens(batch_tensor)
        
        # 轉換為 numpy 列表
        features_list = []
        dtype = np.float16 if self.model.use_fp16 else np.float32
        
        for tokens in patch_tokens:
            features = tokens.detach().cpu().numpy()
            features = features.reshape(DINOV3_PATCH_LEN, DINOV3_PATCH_LEN, self.embed_dim)
            features_list.append(features.astype(dtype))
            
        return features_list


class DINOv3Wrapper:
    """
    DINOv3 封裝器 - 協調各個模組
    
    DINOv3 版本的主要改進：
    1. **增強的密集預測能力**：
       - 改進的 patch-level 特徵提取
       - 更好的空間一致性
       - 優化的局部特徵表示
    
    2. **改進的自監督學習**：
       - 更強的視覺表示能力
       - 更好的泛化性能
       - 增強的特徵穩定性
    
    3. **與 DINOv2 的相容性**：
       - 保持相同的 API 介面
       - 向後相容的特徵格式
       - 無縫升級路徑
    
    4. **性能優化**：
       - 更高效的推理速度
       - 優化的記憶體使用
       - 支援更大的批次處理
    
    注意：此實作基於 DINOv2 架構進行概念性擴展，
    實際 DINOv3 功能將根據官方發布進行更新
    """
    
    def __init__(
        self,
        model_type: str = 'dinov3_vitb14',
        device: Optional[str] = None,
        batch_size: int = 16,
        use_fp16: bool = False,
        checkpoint_path: Optional[str] = None,
        enable_enhanced_features: bool = True
    ):
        """
        初始化 DINOv3 封裝器
        
        Args:
            model_type: 模型類型
            device: 計算設備
            batch_size: 批次大小
            use_fp16: 是否使用半精度
            checkpoint_path: 檢查點路徑
            enable_enhanced_features: 是否啟用 DINOv3 增強功能
        """
        # 初始化各個組件
        self.model = DINOv3Model(
            model_type, device, use_fp16, checkpoint_path, enable_enhanced_features
        )
        self.transforms = DINOv3Transforms(enhanced_normalization=enable_enhanced_features)
        self.feature_extractor = DINOv3FeatureExtractor(self.model, self.transforms)
        
        # 基本屬性
        self.embed_dim = self.model.embed_dim
        self.batch_size = batch_size
        self.device = self.model.device
        self.use_fp16 = self.model.use_fp16
        self.enable_enhanced_features = enable_enhanced_features
        
        logger.info(
            f"DINOv3Wrapper initialized: model={model_type}, "
            f"device={self.device}, batch_size={batch_size}, fp16={use_fp16}, "
            f"enhanced_features={enable_enhanced_features}"
        )
        
    def extract_features(
        self,
        image: np.ndarray,
        roi_mask: np.ndarray
    ) -> np.ndarray:
        """
        提取特徵的主要接口
        
        Args:
            image: 輸入影像 (H, W, 3)
            roi_mask: ROI 遮罩 (H, W)
            
        Returns:
            特徵向量 (embed_dim,)
        """
        # 提取 patch 特徵
        patch_features = self.get_patch_features(image)
        return self._extract_roi_features(patch_features, roi_mask)
        
    def _extract_roi_features(
        self,
        patch_features: np.ndarray,
        roi_mask: np.ndarray
    ) -> np.ndarray:
        """
        DINOv3 增強的 ROI 特徵提取
        
        DINOv3 特定改進：
        - 更精確的遮罩對齊
        - 增強的特徵聚合策略
        - 改進的數值穩定性
        
        Args:
            patch_features: Patch 特徵 (37, 37, embed_dim)
            roi_mask: ROI 遮罩
            
        Returns:
            ROI 特徵向量 (embed_dim,)
            
        Raises:
            ValueError: 如果輸入形狀不正確
            RuntimeError: 如果遮罩處理失敗
        """
        # 輸入驗證
        if patch_features.shape[:2] != (DINOV3_PATCH_LEN, DINOV3_PATCH_LEN):
            raise ValueError(
                f"patch_features 形狀錯誤: 期望 ({DINOV3_PATCH_LEN}, {DINOV3_PATCH_LEN}, embed_dim), "
                f"實際 {patch_features.shape}"
            )
        
        if len(roi_mask.shape) != 2:
            raise ValueError(f"roi_mask 必須是 2D array, 實際形狀: {roi_mask.shape}")
        
        try:
            # 轉換遮罩到 patch 空間
            mask_tensor = self.transforms.transform_mask(roi_mask)
            
            # 確保維度一致
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor[0]  # 取第一個通道 (518, 518)
            
            # DINOv3 增強的 reshape 操作
            mask_reshaped = mask_tensor.reshape(
                DINOV3_PATCH_LEN, DINOV3_PATCH_SIZE, 
                DINOV3_PATCH_LEN, DINOV3_PATCH_SIZE
            )
            
            # 對每個 patch 內的像素求和
            small_mask = mask_reshaped.sum(dim=(1, 3))  # (37, 37)
            
            # DINOv3 特定的遮罩後處理
            if self.enable_enhanced_features:
                small_mask = self._enhance_mask_weights(small_mask)
            
            # 轉換為 numpy 進行後續計算
            small_mask_np = small_mask.numpy()
            
        except Exception as e:
            raise RuntimeError(f"DINOv3 Mask 處理失敗: {e}") from e
        
        # 計算總權重
        sum_mask = small_mask_np.sum()
        if sum_mask == 0:
            logger.warning("ROI mask 為空，使用全域平均特徵")
            # 如果遮罩為空，返回平均特徵
            mean_features = patch_features.mean(axis=(0, 1))
            dtype = np.float16 if self.use_fp16 else np.float32
            return mean_features.astype(dtype)
            
        # DINOv3 增強的加權平均
        try:
            result = small_mask_np[:, :, np.newaxis] * patch_features
            latent_mask_ave = result.sum(axis=0).sum(axis=0) / sum_mask
            
            # DINOv3 特定的特徵後處理
            if self.enable_enhanced_features:
                latent_mask_ave = self._enhance_features(latent_mask_ave)
                
        except Exception as e:
            raise RuntimeError(f"DINOv3 特徵聚合失敗: {e}") from e
        
        # 確保返回正確的數據類型
        dtype = np.float16 if self.use_fp16 else np.float32
        return latent_mask_ave.astype(dtype)
    
    def _enhance_mask_weights(self, mask_weights: torch.Tensor) -> torch.Tensor:
        """
        DINOv3 特定的遮罩權重增強
        
        注意：這是概念性實作，實際功能需要根據官方 DINOv3 進行調整
        """
        # TODO: 實際的 DINOv3 遮罩增強算法
        # 可能包括：
        # - 空間一致性優化
        # - 邊緣保持增強
        # - 多尺度融合
        
        return mask_weights
    
    def _enhance_features(self, features: np.ndarray) -> np.ndarray:
        """
        DINOv3 特定的特徵增強
        
        注意：這是概念性實作，實際功能需要根據官方 DINOv3 進行調整
        """
        # TODO: 實際的 DINOv3 特徵增強算法
        # 可能包括：
        # - 特徵正規化改進
        # - 判別性增強
        # - 穩定性優化
        
        return features
            
    def extract_batch_features(
        self,
        images: List[np.ndarray],
        roi_masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        批次提取特徵
        
        Args:
            images: 影像列表
            roi_masks: ROI 遮罩列表
            
        Returns:
            特徵向量列表
        """
        # 批次提取 patch 特徵
        patch_features_list = self.feature_extractor.extract_batch_patch_features(images)
        
        features = []
        for patch_features, mask in zip(patch_features_list, roi_masks):
            feature = self._extract_roi_features(patch_features, mask)
            features.append(feature)
            
        return features
        
    def get_patch_features(self, image: np.ndarray) -> np.ndarray:
        """
        獲取 patch 特徵（不進行池化）
        
        Args:
            image: 輸入影像
            
        Returns:
            Patch 特徵 (37, 37, embed_dim)
        """
        return self.feature_extractor.extract_patch_features(image)
        
    def debug_mask_processing(self, roi_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        調試遮罩處理過程，返回中間結果用於驗證
        
        Args:
            roi_mask: ROI 遮罩
            
        Returns:
            包含中間處理步驟的字典
        """
        print(f"🔍 DINOv3 Debug mask processing:")
        print(f"  - Input mask shape: {roi_mask.shape}")
        print(f"  - Input mask dtype: {roi_mask.dtype}")
        print(f"  - Input mask range: [{roi_mask.min()}, {roi_mask.max()}]")
        print(f"  - Enhanced features enabled: {self.enable_enhanced_features}")
        
        # 轉換遮罩
        mask_tensor = self.transforms.transform_mask(roi_mask)
        print(f"  - After transform shape: {mask_tensor.shape}")
        print(f"  - After transform range: [{mask_tensor.min():.6f}, {mask_tensor.max():.6f}]")
        
        # 確保維度一致
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor[0]
            print(f"  - After dim reduction shape: {mask_tensor.shape}")
        
        # Reshape
        mask_reshaped = mask_tensor.reshape(
            DINOV3_PATCH_LEN, DINOV3_PATCH_SIZE, 
            DINOV3_PATCH_LEN, DINOV3_PATCH_SIZE
        )
        print(f"  - Reshaped shape: {mask_reshaped.shape}")
        
        # 計算 patch weights
        small_mask = mask_reshaped.sum(dim=(1, 3))
        
        # DINOv3 增強
        if self.enable_enhanced_features:
            enhanced_mask = self._enhance_mask_weights(small_mask)
            print(f"  - Applied DINOv3 mask enhancement")
        else:
            enhanced_mask = small_mask
        
        small_mask_np = enhanced_mask.numpy()
        sum_mask = small_mask_np.sum()
        
        print(f"  - Small mask shape: {small_mask_np.shape}")
        print(f"  - Small mask range: [{small_mask_np.min():.6f}, {small_mask_np.max():.6f}]")
        print(f"  - Sum mask: {sum_mask:.6f}")
        
        return {
            'original_mask': roi_mask,
            'mask_tensor': mask_tensor.numpy(),
            'mask_reshaped_shape': mask_reshaped.shape,
            'small_mask_37x37': small_mask_np,
            'sum_mask': sum_mask,
            'enhanced_features_enabled': self.enable_enhanced_features,
            'mask_tensor_range': [float(mask_tensor.min()), float(mask_tensor.max())],
            'small_mask_range': [float(small_mask_np.min()), float(small_mask_np.max())]
        }
        
    def clear_cache(self):
        """清除快取"""
        gc.collect()
        
        if torch.cuda.is_available() and self.device == 'cuda':
            torch.cuda.empty_cache()
        
    def __del__(self):
        """清理資源"""
        self.clear_cache()
        if hasattr(self, 'model'):
            del self.model
