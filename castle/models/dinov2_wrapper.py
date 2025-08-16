"""
To be tested.
DINOv2 模型封裝
處理視覺特徵提取和方向中和
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class DINOv2Wrapper:
    """DINOv2 視覺基礎模型封裝"""
    
    def __init__(self, model_size: str = 'base', device: str = 'cuda'):
        """
        初始化 DINOv2 模型
        
        Args:
            model_size: 模型大小 ('small', 'base', 'large', 'giant')
            device: 計算設備
        """
        self.model_size = model_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 載入模型
        self.model = self._load_model()
        self.model.eval()
        
        # 特徵維度
        self.feature_dim = self._get_feature_dim()
        
    def _load_model(self):
        """載入預訓練 DINOv2 模型"""
        model_name = f'dinov2_vit{self._get_model_suffix()}14_reg'
        
        try:
            model = torch.hub.load('facebookresearch/dinov2', model_name)
            model = model.to(self.device)
            logger.info(f"Loaded DINOv2 model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model: {e}")
            raise
            
    def _get_model_suffix(self) -> str:
        """獲取模型後綴"""
        suffixes = {
            'small': 's',
            'base': 'b',
            'large': 'l',
            'giant': 'g'
        }
        return suffixes.get(self.model_size, 'b')
    
    def _get_feature_dim(self) -> int:
        """獲取特徵維度"""
        dims = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 1536
        }
        return dims.get(self.model_size, 768)
    
    def extract_features(
        self,
        image: np.ndarray,
        roi_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        提取影像特徵
        
        Args:
            image: 輸入影像 (H, W, 3)
            roi_mask: ROI 遮罩
            
        Returns:
            特徵向量
        """
        # 預處理影像
        image_tensor = self._preprocess_image(image)
        
        with torch.no_grad():
            # 提取 patch 特徵
            features = self.model.forward_features(image_tensor)
            patch_features = features['x_norm_patchtokens']
            
            if roi_mask is not None:
                # 聚焦特徵提取
                focused_features = self._extract_focused_features(
                    patch_features,
                    roi_mask
                )
                return focused_features.cpu().numpy()
            else:
                # 全局平均池化
                global_features = patch_features.mean(dim=1)
                return global_features.cpu().numpy()
                
    def extract_features_with_orientation_neutralization(
        self,
        image: np.ndarray,
        roi_mask: np.ndarray,
        num_rotations: int = 24
    ) -> np.ndarray:
        """
        提取方向中和的特徵
        
        Args:
            image: 輸入影像
            roi_mask: ROI 遮罩
            num_rotations: 旋轉次數
            
        Returns:
            方向中和的特徵
        """
        angle_step = 360.0 / num_rotations
        all_features = []
        
        for i in range(num_rotations):
            angle = i * angle_step
            
            # 旋轉影像和遮罩
            rotated_image = self._rotate_image(image, angle)
            rotated_mask = self._rotate_mask(roi_mask, angle)
            
            # 提取特徵
            features = self.extract_features(rotated_image, rotated_mask)
            all_features.append(features)
            
        # 平均所有旋轉的特徵
        averaged_features = np.mean(all_features, axis=0)
        
        return averaged_features
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """預處理影像為模型輸入格式"""
        # 調整大小為 518x518 (DINOv2 標準輸入)
        import cv2
        resized = cv2.resize(image, (518, 518))
        
        # 正規化
        normalized = (resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # 轉換為 tensor
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _extract_focused_features(
        self,
        patch_features: torch.Tensor,
        roi_mask: np.ndarray
    ) -> torch.Tensor:
        """
        提取聚焦的 ROI 特徵
        
        Args:
            patch_features: Patch 特徵 (1, num_patches, feature_dim)
            roi_mask: ROI 遮罩
            
        Returns:
            聚焦特徵
        """
        # 將遮罩調整為 patch 大小 (37x37 for 518x518 input with patch_size=14)
        patch_size = 14
        h_patches = w_patches = 37
        
        import cv2
        mask_resized = cv2.resize(
            roi_mask.astype(np.float32),
            (w_patches, h_patches)
        )
        
        # 將遮罩展平並正規化
        mask_flat = mask_resized.flatten()
        mask_flat = mask_flat / (mask_flat.sum() + 1e-6)
        
        # 轉換為 tensor
        mask_tensor = torch.from_numpy(mask_flat).float().to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(-1)
        
        # 加權平均
        weighted_features = (patch_features * mask_tensor).sum(dim=1)
        
        return weighted_features
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """旋轉影像"""
        import cv2
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 獲取旋轉矩陣
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 執行旋轉
        rotated = cv2.warpAffine(image, M, (w, h))
        
        return rotated
    
    def _rotate_mask(self, mask: np.ndarray, angle: float) -> np.ndarray:
        """旋轉遮罩"""
        import cv2
        
        # 將布林遮罩轉換為 uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 旋轉
        rotated = self._rotate_image(mask_uint8, angle)
        
        # 轉換回布林
        return rotated > 127
        
    def batch_extract_features(
        self,
        images: List[np.ndarray],
        roi_masks: Optional[List[np.ndarray]] = None,
        batch_size: int = 16
    ) -> np.ndarray:
        """
        批次提取特徵
        
        Args:
            images: 影像列表
            roi_masks: ROI 遮罩列表
            batch_size: 批次大小
            
        Returns:
            特徵陣列
        """
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_masks = roi_masks[i:i+batch_size] if roi_masks else [None] * len(batch_images)
            
            batch_features = []
            for img, mask in zip(batch_images, batch_masks):
                features = self.extract_features(img, mask)
                batch_features.append(features)
                
            all_features.extend(batch_features)
            
        return np.vstack(all_features)