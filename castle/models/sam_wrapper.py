"""
SAM (Segment Anything Model) 封裝
提供互動式分割和 ROI 定義功能
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

from .sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

logger = logging.getLogger(__name__)

class ModelSize(Enum):
    """SAM 模型大小"""
    VIT_H = "vit_h"  # Huge model (~2.4GB)
    VIT_L = "vit_l"  # Large model (~1.2GB)
    VIT_B = "vit_b"  # Base model (~375MB)

@dataclass
class SegmentationResult:
    """分割結果資料類別"""
    mask: np.ndarray  # 二值遮罩
    score: float  # 置信度分數
    area: int  # 遮罩面積
    stability_score: float  # 穩定性分數
    
class SAMWrapper:
    """SAM 模型封裝器
    
    使用範例:
        # 基本初始化
        sam = SAMWrapper(model_size='vit_b', device='cuda')
        
        # 載入影像
        import cv2
        image = cv2.imread('image.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam.set_image(image)
        
        # 使用點提示進行分割
        import numpy as np
        point_coords = np.array([[100, 100], [200, 200]])  # [x, y] 座標
        point_labels = np.array([1, 0])  # 1=前景點, 0=背景點
        mask = sam.predict_with_points(point_coords, point_labels)
        
        # 使用現有遮罩進行精化
        refined_mask = sam.refine_mask(mask)
        
        # 或者結合點提示來精化遮罩
        refined_mask_with_points = sam.refine_mask(
            mask, 
            point_coords=np.array([[150, 150]]),
            point_labels=np.array([1])
        )
        
        # 清除快取
        sam.clear_cache()
    """
    
    def __init__(
        self,
        model_size: Union[str, ModelSize] = ModelSize.VIT_B,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.9
    ):
        """
        初始化 SAM 模型
        
        Args:
            model_size: 模型大小
            checkpoint_path: 模型權重路徑（如果為 None，自動下載）
            device: 計算設備
            points_per_side: 自動分割的採樣點密度
            pred_iou_thresh: IoU 閾值
            stability_score_thresh: 穩定性分數閾值
        """
        self.model_size = ModelSize(model_size) if isinstance(model_size, str) else model_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        
        # 載入模型
        self.sam = self._load_model(checkpoint_path)
        
        # 延遲創建 automatic_generator，避免 cv2 導入問題
        self.automatic_generator = None
        self.predictor = None
        self._predictor_initialized = False
        
        
        # 快取當前影像
        self._current_image = None
        self._image_embedding = None
        
        logger.info(f"Initialized SAM {model_size} on {self.device}")
    
    def _ensure_predictor(self):
        """確保 predictor 已初始化"""
        if not self._predictor_initialized:
            try:
                self.predictor = self._create_predictor()
                self._predictor_initialized = True
                logger.debug("Predictor 初始化成功")
            except Exception as e:
                logger.error(f"Predictor 初始化失敗: {e}")
                raise
        
    def _load_model(self, checkpoint_path: Optional[str] = None):
        """載入 SAM 模型"""
        try:
            # 如果沒有提供路徑，使用預設路徑或下載
            if checkpoint_path is None:
                checkpoint_path = self._get_default_checkpoint_path()
                
            # 載入模型
            sam = sam_model_registry[self.model_size.value](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            sam.eval()
            
            return sam
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise
            
    def _get_default_checkpoint_path(self) -> str:
        """獲取預設模型路徑"""
        # 檢查本地快取
        cache_dir = Path.home() / '.cache' / 'castle' / 'models'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_files = {
            ModelSize.VIT_H: 'sam_vit_h_4b8939.pth',
            ModelSize.VIT_L: 'sam_vit_l_0b3195.pth',
            ModelSize.VIT_B: 'sam_vit_b_01ec64.pth'
        }
        
        checkpoint_path = cache_dir / model_files[self.model_size]
        
        if not checkpoint_path.exists():
            # 下載模型
            self._download_checkpoint(checkpoint_path)
            
        return str(checkpoint_path)
    
    def _download_checkpoint(self, checkpoint_path: Path):
        """下載模型權重（帶有進度條）"""
        import urllib.request
        import sys

        urls = {
            ModelSize.VIT_H: 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            ModelSize.VIT_L: 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            ModelSize.VIT_B: 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
        }

        url = urls[self.model_size]
        logger.info(f"正在下載 SAM 權重檔案：{url}")

        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
            bar_len = 30
            filled_len = int(bar_len * percent // 100)
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            sys.stdout.write(f"\r下載進度: |{bar}| {percent}%")
            sys.stdout.flush()
            if downloaded >= total_size:
                sys.stdout.write('\n')

        urllib.request.urlretrieve(url, checkpoint_path, reporthook=show_progress)
        logger.info(f"已下載權重檔案至 {checkpoint_path}")
        
    def _create_predictor(self):
        """創建預測器"""
        # 直接從 SAM 模型創建預測器，避免依賴 automatic_generator
        from .sam.segment_anything import SamPredictor
        return SamPredictor(self.sam)
    
    def _create_automatic_generator(self):
        """創建自動遮罩生成器"""    
        return SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )
        
    def set_image(self, image: np.ndarray, cache_embedding: bool = True):
        """
        設置當前處理的影像
        
        Args:
            image: 輸入影像 (H, W, 3) RGB 格式
            cache_embedding: 是否快取影像嵌入
        """
        # 確保 predictor 已初始化
        self._ensure_predictor()
        
        # 確保影像格式正確
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        self._current_image = image
        
        # 計算影像嵌入
        self.predictor.set_image(image)
        
        if cache_embedding:
            # 快取嵌入以加速後續預測
            self._image_embedding = self.predictor.features
            
        logger.debug(f"Set image with shape {image.shape}")
        
    def predict_with_points(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool = True
    ) -> np.ndarray:
        """
        使用點提示進行分割
        
        Args:
            point_coords: 點座標 [[x, y], ...]
            point_labels: 點標籤 [1, 0, ...] (1=前景, 0=背景)
            multimask_output: 是否輸出多個遮罩
            
        Returns:
            分割遮罩 (boolean array)
        """
        if self._current_image is None:
            raise ValueError("Please set image first using set_image()")
        
        # 確保 predictor 已初始化
        self._ensure_predictor()
        
        logger.debug(f"預測參數: point_coords={point_coords}, point_labels={point_labels}")
        logger.debug(f"影像形狀: {self._current_image.shape}")
            
        # 第一次預測遮罩
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )
        
        logger.debug(f"第一次預測 - masks 形狀: {masks.shape}, scores: {scores}")
        
        # 選擇最佳遮罩
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        logger.debug(f"最佳遮罩 - index: {best_idx}, score: {best_score}, 分割面積: {np.sum(best_mask)}")
        return best_mask

      
    
    def refine_mask(
        self,
        input_mask: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        multimask_output: bool = False
    ) -> np.ndarray:
        """
        使用現有遮罩進行精化分割
        
        Args:
            input_mask: 輸入遮罩 (H, W) boolean array
            point_coords: 可選的點座標 [[x, y], ...] 用於進一步指導
            point_labels: 可選的點標籤 [1, 0, ...] (1=前景, 0=背景)
            multimask_output: 是否輸出多個遮罩候選
            
        Returns:
            精化後的分割遮罩 (boolean array)
        """
        if self._current_image is None:
            raise ValueError("Please set image first using set_image()")
            
        # 確保 predictor 已初始化
        self._ensure_predictor()
        
        # 確保輸入遮罩格式正確
        if input_mask.dtype != bool:
            input_mask = input_mask.astype(bool)
            
        logger.debug(f"精化遮罩 - 輸入遮罩面積: {np.sum(input_mask)}")
        
        # 將 boolean mask 轉換為 logit 格式 (需要添加 batch 維度)
        # SAM 期望的 mask_input 是 logit 格式，我們將 boolean mask 轉換
        mask_logit = input_mask.astype(np.float32)
        mask_logit = mask_logit[None, :, :]  # 添加 batch 維度 (1, H, W)
        
        try:
            # 使用遮罩和可選的點提示進行預測
            refined_masks, refined_scores, refined_logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_logit,
                multimask_output=multimask_output
            )
            
            if multimask_output and len(refined_masks) > 1:
                # 如果有多個候選遮罩，選擇最佳的
                best_idx = np.argmax(refined_scores)
                refined_mask = refined_masks[best_idx]
                refined_score = refined_scores[best_idx]
                logger.debug(f"多遮罩模式 - 選擇最佳遮罩 index: {best_idx}, score: {refined_score}")
            else:
                # 單一遮罩輸出
                refined_mask = refined_masks[0]
                refined_score = refined_scores[0]
                
            logger.debug(f"遮罩精化完成 - score: {refined_score}, 精化後面積: {np.sum(refined_mask)}")
            
            return refined_mask
            
        except Exception as e:
            logger.error(f"遮罩精化失敗: {e}")
            logger.warning("返回原始輸入遮罩")
            return input_mask
    
    def clear_cache(self):
        """清除快取"""
        self._current_image = None
        self._image_embedding = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        