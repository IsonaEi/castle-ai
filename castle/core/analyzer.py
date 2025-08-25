"""
CASTLE Analyzer - 主要分析入口點
處理從影片到行為分類的完整流程
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import random

from ..models import ModelManager
from ..utils import VideoIO, ROIManager, ConfigLoader
# from .feature_extractor import FeatureExtractor
from .tracker import MultiObjectTracker

logger = logging.getLogger(__name__)

class Analyzer:
    """CASTLE 主分析器類別"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化分析器
        
        Args:
            config_path: 配置檔路徑
        """
        self.config = ConfigLoader.load(config_path) if config_path else ConfigLoader.default()
        self.model_manager = ModelManager(self.config.models)
        self.video_io = VideoIO()
        self.roi_manager = ROIManager()
        self.videos = []

        
    def add_videos(self, video_paths: List[str]):
        """
        添加要分析的影片
        
        Args:
            video_paths: 影片路徑列表
        """
        for path in video_paths:
            video_obj = self.video_io.load_video(path)
            self.videos.append({
                'path': Path(path),
                'video': video_obj,
                'fps': video_obj.fps,
                'total_frames': video_obj.frame_count
            })
        logger.info(f"Added {len(video_paths)} videos for analysis")
    
    def get_image(self, video_idx: int = None, frame_idx: int = None) -> np.ndarray:
        """
        獲取指定影片的幀
        
        Args:
            video_idx: 影片索引
            frame_idx: 幀索引，None 則返回當前 current_image 中的幀索引或 0
            
        Returns:
            影像陣列
        """
        
        if video_idx is None:
            if not self.videos:
                raise ValueError("沒有可用的影片")
            video_idx = random.randint(0, len(self.videos) - 1)
            total_frames = self.videos[video_idx]['total_frames']
            if total_frames <= 0:
                raise ValueError("影片沒有幀數")
            frame_idx = random.randint(0, total_frames - 1)
        elif frame_idx is None:
            frame_idx = self.current_image.get('frame_idx', 0)
            
        return self.video_io.get_frame(self.videos[video_idx]['video'], frame_idx)
    
    def predict_ROI(self, image: np.ndarray, click_list: List[List], update_current: bool = True) -> Dict:
        """
        使用 SAM 預測 ROI
        
        Args:
            image: 輸入影像
            click_list: 點擊列表 [[x, y, type], ...]
            update_current: 是否自動更新到 current_image
            
        Returns:
            ROI 字典
        """
        if self.sam is None:
            self.sam = self.model_manager.get_model('sam')
            
        # 轉換點擊列表為 SAM 格式
        points = np.array([[click[0], click[1]] for click in click_list])
        labels = np.array([1 if click[2] == 'plus' else 0 for click in click_list])
        
        # 預測 mask
        masks, scores, logits = self.sam.predict(
            image=image,
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        
        # 選擇最佳 mask
        best_idx = np.argmax(scores)
        roi = {
            'mask': masks[best_idx],
            'score': scores[best_idx],
            'bbox': self._mask_to_bbox(masks[best_idx])
        }
        
        # 如果需要，自動更新到 current_image
        if update_current:
            self.set_current_roi(roi)
            # 如果 current_image 的 image 還沒設定，也一併設定
            if self.current_image['image'] is None:
                self.current_image['image'] = image
        
        return roi
    
    
    
    def add_roi(self, dir_path: str = None, roi_dict: Dict = None):
        """
        添加 ROI 提示
        
        Args:
            dir_path: ROI 檔案目錄
            roi_dict: ROI 字典
        """
        if dir_path:
            self.roi_manager.load_from_directory(dir_path)
        elif roi_dict:
            self.roi_manager.add_roi(roi_dict)
            
    # def track_video_object(self, batch_size: int = 8):
    #     """
    #     執行影片物件追蹤
        
    #     Args:
    #         batch_size: 批次處理大小
    #     """
    #     if self.deaot is None:
    #         self.deaot = self.model_manager.get_model('deaot')
            
    #     tracker = MultiObjectTracker(self.deaot)
        
    #     for video_info in self.videos:
    #         logger.info(f"Tracking {video_info['path'].name}")
            
    #         # 獲取初始 ROI
    #         initial_rois = self.roi_manager.get_rois_for_video(video_info['path'].name)
            
    #         # 執行追蹤
    #         tracks = tracker.track(
    #             video=video_info['video'],
    #             initial_masks=initial_rois,
    #             batch_size=batch_size
    #         )
            
    #         video_info['tracks'] = tracks
            
    # def generate_focused_visual_latent(
    #     self,
    #     neutralize_orientation: bool = True,
    #     config: str = None
    # ) -> Dict[str, np.ndarray]:
    #     """
    #     生成聚焦視覺潛在特徵
        
    #     Args:
    #         neutralize_orientation: 是否中和方向
    #         config: 預處理配置檔
            
    #     Returns:
    #         特徵字典
    #     """
    #     if self.dinov2 is None:
    #         self.dinov2 = self.model_manager.get_model('dinov2')
            
    #     extractor = FeatureExtractor(self.dinov2)
    #     features = {}
        
    #     for video_info in self.videos:
    #         logger.info(f"Extracting features from {video_info['path'].name}")
            
    #         video_features = extractor.extract(
    #             video=video_info['video'],
    #             tracks=video_info['tracks'],
    #             neutralize_orientation=neutralize_orientation,
    #             config=ConfigLoader.load(config) if config else None
    #         )
            
    #         features[video_info['path'].name] = video_features
            
    #     return features
    
    def save_roi_prompt(self, dir_path: str):
        """保存 ROI 提示"""
        # 檢查 current_image 字典是否設定完整
        if not hasattr(self, 'current_image') or self.current_image is None:
            logger.error("current_image 尚未設定或為 None，無法保存。")
            return
        
        if self.current_image.get('image') is None:
            logger.error("current_image['image'] 尚未設定或為 None，無法保存。")
            return
            
        if self.current_image.get('roi') is None:
            logger.error("current_image['roi'] 尚未設定或為 None，無法保存。")
            return

        # 檢查 dir_path 是否存在，不存在則建立
        dir_path = Path(dir_path)
        dir_path.makedirs(parents=True, exist_ok=True)

        # 建立 images 和 rois 子資料夾
        images_dir = dir_path / "images"
        rois_dir = dir_path / "rois"
        images_dir.makedirs(exist_ok=True)
        rois_dir.makedirs(exist_ok=True)

        # 產生檔名（根據影片和幀資訊命名）
        video_base_name = self.current_image.get('video_base_name', 'unknown')
        frame_idx = self.current_image.get('frame_idx', 0)
        
        # 如果沒有影片名稱，使用 timestamp
        if video_base_name == 'unknown' or video_base_name is None:
            import time
            timestamp = int(time.time())
            filename_base = f"{timestamp}"
        else:
            filename_base = f"{video_base_name}_frame_{frame_idx}"
        
        image_filename = images_dir / f"{filename_base}.png"
        roi_filename = rois_dir / f"{filename_base}.npy"

        # 儲存 current_image['image'] 為 png
        from PIL import Image
        img = self.current_image['image']
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.concatenate([img]*3, axis=2)
            img_pil = Image.fromarray(img)
        else:
            img_pil = img  # 假設已經是 PIL.Image
        img_pil.save(image_filename)

        # 儲存 current_image['roi'] 為 npy
        np.save(roi_filename, self.current_image['roi'])

        logger.info(f"已將 current_image['image'] 儲存至 {image_filename}，roi 儲存至 {roi_filename}")