"""
SAM2 (Segment Anything Model 2) 視頻物件分割封裝
提供視頻物件分割和多目標追蹤功能
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import platform
from enum import Enum
import warnings

# 檢測平台並設定預設設備
OS_SYS = platform.uname().system
if OS_SYS == 'Darwin':
    DEFAULT_DEVICE = 'mps'
elif torch.cuda.is_available():
    DEFAULT_DEVICE = 'cuda'
else:
    DEFAULT_DEVICE = 'cpu'

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)

class SAM2ModelType(Enum):
    """SAM2 模型類型"""
    TINY = "sam2_hiera_tiny"           # Tiny model (~38MB)
    SMALL = "sam2_hiera_small"         # Small model (~184MB) 
    BASE_PLUS = "sam2_hiera_base_plus" # Base+ model (~274MB)
    LARGE = "sam2_hiera_large"         # Large model (~900MB)

# 導入 SAM2 相關模組
def _setup_sam2_imports():
    """設置 SAM2 模組的導入環境"""
    import sys
    import importlib.util
    from pathlib import Path
    
    # 獲取 castle/models 目錄的絕對路徑
    castle_models_dir = Path(__file__).parent
    
    # 臨時添加到 sys.path 以允許 'sam2' 導入
    if str(castle_models_dir) not in sys.path:
        sys.path.insert(0, str(castle_models_dir))
    
    try:
        # 現在嘗試導入 SAM2 模組
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return True, build_sam2_video_predictor, SAM2ImagePredictor
    except ImportError as e:
        logger.warning(f"SAM2 modules not found: {e}")
        # 創建假的類別以避免導入錯誤
        class DummySAM2VideoPredictor:
            pass
        build_sam2_video_predictor = lambda *args, **kwargs: DummySAM2VideoPredictor()
        SAM2ImagePredictor = None
        return False, build_sam2_video_predictor, SAM2ImagePredictor
    finally:
        # 清理 sys.path（可選，避免污染全局環境）
        if str(castle_models_dir) in sys.path:
            sys.path.remove(str(castle_models_dir))

# 執行導入設置
SAM2_AVAILABLE, build_sam2_video_predictor, SAM2ImagePredictor = _setup_sam2_imports()

@dataclass
class TrackingResult:
    """追蹤結果資料類別"""
    frame_idx: int
    object_id: int
    mask: np.ndarray
    confidence: float
    area: int
    center: Tuple[float, float]

@dataclass
class ObjectTrack:
    """物件追蹤軌跡"""
    object_id: int
    start_frame: int
    end_frame: int
    masks: Dict[int, np.ndarray]
    confidences: Dict[int, float]

class SAM2Wrapper:
    """SAM2 (Segment Anything Model 2) 視頻物件分割模型封裝器
    
    使用範例:
        # 基本初始化
        tracker = SAM2Wrapper(model_type='sam2_hiera_base_plus', device='cuda')
        
        # 載入視頻序列
        from castle.utils import video_io
        frames = video_io.read_video('video.mpv')  # 自動轉換為 RGB 格式
        
        # 1. 初始化視頻追蹤器
        tracker.init_video_tracker(frames)
        
        # 2. 在第一幀添加物件（使用點擊或遮罩）
        # 方法 1: 使用點擊點
        obj_id = tracker.add_object_with_points(
            frame_idx=0,
            points=[[x1, y1], [x2, y2]],  # 前景點
            labels=[1, 1]  # 1 表示前景，0 表示背景
        )
        
        # 方法 2: 使用遮罩
        obj_id = tracker.add_object_with_mask(frame_idx=0, mask=initial_mask)
        
        # 3. 執行視頻追蹤
        tracks = tracker.track_video()
        
        # 或逐幀處理
        for frame_idx in range(len(frames)):
            frame_results = tracker.get_frame_results(frame_idx)
        
        # 清除記憶
        tracker.clear_memory()
    """
    
    def __init__(
        self,
        model_type: Union[str, SAM2ModelType] = SAM2ModelType.BASE_PLUS,
        checkpoint_path: Optional[str] = None,
        config_name: Optional[str] = None,
        device: Optional[str] = None,
        apply_postprocessing: bool = True,
        use_amp: bool = True
    ):
        """
        初始化 SAM2 視頻物件分割模型
        
        Args:
            model_type: 模型類型，可以是字符串或 SAM2ModelType 枚舉
            checkpoint_path: 模型權重路徑（如果為 None，自動下載）
            config_name: 配置檔案名稱（如果為 None，使用預設配置）
            device: 計算設備 ('cuda', 'cpu', 'mps')，如果為 None 自動選擇
            apply_postprocessing: 是否應用後處理
            use_amp: 是否使用自動混合精度
            
        Raises:
            ImportError: 當 SAM2 模組未安裝時
            ValueError: 當模型類型不支持時
        """
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 modules not installed. Please install from: https://github.com/facebookresearch/segment-anything-2")
            
        self.model_type = SAM2ModelType(model_type) if isinstance(model_type, str) else model_type
        self.device = device if device else DEFAULT_DEVICE
        self.apply_postprocessing = apply_postprocessing
        self.use_amp = use_amp
        
        # 載入配置和模型
        self._load_model(checkpoint_path, config_name)
        
        # 初始化追蹤狀態
        self._reset_tracking_state()
        
        logger.info(f"Initialized SAM2 {self.model_type.value} on {self.device}")
        
    def _load_model(self, checkpoint_path: Optional[str] = None, config_name: Optional[str] = None):
        """載入 SAM2 模型"""
        # 下載或使用提供的權重
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint()
            
        if config_name is None:
            config_name = self._get_default_config_name()
            
        # 設置 SAM2 配置路徑
        self._setup_sam2_config_path()
        
        # 在導入上下文中建立視頻預測器
        import sys
        from pathlib import Path
        
        castle_models_dir = Path(__file__).parent
        path_added = False
        
        if str(castle_models_dir) not in sys.path:
            sys.path.insert(0, str(castle_models_dir))
            path_added = True
            
        try:
            # 建立視頻預測器
            self.predictor = build_sam2_video_predictor(
                config_file=config_name,
                ckpt_path=checkpoint_path,
                device=self.device
            )
            logger.debug(f"SAM2 model loaded from {checkpoint_path}")
        finally:
            # 清理 sys.path
            if path_added and str(castle_models_dir) in sys.path:
                sys.path.remove(str(castle_models_dir))
        
    def _download_checkpoint(self) -> str:
        """
        下載 SAM2 模型權重檔案
        
        Returns:
            模型權重檔案的本地路徑
            
        Raises:
            ValueError: 當模型類型不支持時
            RuntimeError: 當下載失敗時
        """
        cache_dir = Path.home() / '.cache' / 'castle' / 'models' / 'sam2'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # SAM2 模型下載信息
        model_info = {
            SAM2ModelType.TINY: {
                'filename': 'sam2_hiera_tiny.pt',
                'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt'
            },
            SAM2ModelType.SMALL: {
                'filename': 'sam2_hiera_small.pt', 
                'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt'
            },
            SAM2ModelType.BASE_PLUS: {
                'filename': 'sam2_hiera_base_plus.pt',
                'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt'
            },
            SAM2ModelType.LARGE: {
                'filename': 'sam2_hiera_large.pt',
                'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
            }
        }
        
        if self.model_type not in model_info:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        info = model_info[self.model_type]
        ckpt_path = cache_dir / info['filename']
        
        if not ckpt_path.exists():
            try:
                logger.info(f"正在下載 {self.model_type.value} 模型權重至 {ckpt_path}")
                self._download_file(info['url'], str(ckpt_path))
                logger.info(f"模型權重下載完成")
            except Exception as e:
                logger.error(f"模型權重下載失敗: {e}")
                raise RuntimeError(f"Failed to download checkpoint for {self.model_type.value}: {e}")
        else:
            logger.debug(f"使用現有的模型權重: {ckpt_path}")
            
        return str(ckpt_path)
        
    def _get_default_config_name(self) -> str:
        """獲取預設配置檔案名稱"""
        config_map = {
            SAM2ModelType.TINY: "sam2_hiera_t",
            SAM2ModelType.SMALL: "sam2_hiera_s", 
            SAM2ModelType.BASE_PLUS: "sam2_hiera_b+",
            SAM2ModelType.LARGE: "sam2_hiera_l"
        }
        
        return config_map[self.model_type]
        
    def _setup_sam2_config_path(self):
        """設置 SAM2 配置搜索路徑"""
        import sys
        from pathlib import Path
        
        # 獲取 castle/models 目錄的絕對路徑
        castle_models_dir = Path(__file__).parent
        
        # 臨時添加到 sys.path
        path_added = False
        if str(castle_models_dir) not in sys.path:
            sys.path.insert(0, str(castle_models_dir))
            path_added = True
            
        try:
            # 嘗試導入 sam2 來設置 hydra 配置路徑
            import sam2
            return True
        except ImportError:
            logger.warning("無法設置 SAM2 配置路徑")
            return False
        finally:
            # 清理 sys.path
            if path_added and str(castle_models_dir) in sys.path:
                sys.path.remove(str(castle_models_dir))
        
    def _download_file(self, url: str, destination: str):
        """下載檔案"""
        import urllib.request
        import shutil
        
        # 確保目標目錄存在
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # 下載檔案
        with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        
    def _reset_tracking_state(self):
        """重置追蹤狀態"""
        self.current_frame_idx = 0
        self.object_tracks = {}
        self.active_objects = set()
        self.video_frames = None
        self.video_initialized = False
        self._next_object_id = 1
        self.inference_state = None
        
        # 注意：不在這裡調用 predictor.reset_state()，因為它需要 inference_state 參數
            
    def init_video_tracker(self, frames: List[np.ndarray]):
        """
        初始化視頻追蹤器
        
        Args:
            frames: 影像序列列表，每個元素為 (H, W, 3) RGB 格式
        """
        from collections import OrderedDict
        
        self.video_frames = frames
        
        # 直接創建推理狀態而不使用 load_video_frames
        inference_state = {}
        
        # 預處理圖像
        processed_frames = []
        image_size = 1024  # SAM2 的預期圖像大小
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].to(self.device)
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].to(self.device)
        
        for frame in frames:
            # 確保幀是正確的格式 (H, W, 3) uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # 轉換為 torch tensor 並調整維度順序為 (C, H, W)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # 調整大小到 SAM2 期望的尺寸
            frame_tensor = F.interpolate(
                frame_tensor.unsqueeze(0), 
                size=(image_size, image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # 應用 ImageNet 歸一化
            frame_tensor = (frame_tensor - img_mean) / img_std
            processed_frames.append(frame_tensor)
        
        # 設置推理狀態
        inference_state["images"] = processed_frames
        inference_state["num_frames"] = len(frames)
        inference_state["offload_video_to_cpu"] = True
        inference_state["offload_state_to_cpu"] = False
        inference_state["video_height"] = frames[0].shape[0]
        inference_state["video_width"] = frames[0].shape[1]
        inference_state["device"] = self.device
        inference_state["storage_device"] = self.device
        
        # 初始化其他必要的狀態
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {}
        }
        inference_state["tracking_has_started"] = False
        inference_state["output_dict"] = {}
        
        self.inference_state = inference_state
        self.video_initialized = True
        logger.debug(f"Video tracker initialized with {len(frames)} frames")
        
    def add_object_with_points(
        self, 
        frame_idx: int,
        points: List[List[float]],
        labels: List[int],
        obj_id: Optional[int] = None
    ) -> int:
        """
        使用點擊點添加物件
        
        Args:
            frame_idx: 幀索引
            points: 點擊點座標列表 [[x1, y1], [x2, y2], ...]
            labels: 點擊標籤列表 [1, 1, 0, ...] (1=前景, 0=背景)
            obj_id: 物件ID（如果為None則自動分配）
            
        Returns:
            物件ID
        """
        if not self.video_initialized:
            raise RuntimeError("必須先調用 init_video_tracker() 初始化視頻追蹤器")
            
        if obj_id is None:
            obj_id = self._next_object_id
            self._next_object_id += 1
            
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)
        
        # 添加物件到 SAM2
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points_np,
            labels=labels_np
        )
        
        self.active_objects.add(obj_id)
        logger.debug(f"Added object {obj_id} with {len(points)} points at frame {frame_idx}")
        
        return obj_id
        
    def add_object_with_mask(
        self,
        frame_idx: int, 
        mask: np.ndarray,
        obj_id: Optional[int] = None
    ) -> int:
        """
        使用遮罩添加物件
        
        Args:
            frame_idx: 幀索引
            mask: 二值遮罩 (H, W)
            obj_id: 物件ID（如果為None則自動分配）
            
        Returns:
            物件ID
        """
        if not self.video_initialized:
            raise RuntimeError("必須先調用 init_video_tracker() 初始化視頻追蹤器")
            
        if obj_id is None:
            obj_id = self._next_object_id
            self._next_object_id += 1
            
        # 將遮罩轉換為點擊點（使用質心）
        if np.sum(mask) == 0:
            raise ValueError("Mask is empty")
            
        y_coords, x_coords = np.where(mask > 0)
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        points = [[center_x, center_y]]
        labels = [1]
        
        return self.add_object_with_points(frame_idx, points, labels, obj_id)
        
    def track_video(self) -> Dict[int, ObjectTrack]:
        """
        對整個視頻序列執行物件追蹤
        
        Returns:
            完整的物件追蹤軌跡字典 {object_id: ObjectTrack}
        """
        if not self.video_initialized:
            raise RuntimeError("必須先調用 init_video_tracker() 初始化視頻追蹤器")
            
        if not self.active_objects:
            raise RuntimeError("必須先添加至少一個物件才能開始追蹤")
            
        # 執行視頻分割傳播
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
        # 轉換為 ObjectTrack 格式
        tracks = {}
        for obj_id in self.active_objects:
            masks = {}
            confidences = {}
            
            start_frame = None
            end_frame = None
            
            for frame_idx in range(len(self.video_frames)):
                if frame_idx in video_segments and obj_id in video_segments[frame_idx]:
                    mask = video_segments[frame_idx][obj_id].astype(np.uint8)
                    
                    # 將遮罩調整回原始幀的尺寸
                    original_height, original_width = self.video_frames[frame_idx].shape[:2]
                    if mask.shape != (original_height, original_width):
                        # 使用 torch 進行雙線性插值調整尺寸
                        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                        mask_resized = F.interpolate(
                            mask_tensor, 
                            size=(original_height, original_width), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        mask = (mask_resized.squeeze().numpy() > 0.5).astype(np.uint8)
                    
                    if np.sum(mask) > 0:  # 只記錄非空遮罩
                        masks[frame_idx] = mask
                        confidences[frame_idx] = 0.9  # SAM2 沒有直接提供置信度
                        
                        if start_frame is None:
                            start_frame = frame_idx
                        end_frame = frame_idx
                        
            if masks:  # 只為有效追蹤創建軌跡
                tracks[obj_id] = ObjectTrack(
                    object_id=obj_id,
                    start_frame=start_frame or 0,
                    end_frame=end_frame or 0,
                    masks=masks,
                    confidences=confidences
                )
                
        self.object_tracks = tracks
        logger.debug(f"Video tracking completed for {len(tracks)} objects")
        
        return tracks
        
    def get_frame_results(self, frame_idx: int) -> List[TrackingResult]:
        """
        獲取指定幀的追蹤結果
        
        Args:
            frame_idx: 幀索引
            
        Returns:
            追蹤結果列表
        """
        results = []
        
        for obj_id, track in self.object_tracks.items():
            if frame_idx in track.masks:
                mask = track.masks[frame_idx]
                confidence = track.confidences[frame_idx]
                
                # 計算屬性
                area = int(np.sum(mask))
                
                # 計算質心
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0:
                    center_x = float(np.mean(x_indices))
                    center_y = float(np.mean(y_indices))
                else:
                    center_x, center_y = 0.0, 0.0
                    
                result = TrackingResult(
                    frame_idx=frame_idx,
                    object_id=obj_id,
                    mask=mask,
                    confidence=confidence,
                    area=area,
                    center=(center_x, center_y)
                )
                
                results.append(result)
                
        return results
        
    def restart(self):
        """重啟追蹤器"""
        # 如果有 inference_state，先重置它
        if hasattr(self, 'inference_state') and self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
        self._reset_tracking_state()
        
    def clear_memory(self):
        """清除記憶"""
        if hasattr(self, 'predictor') and hasattr(self, 'inference_state') and self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    def __del__(self):
        """析構函數"""
        if hasattr(self, 'predictor'):
            del self.predictor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
