"""
DeAOT (Decoupling Features in Hierarchical Propagation) 封裝
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
from torchvision import transforms
from enum import Enum

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

class ModelType(Enum):
    """DeAOT 模型類型"""
    R50_DEAOTL = "r50_deaotl"    # ResNet-50 backbone (~300MB)

# 導入 AOT 相關模組
try:
    from .aot.networks.engines.aot_engine import AOTEngine, AOTInferEngine
    from .aot.networks.engines.deaot_engine import DeAOTEngine, DeAOTInferEngine
    from .aot.dataloaders import video_transforms as tr
    from .aot.utils.checkpoint import load_network
    from .aot.networks.models import build_vos_model
    from .aot.networks.engines import build_engine
    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False
    logger.warning("AOT modules not found. Some features may be unavailable.")



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

class DeAOTWrapper:
    """DeAOT (Decoupling Features in Hierarchical Propagation) 視頻物件分割模型封裝器
    
    使用範例:
        # 基本初始化
        tracker = DeAOTWrapper(model_type='r50_deaotl', device='cuda')
        
        # 載入視頻序列
        from castle.utils import video_io
        frames = video_io.read_video('video.mp4')  # 自動轉換為 RGB 格式
        
        # 1. 先設置參考幀和初始遮罩
        # 合併所有物件的遮罩到一個標籤圖中
        combined_mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
        combined_mask[first_object_mask > 0] = 1
        combined_mask[second_object_mask > 0] = 2
        
        tracker.add_reference_frame(frames[0], combined_mask, obj_nums=2)
        
        # 2. 執行序列追蹤（從第二幀開始）
        tracks = tracker.track_sequence(frames[1:], update_memory_interval=5)
        
        # 或者逐幀追蹤
        for i in range(1, len(frames)):
            frame_results = tracker.track_frame(frames[i])
        
        # 清除記憶
        tracker.clear_memory()
    """
    
    def __init__(
        self,
        model_type: Union[str, ModelType] = ModelType.R50_DEAOTL,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        long_term_mem_gap: int = 9999,
        short_term_mem_skip: int = 1,
        max_len_long_term: int = 30,
        max_aot_obj_num: Optional[int] = None,
        use_deaot_engine: bool = True,
        test_flip: bool = False,
        test_multiscale: List[float] = [1.0],
        test_max_short_edge: int = 480,
        test_max_long_edge: int = 854
    ):
        """
        初始化 DeAOT 視頻物件分割模型
        
        Args:
            model_type: 模型類型，可以是字符串或 ModelType 枚舉
                - 'r50_deaotl' / ModelType.R50_DEAOTL: ResNet-50 backbone (~300MB)
            checkpoint_path: 模型權重路徑（如果為 None，自動下載）
            device: 計算設備 ('cuda', 'cpu', 'mps')，如果為 None 自動選擇
            long_term_mem_gap: 長期記憶間隔幀數，控制記憶存儲頻率
            short_term_mem_skip: 短期記憶跳躍幀數，影響追蹤精度
            max_len_long_term: 最大長期記憶長度，限制記憶使用量
            max_aot_obj_num: 每個 AOT 引擎的最大物件數量
            use_deaot_engine: 是否使用 DeAOT 引擎（推薦），否則使用 AOT 引擎
            test_flip: 測試時是否使用水平翻轉增強
            test_multiscale: 多尺度測試比例列表
            test_max_short_edge: 測試時影像短邊最大像素
            test_max_long_edge: 測試時影像長邊最大像素
            
        Raises:
            ImportError: 當 AOT 模組未安裝時
            ValueError: 當模型類型不支持時
        """
        if not AOT_AVAILABLE:
            raise ImportError("AOT modules not installed. Please install from: https://github.com/yoxu515/aot-benchmark")
            
        self.model_type = ModelType(model_type) if isinstance(model_type, str) else model_type
        self.device = device if device else DEFAULT_DEVICE
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.max_len_long_term = max_len_long_term
        self.max_aot_obj_num = max_aot_obj_num
        self.use_deaot_engine = use_deaot_engine
        
        # 測試參數
        self.test_flip = test_flip
        self.test_multiscale = test_multiscale
        self.test_max_short_edge = test_max_short_edge
        self.test_max_long_edge = test_max_long_edge
        
        # 載入配置和模型
        self._load_config()
        self._load_model(checkpoint_path)
        
        # 初始化追蹤狀態
        self._reset_tracking_state()
        
        logger.info(f"Initialized DeAOT {self.model_type.value} on {self.device}")
        
    def _load_config(self):
        """載入模型配置"""
        # 直接導入配置類
        try:
            from .aot.configs.pre_ytb_dav import EngineConfig
            phase = 'PRE_YTB_DAV'
            self.cfg = EngineConfig(phase, self.model_type.value)
        except ImportError:
            # 如果無法導入，創建基本配置
            logger.warning("無法載入 AOT 配置模組，使用預設配置")
            self.cfg = self._create_default_config()
        
        # 更新配置
        self.cfg.TEST_LONG_TERM_MEM_GAP = self.long_term_mem_gap
        self.cfg.MAX_LEN_LONG_TERM = self.max_len_long_term
        self.cfg.TEST_FLIP = self.test_flip
        self.cfg.TEST_MULTISCALE = self.test_multiscale
        self.cfg.TEST_MAX_SHORT_EDGE = self.test_max_short_edge
        self.cfg.TEST_MAX_LONG_EDGE = self.test_max_long_edge
        self.cfg.MODEL_ALIGN_CORNERS = True
        
    def _create_default_config(self):
        """創建預設配置（當無法導入 AOT 配置時使用）"""
        model_type_value = self.model_type.value
        
        class DefaultConfig:
            def __init__(self):
                # 基本配置
                self.TEST_LONG_TERM_MEM_GAP = 9999
                self.MAX_LEN_LONG_TERM = 30
                self.TEST_FLIP = False
                self.TEST_MULTISCALE = [1.0]
                self.TEST_MAX_SHORT_EDGE = 480
                self.TEST_MAX_LONG_EDGE = 854
                self.MODEL_ALIGN_CORNERS = True
                
                # 模型配置
                self.MODEL_VOS = {
                    'name': 'deaot',
                    'backbone': model_type_value
                }
                
                # 引擎配置
                self.MODEL_ENGINE = {
                    'name': 'deaot_engine'
                }
                
                # 測試配置
                self.TEST_CKPT_PATH = None
                
        return DefaultConfig()
        
    def _load_model(self, checkpoint_path: Optional[str] = None):
        """載入 DeAOT 模型"""
        # 下載或使用提供的權重
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint()
            
        self.cfg.TEST_CKPT_PATH = checkpoint_path
        
        # 建立模型
        self.model = build_vos_model(self.cfg.MODEL_VOS, self.cfg)
        self.model, _ = load_network(self.model, checkpoint_path, self.device)
        
        # 建立引擎
        if self.use_deaot_engine:
            # 使用自定義的 DeAOT 引擎
            self.engine = DeAOTInferEngine(
                aot_model=self.model,
                device=self.device,
                long_term_mem_gap=self.long_term_mem_gap,
                short_term_mem_skip=self.short_term_mem_skip,
                max_aot_obj_num=self.max_aot_obj_num
            )
        else:
            # 使用標準引擎
            self.engine = build_engine(
                self.cfg.MODEL_ENGINE,
                phase='eval',
                aot_model=self.model,
                device=self.device,
                short_term_mem_skip=self.short_term_mem_skip,
                long_term_mem_gap=self.long_term_mem_gap,
                max_len_long_term=self.max_len_long_term
            )
        
        # 建立轉換器
        self.transform = transforms.Compose([
            tr.MultiRestrictSize(
                self.cfg.TEST_MAX_SHORT_EDGE,
                self.cfg.TEST_MAX_LONG_EDGE,
                self.cfg.TEST_FLIP,
                self.cfg.TEST_MULTISCALE,
                self.cfg.MODEL_ALIGN_CORNERS
            ),
            tr.MultiToTensor()
        ])
        
        self.model.eval()
        
    def _download_checkpoint(self) -> str:
        """
        下載 DeAOT 模型權重檔案
        
        Returns:
            模型權重檔案的本地路徑
            
        Raises:
            ValueError: 當模型類型不支持時
            RuntimeError: 當下載失敗時
        """
        cache_dir = Path.home() / '.cache' / 'castle' / 'models'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == ModelType.R50_DEAOTL:
            ckpt_path = cache_dir / 'R50_DeAOTL_PRE_YTB_DAV.pth'
            gdown_id = '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ'
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        if not ckpt_path.exists():
            try:
                logger.info(f"正在下載 {self.model_type.value} 模型權重至 {ckpt_path}")
                download_with_gdown(gdown_id, str(ckpt_path))
                logger.info(f"模型權重下載完成")
            except Exception as e:
                logger.error(f"模型權重下載失敗: {e}")
                raise RuntimeError(f"Failed to download checkpoint for {self.model_type.value}: {e}")
        else:
            logger.debug(f"使用現有的模型權重: {ckpt_path}")
            
        return str(ckpt_path)
        
    def _reset_tracking_state(self):
        """重置追蹤狀態"""
        self.current_frame_idx = 0
        self.object_tracks = {}
        self.active_objects = set()
        self.obj_nums = 0
        self._reference_frame_set = False  # 追蹤是否已設置參考幀
        
        # 重啟引擎
        if hasattr(self, 'engine'):
            self.engine.restart_engine()
        
    @torch.no_grad()
    def add_reference_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        obj_nums: int,
        frame_step: int = -1,
    ):
        """
        添加參考幀
        
        Args:
            frame: 影像幀
            mask: 遮罩
            obj_nums: 物件數量
            frame_step: 幀步長
            incremental: 是否增量式添加
        """
        sample = {
            'current_img': frame,
            'current_label': mask,
        }
        
        sample = self.transform(sample)
        frame_tensor = sample[0]['current_img'].unsqueeze(0).float().to(self.device)
        mask_tensor = sample[0]['current_label'].unsqueeze(0).float().to(self.device)
        
        # 調整遮罩大小以匹配幀大小
        mask_tensor = F.interpolate(mask_tensor, size=frame_tensor.shape[-2:], mode='nearest')

        self.engine.add_reference_frame(
            frame_tensor, mask_tensor, obj_nums=[obj_nums], frame_step=frame_step
        )
        
        # 設置物件數量和活躍物件
        self.obj_nums = obj_nums
        self.active_objects = set(range(1, obj_nums + 1))
        
        # 標記已設置參考幀
        self._reference_frame_set = True
        logger.debug(f"參考幀已設置，物件數量: {obj_nums}")
            
    @torch.no_grad()
    def track_frame(
        self,
        frame: np.ndarray,
        update_memory: bool = True
    ) -> Dict[int, TrackingResult]:
        """
        對單一幀執行物件追蹤
        
        Args:
            frame: 當前幀影像，格式為 (H, W, 3) RGB
            update_memory: 是否更新模型記憶
                - True: 將當前預測加入記憶，提高後續追蹤精度
                - False: 僅執行推論，適用於快速預覽
                
        Returns:
            追蹤結果列表 [TrackingResult]
            每個 TrackingResult 包含：
            - frame_idx: 當前幀索引
            - object_id: 物件唯一標識
            - mask: 物件分割遮罩 (H, W)
            - confidence: 追蹤置信度
            - bbox: 邊界框 (x, y, width, height)
            - area: 遮罩面積（像素數）
            - center: 物件中心座標 (x, y)
            
        注意:
            - 必須先調用 add_reference_frame() 設置參考幀
            - 每次調用會自動增加內部幀計數器
            
        Raises:
            RuntimeError: 當尚未調用 add_reference_frame() 時
        """
        # 檢查是否已設置參考幀
        if not self._reference_frame_set:
            raise RuntimeError(
                "必須先調用 add_reference_frame() 設置參考幀後才能進行追蹤。"
                "請先載入第一幀影像和初始遮罩。"
            )
        
        self.current_frame_idx += 1
        output_height, output_width = frame.shape[:2]
        
        # 預處理影像
        sample = {'current_img': frame}
        sample = self.transform(sample)
        image = sample[0]['current_img'].unsqueeze(0).float().to(self.device)
        
        # 執行追蹤
        self.engine.match_propogate_one_frame(image)
        
        # 解碼預測
        pred_logit = self.engine.decode_current_logits((output_height, output_width))
        pred_label = torch.argmax(pred_logit, dim=1, keepdim=True).float()
        
            
        # 處理結果
        results = self._process_predictions(pred_label, self.current_frame_idx)
        
        return list(results.values())
        
    def track_sequence(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[callable] = None
    ) -> Dict[int, ObjectTrack]:
        """
        對整個視頻序列執行物件追蹤
        
        Args:
            frames: 影像序列列表，每個元素為 (H, W, 3) RGB 格式
            progress_callback: 進度回調函數 callback(current_frame, total_frames)
                
        Returns:
            完整的物件追蹤軌跡字典 {object_id: ObjectTrack}
            每個 ObjectTrack 包含：
            - object_id: 物件標識
            - start_frame: 開始幀索引
            - end_frame: 結束幀索引  
            - masks: 每幀的遮罩 {frame_idx: mask}
            - confidences: 每幀的置信度 {frame_idx: confidence}
            - bboxes: 每幀的邊界框 {frame_idx: bbox}
            
        注意:
            - 必須先調用 add_reference_frame() 設置參考幀和初始遮罩
            
        範例:
            # 1. 先設置參考幀
            tracker.add_reference_frame(frames[0], initial_mask, obj_nums=2)
            
            # 2. 執行序列追蹤
            tracks = tracker.track_sequence(
                frames[1:],  # 從第二幀開始追蹤
                update_memory_interval=5,
                progress_callback=lambda i, total: print(f"{i}/{total}")
            )
            
        Raises:
            RuntimeError: 當尚未調用 add_reference_frame() 時
        """
        # 檢查是否已設置參考幀
        if not self._reference_frame_set:
            raise RuntimeError(
                "必須先調用 add_reference_frame() 設置參考幀後才能進行序列追蹤。"
                "請先載入第一幀影像和初始遮罩。"
            )
        
        
        # 初始化物件追蹤軌跡
        for obj_id in self.active_objects:
            self.object_tracks[obj_id] = ObjectTrack(
                object_id=obj_id,
                start_frame=self.current_frame_idx,
                end_frame=self.current_frame_idx + len(frames),
                masks={},
                confidences={}
            )
        
        # 追蹤後續幀
        for i in range(len(frames)):
            # 追蹤當前幀
            results = self.track_frame(frames[i])
            
            # 更新追蹤軌跡
            for result in results:
                obj_id = result.object_id
                if obj_id in self.object_tracks:
                    track = self.object_tracks[obj_id]
                    track.masks[result.frame_idx] = result.mask
                    track.confidences[result.frame_idx] = result.confidence
            
            # 進度回調
            if progress_callback:
                progress_callback(i + 1, len(frames))
                
        # 完成追蹤
        for track in self.object_tracks.values():
            track.end_frame = self.current_frame_idx - 1
            
        return self.object_tracks
        
    def _process_predictions(
        self,
        pred_label: torch.Tensor,
        frame_idx: int
    ) -> Dict[int, TrackingResult]:
        """處理預測結果"""
        results = {}
        
        # 轉換為 numpy
        pred_np = pred_label.squeeze().cpu().numpy().astype(np.uint8)
        
        # 為每個物件提取遮罩
        for obj_id in self.active_objects:
            mask = (pred_np == obj_id).astype(np.uint8)
            
            # 檢查遮罩是否有效
            if np.sum(mask) < 10:
                logger.warning(f"Object {obj_id} lost at frame {frame_idx}")
                continue
                
            # 計算屬性
            area = int(np.sum(mask))
            
            # 計算質心
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                center_x = float(np.mean(x_indices))
                center_y = float(np.mean(y_indices))
            else:
                center_x, center_y = 0.0, 0.0
            
            # 創建結果
            result = TrackingResult(
                frame_idx=frame_idx,
                object_id=obj_id,
                mask=mask,
                confidence=0.9,  # 預設置信度
                area=area,
                center=(center_x, center_y)
            )
            
            results[obj_id] = result
            
            # 更新追蹤軌跡
            if obj_id not in self.object_tracks:
                self.object_tracks[obj_id] = ObjectTrack(
                    object_id=obj_id,
                    start_frame=frame_idx,
                    end_frame=frame_idx,
                    masks={},
                    confidences={}
                )
            
            track = self.object_tracks[obj_id]
            track.masks[frame_idx] = mask
            track.confidences[frame_idx] = result.confidence
            track.end_frame = frame_idx
                
                
        return results
        
    def restart(self):
        """重啟追蹤器"""
        self._reset_tracking_state()
        if hasattr(self, 'engine'):
            self.engine.restart_engine()
            
    def clear_memory(self):
        """清除記憶"""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    def __del__(self):
        """析構函數"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'engine'):
            del self.engine
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# class DeAOTTrackerInferEngine(DeAOTInferEngine):
#     """DeAOT 追蹤推論引擎"""
    
#     def __init__(self, aot_model, device, long_term_mem_gap=9999,
#                  short_term_mem_skip=1, max_aot_obj_num=None):
#         super().__init__(aot_model, device, long_term_mem_gap,
#                         short_term_mem_skip, max_aot_obj_num)
        

def download_with_gdown(file_id, destination):
    """
    使用 gdown 下載 Google Drive 檔案
    
    Args:
        file_id: Google Drive 檔案 ID
        destination: 目標檔案路徑
    """
    import gdown
    
    # 確保目標目錄存在
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # 檢查檔案是否已存在
    if not os.path.isfile(destination):
        print(f"正在下載 {os.path.basename(destination)}...")
        try:
            # 直接使用 gdown 的 Python API
            gdown.download(id=file_id, output=destination, quiet=False)
            print(f"下載完成: {os.path.basename(destination)}")
        except Exception as e:
            print(f"下載失敗: {e}")
            raise
    else:
        print(f"{os.path.basename(destination)} 已存在，跳過下載")