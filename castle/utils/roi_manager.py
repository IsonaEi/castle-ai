"""
ROI Manager 模組 - 管理 Region of Interest (ROI) 相關的資料儲存與讀取

該模組提供了高效的 HDF5 檔案管理功能，用於儲存和管理影像遮罩、配置資料等。
相較於舊版本，新版本提供了更好的錯誤處理、類型安全性和資源管理。
"""

import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

# 設定日誌記錄器
logger = logging.getLogger(__name__)


class ROIManagerError(Exception):
    """ROI Manager 特定的錯誤類別"""
    pass


class ROIManager:
    """
    改進版的 ROI Manager，用於管理 HDF5 格式的遮罩資料和配置
    
    主要功能：
    - 高效儲存和讀取影像遮罩
    - 配置資料管理
    - 自動檔案重置機制防止記憶體問題
    - 支援 context manager 協議
    - 完整的錯誤處理和驗證
    
    範例用法：
        # 作為 context manager 使用，包含 meta 資訊
        with ROIManager("data.h5", 
                       video_basename="experiment.mp4",
                       video_length=1000, 
                       video_height=480, 
                       video_width=640,
                       video_fps=30.0) as roi_manager:
            roi_manager[0] = mask_array  # 自動計算 kinematic 資料
            mask_array = roi_manager[0]
            
            # 獲取運動軌跡
            x_coords, y_coords = roi_manager.get_trajectory()
            
            # 獲取速度分布
            frames, speeds = roi_manager.get_speed_profile()
    """
    
    def __init__(
        self, 
        file_path: Union[str, Path], 
        auto_reset_threshold: int = 5000,
        compression_level: int = 3,
        video_basename: Optional[str] = None,
        video_length: Optional[int] = None,
        video_height: Optional[int] = None,
        video_width: Optional[int] = None,
        video_fps: Optional[float] = None
    ):
        """
        初始化 ROI Manager
        
        Args:
            file_path: HDF5 檔案路徑
            auto_reset_threshold: 自動重置檔案連接的操作次數閾值
            compression_level: GZIP 壓縮等級 (1-9)
            video_basename: 影片檔案名稱
            video_length: 影片總影格數
            video_height: 影片高度
            video_width: 影片寬度
            video_fps: 影片幀率 (frames per second)
        
        Raises:
            ROIManagerError: 當檔案初始化失敗時
        """
        self.file_path = Path(file_path)
        self.auto_reset_threshold = auto_reset_threshold
        self.compression_level = max(1, min(9, compression_level))
        
        self._reset_count = 0
        self._file_handle: Optional[h5py.File] = None
        self._config_cache: Dict[str, Any] = {}
        self._last_centroid: Optional[Tuple[float, float]] = None
        self._last_frame_index: Optional[int] = None
        
        # 確保目錄存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._reset_connection()
            
            # 初始化 meta 資訊（如果提供的話）
            video_meta = {
                'video_basename': video_basename,
                'video_length': video_length,
                'video_height': video_height,
                'video_width': video_width,
                'video_fps': video_fps
            }
            
            if any(v is not None for v in video_meta.values()):
                self._initialize_meta(**video_meta)
            
            # 初始化 kinematic 群組
            self._initialize_kinematic_group()
            
            logger.info(f"ROI Manager 初始化完成: {self.file_path}")
        except Exception as e:
            raise ROIManagerError(f"無法初始化 ROI Manager: {e}")
    
    def __enter__(self) -> 'ROIManager':
        """Context manager 進入"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager 退出"""
        self.close()
    
    def __setitem__(self, index: int, mask: np.ndarray) -> None:
        """支援 roi_manager[index] = mask 語法"""
        self.write_mask(index, mask)
    
    def __getitem__(self, index: int) -> np.ndarray:
        """支援 mask = roi_manager[index] 語法"""
        return self.read_mask(index)
    
    def __len__(self) -> int:
        """返回總影格數量"""
        return self.get_total_frames()
    
    def __del__(self) -> None:
        """解構函數，確保資源被正確釋放"""
        self.close()
    
    # ==================== Private Methods ====================
    
    def _reset_connection(self) -> None:
        """重置 HDF5 檔案連接"""
        self._reset_count = 0
        
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception as e:
                logger.warning(f"關閉舊檔案連接時發生錯誤: {e}")
        
        try:
            mode = 'a' if self.file_path.exists() else 'w'
            self._file_handle = h5py.File(str(self.file_path), mode)
            logger.debug(f"HDF5 檔案連接已重置 (模式: {mode})")
        except Exception as e:
            raise ROIManagerError(f"無法建立 HDF5 檔案連接: {e}")
    
    def _initialize_meta(
        self,
        video_basename: Optional[str] = None,
        video_length: Optional[int] = None,
        video_height: Optional[int] = None,
        video_width: Optional[int] = None,
        video_fps: Optional[float] = None
    ) -> None:
        """初始化 meta 資訊"""
        try:
            # 創建或更新 meta 群組
            if 'meta' not in self._file_handle:
                meta_group = self._file_handle.create_group('meta')
            else:
                meta_group = self._file_handle['meta']
            
            # 存儲 meta 資訊
            meta_fields = {
                'video_basename': video_basename,
                'video_length': video_length,
                'video_height': video_height,
                'video_width': video_width,
                'video_fps': video_fps
            }
            
            for field, value in meta_fields.items():
                if value is not None:
                    if field in meta_group:
                        del meta_group[field]
                    meta_group.create_dataset(field, data=value)
            
            # 特殊處理 video_length -> total_frames
            if video_length is not None:
                if 'total_frames' in self._file_handle:
                    del self._file_handle['total_frames']
                self._file_handle.create_dataset('total_frames', data=video_length)
            
            logger.debug("Meta 資訊初始化完成")
            
        except Exception as e:
            logger.error(f"初始化 meta 資訊失敗: {e}")
    
    def _initialize_kinematic_group(self) -> None:
        """初始化 kinematic 群組和資料集"""
        try:
            if 'kinematic' not in self._file_handle:
                kinematic_group = self._file_handle.create_group('kinematic')
                
                # 創建可擴展的資料集，初始大小為 0
                datasets = {
                    'x': 'float32',          # x 座標 (質心)
                    'y': 'float32',          # y 座標 (質心)
                    'area': 'float32',       # 面積
                    'speed': 'float32',      # 速度
                    'frame_indices': 'int32' # 影格索引對應關係
                }
                
                for name, dtype in datasets.items():
                    kinematic_group.create_dataset(
                        name, (0,), maxshape=(None,), dtype=dtype
                    )
                
                logger.debug("Kinematic 群組初始化完成")
            
        except Exception as e:
            logger.error(f"初始化 kinematic 群組失敗: {e}")
    
    def _calculate_mask_properties(self, mask: np.ndarray) -> Tuple[float, float, float]:
        """
        計算遮罩的屬性
        
        Args:
            mask: 遮罩陣列
            
        Returns:
            (centroid_x, centroid_y, area)
        """
        # 確保是二值遮罩
        binary_mask = mask > 0
        
        # 計算面積
        area = float(np.sum(binary_mask))
        
        if area == 0:
            return 0.0, 0.0, 0.0
        
        # 計算質心
        y_coords, x_coords = np.where(binary_mask)
        centroid_x = float(np.mean(x_coords))
        centroid_y = float(np.mean(y_coords))
        
        return centroid_x, centroid_y, area
    
    def _calculate_speed(
        self, 
        current_centroid: Tuple[float, float], 
        current_frame: int
    ) -> float:
        """
        計算移動速度（像素/影格）
        
        Args:
            current_centroid: 當前質心座標
            current_frame: 當前影格索引
            
        Returns:
            速度值（像素/影格）
        """
        if (self._last_centroid is None or 
            self._last_frame_index is None or 
            current_frame <= self._last_frame_index):
            return 0.0
        
        # 計算歐氏距離
        dx = current_centroid[0] - self._last_centroid[0]
        dy = current_centroid[1] - self._last_centroid[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # 計算時間差
        time_diff = current_frame - self._last_frame_index
        
        # 計算速度
        speed = distance / time_diff if time_diff > 0 else 0.0
        
        return float(speed)
    
    def _update_kinematic_data(
        self,
        frame_index: int,
        centroid_x: float,
        centroid_y: float,
        area: float,
        speed: float
    ) -> None:
        """更新 kinematic 資料"""
        try:
            kinematic_group = self._file_handle['kinematic']
            
            # 檢查該影格是否已存在
            frame_indices = kinematic_group['frame_indices']
            existing_indices = frame_indices[:] if len(frame_indices) > 0 else np.array([])
            
            if frame_index in existing_indices:
                # 更新現有資料
                idx = np.where(existing_indices == frame_index)[0][0]
                kinematic_group['x'][idx] = centroid_x
                kinematic_group['y'][idx] = centroid_y
                kinematic_group['area'][idx] = area
                kinematic_group['speed'][idx] = speed
            else:
                # 添加新資料
                current_size = len(frame_indices)
                new_size = current_size + 1
                
                # 擴展所有資料集
                for dataset_name in ['frame_indices', 'x', 'y', 'area', 'speed']:
                    kinematic_group[dataset_name].resize((new_size,))
                
                # 添加新資料
                kinematic_group['frame_indices'][current_size] = frame_index
                kinematic_group['x'][current_size] = centroid_x
                kinematic_group['y'][current_size] = centroid_y
                kinematic_group['area'][current_size] = area
                kinematic_group['speed'][current_size] = speed
            
            logger.debug(
                f"更新 kinematic 資料: frame {frame_index}, "
                f"pos=({centroid_x:.1f}, {centroid_y:.1f}), "
                f"area={area:.1f}, speed={speed:.2f}"
            )
            
        except Exception as e:
            logger.error(f"更新 kinematic 資料失敗: {e}")
    
    def _check_and_reset(self) -> None:
        """檢查是否需要重置檔案連接"""
        self._reset_count += 1
        if self._reset_count >= self.auto_reset_threshold:
            logger.debug("達到重置閾值，重置檔案連接")
            self._reset_connection()
    
    def _validate_mask(self, mask: np.ndarray) -> None:
        """驗證遮罩資料的有效性"""
        if not isinstance(mask, np.ndarray):
            raise ValueError("遮罩必須是 numpy 陣列")
        
        if mask.ndim < 2:
            raise ValueError("遮罩至少需要是 2D 陣列")
        
        if mask.dtype not in [np.uint8, np.uint16, np.int32, np.bool_]:
            logger.warning(f"遮罩資料類型 {mask.dtype} 可能不是最佳選擇，建議使用 uint8")
    
    # ==================== Public Methods ====================
    
    def write_mask(self, index: int, mask: np.ndarray) -> None:
        """
        寫入遮罩資料到指定索引，並自動更新 kinematic 資料
        
        Args:
            index: 影格索引
            mask: 遮罩陣列
            
        Raises:
            ROIManagerError: 當寫入失敗時
            ValueError: 當遮罩資料無效時
        """
        self._validate_mask(mask)
        self._check_and_reset()
        
        try:
            # 確保 ROI 群組存在
            if 'ROI' not in self._file_handle:
                self._file_handle.create_group('ROI')
            
            dataset_name = f'ROI/{index}'
            
            # 處理現有資料集
            if dataset_name in self._file_handle:
                dataset = self._file_handle[dataset_name]
                if dataset.shape != mask.shape:
                    del self._file_handle[dataset_name]
                    dataset = self._file_handle.create_dataset(
                        dataset_name, 
                        mask.shape, 
                        dtype='uint8',
                        compression="gzip",
                        compression_opts=self.compression_level
                    )
                dataset[:] = mask.astype('uint8')
            else:
                # 建立新的資料集
                dataset = self._file_handle.create_dataset(
                    dataset_name,
                    mask.shape,
                    dtype='uint8',
                    compression="gzip",
                    compression_opts=self.compression_level
                )
                dataset[:] = mask.astype('uint8')
            
            # 計算並更新 kinematic 資料
            centroid_x, centroid_y, area = self._calculate_mask_properties(mask)
            current_centroid = (centroid_x, centroid_y)
            speed = self._calculate_speed(current_centroid, index)
            
            # 更新 kinematic 資料
            self._update_kinematic_data(index, centroid_x, centroid_y, area, speed)
            
            # 更新追蹤狀態
            self._last_centroid = current_centroid
            self._last_frame_index = index
            
            logger.debug(f"成功寫入遮罩到 ROI/{index}")
            
        except Exception as e:
            raise ROIManagerError(f"寫入遮罩失敗 (索引 {index}): {e}")
    
    def read_mask(self, index: int) -> np.ndarray:
        """
        從指定索引讀取遮罩資料
        
        Args:
            index: 影格索引
            
        Returns:
            遮罩陣列
            
        Raises:
            ROIManagerError: 當讀取失敗時
            KeyError: 當索引不存在時
        """
        self._check_and_reset()
        
        try:
            dataset_name = f'ROI/{index}'
            if dataset_name not in self._file_handle:
                raise KeyError(f"索引 {index} 的遮罩不存在")
            
            mask = self._file_handle[dataset_name][:]
            logger.debug(f"成功讀取 ROI/{index} 的遮罩")
            return mask
            
        except KeyError:
            raise
        except Exception as e:
            raise ROIManagerError(f"讀取遮罩失敗 (索引 {index}): {e}")
    
    def has_mask(self, index: int) -> bool:
        """
        檢查指定索引是否存在遮罩
        
        Args:
            index: 影格索引
            
        Returns:
            True 如果遮罩存在，False 否則
        """
        try:
            return f'ROI/{index}' in self._file_handle
        except Exception as e:
            logger.error(f"檢查遮罩存在性時發生錯誤: {e}")
            return False
    
    def write_config(self, key: str, value: Any) -> None:
        """
        寫入配置資料
        
        Args:
            key: 配置鍵名
            value: 配置值
            
        Raises:
            ROIManagerError: 當寫入失敗時
        """
        self._check_and_reset()
        
        try:
            # 刪除舊的配置（如果存在）
            if key in self._file_handle:
                del self._file_handle[key]
            
            # 寫入新的配置
            self._file_handle.create_dataset(key, data=value)
            self._config_cache[key] = value
            
            logger.debug(f"成功寫入配置: {key} = {value}")
            
        except Exception as e:
            raise ROIManagerError(f"寫入配置失敗 ({key}): {e}")
    
    def read_config(self, key: str, default: Any = None) -> Any:
        """
        讀取配置資料
        
        Args:
            key: 配置鍵名
            default: 預設值（當鍵不存在時返回）
            
        Returns:
            配置值或預設值
            
        Raises:
            ROIManagerError: 當讀取失敗時
        """
        # 先檢查快取
        if key in self._config_cache:
            return self._config_cache[key]
        
        self._check_and_reset()
        
        try:
            if key not in self._file_handle:
                if default is not None:
                    return default
                raise KeyError(f"配置鍵 '{key}' 不存在")
            
            value = self._file_handle[key][()]
            self._config_cache[key] = value
            
            logger.debug(f"成功讀取配置: {key} = {value}")
            return value
            
        except KeyError:
            if default is not None:
                return default
            raise
        except Exception as e:
            raise ROIManagerError(f"讀取配置失敗 ({key}): {e}")
    
    # ==================== Property Methods ====================
    
    def get_n_rois(self) -> int:
        """獲取 ROI 數量"""
        try:
            return int(self.read_config('n_rois', 0))
        except Exception:
            return 0
    
    def set_n_rois(self, n_rois: int) -> None:
        """設定 ROI 數量"""
        if n_rois < 0:
            raise ValueError("ROI 數量不能為負數")
        self.write_config('n_rois', n_rois)
    
    def get_total_frames(self) -> int:
        """獲取總影格數量"""
        try:
            return int(self.read_config('total_frames', 0))
        except Exception:
            return 0
    
    def set_total_frames(self, total_frames: int) -> None:
        """設定總影格數量"""
        if total_frames < 0:
            raise ValueError("總影格數量不能為負數")
        self.write_config('total_frames', total_frames)
    
    # ==================== Query Methods ====================
    
    def get_mask_indices(self) -> List[int]:
        """
        獲取所有存在遮罩的索引
        
        Returns:
            包含所有遮罩索引的列表（已排序）
        """
        try:
            indices = []
            
            # 檢查 ROI 群組是否存在
            if 'ROI' not in self._file_handle:
                return indices
            
            roi_group = self._file_handle['ROI']
            for key in roi_group.keys():
                try:
                    indices.append(int(key))
                except ValueError:
                    continue
            
            return sorted(indices)
            
        except Exception as e:
            logger.error(f"獲取遮罩索引時發生錯誤: {e}")
            return []
    
    def get_mask_shape(self, index: int) -> Optional[Tuple[int, ...]]:
        """
        獲取指定索引遮罩的形狀
        
        Args:
            index: 影格索引
            
        Returns:
            遮罩形狀 tuple，如果不存在則返回 None
        """
        try:
            dataset_name = f'ROI/{index}'
            if dataset_name in self._file_handle:
                return self._file_handle[dataset_name].shape
            return None
        except Exception:
            return None
    
    def get_meta_info(self) -> Dict[str, Any]:
        """
        獲取 meta 資訊
        
        Returns:
            包含 meta 資訊的字典
        """
        meta_info = {}
        try:
            if 'meta' in self._file_handle:
                meta_group = self._file_handle['meta']
                for key in meta_group.keys():
                    value = meta_group[key][()]
                    # 處理字串類型
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    meta_info[key] = value
        except Exception as e:
            logger.error(f"讀取 meta 資訊失敗: {e}")
        
        return meta_info
    
    def get_kinematic_data(
        self, 
        frame_indices: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        獲取 kinematic 資料
        
        Args:
            frame_indices: 指定要獲取的影格索引列表，None 表示獲取全部
            
        Returns:
            包含 kinematic 資料的字典
        """
        result = {
            'frame_indices': np.array([]),
            'x': np.array([]),
            'y': np.array([]),
            'area': np.array([]),
            'speed': np.array([])
        }
        
        try:
            if 'kinematic' not in self._file_handle:
                return result
            
            kinematic_group = self._file_handle['kinematic']
            
            # 讀取所有資料
            all_data = {
                key: kinematic_group[key][:]
                for key in result.keys()
            }
            
            if frame_indices is None:
                # 返回所有資料
                result = all_data
            else:
                # 篩選指定的影格索引
                mask = np.isin(all_data['frame_indices'], frame_indices)
                result = {
                    key: values[mask]
                    for key, values in all_data.items()
                }
                
        except Exception as e:
            logger.error(f"讀取 kinematic 資料失敗: {e}")
        
        return result
    
    def get_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        獲取運動軌跡 (x, y 座標序列)
        
        Returns:
            (x_coords, y_coords) 座標陣列
        """
        kinematic_data = self.get_kinematic_data()
        return kinematic_data['x'], kinematic_data['y']
    
    def get_speed_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        獲取速度分布 (影格索引, 速度序列)
        
        Returns:
            (frame_indices, speeds) 陣列
        """
        kinematic_data = self.get_kinematic_data()
        return kinematic_data['frame_indices'], kinematic_data['speed']
    
    def get_area_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        獲取面積變化 (影格索引, 面積序列)
        
        Returns:
            (frame_indices, areas) 陣列
        """
        kinematic_data = self.get_kinematic_data()
        return kinematic_data['frame_indices'], kinematic_data['area']
    
    def get_real_speed_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        獲取實際速度分布 (時間秒數, 像素/秒速度序列)
        
        需要 meta 中存在 video_fps 資訊才能計算
        
        Returns:
            (time_seconds, real_speeds) 陣列
        """
        try:
            # 獲取 fps 資訊
            meta_info = self.get_meta_info()
            if 'video_fps' not in meta_info:
                logger.warning("無法計算實際速度：缺少 video_fps 資訊")
                return np.array([]), np.array([])
            
            fps = float(meta_info['video_fps'])
            if fps <= 0:
                logger.warning("無法計算實際速度：video_fps 值無效")
                return np.array([]), np.array([])
            
            # 獲取原始速度資料
            frame_indices, frame_speeds = self.get_speed_profile()
            
            if len(frame_indices) == 0:
                return np.array([]), np.array([])
            
            # 轉換為實際時間和速度
            time_seconds = frame_indices.astype(float) / fps
            real_speeds = frame_speeds * fps  # 像素/影格 * 影格/秒 = 像素/秒
            
            return time_seconds, real_speeds
            
        except Exception as e:
            logger.error(f"計算實際速度分布失敗: {e}")
            return np.array([]), np.array([])
    
    def get_time_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        獲取基於時間的運動軌跡 (時間秒數, x座標, y座標)
        
        需要 meta 中存在 video_fps 資訊才能計算
        
        Returns:
            (time_seconds, x_coords, y_coords) 陣列
        """
        try:
            # 獲取 fps 資訊
            meta_info = self.get_meta_info()
            if 'video_fps' not in meta_info:
                logger.warning("無法計算時間軌跡：缺少 video_fps 資訊")
                return np.array([]), np.array([]), np.array([])
            
            fps = float(meta_info['video_fps'])
            if fps <= 0:
                logger.warning("無法計算時間軌跡：video_fps 值無效")
                return np.array([]), np.array([]), np.array([])
            
            # 獲取 kinematic 資料
            kinematic_data = self.get_kinematic_data()
            frame_indices = kinematic_data['frame_indices']
            
            if len(frame_indices) == 0:
                return np.array([]), np.array([]), np.array([])
            
            # 轉換為實際時間
            time_seconds = frame_indices.astype(float) / fps
            
            return time_seconds, kinematic_data['x'], kinematic_data['y']
            
        except Exception as e:
            logger.error(f"計算時間軌跡失敗: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        獲取檔案資訊
        
        Returns:
            包含檔案資訊的字典
        """
        info = {
            'file_path': str(self.file_path),
            'file_exists': self.file_path.exists(),
            'file_size_mb': 0,
            'n_rois': self.get_n_rois(),
            'total_frames': self.get_total_frames(),
            'mask_count': len(self.get_mask_indices()),
            'config_keys': [],
            'compression_level': self.compression_level,
            'auto_reset_threshold': self.auto_reset_threshold
        }
        
        if self.file_path.exists():
            info['file_size_mb'] = round(
                self.file_path.stat().st_size / (1024 * 1024), 2
            )
        
        try:
            # 獲取所有配置項目（排除群組）
            config_keys = [
                key for key in self._file_handle.keys()
                if key not in ['ROI', 'meta', 'kinematic']
            ]
            info['config_keys'] = config_keys
            
            # 添加 meta 資訊
            info['meta'] = self.get_meta_info()
            
            # 添加 kinematic 資訊
            if 'kinematic' in self._file_handle:
                kinematic_group = self._file_handle['kinematic']
                info['kinematic_data_count'] = len(kinematic_group['frame_indices'])
            else:
                info['kinematic_data_count'] = 0
                
        except Exception:
            pass
        
        return info
    
    # ==================== Modification Methods ====================
    
    def delete_mask(self, index: int) -> bool:
        """
        刪除指定索引的遮罩
        
        Args:
            index: 影格索引
            
        Returns:
            True 如果刪除成功，False 如果遮罩不存在
            
        Raises:
            ROIManagerError: 當刪除失敗時
        """
        try:
            dataset_name = f'ROI/{index}'
            if dataset_name in self._file_handle:
                del self._file_handle[dataset_name]
                logger.debug(f"成功刪除 ROI/{index} 的遮罩")
                return True
            return False
        except Exception as e:
            raise ROIManagerError(f"刪除遮罩失敗 (索引 {index}): {e}")
    
    def clear_all_masks(self) -> None:
        """
        清除所有遮罩資料（保留配置）
        
        Raises:
            ROIManagerError: 當清除失敗時
        """
        try:
            indices = self.get_mask_indices()
            for index in indices:
                self.delete_mask(index)
            
            # 同時清除 kinematic 資料
            if 'kinematic' in self._file_handle:
                kinematic_group = self._file_handle['kinematic']
                # 重設所有 kinematic 資料集大小為 0
                for dataset_name in ['x', 'y', 'area', 'speed', 'frame_indices']:
                    if dataset_name in kinematic_group:
                        kinematic_group[dataset_name].resize((0,))
            
            # 重置追蹤狀態
            self._last_centroid = None
            self._last_frame_index = None
            
            logger.info(f"成功清除 {len(indices)} 個遮罩和相關 kinematic 資料")
        except Exception as e:
            raise ROIManagerError(f"清除遮罩失敗: {e}")
    
    # ==================== Context Managers ====================
    
    @contextmanager
    def batch_operations(self):
        """
        批次操作 context manager，在批次操作期間暫停自動重置
        
        用法:
            with roi_manager.batch_operations():
                for i in range(1000):
                    roi_manager.write_mask(i, masks[i])
        """
        original_threshold = self.auto_reset_threshold
        self.auto_reset_threshold = float('inf')  # 暫停自動重置
        
        try:
            yield
        finally:
            self.auto_reset_threshold = original_threshold
    
    # ==================== Utility Methods ====================
    
    def flush(self) -> None:
        """強制將資料寫入磁碟"""
        try:
            if self._file_handle is not None:
                self._file_handle.flush()
                logger.debug("資料已強制寫入磁碟")
        except Exception as e:
            logger.error(f"資料寫入磁碟失敗: {e}")
    
    def close(self) -> None:
        """關閉檔案連接並清理資源"""
        try:
            if self._file_handle is not None and self._file_handle.id.valid:
                self._file_handle.close()
                logger.debug("ROI Manager 已關閉")
        except Exception as e:
            logger.error(f"關閉 ROI Manager 時發生錯誤: {e}")
        finally:
            self._file_handle = None
            self._config_cache.clear()


# 向後相容性別名
H5IO = ROIManager  # 提供舊版本的類別名稱作為別名