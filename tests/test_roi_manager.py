# tests/test_roi_manager.py

"""
ROIManager 的完整測試套件

測試覆蓋：
- 基本功能測試
- 錯誤處理測試
- 效能測試
- 邊界條件測試
- 整合測試
"""

import os
import tempfile
from pathlib import Path
from typing import Generator
import time
import warnings

import h5py
import numpy as np
import pytest

from castle.utils.roi_manager import ROIManager, ROIManagerError, H5IO


# ==================== Fixtures ====================

@pytest.fixture
def temp_h5_file() -> Generator[Path, None, None]:
    """建立臨時 HDF5 檔案"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        temp_path = Path(tmp.name)
    
    yield temp_path
    
    # 清理
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_mask() -> np.ndarray:
    """建立範例遮罩"""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # 建立一個圓形遮罩
    center = (50, 50)
    radius = 20
    y, x = np.ogrid[:100, :100]
    mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    mask[mask_circle] = 255
    return mask


@pytest.fixture
def sample_masks() -> list[np.ndarray]:
    """建立多個範例遮罩（模擬移動）"""
    masks = []
    for i in range(10):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # 建立移動的圓形
        center = (40 + i * 2, 40 + i * 2)
        radius = 20
        y, x = np.ogrid[:100, :100]
        mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        mask[mask_circle] = 255
        masks.append(mask)
    return masks


@pytest.fixture
def roi_manager(temp_h5_file) -> ROIManager:
    """建立 ROIManager 實例"""
    return ROIManager(temp_h5_file)


@pytest.fixture
def roi_manager_with_meta(temp_h5_file) -> ROIManager:
    """建立包含 meta 資訊的 ROIManager 實例"""
    return ROIManager(
        temp_h5_file,
        video_basename="test_video.mp4",
        video_length=1000,
        video_height=480,
        video_width=640,
        video_fps=30.0
    )


# ==================== 基本功能測試 ====================

class TestBasicFunctionality:
    """基本功能測試"""
    
    def test_initialization(self, temp_h5_file):
        """測試初始化"""
        manager = ROIManager(temp_h5_file)
        assert manager.file_path == temp_h5_file
        assert manager.auto_reset_threshold == 5000
        assert manager.compression_level == 3
        manager.close()
    
    def test_initialization_with_meta(self, temp_h5_file):
        """測試帶 meta 資訊的初始化"""
        manager = ROIManager(
            temp_h5_file,
            video_basename="test.mp4",
            video_length=100,
            video_height=480,
            video_width=640,
            video_fps=30.0
        )
        
        meta = manager.get_meta_info()
        assert meta['video_basename'] == "test.mp4"
        assert meta['video_length'] == 100
        assert meta['video_height'] == 480
        assert meta['video_width'] == 640
        assert meta['video_fps'] == 30.0
        manager.close()
    
    def test_context_manager(self, temp_h5_file, sample_mask):
        """測試 context manager 協議"""
        with ROIManager(temp_h5_file) as manager:
            manager[0] = sample_mask
            assert manager.has_mask(0)
        
        # 確認檔案已關閉但資料仍存在
        with ROIManager(temp_h5_file) as manager:
            assert manager.has_mask(0)
            retrieved = manager[0]
            np.testing.assert_array_equal(retrieved, sample_mask)
    
    def test_write_and_read_mask(self, roi_manager, sample_mask):
        """測試寫入和讀取遮罩"""
        # 寫入
        roi_manager.write_mask(0, sample_mask)
        
        # 檢查存在性
        assert roi_manager.has_mask(0)
        assert not roi_manager.has_mask(1)
        
        # 讀取
        retrieved = roi_manager.read_mask(0)
        np.testing.assert_array_equal(retrieved, sample_mask)
    
    def test_index_operators(self, roi_manager, sample_mask):
        """測試索引運算子 [] """
        # 使用 [] 寫入
        roi_manager[5] = sample_mask
        
        # 使用 [] 讀取
        retrieved = roi_manager[5]
        np.testing.assert_array_equal(retrieved, sample_mask)
    
    def test_len_operator(self, roi_manager):
        """測試 len() 運算子"""
        roi_manager.set_total_frames(100)
        assert len(roi_manager) == 100
    
    def test_config_operations(self, roi_manager):
        """測試配置操作"""
        # 寫入配置
        roi_manager.write_config('test_key', 'test_value')
        roi_manager.write_config('test_number', 42)
        roi_manager.write_config('test_array', np.array([1, 2, 3]))
        
        # 讀取配置
        assert roi_manager.read_config('test_key') == 'test_value'
        assert roi_manager.read_config('test_number') == 42
        np.testing.assert_array_equal(
            roi_manager.read_config('test_array'),
            np.array([1, 2, 3])
        )
        
        # 預設值
        assert roi_manager.read_config('non_existent', 'default') == 'default'
    
    def test_n_rois_property(self, roi_manager):
        """測試 n_rois 屬性"""
        assert roi_manager.get_n_rois() == 0
        
        roi_manager.set_n_rois(5)
        assert roi_manager.get_n_rois() == 5
    
    def test_total_frames_property(self, roi_manager):
        """測試 total_frames 屬性"""
        assert roi_manager.get_total_frames() == 0
        
        roi_manager.set_total_frames(1000)
        assert roi_manager.get_total_frames() == 1000


# ==================== Kinematic 資料測試 ====================

class TestKinematicData:
    """Kinematic 資料相關測試"""
    
    def test_automatic_kinematic_calculation(self, roi_manager, sample_mask):
        """測試自動計算 kinematic 資料"""
        roi_manager[0] = sample_mask
        
        kinematic = roi_manager.get_kinematic_data()
        assert len(kinematic['frame_indices']) == 1
        assert kinematic['frame_indices'][0] == 0
        assert kinematic['area'][0] > 0
        assert kinematic['x'][0] > 0
        assert kinematic['y'][0] > 0
        assert kinematic['speed'][0] == 0  # 第一幀速度為 0
    
    def test_speed_calculation(self, roi_manager, sample_masks):
        """測試速度計算"""
        for i, mask in enumerate(sample_masks):
            roi_manager[i] = mask
        
        kinematic = roi_manager.get_kinematic_data()
        speeds = kinematic['speed']
        
        # 第一幀速度應為 0
        assert speeds[0] == 0
        
        # 後續幀應有速度
        for i in range(1, len(speeds)):
            assert speeds[i] > 0
    
    def test_trajectory(self, roi_manager, sample_masks):
        """測試軌跡獲取"""
        for i, mask in enumerate(sample_masks):
            roi_manager[i] = mask
        
        x_coords, y_coords = roi_manager.get_trajectory()
        
        assert len(x_coords) == len(sample_masks)
        assert len(y_coords) == len(sample_masks)
        
        # 檢查軌跡是否遞增（因為我們的測試資料是這樣設計的）
        for i in range(1, len(x_coords)):
            assert x_coords[i] > x_coords[i-1]
            assert y_coords[i] > y_coords[i-1]
    
    def test_speed_profile(self, roi_manager, sample_masks):
        """測試速度分布"""
        for i, mask in enumerate(sample_masks):
            roi_manager[i] = mask
        
        frames, speeds = roi_manager.get_speed_profile()
        
        assert len(frames) == len(sample_masks)
        assert len(speeds) == len(sample_masks)
        np.testing.assert_array_equal(frames, np.arange(len(sample_masks)))
    
    def test_area_profile(self, roi_manager, sample_masks):
        """測試面積變化"""
        for i, mask in enumerate(sample_masks):
            roi_manager[i] = mask
        
        frames, areas = roi_manager.get_area_profile()
        
        assert len(frames) == len(sample_masks)
        assert len(areas) == len(sample_masks)
        
        # 所有面積應該相近（因為圓的大小沒變）
        assert np.std(areas) < 10  # 允許小誤差
    
    def test_real_speed_profile(self, roi_manager_with_meta, sample_masks):
        """測試實際速度計算（像素/秒）"""
        for i, mask in enumerate(sample_masks):
            roi_manager_with_meta[i] = mask
        
        time_seconds, real_speeds = roi_manager_with_meta.get_real_speed_profile()
        
        assert len(time_seconds) == len(sample_masks)
        assert len(real_speeds) == len(sample_masks)
        
        # 時間應該符合 fps
        expected_times = np.arange(len(sample_masks)) / 30.0
        np.testing.assert_array_almost_equal(time_seconds, expected_times)
    
    def test_time_trajectory(self, roi_manager_with_meta, sample_masks):
        """測試基於時間的軌跡"""
        for i, mask in enumerate(sample_masks):
            roi_manager_with_meta[i] = mask
        
        time_seconds, x_coords, y_coords = roi_manager_with_meta.get_time_trajectory()
        
        assert len(time_seconds) == len(sample_masks)
        assert len(x_coords) == len(sample_masks)
        assert len(y_coords) == len(sample_masks)


# ==================== 批次操作測試 ====================

class TestBatchOperations:
    """批次操作測試"""
    
    def test_batch_operations_context(self, roi_manager, sample_masks):
        """測試批次操作 context manager"""
        original_threshold = roi_manager.auto_reset_threshold
        
        with roi_manager.batch_operations():
            # 在批次操作中，閾值應該是無限大
            assert roi_manager.auto_reset_threshold == float('inf')
            
            # 執行大量操作
            for i, mask in enumerate(sample_masks * 100):  # 1000 個操作
                roi_manager[i] = mask
        
        # 批次操作結束後，閾值應該恢復
        assert roi_manager.auto_reset_threshold == original_threshold
    
    def test_batch_vs_normal_performance(self, temp_h5_file, sample_masks):
        """比較批次操作和正常操作的效能"""
        # 準備大量資料
        many_masks = sample_masks * 20  # 200 個 masks
        
        # 正常操作
        manager1 = ROIManager(temp_h5_file, auto_reset_threshold=50)
        start = time.time()
        for i, mask in enumerate(many_masks):
            manager1[i] = mask
        normal_time = time.time() - start
        manager1.close()
        
        # 清理檔案
        temp_h5_file.unlink()
        
        # 批次操作
        manager2 = ROIManager(temp_h5_file, auto_reset_threshold=50)
        start = time.time()
        with manager2.batch_operations():
            for i, mask in enumerate(many_masks):
                manager2[i] = mask
        batch_time = time.time() - start
        manager2.close()
        
        # 批次操作應該更快（或至少不慢太多）
        # 注意：在小資料量下可能看不出差異
        assert batch_time <= normal_time * 1.5


# ==================== 錯誤處理測試 ====================

class TestErrorHandling:
    """錯誤處理測試"""
    
    def test_invalid_mask_type(self, roi_manager):
        """測試無效的遮罩類型"""
        with pytest.raises(ValueError, match="遮罩必須是 numpy 陣列"):
            roi_manager[0] = "not an array"
        
        with pytest.raises(ValueError, match="遮罩必須是 numpy 陣列"):
            roi_manager[0] = [1, 2, 3]
    
    def test_invalid_mask_dimension(self, roi_manager):
        """測試無效的遮罩維度"""
        with pytest.raises(ValueError, match="遮罩至少需要是 2D 陣列"):
            roi_manager[0] = np.array([1, 2, 3])
    
    def test_read_non_existent_mask(self, roi_manager):
        """測試讀取不存在的遮罩"""
        with pytest.raises(KeyError, match="索引 999 的遮罩不存在"):
            _ = roi_manager[999]
    
    def test_read_non_existent_config(self, roi_manager):
        """測試讀取不存在的配置"""
        with pytest.raises(KeyError, match="配置鍵 'non_existent' 不存在"):
            roi_manager.read_config('non_existent')
    
    def test_negative_roi_count(self, roi_manager):
        """測試負數 ROI 數量"""
        with pytest.raises(ValueError, match="ROI 數量不能為負數"):
            roi_manager.set_n_rois(-1)
    
    def test_negative_frame_count(self, roi_manager):
        """測試負數影格數量"""
        with pytest.raises(ValueError, match="總影格數量不能為負數"):
            roi_manager.set_total_frames(-1)
    
    def test_invalid_file_path(self):
        """測試無效的檔案路徑"""
        with pytest.raises(ROIManagerError, match="無法初始化 ROI Manager"):
            ROIManager("/invalid/path/that/does/not/exist/file.h5")
    
    def test_mask_type_warning(self, roi_manager):
        """測試遮罩類型警告"""
        # float64 應該產生警告
        float_mask = np.zeros((10, 10), dtype=np.float64)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            roi_manager[0] = float_mask
            assert len(w) == 1
            assert "可能不是最佳選擇" in str(w[0].message)


# ==================== 查詢和資訊測試 ====================

class TestQueryAndInfo:
    """查詢和資訊相關測試"""
    
    def test_get_mask_indices(self, roi_manager, sample_mask):
        """測試獲取遮罩索引"""
        # 寫入一些遮罩
        indices_to_write = [0, 5, 10, 15, 20]
        for idx in indices_to_write:
            roi_manager[idx] = sample_mask
        
        # 獲取索引
        indices = roi_manager.get_mask_indices()
        assert indices == indices_to_write
    
    def test_get_mask_shape(self, roi_manager, sample_mask):
        """測試獲取遮罩形狀"""
        roi_manager[0] = sample_mask
        
        shape = roi_manager.get_mask_shape(0)
        assert shape == sample_mask.shape
        
        # 不存在的遮罩
        assert roi_manager.get_mask_shape(999) is None
    
    def test_get_file_info(self, roi_manager_with_meta, sample_masks):
        """測試獲取檔案資訊"""
        # 寫入一些資料
        for i, mask in enumerate(sample_masks):
            roi_manager_with_meta[i] = mask
        
        roi_manager_with_meta.set_n_rois(3)
        roi_manager_with_meta.write_config('custom_config', 'value')
        
        info = roi_manager_with_meta.get_file_info()
        
        assert info['file_exists'] is True
        assert info['n_rois'] == 3
        assert info['mask_count'] == len(sample_masks)
        assert 'custom_config' in info['config_keys']
        assert info['compression_level'] == 3
        assert info['kinematic_data_count'] == len(sample_masks)
        
        # Meta 資訊
        assert 'meta' in info
        assert info['meta']['video_fps'] == 30.0
    
    def test_get_kinematic_data_with_filter(self, roi_manager, sample_masks):
        """測試篩選特定影格的 kinematic 資料"""
        # 寫入所有遮罩
        for i, mask in enumerate(sample_masks):
            roi_manager[i] = mask
        
        # 只獲取特定影格
        selected_frames = [2, 4, 6]
        kinematic = roi_manager.get_kinematic_data(selected_frames)
        
        assert len(kinematic['frame_indices']) == len(selected_frames)
        np.testing.assert_array_equal(
            kinematic['frame_indices'],
            selected_frames
        )


# ==================== 修改操作測試 ====================

class TestModificationOperations:
    """修改操作測試"""
    
    def test_delete_mask(self, roi_manager, sample_mask):
        """測試刪除遮罩"""
        # 寫入遮罩
        roi_manager[0] = sample_mask
        assert roi_manager.has_mask(0)
        
        # 刪除遮罩
        success = roi_manager.delete_mask(0)
        assert success is True
        assert not roi_manager.has_mask(0)
        
        # 刪除不存在的遮罩
        success = roi_manager.delete_mask(999)
        assert success is False
    
    def test_clear_all_masks(self, roi_manager, sample_masks):
        """測試清除所有遮罩"""
        # 寫入多個遮罩
        for i, mask in enumerate(sample_masks):
            roi_manager[i] = mask
        
        # 確認 kinematic 資料存在
        kinematic_before = roi_manager.get_kinematic_data()
        assert len(kinematic_before['frame_indices']) == len(sample_masks)
        
        # 清除所有
        roi_manager.clear_all_masks()
        
        # 檢查遮罩都被清除
        assert len(roi_manager.get_mask_indices()) == 0
        
        # 檢查 kinematic 資料也被清除
        kinematic_after = roi_manager.get_kinematic_data()
        assert len(kinematic_after['frame_indices']) == 0
    
    def test_overwrite_mask(self, roi_manager, sample_mask):
        """測試覆寫遮罩"""
        # 寫入原始遮罩
        roi_manager[0] = sample_mask
        
        # 建立不同大小的新遮罩
        new_mask = np.ones((50, 50), dtype=np.uint8) * 128
        
        # 覆寫
        roi_manager[0] = new_mask
        
        # 檢查
        retrieved = roi_manager[0]
        assert retrieved.shape == new_mask.shape
        np.testing.assert_array_equal(retrieved, new_mask)
    
    def test_config_overwrite(self, roi_manager):
        """測試配置覆寫"""
        roi_manager.write_config('test_key', 'original')
        assert roi_manager.read_config('test_key') == 'original'
        
        roi_manager.write_config('test_key', 'updated')
        assert roi_manager.read_config('test_key') == 'updated'


# ==================== 資源管理測試 ====================

class TestResourceManagement:
    """資源管理測試"""
    
    def test_flush(self, roi_manager, sample_mask):
        """測試強制寫入磁碟"""
        roi_manager[0] = sample_mask
        roi_manager.flush()
        
        # 檢查資料是否真的被寫入
        # 建立新的 manager 來讀取
        manager2 = ROIManager(roi_manager.file_path)
        assert manager2.has_mask(0)
        manager2.close()
    
    def test_close_and_reopen(self, temp_h5_file, sample_mask):
        """測試關閉和重新開啟"""
        # 第一次開啟並寫入
        manager1 = ROIManager(temp_h5_file)
        manager1[0] = sample_mask
        manager1.close()
        
        # 第二次開啟並讀取
        manager2 = ROIManager(temp_h5_file)
        retrieved = manager2[0]
        np.testing.assert_array_equal(retrieved, sample_mask)
        manager2.close()
    
    def test_auto_reset(self, temp_h5_file, sample_mask):
        """測試自動重置機制"""
        # 使用很小的閾值
        manager = ROIManager(temp_h5_file, auto_reset_threshold=5)
        
        # 執行多次操作觸發重置
        for i in range(10):
            manager[i] = sample_mask
            # 應該在第 5 次操作後重置
        
        # 檢查資料完整性
        for i in range(10):
            assert manager.has_mask(i)
        
        manager.close()
    
    def test_destructor(self, temp_h5_file, sample_mask):
        """測試解構函數"""
        manager = ROIManager(temp_h5_file)
        manager[0] = sample_mask
        
        # 刪除物件，應該自動關閉
        del manager
        
        # 檢查檔案是否可以被其他程序開啟
        manager2 = ROIManager(temp_h5_file)
        assert manager2.has_mask(0)
        manager2.close()


# ==================== 向後相容性測試 ====================

# class TestBackwardCompatibility:
#     """向後相容性測試"""
    
#     def test_h5io_alias(self, temp_h5_file):
#         """測試 H5IO 別名"""
#         # 使用舊名稱
#         manager = H5IO(temp_h5_file)
#         assert isinstance(manager, ROIManager)
#         manager.close()
    
#     def test_legacy_usage(self, temp_h5_file, sample_mask):
#         """測試舊版用法"""
#         # 模擬舊版本的使用方式
#         h5io = H5IO(temp_h5_file)
#         h5io.write_mask(0, sample_mask)
#         h5io.write_config('n_rois', 1)
#         h5io.write_config('total_frames', 100)
        
#         # 讀取
#         mask = h5io.read_mask(0)
#         n_rois = h5io.read_config('n_rois')
#         total_frames = h5io.read_config('total_frames')
        
#         np.testing.assert_array_equal(mask, sample_mask)
#         assert n_rois == 1
#         assert total_frames == 100
        
#         h5io.close()


# ==================== 邊界條件測試 ====================

class TestEdgeCases:
    """邊界條件測試"""
    
    def test_empty_mask(self, roi_manager):
        """測試空遮罩"""
        empty_mask = np.zeros((10, 10), dtype=np.uint8)
        roi_manager[0] = empty_mask
        
        # 檢查 kinematic 資料
        kinematic = roi_manager.get_kinematic_data()
        assert kinematic['area'][0] == 0
        assert kinematic['x'][0] == 0
        assert kinematic['y'][0] == 0
    
    def test_large_mask(self, roi_manager):
        """測試大型遮罩"""
        large_mask = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
        roi_manager[0] = large_mask
        
        retrieved = roi_manager[0]
        np.testing.assert_array_equal(retrieved, large_mask)
    
    def test_single_pixel_mask(self, roi_manager):
        """測試單像素遮罩"""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 255
        
        roi_manager[0] = mask
        
        kinematic = roi_manager.get_kinematic_data()
        assert kinematic['area'][0] == 1
        assert kinematic['x'][0] == 5
        assert kinematic['y'][0] == 5
    
    def test_non_contiguous_frames(self, roi_manager, sample_mask):
        """測試非連續影格"""
        # 寫入非連續的影格
        frames = [0, 10, 20, 100, 500]
        for frame in frames:
            roi_manager[frame] = sample_mask
        
        indices = roi_manager.get_mask_indices()
        assert indices == frames
    
    def test_unicode_config_keys(self, roi_manager):
        """測試 Unicode 配置鍵"""
        roi_manager.write_config('中文鍵', '中文值')
        assert roi_manager.read_config('中文鍵') == '中文值'
        
        roi_manager.write_config('émoji_🎉', 'party')
        assert roi_manager.read_config('émoji_🎉') == 'party'


# ==================== 整合測試 ====================

class TestIntegration:
    """整合測試 - 測試完整工作流程"""
    
    def test_complete_video_processing_workflow(self, temp_h5_file):
        """測試完整的影片處理工作流程"""
        # 1. 初始化
        manager = ROIManager(
            temp_h5_file,
            video_basename="test_video.mp4",
            video_length=100,
            video_height=480,
            video_width=640,
            video_fps=25.0
        )
        
        # 2. 批次寫入遮罩
        with manager.batch_operations():
            for i in range(100):
                # 模擬移動的物體
                mask = np.zeros((480, 640), dtype=np.uint8)
                center_x = 320 + int(100 * np.sin(i * 0.1))
                center_y = 240 + int(50 * np.cos(i * 0.1))
                
                # 建立圓形遮罩
                y, x = np.ogrid[:480, :640]
                circle = (x - center_x)**2 + (y - center_y)**2 <= 30**2
                mask[circle] = 255
                
                manager[i] = mask
        
        # 3. 分析結果
        # 檢查所有遮罩都被寫入
        assert len(manager.get_mask_indices()) == 100
        
        # 檢查軌跡
        x_coords, y_coords = manager.get_trajectory()
        assert len(x_coords) == 100
        
        # 檢查速度
        _, speeds = manager.get_speed_profile()
        assert speeds[0] == 0  # 第一幀
        assert np.mean(speeds[1:]) > 0  # 有移動
        
        # 檢查實際速度（像素/秒）
        time_s, real_speeds = manager.get_real_speed_profile()
        assert len(time_s) == 100
        assert time_s[-1] == pytest.approx(99 / 25.0, rel=1e-5)
        
        # 4. 儲存配置
        manager.write_config('processing_date', '2024-01-01')
        manager.write_config('processing_version', '1.0.0')
        
        # 5. 獲取檔案資訊
        info = manager.get_file_info()
        assert info['mask_count'] == 100
        assert info['kinematic_data_count'] == 100
        assert 'processing_date' in info['config_keys']
        
        manager.close()
        
        # 6. 重新開啟並驗證
        manager2 = ROIManager(temp_h5_file)
        assert manager2.get_total_frames() == 100
        assert len(manager2.get_mask_indices()) == 100
        assert manager2.read_config('processing_date') == '2024-01-01'
        manager2.close()
    
    def test_multi_roi_tracking(self, temp_h5_file):
        """測試多 ROI 追蹤場景"""
        manager = ROIManager(temp_h5_file)
        manager.set_n_rois(3)  # 追蹤 3 個物體
        
        # 為每個 ROI 建立獨立的遮罩序列
        for roi_idx in range(3):
            for frame_idx in range(10):
                # 使用不同的命名策略來區分不同 ROI
                # 例如：frame_idx * 1000 + roi_idx
                combined_idx = frame_idx * 1000 + roi_idx
                
                # 建立遮罩
                mask = np.zeros((100, 100), dtype=np.uint8)
                center = (30 + roi_idx * 20, 30 + frame_idx * 5)
                y, x = np.ogrid[:100, :100]
                circle = (x - center[0])**2 + (y - center[1])**2 <= 10**2
                mask[circle] = 255
                
                manager[combined_idx] = mask
        
        # 驗證
        indices = manager.get_mask_indices()
        assert len(indices) == 30  # 3 ROIs × 10 frames
        
        # 檢查每個 ROI 的軌跡
        for roi_idx in range(3):
            roi_indices = [i for i in indices if i % 1000 == roi_idx]
            assert len(roi_indices) == 10
        
        manager.close()


# ==================== 效能測試 ====================

@pytest.mark.slow
class TestPerformance:
    """效能測試（標記為 slow，可選擇性執行）"""
    
    def test_large_dataset_performance(self, temp_h5_file):
        """測試大型資料集的效能"""
        manager = ROIManager(temp_h5_file)
        
        # 寫入 1000 個遮罩
        start = time.time()
        with manager.batch_operations():
            for i in range(1000):
                mask = np.random.randint(0, 2, (200, 200), dtype=np.uint8) * 255
                manager[i] = mask
        write_time = time.time() - start
        
        # 效能基準：應該在合理時間內完成
        assert write_time < 60  # 60 秒內完成
        
        # 讀取所有遮罩
        start = time.time()
        for i in range(1000):
            _ = manager[i]
        read_time = time.time() - start
        
        assert read_time < 30  # 30 秒內完成
        
        manager.close()
    
    def test_compression_impact(self, temp_h5_file, sample_mask):
        """測試壓縮等級對效能和大小的影響"""
        results = {}
        
        for compression_level in [1, 5, 9]:
            # 清理檔案
            if temp_h5_file.exists():
                temp_h5_file.unlink()
            
            manager = ROIManager(
                temp_h5_file, 
                compression_level=compression_level
            )
            
            # 寫入資料
            start = time.time()
            for i in range(100):
                manager[i] = sample_mask
            write_time = time.time() - start
            
            manager.close()
            
            # 記錄結果
            file_size = temp_h5_file.stat().st_size
            results[compression_level] = {
                'time': write_time,
                'size': file_size
            }
        
        # 驗證：更高的壓縮等級應該產生更小的檔案
        assert results[9]['size'] <= results[1]['size']
        
        # 但寫入時間可能更長
        # （這個斷言可能因硬體而異，所以只是記錄）
        print(f"Compression results: {results}")


# ==================== 執行設定 ====================

if __name__ == "__main__":
    # 執行所有測試
    pytest.main([__file__, "-v"])
    
    # 執行特定測試類別
    # pytest.main([__file__ + "::TestBasicFunctionality", "-v"])
    
    # 執行除了 slow 標記的測試
    # pytest.main([__file__, "-v", "-m", "not slow"])
    
    # 執行並生成覆蓋率報告
    # pytest.main([__file__, "--cov=roi_manager", "--cov-report=html"])