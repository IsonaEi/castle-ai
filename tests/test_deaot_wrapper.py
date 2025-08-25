"""
DeAOT Wrapper 測試模組
完整測試 DeAOTWrapper 類的各種功能，包括：
- 基本初始化和配置
- 多物件追蹤功能
- 與 video_io 模組的整合測試
- 實際影片追蹤測試
- 質心計算和軌跡輸出
- 性能測試和邊界情況測試
"""
import pytest
import numpy as np
import torch
import time
import json
from pathlib import Path
import tempfile
import logging
from dataclasses import asdict
import os

# 檢查可選套件是否可用
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# 環境變數來控制測試行為
IS_CI = os.environ.get('CI', 'false').lower() == 'true'
SKIP_HEAVY_TESTS = os.environ.get('SKIP_HEAVY_TESTS', 'false').lower() == 'true'

# Skip 條件
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 GPU 支援"
)

requires_pandas = pytest.mark.skipif(
    not PANDAS_AVAILABLE,
    reason="需要 pandas 套件"
)

requires_pillow = pytest.mark.skipif(
    not PIL_AVAILABLE,
    reason="需要 Pillow 套件"
)

requires_opencv = pytest.mark.skipif(
    not CV2_AVAILABLE,
    reason="需要 OpenCV 套件"
)

# CI 環境中跳過的測試
skip_on_ci = pytest.mark.skipif(
    IS_CI and SKIP_HEAVY_TESTS,
    reason="在 CI 環境中跳過重量級測試"
)

# 被測試的模組
from castle.models.deaot_wrapper import (
    DeAOTWrapper, ModelType, TrackingResult, ObjectTrack, 
    download_with_gdown, DEFAULT_DEVICE, AOT_AVAILABLE
)

# 可選的可視化工具
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation, PillowWriter
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# 嘗試導入額外的模組進行完整測試
try:
    from castle.utils.video_io import VideoWriter, VideoReader, VideoIO
    VIDEO_IO_AVAILABLE = True
except ImportError:
    VIDEO_IO_AVAILABLE = False

try:
    from castle.models.sam_wrapper import SAMWrapper, ModelSize
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class TestDataGenerator:
    """測試資料生成器"""
    
    @staticmethod
    def create_tracking_result(frame_idx=0, object_id=1, mask_size=(100, 100)):
        """創建 TrackingResult 測試資料"""
        mask = np.ones(mask_size, dtype=np.uint8)
        
        # 計算質心
        y_indices, x_indices = np.where(mask > 0)
        center_x = float(np.mean(x_indices))
        center_y = float(np.mean(y_indices))
        
        return TrackingResult(
            frame_idx=frame_idx,
            object_id=object_id,
            mask=mask,
            confidence=0.95,
            area=int(np.sum(mask)),
            center=(center_x, center_y)
        )
    
    @staticmethod
    def create_object_track(object_id=1, num_frames=5):
        """創建 ObjectTrack 測試資料"""
        masks = {}
        confidences = {}
        
        for i in range(num_frames):
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[20+i:50+i, 20+i:50+i] = 1
            
            masks[i] = mask
            confidences[i] = 0.9 - i * 0.1
        
        return ObjectTrack(
            object_id=object_id,
            start_frame=0,
            end_frame=num_frames-1,
            masks=masks,
            confidences=confidences
        )
    
    @staticmethod
    def create_sample_frames(num_frames=5, height=480, width=640):
        """生成測試用的樣本影像序列"""
        frames = []
        for i in range(num_frames):
            # 創建測試影像：漸變背景上移動的物件
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 創建漸變背景
            gradient = np.linspace(100, 200, width)
            image[:, :, 0] = gradient[np.newaxis, :]  # 紅色通道
            image[:, :, 1] = 150  # 綠色通道
            image[:, :, 2] = np.linspace(200, 100, height)[:, np.newaxis]  # 藍色通道
            
            # 物件 1: 移動的圓形
            center_x = 150 + i * 30
            center_y = 200 + i * 10
            radius = 40
            
            y, x = np.ogrid[:height, :width]
            mask1 = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            image[mask1] = [255, 100, 100]  # 紅色圓形
            
            # 物件 2: 移動的矩形
            rect_x = 400 - i * 20
            rect_y = 300 + i * 15
            rect_w, rect_h = 60, 80
            
            x1, x2 = max(0, rect_x), min(width, rect_x + rect_w)
            y1, y2 = max(0, rect_y), min(height, rect_y + rect_h)
            image[y1:y2, x1:x2] = [100, 255, 100]  # 綠色矩形
            
            frames.append(image)
        
        return frames
    
    @staticmethod
    def create_combined_mask(frame_idx=0, height=480, width=640):
        """創建合併的標籤遮罩"""
        combined = np.zeros((height, width), dtype=np.uint8)
        
        # 物件 1: 圓形遮罩
        center_x = 150 + frame_idx * 30
        center_y = 200 + frame_idx * 10
        radius = 40
        
        y, x = np.ogrid[:height, :width]
        circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        combined[circle_mask] = 1
        
        # 物件 2: 矩形遮罩
        rect_x = 400 - frame_idx * 20
        rect_y = 300 + frame_idx * 15
        rect_w, rect_h = 60, 80
        
        x1, x2 = max(0, rect_x), min(width, rect_x + rect_w)
        y1, y2 = max(0, rect_y), min(height, rect_y + rect_h)
        combined[y1:y2, x1:x2] = 2
            
        return combined


class TestDeAOTWrapperDataStructures:
    """測試資料結構和枚舉"""
    
    def test_model_type_enum(self):
        """測試 ModelType 枚舉"""
        # 測試枚舉值
        assert ModelType.R50_DEAOTL.value == "r50_deaotl"
        
        # 測試字符串轉換
        assert ModelType("r50_deaotl") == ModelType.R50_DEAOTL
        
        # 測試無效類型
        with pytest.raises(ValueError):
            ModelType("invalid_model")
    
    def test_tracking_result_structure(self):
        """測試 TrackingResult 資料結構"""
        result = TestDataGenerator.create_tracking_result(
            frame_idx=5, object_id=2, mask_size=(50, 60)
        )
        
        assert result.frame_idx == 5
        assert result.object_id == 2
        assert result.mask.shape == (50, 60)
        assert result.confidence == 0.95
        assert result.area == 3000  # 50*60
        assert isinstance(result.center, tuple)
        assert len(result.center) == 2
        
        # 測試序列化
        result_dict = asdict(result)
        assert 'frame_idx' in result_dict
        assert 'object_id' in result_dict
        assert 'confidence' in result_dict
    
    def test_object_track_structure(self):
        """測試 ObjectTrack 資料結構"""
        track = TestDataGenerator.create_object_track(object_id=3, num_frames=4)
        
        assert track.object_id == 3
        assert track.start_frame == 0
        assert track.end_frame == 3
        assert len(track.masks) == 4
        assert len(track.confidences) == 4
        
        # 檢查資料一致性
        for frame_idx in range(4):
            assert frame_idx in track.masks
            assert frame_idx in track.confidences
            assert track.confidences[frame_idx] <= 1.0
            assert track.confidences[frame_idx] >= 0.0


@pytest.mark.skipif(not AOT_AVAILABLE, reason="AOT 模組未安裝")
class TestDeAOTWrapperInitialization:
    """測試 DeAOT Wrapper 初始化（需要 AOT 模組）"""
    
    @pytest.mark.parametrize("model_type", [
        ModelType.R50_DEAOTL,
    ])
    @skip_on_ci
    def test_wrapper_initialization_with_models(self, model_type):
        """測試使用不同模型類型初始化"""
        device = 'cpu'  # 使用 CPU 以避免 GPU 依賴
        
        try:
            tracker = DeAOTWrapper(
                model_type=model_type,
                device=device,
                long_term_mem_gap=5000,
                max_len_long_term=10
            )
            
            assert tracker.model_type == model_type
            assert tracker.device == device
            assert tracker.long_term_mem_gap == 5000
            assert tracker.max_len_long_term == 10
            
            # 檢查初始狀態
            assert tracker.current_frame_idx == 0
            assert len(tracker.object_tracks) == 0
            assert len(tracker.active_objects) == 0
            assert tracker.obj_nums == 0
            assert tracker._reference_frame_set == False
            
            # 清理
            tracker.clear_memory()
            del tracker
            
        except FileNotFoundError:
            pytest.skip("模型權重檔案不存在")
    
    def test_device_selection(self):
        """測試設備選擇邏輯"""
        # 測試指定 CPU
        try:
            tracker = DeAOTWrapper(device='cpu')
            assert tracker.device == 'cpu'
            tracker.clear_memory()
            del tracker
        except FileNotFoundError:
            pytest.skip("模型權重檔案不存在")
        
        # 測試自動選擇
        try:
            tracker = DeAOTWrapper(device=None)
            assert tracker.device in ['cpu', 'cuda', 'mps']
            tracker.clear_memory()
            del tracker
        except FileNotFoundError:
            pytest.skip("模型權重檔案不存在")


@pytest.mark.skipif(not AOT_AVAILABLE, reason="AOT 模組未安裝")
class TestDeAOTWrapperTracking:
    """測試追蹤功能（需要 AOT 模組）"""
    
    @pytest.fixture
    def tracker(self):
        """創建測試用的追蹤器"""
        try:
            tracker = DeAOTWrapper(
                model_type=ModelType.R50_DEAOTL,
                device='cpu'
            )
            yield tracker
            tracker.clear_memory()
        except FileNotFoundError:
            pytest.skip("模型權重檔案不存在")
    
    @skip_on_ci
    def test_reference_frame_setup(self, tracker):
        """測試參考幀設置"""
        frames = TestDataGenerator.create_sample_frames(1)
        mask = TestDataGenerator.create_combined_mask()
        
        # 設置參考幀
        tracker.add_reference_frame(frames[0], mask, obj_nums=2)
        
        assert tracker._reference_frame_set == True
        assert tracker.obj_nums == 2
        assert len(tracker.active_objects) > 0
    
    @skip_on_ci
    def test_single_frame_tracking(self, tracker):
        """測試單幀追蹤"""
        frames = TestDataGenerator.create_sample_frames(2)
        mask = TestDataGenerator.create_combined_mask(0)
        
        # 設置參考幀
        tracker.add_reference_frame(frames[0], mask, obj_nums=2)
        
        # 追蹤下一幀
        results = tracker.track_frame(frames[1])
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, TrackingResult)
            assert result.frame_idx == 1
            assert result.mask is not None
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0
    
    @skip_on_ci
    def test_sequence_tracking(self, tracker):
        """測試序列追蹤"""
        frames = TestDataGenerator.create_sample_frames(5)
        mask = TestDataGenerator.create_combined_mask(0)
        
        # 設置參考幀
        tracker.add_reference_frame(frames[0], mask, obj_nums=2)
        
        # 追蹤序列
        tracks = tracker.track_sequence(frames[1:])
        
        assert isinstance(tracks, dict)
        assert len(tracks) > 0
        
        for obj_id, track in tracks.items():
            assert isinstance(track, ObjectTrack)
            assert track.object_id == obj_id
            assert len(track.masks) > 0
            assert len(track.confidences) == len(track.masks)
    
    def test_tracking_without_reference(self, tracker):
        """測試未設置參考幀時追蹤的錯誤處理"""
        frames = TestDataGenerator.create_sample_frames(1)
        
        with pytest.raises(RuntimeError, match="必須先調用 add_reference_frame"):
            tracker.track_frame(frames[0])
    
    def test_restart_functionality(self, tracker):
        """測試重啟功能"""
        frames = TestDataGenerator.create_sample_frames(2)
        mask = TestDataGenerator.create_combined_mask(0)
        
        # 設置初始狀態
        tracker.add_reference_frame(frames[0], mask, obj_nums=2)
        tracker.track_frame(frames[1])
        
        # 重啟
        tracker.restart()
        
        # 檢查狀態重置
        assert tracker.current_frame_idx == 0
        assert tracker._reference_frame_set == False
        assert len(tracker.object_tracks) == 0
        assert len(tracker.active_objects) == 0


class TestDeAOTWrapperUtils:
    """測試工具函數"""
    
    def test_download_function(self, tmp_path):
        """測試下載功能（不實際下載）"""
        from unittest.mock import patch, MagicMock
        
        test_file = tmp_path / "test_model.pth"
        
        with patch('gdown.download') as mock_download, \
             patch('os.path.isfile') as mock_isfile:
            
            # 測試檔案不存在時下載
            mock_isfile.return_value = False
            download_with_gdown("test_id", str(test_file))
            mock_download.assert_called_once()
            
            # 測試檔案存在時不下載
            mock_download.reset_mock()
            mock_isfile.return_value = True
            download_with_gdown("test_id", str(test_file))
            mock_download.assert_not_called()


@pytest.mark.skipif(not AOT_AVAILABLE, reason="AOT 模組未安裝")
@skip_on_ci
class TestDeAOTWrapperPerformance:
    """性能測試"""
    
    @pytest.fixture
    def tracker(self):
        """創建測試用的追蹤器"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            tracker = DeAOTWrapper(
                model_type=ModelType.R50_DEAOTL,
                device=device
            )
            yield tracker
            tracker.clear_memory()
        except FileNotFoundError:
            pytest.skip("模型權重檔案不存在")
    
    def test_tracking_speed(self, tracker):
        """測試追蹤速度"""
        frames = TestDataGenerator.create_sample_frames(10, height=240, width=320)
        mask = TestDataGenerator.create_combined_mask(0, height=240, width=320)
        
        # 設置參考幀
        tracker.add_reference_frame(frames[0], mask, obj_nums=2)
        
        # 測量追蹤時間
        start_time = time.time()
        for frame in frames[1:]:
            tracker.track_frame(frame)
        elapsed_time = time.time() - start_time
        
        # 計算 FPS
        fps = len(frames[1:]) / elapsed_time
        logging.info(f"追蹤速度: {fps:.2f} FPS")
        
        # 基本性能要求（至少 1 FPS）
        assert fps > 1.0
    
    def test_memory_usage(self, tracker):
        """測試記憶體使用"""
        frames = TestDataGenerator.create_sample_frames(20)
        mask = TestDataGenerator.create_combined_mask(0)
        
        # 設置參考幀
        tracker.add_reference_frame(frames[0], mask, obj_nums=2)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # 追蹤多幀
        for frame in frames[1:]:
            tracker.track_frame(frame)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_mb = (peak_memory - initial_memory) / (1024 * 1024)
            logging.info(f"記憶體使用: {memory_mb:.2f} MB")
            
            # 記憶體使用應該在合理範圍內（< 16GB）
            assert memory_mb < 16384


@pytest.mark.skipif(not AOT_AVAILABLE, reason="AOT 模組未安裝")
@pytest.mark.skipif(not MPL_AVAILABLE, reason="需要 matplotlib")
class TestDeAOTWrapperVisualization:
    """DeAOT 視覺化測試 - 生成影片和追蹤結果供人工檢查"""
    
    @pytest.fixture
    def output_dir(self):
        """測試輸出目錄"""
        # 使用當前專案目錄下的 tmp 目錄
        project_root = Path(__file__).parent.parent
        tmp_dir = project_root / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        
        # 在 tmp 目錄下創建唯一的測試目錄
        temp_dir = tempfile.mkdtemp(prefix="deaot_test_", dir=str(tmp_dir))
        output_dir = Path(temp_dir)
        print(f"測試輸出目錄: {output_dir}")
        return output_dir
    
    def _generate_synthetic_video_frames(self, num_frames=30, width=640, height=480):
        """生成合成測試影片的幀序列，包含移動的圓形和矩形"""
        frames = []
        masks = []
        
        for i in range(num_frames):
            # 創建影像：漸變背景
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 漸變背景
            gradient = np.linspace(50, 150, width)
            image[:, :, 0] = gradient[np.newaxis, :]  # 紅色通道
            image[:, :, 1] = 100  # 綠色通道
            image[:, :, 2] = np.linspace(150, 50, height)[:, np.newaxis]  # 藍色通道
            
            # 創建對應的遮罩
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # 物件 1: 移動的圓形（順時針圓周運動）
            center_x = width // 2 + int(150 * np.cos(2 * np.pi * i / num_frames))
            center_y = height // 2 + int(150 * np.sin(2 * np.pi * i / num_frames))
            radius = 40
            
            y, x = np.ogrid[:height, :width]
            circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            image[circle_mask] = [255, 100, 100]  # 紅色圓形
            mask[circle_mask] = 1  # 物件 ID 1
            
            # 物件 2: 移動的矩形（左右移動）
            rect_x = int(100 + 200 * (0.5 + 0.5 * np.sin(4 * np.pi * i / num_frames)))
            rect_y = height // 2 + 100
            rect_w, rect_h = 60, 80
            
            x1, x2 = max(0, rect_x), min(width, rect_x + rect_w)
            y1, y2 = max(0, rect_y), min(height, rect_y + rect_h)
            image[y1:y2, x1:x2] = [100, 255, 100]  # 綠色矩形
            mask[y1:y2, x1:x2] = 2  # 物件 ID 2
            
            # 物件 3: 移動的三角形（對角線移動）
            tri_center_x = int(100 + i * (width - 200) / num_frames)
            tri_center_y = int(100 + i * (height - 200) / num_frames)
            tri_size = 30
            
            # 創建三角形遮罩
            tri_points = np.array([
                [tri_center_x, tri_center_y - tri_size],
                [tri_center_x - tri_size, tri_center_y + tri_size],
                [tri_center_x + tri_size, tri_center_y + tri_size]
            ])
            
            # 使用多邊形填充創建三角形
            for y in range(max(0, tri_center_y - tri_size), min(height, tri_center_y + tri_size + 1)):
                for x in range(max(0, tri_center_x - tri_size), min(width, tri_center_x + tri_size + 1)):
                    if self._point_in_triangle(x, y, tri_points):
                        image[y, x] = [100, 100, 255]  # 藍色三角形
                        mask[y, x] = 3  # 物件 ID 3
            
            frames.append(image)
            masks.append(mask)
        
        return frames, masks
    
    def _point_in_triangle(self, px, py, triangle):
        """檢查點是否在三角形內"""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign([px, py], triangle[0], triangle[1])
        d2 = sign([px, py], triangle[1], triangle[2])
        d3 = sign([px, py], triangle[2], triangle[0])
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    def _save_tracking_visualization(self, frames, tracks, output_dir, video_name="tracking_result"):
        """保存追蹤結果的可視化"""
        num_frames = len(frames)
        
        # 為每個物件分配顏色
        colors = {
            1: [255, 0, 0],     # 紅色
            2: [0, 255, 0],     # 綠色  
            3: [0, 0, 255],     # 藍色
            4: [255, 255, 0],   # 黃色
            5: [255, 0, 255],   # 洋紅色
        }
        
        # 創建可視化幀
        viz_frames = []
        
        for frame_idx in range(num_frames):
            # 複製原始幀
            viz_frame = frames[frame_idx].copy()
            
            # 疊加追蹤遮罩
            overlay = np.zeros_like(viz_frame)
            
            for obj_id, track in tracks.items():
                if frame_idx in track.masks:
                    mask = track.masks[frame_idx]
                    color = colors.get(obj_id, [255, 255, 255])
                    overlay[mask > 0] = color
            
            # 混合原始影像和遮罩
            alpha = 0.5
            viz_frame = (viz_frame * (1 - alpha) + overlay * alpha).astype(np.uint8)
            
            # 添加文字資訊
            viz_frame = self._add_text_to_frame(viz_frame, frame_idx, tracks)
            
            viz_frames.append(viz_frame)
        
        # 保存為影片（如果 VIDEO_IO_AVAILABLE）
        if VIDEO_IO_AVAILABLE:
            video_path = output_dir / f"{video_name}.mp4"
            with VideoWriter(str(video_path), fps=10.0, crf=18) as writer:
                for frame in viz_frames:
                    writer.write_frame(frame)
            print(f"追蹤視覺化影片已保存: {video_path}")
        else:
            print("警告: VIDEO_IO 不可用，跳過影片保存")
            video_path = None
        
        # 同時保存關鍵幀的靜態圖片
        self._save_key_frames(frames, tracks, output_dir, video_name)
        
        return video_path
    
    def _add_text_to_frame(self, frame, frame_idx, tracks):
        """在幀上添加文字資訊（使用 matplotlib）"""
        # 因為 OpenCV 不可用，使用 matplotlib 來添加文字
        fig, ax = plt.subplots(figsize=(frame.shape[1]/100, frame.shape[0]/100), dpi=100)
        ax.imshow(frame)
        
        # 添加幀編號
        ax.text(10, 30, f"Frame: {frame_idx}", color='white', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        # 添加物件追蹤資訊
        y_pos = 60
        for obj_id, track in tracks.items():
            if frame_idx in track.masks:
                confidence = track.confidences.get(frame_idx, 0.0)
                ax.text(10, y_pos, f"Obj {obj_id}: {confidence:.2f}", 
                       color='white', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                y_pos += 25
        
        ax.axis('off')
        
        # 轉換回 numpy array（兼容新版 matplotlib）
        fig.canvas.draw()
        try:
            # 嘗試使用新的 API
            buf = fig.canvas.buffer_rgba()
            result = np.asarray(buf)
            # 轉換 RGBA 到 RGB
            result = result[:, :, :3]
        except AttributeError:
            # 回退到舊的 API（如果存在）
            try:
                result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # 最後的回退方案：使用 PIL
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
                buf.seek(0)
                from PIL import Image
                pil_img = Image.open(buf)
                result = np.array(pil_img)[:, :, :3]  # 移除 alpha 通道
        
        plt.close(fig)
        
        return result
    
    def _save_key_frames(self, frames, tracks, output_dir, video_name):
        """保存關鍵幀的靜態圖片"""
        key_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
        
        fig, axes = plt.subplots(2, len(key_indices), figsize=(20, 8))
        
        for i, frame_idx in enumerate(key_indices):
            if frame_idx >= len(frames):
                continue
                
            # 原始幀
            axes[0, i].imshow(frames[frame_idx])
            axes[0, i].set_title(f'Frame {frame_idx}')
            axes[0, i].axis('off')
            
            # 追蹤結果
            combined_mask = np.zeros(frames[frame_idx].shape[:2])
            for obj_id, track in tracks.items():
                if frame_idx in track.masks:
                    mask = track.masks[frame_idx]
                    combined_mask[mask > 0] = obj_id
            
            axes[1, i].imshow(combined_mask, cmap='tab10')
            axes[1, i].set_title(f'Tracking Mask')
            axes[1, i].axis('off')
        
        plt.suptitle(f'{video_name} - Key Frames', fontsize=16)
        plt.tight_layout()
        
        key_frames_path = output_dir / f"{video_name}_key_frames.png"
        plt.savefig(key_frames_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"關鍵幀圖片已保存: {key_frames_path}")
    
    def _save_tracking_metrics(self, tracks, output_dir, video_name):
        """保存追蹤指標和統計資料"""
        metrics = {}
        
        for obj_id, track in tracks.items():
            # 計算每個物件的統計資料
            confidences = list(track.confidences.values())
            areas = []
            centers = []
            
            for frame_idx, mask in track.masks.items():
                area = np.sum(mask > 0)
                areas.append(area)
                
                # 計算質心
                if area > 0:
                    y_coords, x_coords = np.where(mask > 0)
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    centers.append((center_x, center_y))
            
            metrics[f"object_{obj_id}"] = {
                "start_frame": track.start_frame,
                "end_frame": track.end_frame,
                "total_frames": track.end_frame - track.start_frame + 1,
                "avg_confidence": float(np.mean(confidences)),
                "min_confidence": float(np.min(confidences)),
                "max_confidence": float(np.max(confidences)),
                "avg_area": float(np.mean(areas)),
                "min_area": int(np.min(areas)),
                "max_area": int(np.max(areas)),
                "num_centers": len(centers)
            }
        
        # 保存為 JSON
        metrics_path = output_dir / f"{video_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"追蹤指標已保存: {metrics_path}")
        
        # 創建並保存軌跡圖
        self._plot_trajectories(tracks, output_dir, video_name)
        
        return metrics
    
    def _plot_trajectories(self, tracks, output_dir, video_name):
        """繪製物件軌跡"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        
        for i, (obj_id, track) in enumerate(tracks.items()):
            centers = []
            
            for frame_idx in sorted(track.masks.keys()):
                mask = track.masks[frame_idx]
                if np.sum(mask > 0) > 0:
                    y_coords, x_coords = np.where(mask > 0)
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    centers.append((center_x, center_y))
            
            if centers:
                centers = np.array(centers)
                color = colors[i % len(colors)]
                ax.plot(centers[:, 0], centers[:, 1], color=color, 
                       linewidth=2, label=f'Object {obj_id}')
                ax.scatter(centers[0, 0], centers[0, 1], color=color, 
                          s=100, marker='o', label=f'Start {obj_id}')
                ax.scatter(centers[-1, 0], centers[-1, 1], color=color, 
                          s=100, marker='s', label=f'End {obj_id}')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'{video_name} - Object Trajectories')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # 影像座標系統
        
        trajectory_path = output_dir / f"{video_name}_trajectories.png"
        plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"軌跡圖已保存: {trajectory_path}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    @skip_on_ci
    def test_synthetic_video_tracking(self, output_dir):
        """測試合成影片的多物件追蹤 - 生成圓形、矩形、三角形移動的影片並追蹤"""
        print("\n=== 開始合成影片追蹤測試 ===")
        
        # 生成測試影片
        num_frames = 30
        frames, masks = self._generate_synthetic_video_frames(num_frames=num_frames)
        
        # 保存原始影片
        if VIDEO_IO_AVAILABLE:
            original_video_path = output_dir / "synthetic_original.mp4"
            with VideoWriter(str(original_video_path), fps=10.0, crf=18) as writer:
                for frame in frames:
                    writer.write_frame(frame)
            print(f"原始影片已保存: {original_video_path}")
        
        # 保存第一幀的標註遮罩可視化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(frames[0])
        axes[0].set_title('First Frame')
        axes[0].axis('off')
        
        axes[1].imshow(masks[0], cmap='tab10')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # 顯示每個物件
        for obj_id in [1, 2, 3]:
            obj_mask = (masks[0] == obj_id).astype(np.uint8)
            axes[2].imshow(obj_mask * obj_id, cmap='tab10', alpha=0.3 if obj_id > 1 else 1)
        axes[2].set_title('Objects Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        first_frame_path = output_dir / "synthetic_first_frame_annotation.png"
        plt.savefig(first_frame_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"第一幀標註已保存: {first_frame_path}")
        
        # 初始化 DeAOT 追蹤器
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            tracker = DeAOTWrapper(
                model_type=ModelType.R50_DEAOTL,
                device=device,
                long_term_mem_gap=2,
                max_len_long_term=5
            )
            print(f"DeAOT 追蹤器初始化成功 (device: {device})")
        except FileNotFoundError:
            pytest.skip("DeAOT 模型權重檔案不存在")
        
        # 設置參考幀（第一幀）
        tracker.add_reference_frame(frames[0], masks[0], obj_nums=3)
        print(f"參考幀設置完成，追蹤 3 個物件")
        
        # 執行追蹤
        print("開始追蹤序列...")
        tracks = tracker.track_sequence(frames[1:])
        print(f"追蹤完成，獲得 {len(tracks)} 個物件軌跡")
        
        # 保存追蹤結果可視化
        self._save_tracking_visualization(frames, tracks, output_dir, "synthetic_tracking")
        
        # 保存追蹤指標
        metrics = self._save_tracking_metrics(tracks, output_dir, "synthetic_tracking")
        
        # 驗證追蹤結果
        assert len(tracks) > 0, "應該至少追蹤到一個物件"
        
        for obj_id, track in tracks.items():
            assert len(track.masks) > 0, f"物件 {obj_id} 應該有追蹤遮罩"
            assert len(track.confidences) == len(track.masks), f"物件 {obj_id} 的置信度數量應該與遮罩數量一致"
            
            # 檢查追蹤品質
            avg_confidence = metrics[f"object_{obj_id}"]["avg_confidence"]
            print(f"物件 {obj_id}: 平均置信度 {avg_confidence:.3f}")
            
        print(f"\n測試結果已保存到: {output_dir}")
        print("請檢查生成的影片和圖片以人工確認追蹤效果")
        
        # 清理
        tracker.clear_memory()
        
        return {
            'output_dir': output_dir,
            'num_objects': len(tracks),
            'metrics': metrics
        }
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(not SAM_AVAILABLE, reason="需要 SAM 模組")
    @pytest.mark.skipif(not VIDEO_IO_AVAILABLE, reason="需要 VideoIO 模組")
    @skip_on_ci
    def test_real_video_tracking_with_sam(self, output_dir):
        """測試真實影片追蹤 - 使用 SAM 生成第一幀的遮罩，然後用 DeAOT 追蹤"""
        print("\n=== 開始真實影片追蹤測試 (SAM + DeAOT) ===")
        
        # 設定影片路徑
        project_root = Path(__file__).parent.parent
        video_path = project_root / "notebooks" / "open_field_videos" / "oft_1min.mp4"
        
        if not video_path.exists():
            pytest.skip(f"測試影片不存在: {video_path}")
        
        # 載入影片
        print(f"載入影片: {video_path}")
        video_reader = VideoIO.load_video(video_path)
        video_info = video_reader.get_info()
        print(f"影片資訊: {video_info.width}x{video_info.height}, "
              f"{video_info.fps}fps, {video_info.frame_count} frames")
        
        # 只處理前 60 幀（約 2 秒）以加快測試速度
        num_frames_to_process = min(60, video_info.frame_count)
        
        # 讀取影片幀
        frames = []
        for i in range(num_frames_to_process):
            frame = video_reader.get_frame(i)
            frames.append(frame)
        video_reader.close()
        print(f"已載入 {len(frames)} 幀")
        
        # 步驟 1: 使用 SAM 生成第一幀的分割遮罩
        print("\n步驟 1: 使用 SAM 生成第一幀分割...")
        try:
            sam = SAMWrapper(model_size=ModelSize.VIT_B, device='cuda')
            sam.set_image(frames[0])
            print("SAM 模型載入成功")
        except Exception as e:
            pytest.skip(f"無法初始化 SAM: {e}")
        
        # 定義多個點擊點來分割不同區域
        click_points = [
            (650, 600),  # 第一個物件
        ]
        
        combined_mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
        
        for obj_id, click_point in enumerate(click_points, start=1):
            print(f"  分割物件 {obj_id} (點擊位置: {click_point})")
            
            point_coords = np.array([click_point])
            point_labels = np.array([1])  # 前景點
            
            try:
                mask = sam.predict_with_points(point_coords, point_labels)
                if np.sum(mask) > 0:
                    combined_mask[mask] = obj_id
                    print(f"    物件 {obj_id} 分割成功，區域大小: {np.sum(mask)} 像素")
                else:
                    print(f"    物件 {obj_id} 未檢測到分割區域")
            except Exception as e:
                print(f"    物件 {obj_id} 分割失敗: {e}")
        
        sam.clear_cache()
        
        # 確保至少有一個物件被分割
        num_objects = len(np.unique(combined_mask)) - 1  # 減去背景
        if num_objects == 0:
            print("警告: SAM 未能分割出任何物件，使用預設遮罩")
            # 創建一個預設遮罩（中心區域）
            h, w = frames[0].shape[:2]
            combined_mask[h//3:2*h//3, w//3:2*w//3] = 1
            num_objects = 1
        
        print(f"SAM 分割完成，檢測到 {num_objects} 個物件")
        
        # 保存 SAM 分割結果
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(frames[0])
        axes[0].set_title('First Frame')
        axes[0].axis('off')
        
        axes[1].imshow(combined_mask, cmap='tab10')
        axes[1].set_title(f'SAM Segmentation ({num_objects} objects)')
        axes[1].axis('off')
        
        # 疊加顯示
        axes[2].imshow(frames[0])
        masked = np.ma.masked_where(combined_mask == 0, combined_mask)
        axes[2].imshow(masked, cmap='tab10', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # 標記點擊點
        for click_point in click_points[:num_objects]:
            for ax in axes:
                ax.plot(click_point[0], click_point[1], 'ro', 
                       markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        plt.tight_layout()
        sam_result_path = output_dir / "real_video_sam_segmentation.png"
        plt.savefig(sam_result_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"SAM 分割結果已保存: {sam_result_path}")
        
        # 步驟 2: 使用 DeAOT 進行追蹤
        print("\n步驟 2: 使用 DeAOT 進行追蹤...")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            tracker = DeAOTWrapper(
                model_type=ModelType.R50_DEAOTL,
                device=device,
                long_term_mem_gap=5,
                max_len_long_term=10
            )
            print(f"DeAOT 追蹤器初始化成功 (device: {device})")
        except FileNotFoundError:
            pytest.skip("DeAOT 模型權重檔案不存在")
        
        # 設置參考幀
        tracker.add_reference_frame(frames[0], combined_mask, obj_nums=num_objects)
        print(f"參考幀設置完成，開始追蹤 {num_objects} 個物件")
        
        # 執行追蹤
        print(f"追蹤 {len(frames)-1} 幀...")
        tracks = tracker.track_sequence(frames[1:])
        print(f"追蹤完成，獲得 {len(tracks)} 個物件軌跡")
        
        # 保存追蹤結果可視化
        self._save_tracking_visualization(frames, tracks, output_dir, "real_video_tracking")
        
        # 保存追蹤指標
        metrics = self._save_tracking_metrics(tracks, output_dir, "real_video_tracking")
        
        # 保存部分原始影片供比較
        original_clip_path = output_dir / "real_video_original_clip.mp4"
        with VideoWriter(str(original_clip_path), fps=video_info.fps, crf=18) as writer:
            for frame in frames:
                writer.write_frame(frame)
        print(f"原始影片片段已保存: {original_clip_path}")
        
        # 驗證結果
        assert len(tracks) > 0, "應該至少追蹤到一個物件"
        
        for obj_id, track in tracks.items():
            print(f"\n物件 {obj_id} 統計:")
            print(f"  追蹤幀數: {len(track.masks)}/{len(frames)-1}")
            print(f"  平均置信度: {metrics[f'object_{obj_id}']['avg_confidence']:.3f}")
            print(f"  平均區域大小: {metrics[f'object_{obj_id}']['avg_area']:.0f} 像素")
        
        print(f"\n所有測試結果已保存到: {output_dir}")
        print("請檢查以下檔案：")
        print(f"  1. SAM 分割結果: real_video_sam_segmentation.png")
        print(f"  2. 追蹤結果影片: real_video_tracking.mp4")
        print(f"  3. 關鍵幀對比: real_video_tracking_key_frames.png")
        print(f"  4. 物件軌跡圖: real_video_tracking_trajectories.png")
        print(f"  5. 追蹤指標: real_video_tracking_metrics.json")
        
        # 清理
        tracker.clear_memory()
        
        return {
            'output_dir': output_dir,
            'num_objects_detected': num_objects,
            'num_objects_tracked': len(tracks),
            'frames_processed': len(frames),
            'metrics': metrics
        }


def run_manual_visualization_tests():
    """手動執行視覺化測試"""
    import tempfile
    
    tester = TestDeAOTWrapperVisualization()
    
    # 設置輸出目錄
    project_root = Path(__file__).parent.parent
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    
    temp_dir = tempfile.mkdtemp(prefix="deaot_manual_test_", dir=str(tmp_dir))
    output_dir = Path(temp_dir)
    
    print(f"輸出目錄: {output_dir}")
    
    # 執行合成影片測試
    print("\n執行合成影片追蹤測試...")
    try:
        result1 = tester.test_synthetic_video_tracking(output_dir)
        print(f"合成影片測試成功: {result1}")
    except Exception as e:
        print(f"合成影片測試失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 執行真實影片測試
    print("\n執行真實影片追蹤測試...")
    try:
        result2 = tester.test_real_video_tracking_with_sam(output_dir)
        print(f"真實影片測試成功: {result2}")
    except Exception as e:
        print(f"真實影片測試失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n所有結果已保存到: {output_dir}")


if __name__ == "__main__":
    # 當直接執行此檔案時，可選擇運行視覺化測試或所有測試
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "viz":
        print("執行 DeAOT Wrapper 視覺化測試...")
        run_manual_visualization_tests()
    else:
        # 運行所有測試
        pytest.main([__file__, "-v", "--tb=short"])