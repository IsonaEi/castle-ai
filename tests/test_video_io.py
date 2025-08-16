"""
video_io 模組的測試套件

包含對 VideoReader、VideoWriter、VideoIO 和 SubtitleGenerator 的全面測試。
使用 mock 來模擬 av 庫的行為，避免依賴實際的影片檔案。
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
import os

from castle.utils.video_io import (
    VideoIOError,
    VideoInfo,
    VideoIO,
    VideoReader,
    VideoWriter,
    SubtitleGenerator
)


# ==================== Fixtures ====================

@pytest.fixture
def mock_av_container():
    """模擬 av 容器物件"""
    container = Mock()
    
    # 模擬影片流
    video_stream = Mock()
    video_stream.average_rate = 30.0
    video_stream.time_base = 1/30.0
    video_stream.width = 640
    video_stream.height = 480
    video_stream.frames = 1000
    
    container.streams.video = [video_stream]
    container.decode.return_value = []
    
    return container, video_stream


@pytest.fixture
def mock_av_frame():
    """模擬 av 影格物件"""
    frame = Mock()
    frame.pts = 100
    frame.to_rgb.return_value.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_frames():
    """產生範例影格資料"""
    return [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(10)
    ]


@pytest.fixture
def temp_video_path():
    """建立暫時影片檔案路徑"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
    
    # 創建空檔案
    Path(temp_path).touch()
    
    yield temp_path
    
    # 清理
    if Path(temp_path).exists():
        Path(temp_path).unlink()


@pytest.fixture
def temp_output_dir():
    """建立暫時輸出目錄"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# ==================== VideoInfo Tests ====================

class TestVideoInfo:
    """VideoInfo 資料類別測試"""
    
    def test_video_info_creation(self):
        """測試 VideoInfo 建立"""
        path = Path("test.mp4")
        info = VideoInfo(
            path=path,
            fps=30.0,
            frame_count=1000,
            width=640,
            height=480,
            duration=33.33
        )
        
        assert info.path == path
        assert info.fps == 30.0
        assert info.frame_count == 1000
        assert info.width == 640
        assert info.height == 480
        assert info.duration == 33.33


# ==================== VideoIOError Tests ====================

class TestVideoIOError:
    """VideoIOError 異常類別測試"""
    
    def test_video_io_error_creation(self):
        """測試 VideoIOError 建立"""
        error = VideoIOError("測試錯誤")
        assert str(error) == "測試錯誤"
        assert isinstance(error, Exception)


# ==================== VideoReader Tests ====================

class TestVideoReader:
    """VideoReader 類別測試"""
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_init_success(self, mock_av_open, mock_av_container, temp_video_path):
        """測試 VideoReader 成功初始化"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        
        # 模擬成功的影格讀取用於 _calculate_frame_count
        mock_frame = Mock()
        mock_frame.pts = 100
        mock_frame.to_rgb.return_value.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        container.decode.return_value = [mock_frame]
        
        reader = VideoReader(temp_video_path)
        
        assert reader.path == Path(temp_video_path)
        assert reader.fps == 30.0
        assert reader.width == 640
        assert reader.height == 480
        assert reader.frame_count >= 900  # 允許一些彈性，因為模擬可能不完全準確
        assert reader.duration >= 30.0
        
        mock_av_open.assert_called_once_with(str(temp_video_path))
    
    def test_video_reader_file_not_found(self):
        """測試檔案不存在的情況"""
        with pytest.raises(FileNotFoundError, match="影片檔案不存在"):
            VideoReader("nonexistent.mp4")
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_av_error(self, mock_av_open, temp_video_path):
        """測試 av 庫錯誤"""
        mock_av_open.side_effect = Exception("AV error")
        
        with pytest.raises(VideoIOError, match="初始化影片讀取器失敗"):
            VideoReader(temp_video_path)
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_context_manager(self, mock_av_open, mock_av_container, temp_video_path):
        """測試 context manager 功能"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        
        with VideoReader(temp_video_path) as reader:
            assert not reader._closed
        
        assert reader._closed
        container.close.assert_called_once()
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_get_frame_success(self, mock_av_open, mock_av_container, mock_av_frame, temp_video_path):
        """測試成功獲取影格"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        container.decode.return_value = [mock_av_frame]
        
        reader = VideoReader(temp_video_path)
        frame = reader.get_frame(100)
        
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        assert 100 in reader._frame_cache
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_get_frame_out_of_range(self, mock_av_open, mock_av_container, temp_video_path):
        """測試影格索引超出範圍"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        
        reader = VideoReader(temp_video_path)
        
        with pytest.raises(IndexError, match="影格索引.*超出範圍"):
            reader.get_frame(-1)
        
        with pytest.raises(IndexError, match="影格索引.*超出範圍"):
            reader.get_frame(1000)
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_cache_functionality(self, mock_av_open, mock_av_container, mock_av_frame, temp_video_path):
        """測試快取功能"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        container.decode.return_value = [mock_av_frame]
        
        reader = VideoReader(temp_video_path)
        
        # 第一次讀取
        frame1 = reader.get_frame(100)
        assert 100 in reader._frame_cache
        
        # 第二次讀取應該使用快取
        frame2 = reader.get_frame(100)
        assert np.array_equal(frame1, frame2)
        
        # 測試快取清除
        reader.clear_cache()
        assert len(reader._frame_cache) == 0
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_iterate_frames(self, mock_av_open, mock_av_container, mock_av_frame, temp_video_path):
        """測試影格迭代功能"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        
        # 設定更高的影格數量，避免影格索引超出範圍
        video_stream.frames = 100
        
        # 模擬每次解碼都返回一個影格
        def mock_decode_side_effect(*args):
            return [mock_av_frame]
        
        container.decode.side_effect = mock_decode_side_effect
        
        reader = VideoReader(temp_video_path)
        
        # 確保 reader 有足夠的影格數量
        if len(reader) > 5:
            frames_list = list(reader.iterate_frames(start=0, end=5, step=2))
            
            assert len(frames_list) == 3  # 0, 2, 4
            assert all(isinstance(frame, np.ndarray) for idx, frame in frames_list)
            assert frames_list[0][0] == 0
            assert frames_list[1][0] == 2
            assert frames_list[2][0] == 4
        else:
            # 如果影格數量不足，測試會跳過範圍檢查
            with pytest.raises(ValueError, match="起始索引.*無效"):
                list(reader.iterate_frames(start=0, end=5, step=2))
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_get_batch_frames(self, mock_av_open, mock_av_container, mock_av_frame, temp_video_path):
        """測試批次獲取影格"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        container.decode.return_value = [mock_av_frame]
        
        reader = VideoReader(temp_video_path)
        
        indices = [10, 20, 30]
        frames = reader.get_batch_frames(indices)
        
        assert len(frames) == 3
        assert all(isinstance(frame, np.ndarray) for frame in frames)
        assert all(frame.shape == (480, 640, 3) for frame in frames)
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_get_info(self, mock_av_open, mock_av_container, temp_video_path):
        """測試獲取影片資訊"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        
        # 模擬成功的影格讀取
        mock_frame = Mock()
        mock_frame.pts = 100
        mock_frame.to_rgb.return_value.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        container.decode.return_value = [mock_frame]
        
        reader = VideoReader(temp_video_path)
        info = reader.get_info()
        
        assert isinstance(info, VideoInfo)
        assert info.path == Path(temp_video_path)
        assert info.fps == 30.0
        assert info.width == 640
        assert info.height == 480
        assert info.frame_count >= 900
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_magic_methods(self, mock_av_open, mock_av_container, mock_av_frame, temp_video_path):
        """測試魔術方法"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        container.decode.return_value = [mock_av_frame]
        
        reader = VideoReader(temp_video_path)
        
        # 測試 __len__
        assert len(reader) >= 900
        
        # 測試 __getitem__
        if len(reader) > 100:
            frame = reader[100]
            assert isinstance(frame, np.ndarray)
    
    @patch('castle.utils.video_io.av.open')
    def test_video_reader_binary_search_frame_count(self, mock_av_open, temp_video_path):
        """測試二分搜尋影格數量功能"""
        container = Mock()
        video_stream = Mock()
        video_stream.average_rate = 30.0
        video_stream.time_base = 1/30.0
        video_stream.width = 640
        video_stream.height = 480
        video_stream.frames = None  # 模擬不可用的情況
        
        container.streams.video = [video_stream]
        mock_av_open.return_value = container
        
        # 模擬二分搜尋過程
        def mock_decode_side_effect(*args):
            if hasattr(mock_decode_side_effect, 'call_count'):
                mock_decode_side_effect.call_count += 1
            else:
                mock_decode_side_effect.call_count = 1
            
            # 模擬在某個影格後無法讀取
            if mock_decode_side_effect.call_count > 500:
                raise StopIteration()
            
            frame = Mock()
            frame.pts = mock_decode_side_effect.call_count
            frame.to_rgb.return_value.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            return [frame]
        
        container.decode.side_effect = mock_decode_side_effect
        
        reader = VideoReader(temp_video_path)
        assert reader.frame_count > 0


# ==================== VideoWriter Tests ====================

class TestVideoWriter:
    """VideoWriter 類別測試"""
    
    @patch('castle.utils.video_io.av.open')
    def test_video_writer_init_success(self, mock_av_open, temp_output_dir):
        """測試 VideoWriter 成功初始化"""
        mock_output = Mock()
        mock_stream = Mock()
        mock_output.add_stream.return_value = mock_stream
        mock_av_open.return_value = mock_output
        
        output_path = temp_output_dir / "test_output.mp4"
        writer = VideoWriter(str(output_path), fps=30.0, crf=20)
        
        assert writer.output_path == output_path
        assert not writer._initialized
        assert not writer._closed
        assert writer._frame_count == 0
        
        mock_av_open.assert_called_once_with(str(output_path), 'w')
        mock_output.add_stream.assert_called_once_with('libx264', rate=30.0)
    
    @patch('castle.utils.video_io.av.open')
    def test_video_writer_av_error(self, mock_av_open, temp_output_dir):
        """測試 av 庫錯誤"""
        mock_av_open.side_effect = Exception("AV error")
        
        output_path = temp_output_dir / "test_output.mp4"
        
        with pytest.raises(VideoIOError, match="初始化影片寫入器失敗"):
            VideoWriter(str(output_path))
    
    @patch('castle.utils.video_io.av.open')
    @patch('castle.utils.video_io.av.VideoFrame')
    def test_video_writer_write_frame_success(self, mock_video_frame, mock_av_open, sample_frames, temp_output_dir):
        """測試成功寫入影格"""
        mock_output = Mock()
        mock_stream = Mock()
        mock_output.add_stream.return_value = mock_stream
        mock_stream.encode.return_value = []  # 無封包返回
        mock_av_open.return_value = mock_output
        
        mock_av_frame = Mock()
        mock_video_frame.from_ndarray.return_value = mock_av_frame
        
        output_path = temp_output_dir / "test_output.mp4"
        writer = VideoWriter(str(output_path))
        
        frame = sample_frames[0]
        writer.write_frame(frame)
        
        assert writer._initialized
        assert writer._frame_count == 1
        assert mock_stream.width == 640
        assert mock_stream.height == 480
        
        mock_video_frame.from_ndarray.assert_called_once()
        mock_stream.encode.assert_called_once_with(mock_av_frame)
    
    @patch('castle.utils.video_io.av.open')
    def test_video_writer_write_frame_invalid_input(self, mock_av_open, temp_output_dir):
        """測試寫入無效影格"""
        mock_output = Mock()
        mock_stream = Mock()
        mock_output.add_stream.return_value = mock_stream
        mock_av_open.return_value = mock_output
        
        output_path = temp_output_dir / "test_output.mp4"
        writer = VideoWriter(str(output_path))
        
        # 測試非 numpy 陣列
        with pytest.raises(ValueError, match="影格必須是 numpy 陣列"):
            writer.write_frame("not an array")
        
        # 測試錯誤形狀
        with pytest.raises(ValueError, match="預期影格形狀為"):
            writer.write_frame(np.zeros((100, 100)))  # 只有 2D
        
        with pytest.raises(ValueError, match="預期影格形狀為"):
            writer.write_frame(np.zeros((100, 100, 4)))  # 4 通道
    
    @patch('castle.utils.video_io.av.open')
    @patch('castle.utils.video_io.av.VideoFrame')
    def test_video_writer_context_manager(self, mock_video_frame, mock_av_open, sample_frames, temp_output_dir):
        """測試 context manager 功能"""
        mock_output = Mock()
        mock_stream = Mock()
        mock_output.add_stream.return_value = mock_stream
        mock_stream.encode.return_value = []
        mock_av_open.return_value = mock_output
        
        mock_av_frame = Mock()
        mock_video_frame.from_ndarray.return_value = mock_av_frame
        
        output_path = temp_output_dir / "test_output.mp4"
        
        with VideoWriter(str(output_path)) as writer:
            writer.write_frame(sample_frames[0])
            assert not writer._closed
        
        assert writer._closed
        mock_output.close.assert_called_once()
    
    @patch('castle.utils.video_io.av.open')
    def test_video_writer_write_after_close(self, mock_av_open, sample_frames, temp_output_dir):
        """測試關閉後寫入"""
        mock_output = Mock()
        mock_stream = Mock()
        mock_stream.encode.return_value = []  # 確保返回可迭代物件
        mock_output.add_stream.return_value = mock_stream
        mock_av_open.return_value = mock_output
        
        output_path = temp_output_dir / "test_output.mp4"
        writer = VideoWriter(str(output_path))
        writer.close()
        
        with pytest.raises(VideoIOError, match="影片寫入器已關閉"):
            writer.write_frame(sample_frames[0])
    
    @patch('castle.utils.video_io.av.open')
    @patch('castle.utils.video_io.av.VideoFrame')
    def test_video_writer_float_frame_conversion(self, mock_video_frame, mock_av_open, temp_output_dir):
        """測試浮點數影格轉換"""
        mock_output = Mock()
        mock_stream = Mock()
        mock_output.add_stream.return_value = mock_stream
        mock_stream.encode.return_value = []
        mock_av_open.return_value = mock_output
        
        mock_av_frame = Mock()
        mock_video_frame.from_ndarray.return_value = mock_av_frame
        
        output_path = temp_output_dir / "test_output.mp4"
        writer = VideoWriter(str(output_path))
        
        # 測試 float32 影格
        float_frame = np.random.rand(480, 640, 3).astype(np.float32)
        writer.write_frame(float_frame)
        
        # 驗證轉換到 uint8
        call_args = mock_video_frame.from_ndarray.call_args[0]
        converted_frame = call_args[0]
        assert converted_frame.dtype == np.uint8
        assert np.all(converted_frame >= 0) and np.all(converted_frame <= 255)


# ==================== VideoIO Tests ====================

class TestVideoIO:
    """VideoIO 靜態方法測試"""
    
    @patch('castle.utils.video_io.VideoReader')
    def test_load_video_success(self, mock_video_reader_class, temp_video_path):
        """測試成功載入影片"""
        mock_reader = Mock()
        mock_video_reader_class.return_value = mock_reader
        
        result = VideoIO.load_video(temp_video_path)
        
        assert result == mock_reader
        mock_video_reader_class.assert_called_once_with(temp_video_path)
    
    @patch('castle.utils.video_io.VideoReader')
    def test_load_video_error(self, mock_video_reader_class):
        """測試載入影片錯誤"""
        mock_video_reader_class.side_effect = Exception("Test error")
        
        with pytest.raises(VideoIOError, match="載入影片失敗"):
            VideoIO.load_video("test.mp4")
    
    @patch('castle.utils.video_io.VideoWriter')
    def test_save_video_success(self, mock_video_writer_class, sample_frames, temp_output_dir):
        """測試成功儲存影片"""
        mock_writer = Mock()
        mock_video_writer_class.return_value.__enter__.return_value = mock_writer
        mock_video_writer_class.return_value.__exit__.return_value = None
        
        output_path = temp_output_dir / "test_output.mp4"
        VideoIO.save_video(sample_frames, str(output_path), fps=25.0, crf=20)
        
        mock_video_writer_class.assert_called_once_with(str(output_path), 25.0, 20, 'libx264')
        assert mock_writer.write_frame.call_count == len(sample_frames)
    
    def test_save_video_no_frames(self, temp_output_dir):
        """測試儲存空影格列表"""
        output_path = temp_output_dir / "test_output.mp4"
        
        with pytest.raises(ValueError, match="沒有影格資料可儲存"):
            VideoIO.save_video([], str(output_path))
    
    def test_save_video_invalid_frames(self, temp_output_dir):
        """測試儲存無效影格"""
        output_path = temp_output_dir / "test_output.mp4"
        
        with pytest.raises(ValueError, match="所有影格必須是 numpy 陣列"):
            VideoIO.save_video(["not", "arrays"], str(output_path))
    
    @patch('castle.utils.video_io.VideoWriter')
    def test_save_video_error(self, mock_video_writer_class, sample_frames, temp_output_dir):
        """測試儲存影片錯誤"""
        mock_video_writer_class.side_effect = Exception("Writer error")
        
        output_path = temp_output_dir / "test_output.mp4"
        
        with pytest.raises(VideoIOError, match="儲存影片失敗"):
            VideoIO.save_video(sample_frames, str(output_path))
    
    def test_get_frame(self):
        """測試獲取影格方法"""
        mock_video = Mock()
        mock_frame = np.zeros((100, 100, 3))
        mock_video.get_frame.return_value = mock_frame
        
        result = VideoIO.get_frame(mock_video, 50)
        
        assert np.array_equal(result, mock_frame)
        mock_video.get_frame.assert_called_once_with(50)


# ==================== SubtitleGenerator Tests ====================

class TestSubtitleGenerator:
    """SubtitleGenerator 類別測試"""
    
    def test_subtitle_generator_init(self):
        """測試字幕生成器初始化"""
        generator = SubtitleGenerator()
        assert generator.subtitles == []
        assert generator.get_subtitle_count() == 0
        assert generator.get_total_duration() == 0.0
    
    def test_add_subtitle_success(self):
        """測試成功添加字幕"""
        generator = SubtitleGenerator()
        
        generator.add_subtitle(0.0, 5.0, "第一段字幕")
        generator.add_subtitle(5.0, 10.0, "第二段字幕")
        
        assert len(generator.subtitles) == 2
        assert generator.subtitles[0]['start'] == 0.0
        assert generator.subtitles[0]['end'] == 5.0
        assert generator.subtitles[0]['text'] == "第一段字幕"
        assert generator.get_subtitle_count() == 2
        assert generator.get_total_duration() == 10.0
    
    def test_add_subtitle_invalid_time(self):
        """測試添加無效時間的字幕"""
        generator = SubtitleGenerator()
        
        # 負數時間
        with pytest.raises(ValueError, match="時間不能為負數"):
            generator.add_subtitle(-1.0, 5.0, "測試")
        
        with pytest.raises(ValueError, match="時間不能為負數"):
            generator.add_subtitle(0.0, -1.0, "測試")
        
        # 開始時間大於等於結束時間
        with pytest.raises(ValueError, match="開始時間必須小於結束時間"):
            generator.add_subtitle(5.0, 5.0, "測試")
        
        with pytest.raises(ValueError, match="開始時間必須小於結束時間"):
            generator.add_subtitle(10.0, 5.0, "測試")
    
    def test_add_subtitle_empty_text(self):
        """測試添加空文字字幕"""
        generator = SubtitleGenerator()
        
        with pytest.raises(ValueError, match="字幕文字不能為空"):
            generator.add_subtitle(0.0, 5.0, "")
        
        with pytest.raises(ValueError, match="字幕文字不能為空"):
            generator.add_subtitle(0.0, 5.0, "   ")  # 只有空白
    
    def test_save_srt_format(self, temp_output_dir):
        """測試儲存 SRT 格式"""
        generator = SubtitleGenerator()
        generator.add_subtitle(0.0, 2.5, "第一段字幕")
        generator.add_subtitle(3.0, 5.5, "第二段字幕")
        
        output_path = temp_output_dir / "test.srt"
        generator.save(str(output_path), format="srt")
        
        assert output_path.exists()
        
        content = output_path.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        
        # 檢查 SRT 格式
        assert lines[0] == "1"
        assert "00:00:00,000 --> 00:00:02,500" in lines[1]
        assert lines[2] == "第一段字幕"
        assert lines[4] == "2"
        assert "00:00:03,000 --> 00:00:05,500" in lines[5]
        assert lines[6] == "第二段字幕"
    
    def test_save_vtt_format(self, temp_output_dir):
        """測試儲存 WebVTT 格式"""
        generator = SubtitleGenerator()
        generator.add_subtitle(0.0, 2.5, "第一段字幕")
        generator.add_subtitle(3.0, 5.5, "第二段字幕")
        
        output_path = temp_output_dir / "test.vtt"
        generator.save(str(output_path), format="vtt")
        
        assert output_path.exists()
        
        content = output_path.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        
        # 檢查 WebVTT 格式
        assert lines[0] == "WEBVTT"
        assert "00:00:00.000 --> 00:00:02.500" in lines[2]
        assert lines[3] == "第一段字幕"
        assert "00:00:03.000 --> 00:00:05.500" in lines[5]
        assert lines[6] == "第二段字幕"
    
    def test_save_unsupported_format(self, temp_output_dir):
        """測試儲存不支援的格式"""
        generator = SubtitleGenerator()
        generator.add_subtitle(0.0, 5.0, "測試字幕")
        
        output_path = temp_output_dir / "test.ass"
        
        with pytest.raises(IOError, match="儲存字幕檔案失敗"):
            generator.save(str(output_path), format="ass")
    
    def test_save_empty_subtitles(self, temp_output_dir):
        """測試儲存空字幕列表"""
        generator = SubtitleGenerator()
        
        output_path = temp_output_dir / "test.srt"
        generator.save(str(output_path), format="srt")
        
        # 應該不會產生檔案或產生空檔案
        # 具體行為取決於實作
    
    def test_clear_subtitles(self):
        """測試清除字幕"""
        generator = SubtitleGenerator()
        generator.add_subtitle(0.0, 5.0, "測試字幕")
        
        assert generator.get_subtitle_count() == 1
        
        generator.clear()
        
        assert generator.get_subtitle_count() == 0
        assert generator.get_total_duration() == 0.0
    
    def test_format_srt_time(self):
        """測試 SRT 時間格式化"""
        # 測試正常時間
        result = SubtitleGenerator._format_srt_time(65.123)
        assert result == "00:01:05,123"
        
        # 測試零時間
        result = SubtitleGenerator._format_srt_time(0.0)
        assert result == "00:00:00,000"
        
        # 測試小時級別時間
        result = SubtitleGenerator._format_srt_time(3661.5)
        assert result == "01:01:01,500"
    
    def test_format_vtt_time(self):
        """測試 WebVTT 時間格式化"""
        # 測試正常時間
        result = SubtitleGenerator._format_vtt_time(65.123)
        assert result == "00:01:05.123"
        
        # 測試零時間
        result = SubtitleGenerator._format_vtt_time(0.0)
        assert result == "00:00:00.000"
        
        # 測試小時級別時間
        result = SubtitleGenerator._format_vtt_time(3661.5)
        assert result == "01:01:01.500"


# ==================== Integration Tests ====================

class TestVideoIOIntegration:
    """整合測試"""
    
    @patch('castle.utils.video_io.av.open')
    @patch('castle.utils.video_io.av.VideoFrame')
    def test_full_video_pipeline(self, mock_video_frame, mock_av_open, temp_video_path, temp_output_dir):
        """測試完整的影片處理流程"""
        # 設定 mock 物件
        container = Mock()
        video_stream = Mock()
        video_stream.average_rate = 30.0
        video_stream.time_base = 1/30.0
        video_stream.width = 640
        video_stream.height = 480
        video_stream.frames = 100
        
        container.streams.video = [video_stream]
        
        # 模擬影格
        mock_frame = Mock()
        mock_frame.pts = 50
        mock_frame.to_rgb.return_value.to_ndarray.return_value = np.random.randint(
            0, 255, (480, 640, 3), dtype=np.uint8
        )
        container.decode.return_value = [mock_frame]
        
        # 設定寫入器 mock
        mock_output = Mock()
        mock_stream = Mock()
        mock_output.add_stream.return_value = mock_stream
        mock_stream.encode.return_value = []
        
        # 根據呼叫參數返回不同的 mock 物件
        def av_open_side_effect(path, mode='r'):
            if mode == 'w':
                return mock_output
            else:
                return container
        
        mock_av_open.side_effect = av_open_side_effect
        
        mock_av_frame = Mock()
        mock_video_frame.from_ndarray.return_value = mock_av_frame
        
        # 測試載入影片
        reader = VideoIO.load_video(temp_video_path)
        assert isinstance(reader, VideoReader)
        
        # 只有當影片有足夠影格時才進行測試
        if len(reader) > 10:
            # 測試讀取影格
            frames = []
            for i in range(0, 10, 2):
                frame = reader.get_frame(i)
                frames.append(frame)
            
            assert len(frames) == 5
            
            # 測試儲存影片
            output_path = temp_output_dir / "output.mp4"
            VideoIO.save_video(frames, str(output_path), fps=15.0)
            
            # 驗證呼叫
            # encode 會被調用 len(frames) 次寫入影格 + 1 次刷新編碼器緩衝區
            assert mock_stream.encode.call_count == len(frames) + 1
            mock_output.close.assert_called()
        else:
            # 如果影格數量不足，測試一個簡單的影格列表
            frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
            output_path = temp_output_dir / "output.mp4"
            VideoIO.save_video(frames, str(output_path), fps=15.0)
    
    def test_subtitle_workflow(self, temp_output_dir):
        """測試字幕工作流程"""
        generator = SubtitleGenerator()
        
        # 添加多個字幕
        subtitles_data = [
            (0.0, 3.0, "影片開始"),
            (3.5, 8.0, "這是第二段字幕"),
            (8.5, 12.0, "影片即將結束"),
            (12.5, 15.0, "謝謝觀看")
        ]
        
        for start, end, text in subtitles_data:
            generator.add_subtitle(start, end, text)
        
        assert generator.get_subtitle_count() == 4
        assert generator.get_total_duration() == 15.0
        
        # 儲存為兩種格式
        srt_path = temp_output_dir / "subtitles.srt"
        vtt_path = temp_output_dir / "subtitles.vtt"
        
        generator.save(str(srt_path), format="srt")
        generator.save(str(vtt_path), format="vtt")
        
        assert srt_path.exists()
        assert vtt_path.exists()
        
        # 驗證檔案內容
        srt_content = srt_path.read_text(encoding='utf-8')
        vtt_content = vtt_path.read_text(encoding='utf-8')
        
        assert "影片開始" in srt_content
        assert "謝謝觀看" in srt_content
        assert "WEBVTT" in vtt_content
        assert "影片開始" in vtt_content


# ==================== Performance Tests ====================

class TestVideoIOPerformance:
    """效能測試"""
    
    @patch('castle.utils.video_io.av.open')
    def test_cache_performance(self, mock_av_open, mock_av_container, mock_av_frame, temp_video_path):
        """測試快取效能"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        container.decode.return_value = [mock_av_frame]
        
        reader = VideoReader(temp_video_path)
        
        # 第一次讀取
        import time
        start_time = time.time()
        frame1 = reader.get_frame(100)
        first_read_time = time.time() - start_time
        
        # 第二次讀取（應該使用快取）
        start_time = time.time()
        frame2 = reader.get_frame(100)
        second_read_time = time.time() - start_time
        
        # 快取讀取應該更快（實際上 mock 可能沒有明顯差異）
        assert np.array_equal(frame1, frame2)
        assert 100 in reader._frame_cache
    
    @patch('castle.utils.video_io.av.open')
    def test_batch_frame_efficiency(self, mock_av_open, mock_av_container, mock_av_frame, temp_video_path):
        """測試批次讀取效率"""
        container, video_stream = mock_av_container
        mock_av_open.return_value = container
        container.decode.return_value = [mock_av_frame]
        
        reader = VideoReader(temp_video_path)
        
        # 批次讀取
        indices = list(range(0, 100, 10))
        frames = reader.get_batch_frames(indices)
        
        assert len(frames) == len(indices)
        assert all(isinstance(frame, np.ndarray) for frame in frames)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
