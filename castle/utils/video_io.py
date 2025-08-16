"""
Video input/output utilities
Handles video reading, writing, and subtitle generation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Generator
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoInfo:
    """Video information data class"""
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float
    
class VideoIO:
    """Video input/output processor"""
    
    @staticmethod
    def load_video(video_path: str) -> 'VideoReader':
        """
        Load video file
        
        Args:
            video_path: video path
            
        Returns:
            VideoReader object
        """
        return VideoReader(video_path)
    
    @staticmethod
    def save_video(
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        codec: str = 'mp4v'
    ):
        """
        Save video
        
        Args:
            frames: frame list
            output_path: output path
            fps: frame rate
            codec: codec
        """
        if not frames:
            raise ValueError("No frames to save")
            
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in frames:
            if frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 3 and frame.dtype == np.float32:
                frame = (frame * 255).astype(np.uint8)
                
            writer.write(frame)
            
        writer.release()
        logger.info(f"Saved video to {output_path}")
        
    @staticmethod
    def get_frame(video: 'VideoReader', frame_idx: int) -> np.ndarray:
        """Get specified frame"""
        return video.get_frame(frame_idx)
        
class VideoReader:
    """Video reader"""
    
    def __init__(self, video_path: str):
        """
        Initialize video reader
        
        Args:
            video_path: video path
        """
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.cap = cv2.VideoCapture(str(self.path))
        
        # Get video information
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        self._frame_cache = {}
        
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get specified frame
        
        Args:
            frame_idx: frame index
            
        Returns:
            frame array (RGB)
        """
        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx]
            
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"Frame index {frame_idx} out of range")
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Cache the nearest frame
        if len(self._frame_cache) > 100:
            self._frame_cache.clear()
        self._frame_cache[frame_idx] = frame
        
        return frame
    
    def iterate_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterate video frames
        
        Args:
            start: start frame
            end: end frame
            step: step
            
        Yields:
            (frame index, frame array)
        """
        if end is None:
            end = self.frame_count
            
        for i in range(start, end, step):
            yield i, self.get_frame(i)
            
    def get_batch_frames(
        self,
        indices: List[int]
    ) -> List[np.ndarray]:
        """
        Batch get frames
        
        Args:
            indices: frame index list
            
        Returns:
            frame list
        """
        frames = []
        for idx in indices:
            frames.append(self.get_frame(idx))
        return frames
    
    def release(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
            
    def __del__(self):
        """Destructor"""
        self.release()
        
    def get_info(self) -> VideoInfo:
        """Get video information"""
        return VideoInfo(
            path=self.path,
            fps=self.fps,
            frame_count=self.frame_count,
            width=self.width,
            height=self.height,
            duration=self.duration
        )
        
class SubtitleGenerator:
    """Subtitle generator"""
    
    def __init__(self):
        """Initialize subtitle generator"""
        self.subtitles = []
        
    def add_subtitle(
        self,
        start_time: float,
        end_time: float,
        text: str
    ):
        """
        Add subtitle
        
        Args:
            start_time: start time (seconds)
            end_time: end time (seconds)
            text: subtitle text
        """
        self.subtitles.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })
        
    def save(self, output_path: str, format: str = 'srt'):
        """
        Save subtitle file
        
        Args:
            output_path: output path
            format: subtitle format ('srt', 'vtt')
        """
        output_path = Path(output_path)
        
        if format == 'srt':
            self._save_srt(output_path)
        elif format == 'vtt':
            self._save_vtt(output_path)
        else:
            raise ValueError(f"Unsupported subtitle format: {format}")
            
    def _save_srt(self, output_path: Path):
        """Save as SRT format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(self.subtitles, 1):
                f.write(f"{i}\n")
                f.write(f"{self._format_srt_time(sub['start'])} --> ")
                f.write(f"{self._format_srt_time(sub['end'])}\n")
                f.write(f"{sub['text']}\n\n")
                
    def _save_vtt(self, output_path: Path):
        """Save as WebVTT format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for sub in self.subtitles:
                f.write(f"{self._format_vtt_time(sub['start'])} --> ")
                f.write(f"{self._format_vtt_time(sub['end'])}\n")
                f.write(f"{sub['text']}\n\n")
                
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format SRT time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """Format WebVTT time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"