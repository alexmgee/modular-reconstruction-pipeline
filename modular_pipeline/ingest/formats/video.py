"""
Video Frame Extraction
======================

Extract frames from standard video formats (MP4, MOV, AVI, etc.) using
ffmpeg or OpenCV with quality control and temporal sampling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import subprocess
import json
import warnings
import shutil

import numpy as np
import cv2
from tqdm import tqdm

from modular_pipeline.ingest.quality import QualityAnalyzer, MotionDetector
from modular_pipeline.ingest.metadata import MetadataExtractor


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which('ffmpeg') is not None


HAS_FFMPEG = check_ffmpeg()


@dataclass
class VideoInfo:
    """Video file information."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # seconds
    codec: str
    format: str
    bitrate: int  # kbps
    
    def to_dict(self) -> Dict:
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration': self.duration,
            'codec': self.codec,
            'format': self.format,
            'bitrate': self.bitrate,
        }


class VideoExtractor:
    """Extract frames from video files."""
    
    def __init__(
        self,
        use_ffmpeg: bool = True,
        quality_threshold: float = 0.5,
        analyze_quality: bool = True,
    ):
        """Initialize video extractor.
        
        Args:
            use_ffmpeg: Prefer ffmpeg over OpenCV (faster, better quality)
            quality_threshold: Minimum quality score for frames
            analyze_quality: Perform quality analysis on extracted frames
        """
        self.use_ffmpeg = use_ffmpeg and HAS_FFMPEG
        self.quality_threshold = quality_threshold
        self.analyze_quality = analyze_quality
        
        if self.use_ffmpeg and not HAS_FFMPEG:
            warnings.warn("ffmpeg not found, falling back to OpenCV")
            self.use_ffmpeg = False
    
    def get_info(self, video_path: Path) -> VideoInfo:
        """Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoInfo object
        """
        if self.use_ffmpeg:
            return self._get_info_ffprobe(video_path)
        else:
            return self._get_info_opencv(video_path)
    
    def _get_info_ffprobe(self, video_path: Path) -> VideoInfo:
        """Get video info using ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            raise ValueError("No video stream found")
        
        # Extract info
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Parse fps (can be fractional like "30000/1001")
        fps_str = video_stream['r_frame_rate']
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
        
        duration = float(data['format'].get('duration', 0))
        frame_count = int(video_stream.get('nb_frames', duration * fps))
        codec = video_stream['codec_name']
        format_name = data['format']['format_name']
        bitrate = int(data['format'].get('bit_rate', 0)) // 1000  # Convert to kbps
        
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec,
            format=format_name,
            bitrate=bitrate
        )
    
    def _get_info_opencv(self, video_path: Path) -> VideoInfo:
        """Get video info using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Get codec (fourcc)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec,
            format=video_path.suffix[1:],
            bitrate=0  # Not available from OpenCV
        )
    
    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        skip_frames: int = 0,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        output_format: str = 'jpg',
        output_quality: int = 95,
    ) -> Tuple[List[Path], Dict]:
        """Extract frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all)
            skip_frames: Number of frames to skip between extractions
            target_fps: Target FPS for extraction (None = native)
            max_frames: Maximum number of frames to extract
            output_format: Output format (jpg, png)
            output_quality: JPEG quality (1-100)
            
        Returns:
            (frame_paths, statistics)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info
        info = self.get_info(video_path)
        
        # Determine extraction method
        if self.use_ffmpeg and skip_frames == 0 and target_fps is None:
            return self._extract_ffmpeg(
                video_path, output_dir, start_frame, end_frame,
                max_frames, output_format, output_quality, info
            )
        else:
            return self._extract_opencv(
                video_path, output_dir, start_frame, end_frame,
                skip_frames, target_fps, max_frames, output_format,
                output_quality, info
            )
    
    def _extract_ffmpeg(
        self,
        video_path: Path,
        output_dir: Path,
        start_frame: int,
        end_frame: Optional[int],
        max_frames: Optional[int],
        output_format: str,
        output_quality: int,
        info: VideoInfo,
    ) -> Tuple[List[Path], Dict]:
        """Extract frames using ffmpeg (fastest, best quality)."""
        
        # Build ffmpeg command
        cmd = ['ffmpeg', '-i', str(video_path)]
        
        # Start time
        if start_frame > 0:
            start_time = start_frame / info.fps
            cmd.extend(['-ss', str(start_time)])
        
        # Duration/frame count
        if end_frame is not None:
            duration = (end_frame - start_frame) / info.fps
            cmd.extend(['-t', str(duration)])
        elif max_frames is not None:
            cmd.extend(['-frames:v', str(max_frames)])
        
        # Quality settings
        if output_format == 'jpg':
            cmd.extend(['-q:v', str(101 - output_quality)])  # ffmpeg quality scale
        
        # Output pattern
        output_pattern = output_dir / f'frame_%06d.{output_format}'
        cmd.append(str(output_pattern))
        
        # Run ffmpeg
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Get extracted frames
        frame_paths = sorted(output_dir.glob(f'frame_*.{output_format}'))
        
        # Statistics
        stats = {
            'total_frames': len(frame_paths),
            'video_info': info.to_dict(),
            'method': 'ffmpeg',
        }
        
        return frame_paths, stats
    
    def _extract_opencv(
        self,
        video_path: Path,
        output_dir: Path,
        start_frame: int,
        end_frame: Optional[int],
        skip_frames: int,
        target_fps: Optional[float],
        max_frames: Optional[int],
        output_format: str,
        output_quality: int,
        info: VideoInfo,
    ) -> Tuple[List[Path], Dict]:
        """Extract frames using OpenCV (more flexible, slower)."""
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Calculate frame sampling
        if target_fps is not None and target_fps < info.fps:
            sample_interval = int(info.fps / target_fps)
        else:
            sample_interval = skip_frames + 1
        
        # Set start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Determine end frame
        if end_frame is None:
            end_frame = info.frame_count
        
        # Quality analyzer
        quality_analyzer = None
        motion_detector = None
        if self.analyze_quality:
            quality_analyzer = QualityAnalyzer()
            motion_detector = MotionDetector()
        
        # Statistics
        stats = {
            'total_extracted': 0,
            'total_skipped': 0,
            'quality_rejected': 0,
            'motion_scores': [],
            'quality_scores': [],
            'video_info': info.to_dict(),
            'method': 'opencv',
        }
        
        frame_paths = []
        frame_idx = start_frame
        extracted_count = 0
        
        # Encode parameters
        if output_format == 'jpg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, output_quality]
        else:
            encode_params = []
        
        # Progress bar
        total_to_process = (end_frame - start_frame) // sample_interval
        if max_frames:
            total_to_process = min(total_to_process, max_frames)
        
        pbar = tqdm(total=total_to_process, desc="Extracting frames")
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames
            if (frame_idx - start_frame) % sample_interval != 0:
                frame_idx += 1
                continue
            
            # Quality analysis
            if quality_analyzer:
                quality_metrics = quality_analyzer.analyze(frame)
                stats['quality_scores'].append(quality_metrics.quality_score)
                
                if quality_metrics.quality_score < self.quality_threshold:
                    stats['quality_rejected'] += 1
                    stats['total_skipped'] += 1
                    frame_idx += 1
                    pbar.update(1)
                    continue
            
            # Motion detection
            if motion_detector:
                motion_score = motion_detector.detect(frame)
                stats['motion_scores'].append(motion_score)
            
            # Save frame
            output_path = output_dir / f'frame_{extracted_count:06d}.{output_format}'
            cv2.imwrite(str(output_path), frame, encode_params)
            frame_paths.append(output_path)
            
            stats['total_extracted'] += 1
            extracted_count += 1
            frame_idx += 1
            pbar.update(1)
            
            # Check max frames
            if max_frames and extracted_count >= max_frames:
                break
        
        pbar.close()
        cap.release()
        
        # Compute statistics
        if stats['quality_scores']:
            stats['quality_mean'] = np.mean(stats['quality_scores'])
            stats['quality_std'] = np.std(stats['quality_scores'])
        
        if stats['motion_scores']:
            stats['motion_mean'] = np.mean(stats['motion_scores'])
            stats['motion_std'] = np.std(stats['motion_scores'])
        
        return frame_paths, stats


__all__ = [
    'VideoInfo',
    'VideoExtractor',
    'check_ffmpeg',
]
