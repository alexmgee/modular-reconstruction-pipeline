"""
DJI Osmo .osv Container Handler
================================

Handle DJI Osmo Action .osv container files which are essentially renamed MP4 files
with specific metadata structure used by DJI 360° cameras.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import shutil
import warnings

from modular_pipeline.ingest.formats.video import VideoExtractor, VideoInfo


class OSVHandler:
    """Handle DJI Osmo .osv files."""
    
    def __init__(self):
        """Initialize OSV handler."""
        self.video_extractor = VideoExtractor()
    
    def is_osv_file(self, file_path: Path) -> bool:
        """Check if file is a .osv file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file has .osv extension
        """
        return file_path.suffix.lower() == '.osv'
    
    def convert_to_mp4(
        self,
        osv_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Convert .osv to .mp4 (just rename, they're the same format).
        
        Args:
            osv_path: Path to .osv file
            output_path: Output path (defaults to same name with .mp4)
            
        Returns:
            Path to converted MP4 file
        """
        osv_path = Path(osv_path)
        
        if not self.is_osv_file(osv_path):
            raise ValueError(f"Not an OSV file: {osv_path}")
        
        if output_path is None:
            output_path = osv_path.with_suffix('.mp4')
        else:
            output_path = Path(output_path)
        
        # OSV files are just MP4 containers with .osv extension
        # Simply copy and rename
        shutil.copy2(osv_path, output_path)
        
        return output_path
    
    def extract_frames(
        self,
        osv_path: Path,
        output_dir: Path,
        convert_first: bool = True,
        cleanup_mp4: bool = True,
        **kwargs
    ) -> Tuple[List[Path], Dict]:
        """Extract frames from .osv file.
        
        Args:
            osv_path: Path to .osv file
            output_dir: Output directory for frames
            convert_first: Convert to MP4 first (recommended)
            cleanup_mp4: Delete temporary MP4 after extraction
            **kwargs: Additional arguments passed to VideoExtractor
            
        Returns:
            (frame_paths, statistics)
        """
        osv_path = Path(osv_path)
        output_dir = Path(output_dir)
        
        if not self.is_osv_file(osv_path):
            raise ValueError(f"Not an OSV file: {osv_path}")
        
        # Convert to MP4 if requested
        if convert_first:
            mp4_path = osv_path.with_suffix('.mp4')
            self.convert_to_mp4(osv_path, mp4_path)
            video_path = mp4_path
            created_mp4 = True
        else:
            video_path = osv_path
            created_mp4 = False
        
        # Extract frames using video extractor
        try:
            frame_paths, stats = self.video_extractor.extract_frames(
                video_path,
                output_dir,
                **kwargs
            )
            
            # Add OSV-specific info to stats
            stats['source_format'] = 'osv'
            stats['osv_file'] = str(osv_path)
            
            return frame_paths, stats
            
        finally:
            # Cleanup temporary MP4 if created
            if created_mp4 and cleanup_mp4 and video_path.exists():
                video_path.unlink()
    
    def get_info(self, osv_path: Path) -> VideoInfo:
        """Get video info from .osv file.
        
        Args:
            osv_path: Path to .osv file
            
        Returns:
            VideoInfo object
        """
        osv_path = Path(osv_path)
        
        if not self.is_osv_file(osv_path):
            raise ValueError(f"Not an OSV file: {osv_path}")
        
        # OSV files can be read directly as MP4
        return self.video_extractor.get_info(osv_path)
    
    def detect_camera_model(self, osv_path: Path) -> str:
        """Detect DJI camera model from OSV file.
        
        Args:
            osv_path: Path to .osv file
            
        Returns:
            Camera model string
        """
        # Get video info
        info = self.get_info(osv_path)
        
        # Heuristics based on resolution
        width, height = info.width, info.height
        
        if width == 5312 and height == 2656:
            return "DJI Osmo Action 5 Pro"
        elif width == 3840 and height == 1920:
            return "DJI Osmo Action 4"
        elif width == 2880 and height == 1440:
            return "DJI Osmo Action 3"
        else:
            return f"DJI Osmo Action (Unknown {width}x{height})"


def batch_convert_osv(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    recursive: bool = True,
) -> List[Path]:
    """Batch convert .osv files to .mp4.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory (defaults to input_dir)
        recursive: Search subdirectories
        
    Returns:
        List of converted MP4 paths
    """
    input_dir = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find OSV files
    if recursive:
        osv_files = list(input_dir.rglob('*.osv')) + list(input_dir.rglob('*.OSV'))
    else:
        osv_files = list(input_dir.glob('*.osv')) + list(input_dir.glob('*.OSV'))
    
    osv_files = sorted(set(osv_files))
    
    if not osv_files:
        warnings.warn(f"No .osv files found in {input_dir}")
        return []
    
    # Convert each file
    handler = OSVHandler()
    converted_paths = []
    
    for osv_path in osv_files:
        # Determine output path
        if output_dir == input_dir:
            mp4_path = osv_path.with_suffix('.mp4')
        else:
            rel_path = osv_path.relative_to(input_dir)
            mp4_path = output_dir / rel_path.with_suffix('.mp4')
            mp4_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert
        result_path = handler.convert_to_mp4(osv_path, mp4_path)
        converted_paths.append(result_path)
    
    return converted_paths


__all__ = [
    'OSVHandler',
    'batch_convert_osv',
]
