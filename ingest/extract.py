"""
Main Ingest Module
==================

Orchestrates frame extraction, metadata collection, and quality filtering
from various source formats (video, images, OSV containers).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json

from modular_pipeline.core import (
    BaseModule,
    BaseConfig,
    StageResult,
    QualityLevel,
    ImageGeometry,
    ensure_directory,
)

from modular_pipeline.ingest.camera_db import camera_db, detect_camera_from_exif
from modular_pipeline.ingest.metadata import MetadataExtractor, ImageMetadata
from modular_pipeline.ingest.quality import QualityAnalyzer
from modular_pipeline.ingest.formats import (
    VideoExtractor,
    ImageImporter,
    OSVHandler,
    IMAGE_EXTENSIONS,
)


@dataclass
class IngestConfig(BaseConfig):
    """Configuration for ingest module."""
    
    # Source handling
    source_type: str = "auto"  # auto, video, images, osv
    
    # Frame extraction (video)
    skip_frames: int = 0
    max_frames: Optional[int] = None
    target_fps: Optional[float] = None
    start_frame: int = 0
    end_frame: Optional[int] = None
    
    # Quality filtering
    blur_threshold: float = 100.0
    exposure_range: Tuple[float, float] = (0.1, 0.9)
    min_quality_score: float = 0.5
    filter_quality: bool = True
    
    # Metadata
    extract_metadata: bool = True
    sort_by_timestamp: bool = True
    
    # Image import
    copy_files: bool = True
    recursive_search: bool = True
    
    # Output
    output_format: str = "jpg"
    output_quality: int = 95
    rename_pattern: Optional[str] = "frame_{:06d}.jpg"


class IngestModule(BaseModule):
    """Main ingest module for frame extraction and organization."""
    
    def _default_config(self) -> IngestConfig:
        """Return default configuration."""
        return IngestConfig()
    
    def _initialize(self) -> None:
        """Initialize module resources."""
        self.config: IngestConfig
        
        # Initialize format handlers
        self.video_extractor = VideoExtractor(
            quality_threshold=self.config.min_quality_score,
            analyze_quality=self.config.filter_quality,
        )
        
        self.image_importer = ImageImporter(
            quality_threshold=self.config.min_quality_score,
            analyze_quality=self.config.filter_quality,
            extract_metadata=self.config.extract_metadata,
            copy_files=self.config.copy_files,
        )
        
        self.osv_handler = OSVHandler()
        
        # Initialize metadata extractor
        if self.config.extract_metadata:
            self.metadata_extractor = MetadataExtractor()
        else:
            self.metadata_extractor = None
        
        self.log("Ingest module initialized")
    
    def process(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> StageResult:
        """Process input source and extract frames.
        
        Args:
            input_path: Path to video file or image directory
            output_path: Output directory for frames
            **kwargs: Additional options
            
        Returns:
            StageResult with extraction outcome
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        self.log(f"Processing input: {input_path}")
        
        # Detect source type
        source_type = self._detect_source_type(input_path)
        self.log(f"Detected source type: {source_type}")
        
        # Create output directory
        frames_dir = ensure_directory(output_path / "frames")
        
        # Process based on source type
        start_time = self._timed_process.__self__.__class__
        import time
        start = time.perf_counter()
        
        if source_type == "video":
            frame_paths, stats = self._process_video(input_path, frames_dir)
        elif source_type == "osv":
            frame_paths, stats = self._process_osv(input_path, frames_dir)
        elif source_type == "images":
            frame_paths, stats = self._process_images(input_path, frames_dir)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        elapsed = time.perf_counter() - start
        
        # Extract metadata from first frame for camera detection
        camera_profile = None
        geometry = ImageGeometry.PINHOLE
        
        if frame_paths and self.metadata_extractor:
            try:
                metadata = self.metadata_extractor.extract(frame_paths[0])
                camera_profile = detect_camera_from_exif({
                    'Make': metadata.make,
                    'Model': metadata.model
                })
                
                if camera_profile:
                    geometry = camera_profile.geometry
                    self.log(f"Detected camera: {camera_profile.make} {camera_profile.model}")
                    self.log(f"Geometry: {geometry.value}")
            except Exception as e:
                self.log(f"Failed to detect camera: {e}", "warning")
        
        # Save manifest
        manifest = self._create_manifest(
            input_path, source_type, frame_paths, stats,
            camera_profile, geometry
        )
        manifest_path = output_path / "ingest_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Determine quality level
        items_processed = stats.get('total_imported', stats.get('total_extracted', len(frame_paths)))
        items_failed = stats.get('total_rejected', stats.get('quality_rejected', 0))
        
        if items_processed == 0:
            quality = QualityLevel.REJECT
            confidence = 0.0
        else:
            success_rate = items_processed / (items_processed + items_failed) if items_failed > 0 else 1.0
            confidence = success_rate
            quality = self.config.confidence_to_quality(confidence)
        
        # Create result
        result = self._create_result(
            success=len(frame_paths) > 0,
            output_path=output_path,
            items_processed=items_processed,
            items_failed=items_failed,
            processing_time=elapsed,
            quality=quality,
            confidence=confidence,
            metrics={
                'frame_count': len(frame_paths),
                'source_type': source_type,
                'geometry': geometry.value if geometry else None,
                'camera_make': camera_profile.make if camera_profile else None,
                'camera_model': camera_profile.model if camera_profile else None,
                **stats
            }
        )
        
        # Save result manifest
        result.save_manifest(output_path / "stage_result.json")
        
        self.log(f"Ingest complete: {len(frame_paths)} frames extracted")
        
        return result
    
    def _detect_source_type(self, input_path: Path) -> str:
        """Detect source type from input path."""
        
        if self.config.source_type != "auto":
            return self.config.source_type
        
        # Check if directory
        if input_path.is_dir():
            return "images"
        
        # Check file extension
        suffix = input_path.suffix.lower()
        
        if suffix == '.osv':
            return "osv"
        elif suffix in ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm']:
            return "video"
        elif suffix in IMAGE_EXTENSIONS:
            # Single image - treat as directory
            return "images"
        else:
            raise ValueError(f"Unknown source format: {suffix}")
    
    def _process_video(
        self,
        video_path: Path,
        output_dir: Path
    ) -> Tuple[List[Path], Dict]:
        """Process video file."""
        
        self.log(f"Extracting frames from video: {video_path.name}")
        
        frame_paths, stats = self.video_extractor.extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            start_frame=self.config.start_frame,
            end_frame=self.config.end_frame,
            skip_frames=self.config.skip_frames,
            target_fps=self.config.target_fps,
            max_frames=self.config.max_frames,
            output_format=self.config.output_format,
            output_quality=self.config.output_quality,
        )
        
        return frame_paths, stats
    
    def _process_osv(
        self,
        osv_path: Path,
        output_dir: Path
    ) -> Tuple[List[Path], Dict]:
        """Process OSV file."""
        
        self.log(f"Extracting frames from OSV: {osv_path.name}")
        
        # Detect camera model
        camera_model = self.osv_handler.detect_camera_model(osv_path)
        self.log(f"Detected: {camera_model}")
        
        frame_paths, stats = self.osv_handler.extract_frames(
            osv_path=osv_path,
            output_dir=output_dir,
            start_frame=self.config.start_frame,
            end_frame=self.config.end_frame,
            skip_frames=self.config.skip_frames,
            target_fps=self.config.target_fps,
            max_frames=self.config.max_frames,
            output_format=self.config.output_format,
            output_quality=self.config.output_quality,
        )
        
        stats['detected_camera'] = camera_model
        
        return frame_paths, stats
    
    def _process_images(
        self,
        input_path: Path,
        output_dir: Path
    ) -> Tuple[List[Path], Dict]:
        """Process image directory."""
        
        self.log(f"Importing images from: {input_path}")
        
        # If single file, use parent directory
        if input_path.is_file():
            search_dir = input_path.parent
        else:
            search_dir = input_path
        
        frame_paths, stats = self.image_importer.import_images(
            input_dir=search_dir,
            output_dir=output_dir,
            recursive=self.config.recursive_search,
            sort_by='exif' if self.config.sort_by_timestamp else 'name',
            rename_pattern=self.config.rename_pattern,
            max_images=self.config.max_frames,
        )
        
        return frame_paths, stats
    
    def _create_manifest(
        self,
        input_path: Path,
        source_type: str,
        frame_paths: List[Path],
        stats: Dict,
        camera_profile,
        geometry: ImageGeometry,
    ) -> Dict:
        """Create ingest manifest."""
        
        manifest = {
            'input': {
                'path': str(input_path),
                'type': source_type,
            },
            'output': {
                'frame_count': len(frame_paths),
                'frames_directory': str(frame_paths[0].parent) if frame_paths else None,
            },
            'camera': {
                'detected': camera_profile is not None,
                'make': camera_profile.make if camera_profile else None,
                'model': camera_profile.model if camera_profile else None,
                'geometry': geometry.value,
            },
            'reconstruction_hints': {},
            'statistics': stats,
        }
        
        # Add reconstruction hints if camera detected
        if camera_profile:
            hints = camera_db.get_reconstruction_hints(
                camera_profile.make,
                camera_profile.model
            )
            manifest['reconstruction_hints'] = {
                'requires_reframe': hints.get('requires_reframe', False),
                'requires_undistort': hints.get('requires_undistort', False),
                'suggested_masks': hints.get('typical_masks', []),
                'camera_model': hints.get('camera_model').value if hints.get('camera_model') else None,
            }
        
        return manifest


__all__ = [
    'IngestConfig',
    'IngestModule',
]
