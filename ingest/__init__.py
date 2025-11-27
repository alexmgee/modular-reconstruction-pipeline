"""
Ingest Module
=============

Frame extraction, metadata collection, and quality filtering from various
source formats: video files, image directories, and DJI OSV containers.

Main Components:
- IngestModule: Main orchestrator for frame extraction
- Camera Database: Known camera profiles and reconstruction hints
- Metadata Extraction: EXIF, GPS, and timestamp extraction
- Quality Analysis: Blur detection, exposure analysis, motion estimation
- Format Handlers: Video, images, OSV containers
"""

from modular_pipeline.ingest.extract import (
    IngestModule,
    IngestConfig,
)

from modular_pipeline.ingest.camera_db import (
    CameraProfile,
    CameraDatabase,
    camera_db,
    detect_camera_from_exif,
    get_camera_hints,
)

from modular_pipeline.ingest.metadata import (
    GPSPoint,
    ImageMetadata,
    MetadataExtractor,
    GPSTrack,
)

from modular_pipeline.ingest.quality import (
    QualityMetrics,
    QualityAnalyzer,
    MotionDetector,
    analyze_image_quality,
    filter_quality_images,
)

from modular_pipeline.ingest.formats import (
    VideoExtractor,
    VideoInfo,
    ImageImporter,
    OSVHandler,
    IMAGE_EXTENSIONS,
)


__all__ = [
    # Main module
    'IngestModule',
    'IngestConfig',
    
    # Camera database
    'CameraProfile',
    'CameraDatabase',
    'camera_db',
    'detect_camera_from_exif',
    'get_camera_hints',
    
    # Metadata
    'GPSPoint',
    'ImageMetadata',
    'MetadataExtractor',
    'GPSTrack',
    
    # Quality
    'QualityMetrics',
    'QualityAnalyzer',
    'MotionDetector',
    'analyze_image_quality',
    'filter_quality_images',
    
    # Format handlers
    'VideoExtractor',
    'VideoInfo',
    'ImageImporter',
    'OSVHandler',
    'IMAGE_EXTENSIONS',
]
