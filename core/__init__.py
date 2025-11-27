"""
SplatForge Core: Shared foundation for all pipeline modules.

This package provides base classes, configurations, and utilities used
throughout the reconstruction pipeline.
"""

from .base import (
    # Enums
    QualityLevel,
    Backend,
    ImageGeometry,
    CameraModel,
    
    # Result types
    StageResult,
    ItemResult,
    
    # Configuration
    BaseConfig,
    
    # Base class
    BaseModule,
    
    # Device management
    DeviceManager,
    
    # Utilities
    get_image_files,
    ensure_directory,
    format_time,
    format_size,
    
    # Feature flags
    HAS_TORCH,
    HAS_NUMPY,
    HAS_YAML,
)

__all__ = [
    # Enums
    'QualityLevel',
    'Backend',
    'ImageGeometry',
    'CameraModel',
    
    # Result types
    'StageResult',
    'ItemResult',
    
    # Configuration
    'BaseConfig',
    
    # Base class
    'BaseModule',
    
    # Device management
    'DeviceManager',
    
    # Utilities
    'get_image_files',
    'ensure_directory',
    'format_time',
    'format_size',
    
    # Feature flags
    'HAS_TORCH',
    'HAS_NUMPY',
    'HAS_YAML',
]
