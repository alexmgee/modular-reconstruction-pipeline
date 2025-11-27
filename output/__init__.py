"""
SplatForge Output: Final output generation modules

This package contains modules for the output stage:
- splat: Gaussian splatting training
- mesh: Mesh extraction from splats
- export: Format conversion and viewer integration
"""

from .splat import SplatModule, SplatConfig, SplatBackend
from .mesh import MeshModule, MeshConfig, MeshBackend
from .export import ExportModule, ExportConfig, ExportFormat

__all__ = [
    'SplatModule',
    'SplatConfig',
    'SplatBackend',
    'MeshModule',
    'MeshConfig',
    'MeshBackend',
    'ExportModule',
    'ExportConfig',
    'ExportFormat',
]
