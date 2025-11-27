"""
Prepare Module
==============

Preprocessing stage for reconstruction pipelines, including:
- Reframing: Convert 360° content to pinhole rig views
- Masking: Remove unwanted objects (tripods, operators, etc.)
- COLMAP rig JSON generation for bundle adjustment
"""

from modular_pipeline.prepare.reframe import ReframeConfig, ReframeModule
from modular_pipeline.prepare.masking import MaskingConfig, MaskingModule
from modular_pipeline.prepare.rig_json import RigJSONGenerator


__all__ = [
    # Reframe
    'ReframeConfig',
    'ReframeModule',
    
    # Masking
    'MaskingConfig',
    'MaskingModule',
    
    # Utilities
    'RigJSONGenerator',
]
