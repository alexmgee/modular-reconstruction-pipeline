"""
SplatForge: Modular 3D Reconstruction Pipeline

A cohesive, modular system for converting 360° video and images into
3D reconstructions (Gaussian splats, meshes) with state-of-the-art methods.

Example:
    >>> from modular_pipeline import Pipeline
    >>> pipeline = Pipeline("./my_project", preset="osmo_360")
    >>> result = pipeline.run()
    >>> print(result.summary())
"""

from .pipeline import Pipeline, PipelinePreset, PipelineResult, Project

__version__ = "2.0.0"

__all__ = [
    'Pipeline',
    'PipelinePreset', 
    'PipelineResult',
    'Project',
]
