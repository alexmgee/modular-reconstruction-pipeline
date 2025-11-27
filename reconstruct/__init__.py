"""
SplatForge Reconstruct: 3D Reconstruction Modules

This package contains modules for the reconstruction stage:
- retrieve: Image pair generation via NetVLAD/vocab tree
- instant: InstantSplat fallback for challenging cases

Note: extract.py, match.py, and sfm.py are at the top level for now
      and will be moved here in a future refactor.
"""

from .retrieve import RetrievalModule, RetrievalConfig, RetrievalBackend
from .instant import InstantSplatModule, InstantSplatConfig

__all__ = [
    'RetrievalModule',
    'RetrievalConfig',
    'RetrievalBackend',
    'InstantSplatModule',
    'InstantSplatConfig',
]
