"""
SplatForge Mesh: Mesh Extraction Module

Extracts meshes from Gaussian splats using various SOTA methods.

Backends (from SOTA_2025.pdf):
- GOF: Best for outdoor/unbounded scenes (~45 min, SOTA recall)
- PGSR: Best textures (~30 min, 0.47 Chamfer distance)
- 2DGS: Best for thin surfaces (~30 min, clean geometry)
- SuGaR: Best for Blender/Unity editing (~20 min, editable mesh)

Usage:
    # CLI
    python -m modular_pipeline.output.mesh ./splat ./output --backend pgsr
    
    # Python API
    from modular_pipeline.output.mesh import MeshModule, MeshConfig
    
    config = MeshConfig(backend="pgsr", texture_resolution=2048)
    mesh = MeshModule(config)
    result = mesh.process(splat_path, output_path)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import logging

# Core imports
from modular_pipeline.core import (
    BaseModule, BaseConfig, StageResult, ItemResult,
    QualityLevel, Backend,
    ensure_directory, format_time,
    HAS_TORCH, HAS_NUMPY
)

try:
    import numpy as np
except ImportError:
    np = None


# =============================================================================
# CONFIGURATION
# =============================================================================

class MeshBackend(Enum):
    """Mesh extraction backends."""
    GOF = "gof"             # Gaussian Opaque Fusion (outdoor/unbounded)
    PGSR = "pgsr"           # Point-based Gaussian Splatting Rasterization
    TWODJGS = "2dgs"        # 2D Gaussian Splatting (thin surfaces)
    SUGAR = "sugar"         # Editable mesh for Blender/Unity
    AUTO = "auto"


@dataclass
class MeshConfig(BaseConfig):
    """Configuration for mesh extraction.
    
    Attributes:
        backend: Mesh extraction method (pgsr recommended for quality)
        
        # Mesh resolution
        resolution: Marching cubes grid resolution
        texture_resolution: Output texture size (width, height)
        
        # Quality settings
        poisson_depth: Poisson reconstruction depth (higher = more detail)
        simplify_ratio: Mesh decimation ratio (0.0-1.0, 0=no simplification)
        min_face_count: Minimum output face count
        max_face_count: Maximum output face count (triggers decimation)
        
        # Texture baking
        bake_textures: Generate texture maps
        bake_normals: Generate normal maps
        bake_albedo: Generate albedo maps
        
        # Export
        export_obj: Export as OBJ with materials
        export_ply: Export as PLY
    """
    # Backend selection
    backend: MeshBackend = MeshBackend.PGSR
    
    # Mesh resolution
    resolution: int = 512               # Marching cubes grid
    texture_resolution: int = 2048      # Texture size
    
    # Quality settings
    poisson_depth: int = 10             # Poisson reconstruction depth
    simplify_ratio: float = 0.0         # 0.0 = no decimation, 0.5 = half faces
    min_face_count: int = 10_000
    max_face_count: int = 1_000_000
    
    # Texture baking
    bake_textures: bool = True
    bake_normals: bool = False
    bake_albedo: bool = True
    
    # Export formats
    export_obj: bool = True
    export_ply: bool = False
    export_glb: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.backend, str):
            self.backend = MeshBackend(self.backend)


# =============================================================================
# MESH MODULE
# =============================================================================

class MeshModule(BaseModule):
    """Extract mesh from trained Gaussian splat.
    
    Supports multiple extraction methods optimized for different use cases.
    Generates textured meshes suitable for real-time rendering or editing.
    
    Example:
        >>> mesh = MeshModule(MeshConfig(backend="pgsr"))
        >>> result = mesh.process(
        ...     splat_path=Path("./splat"),
        ...     output_path=Path("./mesh")
        ... )
        >>> print(f"Mesh: {result.metrics['num_faces']:,} faces")
    """
    
    def __init__(self, config: Optional[MeshConfig] = None):
        super().__init__(config)
    
    def _default_config(self) -> MeshConfig:
        return MeshConfig()
    
    def _initialize(self) -> None:
        """Initialize mesh extraction backend."""
        self._extractor = None
        self._backend_name = self.config.backend.value
        
        try:
            self._load_backend()
        except Exception as e:
            self.log(f"Failed to load mesh extraction backend: {e}", "warning")
            self.log("Mesh extraction will not be available", "warning")
    
    def _load_backend(self) -> None:
        """Load mesh extraction backend."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required for mesh extraction")
        
        import torch
        device = self.device
        
        if self.config.backend == MeshBackend.GOF:
            try:
                # GOF import (stub)
                # from gof import GaussianOpaqueFusion
                
                self.log("GOF backend loading (stub implementation)")
                self.log("Full GOF integration requires installation", "warning")
                
                self._extractor = None
                return
                
            except ImportError as e:
                self.log(f"GOF not available: {e}", "warning")
                raise
        
        elif self.config.backend == MeshBackend.PGSR:
            try:
                # PGSR import (stub)
                # from pgsr import PointBasedGaussianRenderer
                
                self.log("PGSR backend loading (stub implementation)")
                self.log("Full PGSR integration requires installation", "warning")
                
                self._extractor = None
                return
                
            except ImportError as e:
                self.log(f"PGSR not available: {e}", "warning")
                raise
        
        elif self.config.backend == MeshBackend.TWODJGS:
            try:
                # 2DGS import (stub)
                # from gs2d import TwoDGaussianSplatting
                
                self.log("2DGS backend loading (stub implementation)")
                self.log("Full 2DGS integration requires installation", "warning")
                
                self._extractor = None
                return
                
            except ImportError as e:
                self.log(f"2DGS not available: {e}", "warning")
                raise
        
        elif self.config.backend == MeshBackend.SUGAR:
            try:
                # SuGaR import (stub)
                # from sugar import SuGaR
                
                self.log("SuGaR backend loading (stub implementation)")
                self.log("Full SuGaR integration requires installation", "warning")
                
                self._extractor = None
                return
                
            except ImportError as e:
                self.log(f"SuGaR not available: {e}", "warning")
                raise
    
    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------
    
    def process(
        self,
        splat_path: Path,
        output_path: Path,
        **kwargs
    ) -> StageResult:
        """Extract mesh from Gaussian splat.
        
        Args:
            splat_path: Path to trained Gaussian splat
            output_path: Output directory for mesh
        
        Returns:
            StageResult with mesh statistics
        """
        splat_path = Path(splat_path)
        output_path = Path(output_path)
        ensure_directory(output_path)
        
        # Check if extractor is available
        if self._extractor is None:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=["Mesh extraction backend not loaded. Install backend library."]
            )
        
        # Validate splat
        if not splat_path.exists():
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=[f"Splat not found: {splat_path}"]
            )
        
        self.log(f"Extracting mesh using {self._backend_name}")
        self.log(f"  Resolution: {self.config.resolution}³")
        self.log(f"  Texture: {self.config.texture_resolution}px")
        
        start_time = time.perf_counter()
        
        # Extract (stub)
        try:
            result = self._extract(splat_path, output_path)
        except Exception as e:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=[f"Extraction failed: {e}"]
            )
        
        elapsed = time.perf_counter() - start_time
        
        # Quality assessment based on face count
        num_faces = result.get('num_faces', 0)
        if self.config.min_face_count <= num_faces <= self.config.max_face_count:
            quality = QualityLevel.GOOD
            confidence = 0.85
        elif num_faces > 0:
            quality = QualityLevel.REVIEW
            confidence = 0.70
        else:
            quality = QualityLevel.POOR
            confidence = 0.50
        
        self.log(f"\nMesh extraction complete in {format_time(elapsed)}")
        self.log(f"  Vertices: {result.get('num_vertices', 0):,}")
        self.log(f"  Faces: {num_faces:,}")
        
        return self._create_result(
            success=result['success'],
            output_path=output_path,
            processing_time=elapsed,
            quality=quality,
            confidence=confidence,
            metrics=result
        )
    
    def _extract(
        self,
        splat_path: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """Run mesh extraction (stub implementation).
        
        Full implementation would:
        1. Load trained Gaussian splat
        2. Run marching cubes or Poisson reconstruction
        3. Generate UV maps
        4. Bake textures from Gaussians
        5. Simplify mesh if needed
        6. Export in requested formats
        
        Returns:
            Dict with mesh statistics
        """
        self.log("Mesh extraction (stub implementation)", "warning")
        self.log("This is a placeholder - full backend integration needed", "warning")
        
        # Return stub results
        return {
            'success': False,
            'num_vertices': 0,
            'num_faces': 0,
            'texture_size': 0,
            'error': f'Not yet implemented - requires {self._backend_name} installation'
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for mesh extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract mesh from Gaussian splat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("splat", type=Path, help="Path to trained splat")
    parser.add_argument("output", type=Path, help="Output directory")
    
    # Backend settings
    parser.add_argument("--backend", type=str, default="pgsr",
                       choices=["gof", "pgsr", "2dgs", "sugar"],
                       help="Mesh extraction method")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Marching cubes resolution")
    parser.add_argument("--texture-resolution", type=int, default=2048,
                       help="Texture resolution")
    
    # Quality settings
    parser.add_argument("--simplify", type=float, default=0.0,
                       help="Mesh decimation ratio (0.0-1.0)")
    parser.add_argument("--poisson-depth", type=int, default=10,
                       help="Poisson reconstruction depth")
    
    # Export
    parser.add_argument("--export-obj", action="store_true", default=True,
                       help="Export as OBJ")
    parser.add_argument("--export-glb", action="store_true", default=True,
                       help="Export as GLB")
    
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    config = MeshConfig(
        backend=MeshBackend(args.backend),
        resolution=args.resolution,
        texture_resolution=args.texture_resolution,
        simplify_ratio=args.simplify,
        poisson_depth=args.poisson_depth,
        export_obj=args.export_obj,
        export_glb=args.export_glb,
        device=args.device,
        verbose=args.verbose,
    )
    
    mesh = MeshModule(config)
    result = mesh.process(args.splat, args.output)
    
    if result.success:
        print(f"\n✓ Mesh extraction complete: {result.output_path}")
        print(f"  Vertices: {result.metrics.get('num_vertices', 0):,}")
        print(f"  Faces: {result.metrics.get('num_faces', 0):,}")
        print(f"  Time: {format_time(result.processing_time_seconds)}")
    else:
        print(f"\n✗ Mesh extraction failed")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
