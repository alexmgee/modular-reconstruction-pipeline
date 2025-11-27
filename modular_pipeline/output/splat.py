"""
SplatForge Splat: Gaussian Splatting Training Module

Trains Gaussian splats from sparse COLMAP reconstructions using gsplat or nerfstudio.

Key features:
- Fast training with gsplat (CUDA-optimized)
- Configurable training parameters
- Real-time viewer support
- Quality metrics (PSNR, SSIM, LPIPS)
- Export to standard formats

Usage:
    # CLI
    python -m modular_pipeline.output.splat ./sparse ./output --iterations 30000
    
    # Python API
    from modular_pipeline.output.splat import SplatModule, SplatConfig
    
    config = SplatConfig(num_iterations=30000, sh_degree=3)
    splat = SplatModule(config)
    result = splat.process(sparse_path, output_path)
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

class SplatBackend(Enum):
    """Gaussian splatting training backends."""
    GSPLAT = "gsplat"                 # CUDA-optimized (recommended)
    NERFSTUDIO = "nerfstudio"         # nerfstudio 3DGS
    INRIA_3DGS = "inria_3dgs"         # Original 3DGS implementation
    AUTO = "auto"


@dataclass
class SplatConfig(BaseConfig):
    """Configuration for Gaussian splatting training.
    
    Attributes:
        backend: Training backend (gsplat recommended)
        num_iterations: Total training iterations
        densification_interval: Densify every N iterations
        densification_start: Start densification at iteration N
        densification_end: Stop densification at iteration N
        
        # Model settings
        sh_degree: Spherical harmonics degree (0-3, higher = better view-dependence)
        
        # Learning rates
        position_lr_init: Initial position learning rate
        position_lr_final: Final position learning rate
        feature_lr: Feature learning rate
        opacity_lr: Opacity learning rate
        scaling_lr: Scaling learning rate
        rotation_lr: Rotation learning rate
        
        # Output
        save_checkpoint_interval: Save checkpoint every N iterations
        export_ply: Export final splat as PLY
        viewer_enabled: Enable real-time viewer
    """
    # Backend selection
    backend: SplatBackend = SplatBackend.GSPLAT
    
    # Training iterations
    num_iterations: int = 30_000
    
    # Densification schedule
    densification_interval: int = 100
    densification_start: int = 500
    densification_end: int = 15_000
    densify_grad_threshold: float = 0.0002
    
    # Model settings
    sh_degree: int = 3                # 0=diffuse, 3=view-dependent
    
    # Learning rates
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    
    # Pruning
    opacity_cull_threshold: float = 0.005
    
    # Output
    save_checkpoint_interval: int = 5_000
    export_ply: bool = True
    viewer_enabled: bool = False
    viewer_port: int = 7007
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.backend, str):
            self.backend = SplatBackend(self.backend)


# =============================================================================
# SPLAT MODULE
# =============================================================================

class SplatModule(BaseModule):
    """Train Gaussian splats from COLMAP sparse reconstruction.
    
    Loads camera poses and point cloud from COLMAP, then trains
    3D Gaussians to represent the scene with novel view synthesis.
    
    Example:
        >>> splat = SplatModule(SplatConfig(num_iterations=30000))
        >>> result = splat.process(
        ...     sparse_path=Path("./sparse/0"),
        ...     output_path=Path("./output")
        ... )
        >>> print(f"Final PSNR: {result.metrics['psnr']:.2f}dB")
    """
    
    def __init__(self, config: Optional[SplatConfig] = None):
        super().__init__(config)
    
    def _default_config(self) -> SplatConfig:
        return SplatConfig()
    
    def _initialize(self) -> None:
        """Initialize training backend."""
        self._trainer = None
        self._backend_name = self.config.backend.value
        
        try:
            self._load_backend()
        except Exception as e:
            self.log(f"Failed to load training backend: {e}", "warning")
            self.log("Splat training will not be available", "warning")
    
    def _load_backend(self) -> None:
        """Load Gaussian splatting training backend."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required for Gaussian splatting")
        
        import torch
        device = self.device
        
        if self.config.backend == SplatBackend.GSPLAT:
            try:
                # gsplat import (stub - requires installation)
                # from gsplat import GaussianSplatTrainer
                
                # self._trainer = GaussianSplatTrainer(
                #     sh_degree=self.config.sh_degree,
                #     ...
                # )
                
                self.log("gsplat backend loading (stub implementation)")
                self.log("Full gsplat integration requires: pip install gsplat", "warning")
                
                self._trainer = None
                return
                
            except ImportError as e:
                self.log(f"gsplat not available: {e}", "warning")
                raise
        
        elif self.config.backend == SplatBackend.NERFSTUDIO:
            try:
                # nerfstudio import (stub)
                # from nerfstudio.models.splatfacto import SplatfactoModel
                
                self.log("nerfstudio backend loading (stub implementation)")
                self.log("Full nerfstudio integration requires installation", "warning")
                
                self._trainer = None
                return
                
            except ImportError as e:
                self.log(f"nerfstudio not available: {e}", "warning")
                raise
    
    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------
    
    def process(
        self,
        sparse_path: Path,
        output_path: Path,
        images_path: Optional[Path] = None,
        **kwargs
    ) -> StageResult:
        """Train Gaussian splat from COLMAP reconstruction.
        
        Args:
            sparse_path: Path to COLMAP sparse reconstruction (e.g., sparse/0)
            output_path: Output directory for trained splat
            images_path: Path to training images (auto-detected if None)
        
        Returns:
            StageResult with training statistics
        """
        sparse_path = Path(sparse_path)
        output_path = Path(output_path)
        ensure_directory(output_path)
        
        # Check if trainer is available
        if self._trainer is None:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=["Splat training backend not loaded. Install gsplat or nerfstudio."]
            )
        
        # Validate sparse reconstruction
        if not sparse_path.exists():
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=[f"Sparse reconstruction not found: {sparse_path}"]
            )
        
        # Auto-detect images path if not provided
        if images_path is None:
            images_path = self._detect_images_path(sparse_path)
        
        if not images_path or not images_path.exists():
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=["Images path not found. Specify --images argument."]
            )
        
        self.log(f"Training Gaussian splat for {self.config.num_iterations} iterations")
        self.log(f"  Sparse model: {sparse_path}")
        self.log(f"  Images: {images_path}")
        
        start_time = time.perf_counter()
        
        # Train (stub)
        try:
            result = self._train(sparse_path, images_path, output_path)
        except Exception as e:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=[f"Training failed: {e}"]
            )
        
        elapsed = time.perf_counter() - start_time
        
        # Quality assessment based on PSNR
        psnr = result.get('psnr', 0)
        if psnr >= 30.0:
            quality = QualityLevel.EXCELLENT
            confidence = 0.95
        elif psnr >= 25.0:
            quality = QualityLevel.GOOD
            confidence = 0.85
        elif psnr >= 20.0:
            quality = QualityLevel.REVIEW
            confidence = 0.70
        else:
            quality = QualityLevel.POOR
            confidence = 0.50
        
        self.log(f"\nTraining complete in {format_time(elapsed)}")
        self.log(f"  Final PSNR: {psnr:.2f}dB")
        if 'ssim' in result:
            self.log(f"  Final SSIM: {result['ssim']:.4f}")
        
        return self._create_result(
            success=result['success'],
            output_path=output_path,
            processing_time=elapsed,
            quality=quality,
            confidence=confidence,
            metrics=result
        )
    
    def _train(
        self,
        sparse_path: Path,
        images_path: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """Run Gaussian splatting training (stub implementation).
        
        Full implementation would:
        1. Load COLMAP reconstruction (cameras, images, points)
        2. Initialize Gaussians from point cloud
        3. Setup optimizers with learning rate schedules
        4. Training loop with densification/pruning
        5. Validation with PSNR/SSIM metrics
        6. Export final model
        
        Returns:
            Dict with training statistics
        """
        self.log("Gaussian splatting training (stub implementation)", "warning")
        self.log("This is a placeholder - full backend integration needed", "warning")
        
        # Return stub results
        return {
            'success': False,
            'iterations': 0,
            'psnr': 0.0,
            'ssim': 0.0,
            'num_gaussians': 0,
            'error': 'Not yet implemented - requires gsplat/nerfstudio installation'
        }
    
    def _detect_images_path(self, sparse_path: Path) -> Optional[Path]:
        """Auto-detect images directory from sparse reconstruction path."""
        # Common patterns:
        # sparse_path: ./output/sparse/0
        # images:      ./output/../images or ./images
        
        project_root = sparse_path.parent.parent
        
        # Try common locations
        for images_dir in ['images', 'prepared', 'frames', 'masked']:
            path = project_root / images_dir
            if path.exists() and path.is_dir():
                return path
        
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for Gaussian splatting training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Gaussian splats from COLMAP reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("sparse", type=Path, help="Path to COLMAP sparse reconstruction")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--images", type=Path, help="Path to training images (auto-detected if omitted)")
    
    # Training settings
    parser.add_argument("--backend", type=str, default="gsplat",
                       choices=["gsplat", "nerfstudio", "inria_3dgs"],
                       help="Training backend")
    parser.add_argument("--iterations", type=int, default=30_000,
                       help="Training iterations")
    parser.add_argument("--sh-degree", type=int, default=3,
                       choices=[0, 1, 2, 3],
                       help="Spherical harmonics degree")
    
    # Viewer
    parser.add_argument("--viewer", action="store_true",
                       help="Enable real-time training viewer")
    parser.add_argument("--viewer-port", type=int, default=7007,
                       help="Viewer port")
    
    # Output
    parser.add_argument("--export-ply", action="store_true", default=True,
                       help="Export final splat as PLY")
    
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    config = SplatConfig(
        backend=SplatBackend(args.backend),
        num_iterations=args.iterations,
        sh_degree=args.sh_degree,
        viewer_enabled=args.viewer,
        viewer_port=args.viewer_port,
        export_ply=args.export_ply,
        device=args.device,
        verbose=args.verbose,
    )
    
    splat = SplatModule(config)
    result = splat.process(args.sparse, args.output, images_path=args.images)
    
    if result.success:
        print(f"\n✓ Training complete: {result.output_path}")
        print(f"  PSNR: {result.metrics.get('psnr', 0):.2f}dB")
        print(f"  Gaussians: {result.metrics.get('num_gaussians', 0):,}")
        print(f"  Time: {format_time(result.processing_time_seconds)}")
    else:
        print(f"\n✗ Training failed")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
