"""
SplatForge InstantSplat: Dense 3D Reconstruction Fallback

Provides fast, robust reconstruction when GLOMAP/COLMAP fails or
produces poor results. Uses DUSt3R/MASt3R for dense 3D estimation.

Key advantages:
- 30x faster than traditional SfM
- Handles sparse views (3-10 images)
- Works with challenging cases (texture-poor, extreme motion)
- No feature matching required

Usage:
    # CLI
    python -m modular_pipeline.reconstruct.instant ./images ./output
    
    # Python API
    from modular_pipeline.reconstruct.instant import InstantSplatModule, InstantSplatConfig
    
    config = InstantSplatConfig(model_name="mast3r")
    instant = InstantSplatModule(config)
    result = instant.process(images_path, output_path)
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
    get_image_files, ensure_directory, format_time,
    HAS_TORCH, HAS_NUMPY
)

try:
    import numpy as np
except ImportError:
    np = None


# =============================================================================
# CONFIGURATION
# =============================================================================

class InstantBackend(Enum):
    """InstantSplat backend models."""
    MAST3R = "mast3r"         # MASt3R (recommended, most robust)
    DUST3R = "dust3r"         # DUSt3R (faster, less accurate)
    AUTO = "auto"


@dataclass
class InstantSplatConfig(BaseConfig):
    """Configuration for InstantSplat reconstruction.
    
    Attributes:
        backend: Model to use (mast3r recommended)
        model_name: Specific model checkpoint
        
        # Image settings
        image_size: Resize images to this size (512 or 224)
        max_images: Maximum images to process at once
        
        # Reconstruction settings
        confidence_threshold: Minimum confidence for points
        point_cloud_size: Target point cloud size
        
        # Output
        export_colmap: Export COLMAP-compatible reconstruction
        export_pointcloud: Export dense point cloud (PLY)
    """
    # Backend selection
    backend: InstantBackend = InstantBackend.MAST3R
    model_name: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    
    # Image processing
    image_size: int = 512         # 512 or 224
    max_images: int = 100         # Max images per batch
    
    # Reconstruction quality
    confidence_threshold: float = 1.0   # Min confidence
    point_cloud_size: int = 100_000     # Target points
    min_points_per_image: int = 1000    # Min points per image
    
    # Output options
    export_colmap: bool = True          # COLMAP format output
    export_pointcloud: bool = True      # PLY point cloud
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.backend, str):
            self.backend = InstantBackend(self.backend)


# =============================================================================
# INSTANTSPLAT MODULE
# =============================================================================

class InstantSplatModule(BaseModule):
    """Fast dense 3D reconstruction using MASt3R/DUSt3R.
    
    Provides fallback when traditional SfM fails or produces
    poor results. Uses dense matching without feature extraction.
    
    Example:
        >>> instant = InstantSplatModule(InstantSplatConfig())
        >>> result = instant.process(
        ...     images_path=Path("./images"),
        ...     output_path=Path("./sparse")
        ... )
        >>> print(f"Reconstructed {result.metrics['num_points3d']} points")
    """
    
    def __init__(self, config: Optional[InstantSplatConfig] = None):
        super().__init__(config)
    
    def _default_config(self) -> InstantSplatConfig:
        return InstantSplatConfig()
    
    def _initialize(self) -> None:
        """Load InstantSplat model."""
        self._model = None
        self._model_name = self.config.backend.value
        
        try:
            self._load_model()
        except Exception as e:
            self.log(f"Failed to load InstantSplat model: {e}", "warning")
            self.log("InstantSplat will not be available as fallback", "warning")
    
    def _load_model(self) -> None:
        """Load MASt3R or DUSt3R model."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required for InstantSplat")
        
        import torch
        device = self.device
        
        if self.config.backend == InstantBackend.MAST3R:
            try:
                # MASt3R import (stub - requires actual installation)
                # from mast3r.model import AsymmetricMASt3R
                
                # self._model = AsymmetricMASt3R.from_pretrained(
                #     self.config.model_name
                # ).eval().to(device)
                
                self.log(f"MASt3R model loading (stub implementation)")
                self.log("Full MASt3R integration requires: pip install mast3r", "warning")
                
                # Mark as not available
                self._model = None
                return
                
            except ImportError as e:
                self.log(f"MASt3R not available: {e}", "warning")
                raise
        
        elif self.config.backend == InstantBackend.DUST3R:
            try:
                # DUSt3R import (stub)
                # from dust3r.model import DUSt3R
                
                self.log("DUSt3R model loading (stub implementation)")
                self.log("Full DUSt3R integration requires installation", "warning")
                
                self._model = None
                return
                
            except ImportError as e:
                self.log(f"DUSt3R not available: {e}", "warning")
                raise
    
    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------
    
    def process(
        self,
        images_path: Path,
        output_path: Path,
        image_list: Optional[List[str]] = None,
        **kwargs
    ) -> StageResult:
        """Run InstantSplat reconstruction.
        
        Args:
            images_path: Directory containing images
            output_path: Output directory for reconstruction
            image_list: Optional list of specific images
        
        Returns:
            StageResult with reconstruction statistics
        """
        images_path = Path(images_path)
        output_path = Path(output_path)
        ensure_directory(output_path)
        
        # Check if model is available
        if self._model is None:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=["InstantSplat model not loaded. Install mast3r or dust3r."]
            )
        
        # Get image list
        if image_list:
            image_files = [images_path / img for img in image_list]
        else:
            image_files = get_image_files(images_path)
        
        if not image_files:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=["No images found"]
            )
        
        # Limit images if needed
        if len(image_files) > self.config.max_images:
            self.log(f"Limiting to {self.config.max_images} images (have {len(image_files)})", "warning")
            image_files = image_files[:self.config.max_images]
        
        self.log(f"Running InstantSplat on {len(image_files)} images")
        
        start_time = time.perf_counter()
        
        # Run reconstruction (stub)
        try:
            result = self._reconstruct(image_files, output_path)
        except Exception as e:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=[f"Reconstruction failed: {e}"]
            )
        
        elapsed = time.perf_counter() - start_time
        
        # Quality assessment
        if result['num_registered'] >= len(image_files) * 0.8:
            quality = QualityLevel.GOOD
            confidence = 0.85
        elif result['num_registered'] >= len(image_files) * 0.5:
            quality = QualityLevel.REVIEW
            confidence = 0.70
        else:
            quality = QualityLevel.POOR
            confidence = 0.50
        
        self.log(f"\nInstantSplat reconstruction complete in {format_time(elapsed)}")
        self.log(f"  Registered images: {result['num_registered']}/{len(image_files)}")
        self.log(f"  3D points: {result['num_points3d']:,}")
        
        return self._create_result(
            success=result['success'],
            output_path=output_path,
            items_processed=result['num_registered'],
            items_failed=len(image_files) - result['num_registered'],
            processing_time=elapsed,
            quality=quality,
            confidence=confidence,
            metrics=result
        )
    
    def _reconstruct(
        self,
        image_files: List[Path],
        output_path: Path
    ) -> Dict[str, Any]:
        """Run dense reconstruction (stub implementation).
        
        Full implementation would:
        1. Load and preprocess images
        2. Run MASt3R/DUSt3R inference
        3. Triangulate dense point cloud
        4. Estimate camera poses
        5. Export to COLMAP format
        
        Returns:
            Dict with reconstruction statistics
        """
        self.log("InstantSplat reconstruction (stub implementation)", "warning")
        self.log("This is a placeholder - full MASt3R/DUSt3R integration needed", "warning")
        
        # Return stub results
        return {
            'success': False,
            'num_images': len(image_files),
            'num_registered': 0,
            'num_points3d': 0,
            'mean_reproj_error': 0.0,
            'error': 'Not yet implemented - requires MASt3R/DUSt3R installation'
        }


# =============================================================================
# FAILURE DETECTION
# =============================================================================

def detect_sfm_failure(sfm_result: StageResult) -> Tuple[bool, str]:
    """Detect if SfM failed and should trigger InstantSplat fallback.
    
    Args:
        sfm_result: Result from SfMPipeline
    
    Returns:
        (should_fallback, reason) tuple
    """
    # SfM completely failed
    if not sfm_result.success:
        return True, "SfM failed to produce reconstruction"
    
    # Low registration rate
    if sfm_result.metrics.get('num_registered', 0) < sfm_result.metrics.get('num_images', 0) * 0.5:
        return True, f"Low registration rate: {sfm_result.metrics['num_registered']}/{sfm_result.metrics['num_images']}"
    
    # Very few 3D points
    if sfm_result.metrics.get('num_points3d', 0) < 100:
        return True, f"Insufficient 3D points: {sfm_result.metrics['num_points3d']}"
    
    # High reprojection error
    if sfm_result.metrics.get('mean_reproj_error', 0) > 2.0:
        return True, f"High reprojection error: {sfm_result.metrics['mean_reproj_error']:.2f}px"
    
    # Poor quality flag
    if sfm_result.quality == QualityLevel.POOR or sfm_result.quality == QualityLevel.REJECT:
        return True, f"Poor quality reconstruction ({sfm_result.quality.name})"
    
    return False, ""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for InstantSplat."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fast dense 3D reconstruction using MASt3R/DUSt3R",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("images", type=Path, help="Directory with images")
    parser.add_argument("output", type=Path, help="Output directory")
    
    # Model settings
    parser.add_argument("--backend", type=str, default="mast3r",
                       choices=["mast3r", "dust3r"],
                       help="Model backend")
    parser.add_argument("--image-size", type=int, default=512,
                       choices=[224, 512],
                       help="Image resolution")
    parser.add_argument("--max-images", type=int, default=100,
                       help="Maximum images to process")
    
    # Output settings
    parser.add_argument("--export-colmap", action="store_true", default=True,
                       help="Export COLMAP format")
    parser.add_argument("--export-pointcloud", action="store_true", default=True,
                       help="Export PLY point cloud")
    
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    config = InstantSplatConfig(
        backend=InstantBackend(args.backend),
        image_size=args.image_size,
        max_images=args.max_images,
        export_colmap=args.export_colmap,
        export_pointcloud=args.export_pointcloud,
        device=args.device,
        verbose=args.verbose,
    )
    
    instant = InstantSplatModule(config)
    result = instant.process(args.images, args.output)
    
    if result.success:
        print(f"\n✓ Reconstruction complete: {result.output_path}")
        print(f"  Registered: {result.metrics['num_registered']}/{result.metrics['num_images']} images")
        print(f"  3D points: {result.metrics['num_points3d']:,}")
        print(f"  Time: {format_time(result.processing_time_seconds)}")
    else:
        print(f"\n✗ Reconstruction failed")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
