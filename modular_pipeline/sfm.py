"""
SplatForge SfM: Structure from Motion Module

Runs sparse reconstruction using GLOMAP (recommended) or COLMAP.
Handles database creation, camera configuration, and mapper execution.

GLOMAP advantages over COLMAP:
- 8% higher recall on ETH3D
- 10-50x faster (joint global positioning)
- Better handling of sequential video
- Reduced drift accumulation

Usage:
    # CLI
    python -m splatforge.reconstruct.sfm ./prepared ./sparse --mapper glomap
    
    # Python API
    from splatforge.reconstruct.sfm import SfMPipeline, SfMConfig
    
    config = SfMConfig(mapper="glomap", camera_model="SIMPLE_RADIAL")
    sfm = SfMPipeline(config)
    result = sfm.process(images_path, features_path, matches_path, output_path)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import subprocess
import shutil
import time
import json
import logging

# Core imports
from modular_pipeline.core import (
    BaseModule, BaseConfig, StageResult, ItemResult,
    QualityLevel, Backend, CameraModel,
    ensure_directory, format_time,
    HAS_TORCH, HAS_NUMPY
)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class MapperBackend(Enum):
    """SfM mapper backends."""
    GLOMAP = "glomap"       # Global SfM (recommended)
    COLMAP = "colmap"       # Incremental SfM (fallback)
    AUTO = "auto"           # Auto-select based on dataset


class CameraModeEnum(Enum):
    """COLMAP camera mode for intrinsics."""
    AUTO = "auto"                   # One camera per EXIF
    SINGLE = "single"               # All images share one camera
    PER_FOLDER = "per_folder"       # One camera per subfolder
    PER_IMAGE = "per_image"         # Each image has own camera


@dataclass
class SfMConfig(BaseConfig):
    """Configuration for Structure from Motion.
    
    Attributes:
        mapper: Which SfM backend to use (glomap recommended)
        camera_model: COLMAP camera model type
        camera_mode: How to share camera intrinsics
        
        # GLOMAP options
        max_num_tracks: Maximum tracks for large datasets
        max_num_iterations: Bundle adjustment iterations
        
        # Post-processing
        run_point_triangulator: Re-triangulate for denser points
        run_bundle_adjustment: Final bundle adjustment
    """
    # Mapper selection
    mapper: MapperBackend = MapperBackend.GLOMAP
    
    # Camera settings
    camera_model: CameraModel = CameraModel.SIMPLE_RADIAL
    camera_mode: CameraModeEnum = CameraModeEnum.AUTO
    camera_params: Optional[str] = None  # Manual camera params
    
    # GLOMAP settings
    max_num_tracks: int = 6_000_000
    global_positioning_max_iter: int = 100
    bundle_adjustment_max_iter: int = 100
    
    # COLMAP settings (fallback)
    colmap_mapper_options: Dict[str, Any] = field(default_factory=dict)
    
    # Post-processing
    run_point_triangulator: bool = True    # Denser point cloud
    run_bundle_adjustment: bool = False    # Extra BA pass
    
    # Triangulator settings
    tri_min_angle: float = 1.5            # Min triangulation angle (degrees)
    tri_max_error: float = 4.0            # Max reprojection error (pixels)
    tri_max_dist: float = 100.0           # Max triangulation distance
    
    # Executables
    glomap_path: str = "glomap"
    colmap_path: str = "colmap"
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.mapper, str):
            self.mapper = MapperBackend(self.mapper)
        if isinstance(self.camera_model, str):
            self.camera_model = CameraModel(self.camera_model)
        if isinstance(self.camera_mode, str):
            self.camera_mode = CameraModeEnum(self.camera_mode)


# =============================================================================
# SFM PIPELINE
# =============================================================================

class SfMPipeline(BaseModule):
    """Run Structure from Motion reconstruction.
    
    Pipeline:
    1. Create COLMAP database
    2. Import images with camera model
    3. Import features and matches
    4. Run mapper (GLOMAP or COLMAP)
    5. Optional: Point triangulator for denser cloud
    
    Example:
        >>> sfm = SfMPipeline(SfMConfig(mapper="glomap"))
        >>> result = sfm.process(
        ...     images_path=Path("./images"),
        ...     features_path=Path("./features.h5"),
        ...     matches_path=Path("./matches.h5"),
        ...     pairs_path=Path("./pairs.txt"),
        ...     output_path=Path("./sparse")
        ... )
    """
    
    def __init__(self, config: Optional[SfMConfig] = None):
        super().__init__(config)
    
    def _default_config(self) -> SfMConfig:
        return SfMConfig()
    
    def _initialize(self) -> None:
        """Check for required executables."""
        self._glomap_available = self._check_executable(self.config.glomap_path)
        self._colmap_available = self._check_executable(self.config.colmap_path)
        
        if self.config.mapper == MapperBackend.GLOMAP and not self._glomap_available:
            self.log("GLOMAP not found, will fall back to COLMAP", "warning")
        
        if not self._colmap_available and not self._glomap_available:
            self.log("Neither GLOMAP nor COLMAP found in PATH", "error")
    
    def _check_executable(self, name: str) -> bool:
        """Check if executable is available."""
        return shutil.which(name) is not None
    
    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------
    
    def process(
        self,
        images_path: Path,
        features_path: Path,
        matches_path: Path,
        pairs_path: Path,
        output_path: Path,
        masks_path: Optional[Path] = None,
        **kwargs
    ) -> StageResult:
        """Run complete SfM pipeline.
        
        Args:
            images_path: Directory containing images
            features_path: Path to features.h5
            matches_path: Path to matches.h5
            pairs_path: Path to pairs.txt
            output_path: Output directory for sparse reconstruction
            masks_path: Optional path to mask images
        
        Returns:
            StageResult with reconstruction statistics
        """
        images_path = Path(images_path)
        output_path = Path(output_path)
        ensure_directory(output_path)
        
        database_path = output_path / 'database.db'
        sparse_path = output_path / 'sparse'
        
        start_time = time.perf_counter()
        errors = []
        warnings = []
        
        try:
            # Step 1: Create database
            self.log("Creating COLMAP database...")
            self._create_database(database_path, images_path)
            
            # Step 2: Import features and matches
            self.log("Importing features and matches...")
            self._import_features(database_path, images_path, features_path)
            self._import_matches(database_path, images_path, pairs_path, matches_path)
            
            # Step 3: Run mapper
            self.log(f"Running {self.config.mapper.value} mapper...")
            ensure_directory(sparse_path)
            
            if self.config.mapper == MapperBackend.GLOMAP and self._glomap_available:
                success = self._run_glomap(database_path, images_path, sparse_path)
            else:
                if self.config.mapper == MapperBackend.GLOMAP:
                    warnings.append("GLOMAP unavailable, using COLMAP")
                success = self._run_colmap(database_path, images_path, sparse_path)
            
            if not success:
                errors.append("Mapper failed")
                return self._create_result(
                    success=False,
                    output_path=output_path,
                    errors=errors,
                    warnings=warnings
                )
            
            # Step 4: Post-processing
            if self.config.run_point_triangulator:
                self.log("Running point triangulator...")
                self._run_triangulator(database_path, images_path, sparse_path)
            
            # Gather statistics
            stats = self._gather_reconstruction_stats(sparse_path)
            
        except Exception as e:
            errors.append(str(e))
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=errors
            )
        
        elapsed = time.perf_counter() - start_time
        
        # Quality assessment
        if stats['num_registered'] >= stats['num_images'] * 0.9:
            quality = QualityLevel.EXCELLENT
            confidence = 0.95
        elif stats['num_registered'] >= stats['num_images'] * 0.7:
            quality = QualityLevel.GOOD
            confidence = 0.85
        elif stats['num_registered'] >= stats['num_images'] * 0.5:
            quality = QualityLevel.REVIEW
            confidence = 0.70
        else:
            quality = QualityLevel.POOR
            confidence = 0.50
        
        self.log(f"\nReconstruction complete in {format_time(elapsed)}")
        self.log(f"  Registered images: {stats['num_registered']}/{stats['num_images']}")
        self.log(f"  3D points: {stats['num_points3d']:,}")
        self.log(f"  Mean track length: {stats['mean_track_length']:.1f}")
        self.log(f"  Mean reprojection error: {stats['mean_reproj_error']:.2f}px")
        
        result = self._create_result(
            success=True,
            output_path=sparse_path,
            items_processed=stats['num_registered'],
            items_failed=stats['num_images'] - stats['num_registered'],
            processing_time=elapsed,
            quality=quality,
            confidence=confidence,
            needs_review=quality in [QualityLevel.REVIEW, QualityLevel.POOR],
            metrics=stats,
            warnings=warnings
        )
        
        if self.config.save_manifests:
            result.save_manifest()
        
        return result
    
    # -------------------------------------------------------------------------
    # Database Operations
    # -------------------------------------------------------------------------
    
    def _create_database(self, database_path: Path, images_path: Path) -> None:
        """Create COLMAP database and import images."""
        if database_path.exists():
            database_path.unlink()
        
        if HAS_PYCOLMAP:
            # Use pycolmap
            import pycolmap
            
            pycolmap.Database(database_path).close()
            
            # Determine camera mode
            if self.config.camera_mode == CameraModeEnum.SINGLE:
                camera_mode = pycolmap.CameraMode.SINGLE
            elif self.config.camera_mode == CameraModeEnum.PER_FOLDER:
                camera_mode = pycolmap.CameraMode.PER_FOLDER
            elif self.config.camera_mode == CameraModeEnum.PER_IMAGE:
                camera_mode = pycolmap.CameraMode.PER_IMAGE
            else:
                camera_mode = pycolmap.CameraMode.AUTO
            
            # Get camera model
            camera_model = self.config.camera_model.value
            
            # Import images
            pycolmap.import_images(
                database_path,
                images_path,
                camera_mode=camera_mode,
                camera_model=camera_model,
            )
        else:
            # Use COLMAP CLI
            cmd = [
                self.config.colmap_path, 'feature_extractor',
                '--database_path', str(database_path),
                '--image_path', str(images_path),
                '--ImageReader.single_camera', '0',
                '--ImageReader.camera_model', self.config.camera_model.value,
                '--SiftExtraction.use_gpu', '0',  # Skip SIFT, we import our features
            ]
            subprocess.run(cmd, check=True, capture_output=True)
    
    def _import_features(
        self,
        database_path: Path,
        images_path: Path,
        features_path: Path
    ) -> None:
        """Import pre-extracted features into database."""
        if HAS_PYCOLMAP:
            from hloc import reconstruction
            reconstruction.import_features(images_path, database_path, features_path)
        else:
            raise RuntimeError("pycolmap required for feature import")
    
    def _import_matches(
        self,
        database_path: Path,
        images_path: Path,
        pairs_path: Path,
        matches_path: Path
    ) -> None:
        """Import pre-computed matches into database."""
        if HAS_PYCOLMAP:
            from hloc import reconstruction
            reconstruction.import_matches(
                images_path, database_path, pairs_path, matches_path
            )
        else:
            raise RuntimeError("pycolmap required for match import")
    
    # -------------------------------------------------------------------------
    # Mapper Execution
    # -------------------------------------------------------------------------
    
    def _run_glomap(
        self,
        database_path: Path,
        images_path: Path,
        output_path: Path
    ) -> bool:
        """Run GLOMAP mapper."""
        cmd = [
            self.config.glomap_path, 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(images_path),
            '--output_path', str(output_path),
            '--TrackEstablishment.max_num_tracks', str(self.config.max_num_tracks),
            '--GlobalPositioning.max_num_iterations', str(self.config.global_positioning_max_iter),
            '--BundleAdjustment.max_num_iterations', str(self.config.bundle_adjustment_max_iter),
        ]
        
        self.log(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            if self.config.verbose:
                self.log(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"GLOMAP failed: {e.stderr}", "error")
            return False
    
    def _run_colmap(
        self,
        database_path: Path,
        images_path: Path,
        output_path: Path
    ) -> bool:
        """Run COLMAP incremental mapper."""
        cmd = [
            self.config.colmap_path, 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(images_path),
            '--output_path', str(output_path),
        ]
        
        # Add any custom options
        for key, value in self.config.colmap_mapper_options.items():
            cmd.extend([f'--{key}', str(value)])
        
        self.log(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"COLMAP failed: {e.stderr}", "error")
            return False
    
    def _run_triangulator(
        self,
        database_path: Path,
        images_path: Path,
        sparse_path: Path
    ) -> Dict[str, int]:
        """Run point triangulator for denser point cloud.
        
        Returns:
            Dict with before/after point counts
        """
        # Find all model directories
        model_dirs = sorted([d for d in sparse_path.iterdir() if d.is_dir()])
        if not model_dirs:
            self.log("No model found for triangulation", "warning")
            return {'points_before': 0, 'points_after': 0}
        
        stats = {}
        
        for model_path in model_dirs:
            # Get initial point count
            points_before = self._count_points(model_path)
            
            cmd = [
                self.config.colmap_path, 'point_triangulator',
                '--database_path', str(database_path),
                '--image_path', str(images_path),
                '--input_path', str(model_path),
                '--output_path', str(model_path),
                '--Mapper.tri_min_angle', str(self.config.tri_min_angle),
                '--Mapper.tri_max_transitivity', str(self.config.tri_max_error),
                '--Mapper.filter_max_reproj_error', str(self.config.tri_max_error),
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, check=True, text=True)
                
                # Get final point count
                points_after = self._count_points(model_path)
                points_added = points_after - points_before
                
                self.log(f"Point triangulation ({model_path.name}): "
                        f"{points_before:,} → {points_after:,} points (+{points_added:,})")
                
                stats[model_path.name] = {
                    'before': points_before,
                    'after': points_after,
                    'added': points_added
                }
                
            except subprocess.CalledProcessError as e:
                self.log(f"Triangulation failed for {model_path.name}: {e.stderr}", "warning")
        
        return stats
    
    def _count_points(self, model_path: Path) -> int:
        """Count 3D points in a reconstruction."""
        if HAS_PYCOLMAP:
            try:
                import pycolmap
                reconstruction = pycolmap.Reconstruction(model_path)
                return len(reconstruction.points3D)
            except Exception as e:
                self.log(f"Could not count points with pycolmap: {e}", "warning")
        
        # Fallback: read points3D.bin or points3D.txt
        points_bin = model_path / 'points3D.bin'
        points_txt = model_path / 'points3D.txt'
        
        if points_bin.exists():
            # Rough estimate from binary file size
            file_size = points_bin.stat().st_size
            # Each point is ~43 bytes in binary format
            return file_size // 43
        elif points_txt.exists():
            # Count lines in text file (skip header/comments)
            with open(points_txt) as f:
                return sum(1 for line in f if line.strip() and not line.startswith('#'))
        
        return 0
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def _gather_reconstruction_stats(self, sparse_path: Path) -> Dict[str, Any]:
        """Gather reconstruction statistics."""
        stats = {
            'num_images': 0,
            'num_registered': 0,
            'num_points3d': 0,
            'mean_track_length': 0.0,
            'mean_reproj_error': 0.0,
            'num_observations': 0,
        }
        
        # Find model directory
        model_dirs = list(sparse_path.glob('*'))
        if not model_dirs:
            return stats
        
        model_path = model_dirs[0]
        
        if HAS_PYCOLMAP:
            try:
                import pycolmap
                reconstruction = pycolmap.Reconstruction(model_path)
                
                stats['num_images'] = len(reconstruction.images)
                stats['num_registered'] = len([
                    img for img in reconstruction.images.values()
                    if img.registered
                ])
                stats['num_points3d'] = len(reconstruction.points3D)
                
                # Compute mean track length and reprojection error
                if reconstruction.points3D:
                    track_lengths = []
                    reproj_errors = []
                    for point in reconstruction.points3D.values():
                        track_lengths.append(len(point.track.elements))
                        reproj_errors.append(point.error)
                    
                    stats['mean_track_length'] = np.mean(track_lengths)
                    stats['mean_reproj_error'] = np.mean(reproj_errors)
                    stats['num_observations'] = sum(track_lengths)
            except Exception as e:
                self.log(f"Could not read reconstruction stats: {e}", "warning")
        
        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_colmap_database(
    database_path: Path,
    images_path: Path,
    features_path: Path,
    matches_path: Path,
    pairs_path: Path,
    camera_model: str = "SIMPLE_RADIAL",
    camera_mode: str = "AUTO"
) -> Path:
    """Create a COLMAP database from pre-extracted features.
    
    This is a convenience function for creating a database
    without running the full pipeline.
    """
    config = SfMConfig(
        camera_model=CameraModel(camera_model),
        camera_mode=CameraModeEnum(camera_mode),
    )
    
    sfm = SfMPipeline(config)
    sfm._create_database(database_path, images_path)
    sfm._import_features(database_path, images_path, features_path)
    sfm._import_matches(database_path, images_path, pairs_path, matches_path)
    
    return database_path


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for SfM."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Structure from Motion reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("images", type=Path, help="Directory with images")
    parser.add_argument("features", type=Path, help="Path to features.h5")
    parser.add_argument("matches", type=Path, help="Path to matches.h5")
    parser.add_argument("pairs", type=Path, help="Path to pairs.txt")
    parser.add_argument("output", type=Path, help="Output directory")
    
    # Mapper settings
    parser.add_argument("--mapper", type=str, default="glomap",
                       choices=["glomap", "colmap", "auto"],
                       help="SfM mapper to use")
    parser.add_argument("--camera-model", type=str, default="SIMPLE_RADIAL",
                       help="COLMAP camera model")
    parser.add_argument("--camera-mode", type=str, default="auto",
                       choices=["auto", "single", "per_folder", "per_image"],
                       help="Camera intrinsic sharing mode")
    
    # GLOMAP settings
    parser.add_argument("--max-tracks", type=int, default=6_000_000,
                       help="Maximum number of tracks")
    
    # Post-processing
    parser.add_argument("--triangulate", action="store_true", default=True,
                       help="Run point triangulator after mapping")
    parser.add_argument("--no-triangulate", action="store_false", dest="triangulate")
    
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    config = SfMConfig(
        mapper=MapperBackend(args.mapper),
        camera_model=CameraModel(args.camera_model),
        camera_mode=CameraModeEnum(args.camera_mode),
        max_num_tracks=args.max_tracks,
        run_point_triangulator=args.triangulate,
        verbose=args.verbose,
    )
    
    sfm = SfMPipeline(config)
    result = sfm.process(
        images_path=args.images,
        features_path=args.features,
        matches_path=args.matches,
        pairs_path=args.pairs,
        output_path=args.output
    )
    
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
