"""
Reframe Module
==============

Adapter module that wraps reframe_v2.py with BaseModule interface.
Converts equirectangular 360° images into pinhole camera rig views for
computer vision models and generates COLMAP rig configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import sys

# Import reframe_v2 components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from reframe_v2 import (
    EquirectToRig,
    RigGenerator,
    RigConfig as ReframeRigConfig,
    ProjectionBackend,
)

from modular_pipeline.core import (
    BaseModule,
    BaseConfig,
    StageResult,
    QualityLevel,
    ImageGeometry,
    get_image_files,
    ensure_directory,
)

from modular_pipeline.prepare.rig_json import RigJSONGenerator


@dataclass
class ReframeConfig(BaseConfig):
    """Configuration for reframe module."""
    
    # Rig pattern selection
    rig_pattern: str = "ring12"  # cube, ring8, ring12, geodesic, custom
    custom_rig_config: Optional[str] = None  # Path to custom rig JSON
    
    # Output dimensions
    output_width: int = 1920
    output_height: int = 1080
    fov_h: float = 90.0  # Horizontal FOV in degrees
    fov_v: float = 60.0  # Vertical FOV in degrees
    
    # Processing options
    projection_backend: str = "auto"  # auto, torch_gpu, py360, equilib, opencv
    interpolation: str = "bilinear"  # bilinear, nearest, cubic
    cache_projection_maps: bool = True
    
    # Passthrough behavior
    auto_detect_geometry: bool = True
    passthrough_pinhole: bool = True  # Skip reframe if already pinhole
    
    # COLMAP rig JSON generation
    generate_colmap_rig: bool = True
    
    # Output settings
    save_rig_visualization: bool = False


class ReframeModule(BaseModule):
    """Reframe module for converting 360° content to pinhole rig views."""
    
    def _default_config(self) -> ReframeConfig:
        """Return default configuration."""
        return ReframeConfig()
    
    def _initialize(self) -> None:
        """Initialize module resources."""
        self.config: ReframeConfig
        
        # Create rig configuration
        if self.config.custom_rig_config:
            self.rig_config = ReframeRigConfig.load(Path(self.config.custom_rig_config))
        else:
            self.rig_config = self._create_default_rig()
        
        # Map backend string to enum
        backend_map = {
            'auto': None,
            'torch_gpu': ProjectionBackend.TORCH_GPU,
            'py360': ProjectionBackend.PY360,
            'equilib': ProjectionBackend.EQUILIB,
            'opencv': ProjectionBackend.OPENCV,
            'numpy': ProjectionBackend.NUMPY,
        }
        backend = backend_map.get(self.config.projection_backend)
        
        # Initialize reframe processor
        self.processor = EquirectToRig(
            rig_config=self.rig_config,
            backend=backend,
            num_workers=self.config.num_workers,
            cache_maps=self.config.cache_projection_maps,
            debug=False,
        )
        
        self.log("Reframe module initialized")
        self.log(f"Rig pattern: {self.config.rig_pattern}")
        self.log(f"Backend: {self.processor.backend.value}")
    
    def _create_default_rig(self) -> ReframeRigConfig:
        """Create default rig based on pattern."""
        
        if self.config.rig_pattern == "cube":
            return RigGenerator.create_cube_rig(
                resolution=min(self.config.output_width, self.config.output_height),
                fov=self.config.fov_h
            )
        elif self.config.rig_pattern == "ring8":
            return RigGenerator.create_ring_rig(
                num_cameras=8,
                width=self.config.output_width,
                height=self.config.output_height,
                fov_h=self.config.fov_h,
                fov_v=self.config.fov_v,
            )
        elif self.config.rig_pattern == "ring12":
            return RigGenerator.create_ring_rig(
                num_cameras=12,
                width=self.config.output_width,
                height=self.config.output_height,
                fov_h=self.config.fov_h,
                fov_v=self.config.fov_v,
            )
        elif self.config.rig_pattern == "geodesic":
            return RigGenerator.create_geodesic_rig(
                resolution=min(self.config.output_width, self.config.output_height),
                fov=self.config.fov_h
            )
        else:
            raise ValueError(f"Unknown rig pattern: {self.config.rig_pattern}")
    
    def process(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> StageResult:
        """
        Process frames through reframe pipeline.
        
        Args:
            input_path: Input directory with frames and ingest_manifest.json
            output_path: Output directory for rig views
            **kwargs: Additional options
            
        Returns:
            StageResult with reframe outcome
        """
        import time
        import cv2
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        self.log(f"Processing reframe: {input_path}")
        
        # Check for ingest manifest to determine geometry
        manifest_path = input_path / "ingest_manifest.json"
        geometry = self._detect_geometry(manifest_path)
        
        self.log(f"Detected geometry: {geometry.value if geometry else 'unknown'}")
        
        # Passthrough if already pinhole
        if self.config.passthrough_pinhole and geometry == ImageGeometry.PINHOLE:
            self.log("Geometry is already pinhole, passing through")
            return self._passthrough(input_path, output_path)
        
        # Check if geometry requires reframing
        if geometry not in [ImageGeometry.EQUIRECTANGULAR, ImageGeometry.DUAL_FISHEYE]:
            self.log(f"Geometry {geometry.value} does not require reframing")
            return self._passthrough(input_path, output_path)
        
        # Find frames
        frames_dir = input_path / "frames"
        if not frames_dir.exists():
            frames_dir = input_path
        
        image_files = get_image_files(frames_dir)
        
        if not image_files:
            return self._create_result(
                success=False,
                output_path=output_path,
                quality=QualityLevel.REJECT,
                confidence=0.0,
                errors=["No image files found"]
            )
        
        self.log(f"Found {len(image_files)} frames to reframe")
        
        # Create output directories for each camera
        rig_views_dir = ensure_directory(output_path / "rig_views")
        for camera in self.rig_config.cameras:
            ensure_directory(rig_views_dir / camera.name)
        
        # Process frames
        start_time = time.perf_counter()
        
        processed = 0
        failed = 0
        
        for idx, image_path in enumerate(image_files):
            try:
                # Load equirectangular image
                equirect = cv2.imread(str(image_path))
                if equirect is None:
                    self.log(f"Failed to load: {image_path}", "warning")
                    failed += 1
                    continue
                
                # Process all views
                views = self.processor.process_image(
                    equirect,
                    camera=None,
                    interpolation=self.config.interpolation
                )
                
                # Save views
                frame_name = f"frame_{idx:06d}.jpg"
                for camera_name, view_img in views.items():
                    output_file = rig_views_dir / camera_name / frame_name
                    cv2.imwrite(str(output_file), view_img)
                
                processed += 1
                
                if (idx + 1) % 50 == 0:
                    self.log(f"Reframed {idx + 1}/{len(image_files)} frames")
                    
            except Exception as e:
                self.log(f"Error processing {image_path}: {e}", "error")
                failed += 1
        
        elapsed = time.perf_counter() - start_time
        
        # Save rig configuration
        rig_config_path = output_path / "rig_config.json"
        self.rig_config.save(rig_config_path)
        
        # Generate COLMAP rig JSON if requested
        colmap_rig_path = None
        if self.config.generate_colmap_rig:
            colmap_rig_path = output_path / "colmap_rig.json"
            rig_dict = self.rig_config.to_dict()
            RigJSONGenerator.from_rig_config(rig_dict, colmap_rig_path)
            self.log(f"Generated COLMAP rig JSON: {colmap_rig_path}")
        
        # Create visualization if requested
        if self.config.save_rig_visualization and image_files:
            viz_path = output_path / "rig_visualization.jpg"
            first_frame = cv2.imread(str(image_files[0]))
            if first_frame is not None:
                self.processor.visualize_rig(first_frame, viz_path)
                self.log(f"Saved rig visualization: {viz_path}")
        
        # Calculate quality metrics
        success_rate = processed / len(image_files) if image_files else 0.0
        confidence = success_rate
        quality = self.config.confidence_to_quality(confidence)
        
        # Create manifest
        manifest = {
            'input': {
                'path': str(input_path),
                'geometry': geometry.value if geometry else 'unknown',
                'total_frames': len(image_files),
            },
            'output': {
                'rig_views_dir': str(rig_views_dir),
                'rig_config': str(rig_config_path),
                'colmap_rig': str(colmap_rig_path) if colmap_rig_path else None,
            },
            'rig': {
                'pattern': self.config.rig_pattern,
                'num_cameras': len(self.rig_config.cameras),
                'camera_names': [cam.name for cam in self.rig_config.cameras],
            },
            'statistics': {
                'processed': processed,
                'failed': failed,
                'processing_time': elapsed,
                'average_time_per_frame': elapsed / processed if processed > 0 else 0,
            }
        }
        
        manifest_path = output_path / "reframe_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create result
        result = self._create_result(
            success=processed > 0,
            output_path=output_path,
            items_processed=processed,
            items_failed=failed,
            processing_time=elapsed,
            quality=quality,
            confidence=confidence,
            metrics={
                'rig_pattern': self.config.rig_pattern,
                'num_cameras': len(self.rig_config.cameras),
                'backend_used': self.processor.backend.value,
                'total_views_generated': processed * len(self.rig_config.cameras),
            }
        )
        
        # Save result manifest
        result.save_manifest(output_path / "stage_result.json")
        
        self.log(f"Reframe complete: {processed} frames → {processed * len(self.rig_config.cameras)} views")
        
        return result
    
    def _detect_geometry(self, manifest_path: Path) -> Optional[ImageGeometry]:
        """Detect geometry from ingest manifest."""
        
        if not self.config.auto_detect_geometry:
            return ImageGeometry.EQUIRECTANGULAR
        
        if not manifest_path.exists():
            self.log("No ingest manifest found, assuming equirectangular", "warning")
            return ImageGeometry.EQUIRECTANGULAR
        
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            geometry_str = manifest.get('camera', {}).get('geometry', 'pinhole')
            
            # Map string to enum
            geometry_map = {
                'pinhole': ImageGeometry.PINHOLE,
                'fisheye': ImageGeometry.FISHEYE,
                'dual_fisheye': ImageGeometry.DUAL_FISHEYE,
                'equirect': ImageGeometry.EQUIRECTANGULAR,
                'equirectangular': ImageGeometry.EQUIRECTANGULAR,
                'cubemap': ImageGeometry.CUBEMAP,
            }
            
            return geometry_map.get(geometry_str.lower(), ImageGeometry.PINHOLE)
            
        except Exception as e:
            self.log(f"Error reading manifest: {e}", "warning")
            return ImageGeometry.EQUIRECTANGULAR
    
    def _passthrough(self, input_path: Path, output_path: Path) -> StageResult:
        """Passthrough mode - copy frames without reframing."""
        import shutil
        import time
        
        self.log("Passthrough mode: copying frames without reframing")
        
        # Find frames
        frames_dir = input_path / "frames"
        if not frames_dir.exists():
            frames_dir = input_path
        
        image_files = get_image_files(frames_dir)
        
        # Create output directory
        output_frames_dir = ensure_directory(output_path / "frames")
        
        start_time = time.perf_counter()
        
        # Copy frames
        for src in image_files:
            dst = output_frames_dir / src.name
            shutil.copy2(src, dst)
        
        elapsed = time.perf_counter() - start_time
        
        # Copy ingest manifest if it exists
        manifest_src = input_path / "ingest_manifest.json"
        if manifest_src.exists():
            shutil.copy2(manifest_src, output_path / "ingest_manifest.json")
        
        # Create reframe manifest indicating passthrough
        manifest = {
            'mode': 'passthrough',
            'reason': 'input already in pinhole geometry',
            'input': str(input_path),
            'output': str(output_frames_dir),
            'frame_count': len(image_files),
            'processing_time': elapsed,
        }
        
        with open(output_path / "reframe_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create result
        result = self._create_result(
            success=True,
            output_path=output_path,
            items_processed=len(image_files),
            items_failed=0,
            processing_time=elapsed,
            quality=QualityLevel.EXCELLENT,
            confidence=1.0,
            metrics={
                'mode': 'passthrough',
                'frame_count': len(image_files),
            }
        )
        
        result.save_manifest(output_path / "stage_result.json")
        
        self.log(f"Passthrough complete: {len(image_files)} frames copied")
        
        return result


__all__ = [
    'ReframeConfig',
    'ReframeModule',
]
