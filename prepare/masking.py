"""
Masking Module
==============

Adapter module that wraps masking_v2.py with BaseModule interface.
Generates segmentation masks using SAM3/FastSAM to remove tripods,
operators, and unwanted objects from reconstruction pipelines.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import sys

# Import masking_v2 components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from masking_v2 import (
    MaskingPipeline,
    MaskConfig as MaskingV2Config,
    SegmentationModel,
    MaskQuality as V2MaskQuality,
    ImageGeometry as V2ImageGeometry,
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


@dataclass
class MaskingConfig(BaseConfig):
    """Configuration for masking module."""
    
    # Model settings
    model: str = "auto"  # auto, sam3, fastsam, efficient, sam2
    model_checkpoint: Optional[str] = None
    
    # Text prompts for SAM3
    remove_prompts: List[str] = field(default_factory=lambda: [
        "tripod",
        "camera operator person",
        "equipment gear",
        "shadow of tripod",
        "camera rover vehicle",
        "photographer",
        "selfie stick"
    ])
    keep_prompts: List[str] = field(default_factory=list)
    
    # Quality control
    confidence_threshold: float = 0.70
    min_mask_area: int = 100
    max_mask_area_ratio: float = 0.5
    
    # Processing options
    use_temporal_consistency: bool = True
    temporal_window: int = 5
    
    # Geometry handling
    geometry_aware: bool = True
    handle_distortion: bool = True
    pole_mask_expand: float = 1.2
    
    # Output settings
    output_format: str = "png"
    save_confidence_maps: bool = False
    save_review_images: bool = True


class MaskingModule(BaseModule):
    """Masking module for removing unwanted objects."""
    
    def _default_config(self) -> MaskingConfig:
        """Return default configuration."""
        return MaskingConfig()
    
    def _initialize(self) -> None:
        """Initialize module resources."""
        self.config: MaskingConfig
        
        # Map model string to enum
        model_map = {
            'auto': None,
            'sam3': SegmentationModel.SAM3,
            'fastsam': SegmentationModel.FASTSAM,
            'efficient': SegmentationModel.EFFICIENTSAM,
            'sam2': SegmentationModel.SAM2,
        }
        
        model = model_map.get(self.config.model)
        
        # Create masking_v2 config
        v2_config = MaskingV2Config(
            model=model if model else SegmentationModel.SAM3,
            model_checkpoint=self.config.model_checkpoint,
            device=self.device,
            remove_prompts=self.config.remove_prompts,
            keep_prompts=self.config.keep_prompts,
            confidence_threshold=self.config.confidence_threshold,
            review_threshold=self.config.review_threshold,
            min_mask_area=self.config.min_mask_area,
            max_mask_area_ratio=self.config.max_mask_area_ratio,
            use_temporal_consistency=self.config.use_temporal_consistency,
            temporal_window=self.config.temporal_window,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            geometry_aware=self.config.geometry_aware,
            handle_distortion=self.config.handle_distortion,
            pole_mask_expand=self.config.pole_mask_expand,
            save_confidence_maps=self.config.save_confidence_maps,
            save_review_images=self.config.save_review_images,
            output_format=self.config.output_format,
        )
        
        # Initialize masking pipeline
        self.pipeline = MaskingPipeline(
            config=v2_config,
            auto_select_model=(self.config.model == "auto")
        )
        
        self.log("Masking module initialized")
        self.log(f"Model: {self.pipeline.config.model.value}")
    
    def process(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> StageResult:
        """
        Process frames through masking pipeline.
        
        Args:
            input_path: Input directory with frames or rig views
            output_path: Output directory for masks
            **kwargs: Additional options
                - geometry: Override geometry detection
            
        Returns:
            StageResult with masking outcome
        """
        import time
        import cv2
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        self.log(f"Processing masking: {input_path}")
        
        # Detect geometry
        geometry = self._detect_geometry(input_path, kwargs.get('geometry'))
        self.log(f"Processing geometry: {geometry.value if geometry else 'pinhole'}")
        
        # Convert to V2 geometry enum
        v2_geometry = self._core_to_v2_geometry(geometry)
        
        # Find input images
        frames_to_process = self._find_input_images(input_path)
        
        if not frames_to_process:
            return self._create_result(
                success=False,
                output_path=output_path,
                quality=QualityLevel.REJECT,
                confidence=0.0,
                errors=["No images found to process"]
            )
        
        self.log(f"Found {len(frames_to_process)} images to mask")
        
        # Create output directories
        masks_dir = ensure_directory(output_path / "masks")
        
        if self.config.save_review_images:
            review_dir = ensure_directory(output_path / "review")
        
        # Process images
        start_time = time.perf_counter()
        
        processed = 0
        failed = 0
        review_count = 0
        rejected_count = 0
        
        confidences = []
        
        for idx, image_path in enumerate(frames_to_process):
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    self.log(f"Failed to load: {image_path}", "warning")
                    failed += 1
                    continue
                
                # Process image
                result = self.pipeline.process_image(
                    image,
                    geometry=v2_geometry,
                    custom_prompts=None
                )
                
                # Convert V2 quality to core quality
                quality = self._v2_to_core_quality(result.quality)
                
                # Save mask
                mask_name = f"mask_{image_path.stem}.{self.config.output_format}"
                mask_path = masks_dir / mask_name
                
                if self.config.output_format == 'npy':
                    import numpy as np
                    np.save(mask_path, result.mask)
                else:
                    cv2.imwrite(str(mask_path), result.mask * 255)
                
                # Save confidence map if requested
                if self.config.save_confidence_maps:
                    import numpy as np
                    conf_path = masks_dir / f"conf_{image_path.stem}.npy"
                    np.save(conf_path, result.confidence)
                
                # Handle review
                if result.needs_review and self.config.save_review_images:
                    review_path = review_dir / f"review_{image_path.stem}.jpg"
                    review_img = self._create_review_image(image, result.mask)
                    cv2.imwrite(str(review_path), review_img)
                    review_count += 1
                
                if quality == QualityLevel.REJECT:
                    rejected_count += 1
                else:
                    processed += 1
                    confidences.append(result.confidence)
                
                if (idx + 1) % 50 == 0:
                    self.log(f"Masked {idx + 1}/{len(frames_to_process)} images")
                    
            except Exception as e:
                self.log(f"Error processing {image_path}: {e}", "error")
                failed += 1
        
        elapsed = time.perf_counter() - start_time
        
        # Calculate quality metrics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        overall_quality = self.config.confidence_to_quality(avg_confidence)
        
        # Create manifest
        manifest = {
            'input': {
                'path': str(input_path),
                'geometry': geometry.value if geometry else 'unknown',
                'total_images': len(frames_to_process),
            },
            'output': {
                'masks_dir': str(masks_dir),
                'review_dir': str(review_dir) if self.config.save_review_images else None,
            },
            'model': {
                'type': self.pipeline.config.model.value,
                'checkpoint': self.pipeline.config.model_checkpoint,
                'prompts': {
                    'remove': self.config.remove_prompts,
                    'keep': self.config.keep_prompts,
                }
            },
            'statistics': {
                'processed': processed,
                'failed': failed,
                'rejected': rejected_count,
                'review_needed': review_count,
                'average_confidence': avg_confidence,
                'processing_time': elapsed,
                'average_time_per_image': elapsed / len(frames_to_process) if frames_to_process else 0,
            }
        }
        
        manifest_path = output_path / "masking_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Determine if review is needed
        needs_review = review_count > 0 or overall_quality in [QualityLevel.REVIEW, QualityLevel.POOR]
        
        # Create result
        result = self._create_result(
            success=processed > 0,
            output_path=output_path,
            items_processed=processed,
            items_failed=failed,
            items_skipped=rejected_count,
            processing_time=elapsed,
            quality=overall_quality,
            confidence=avg_confidence,
            needs_review=needs_review,
            review_items=[f"review_{i:06d}" for i in range(review_count)],
            metrics={
                'model': self.pipeline.config.model.value,
                'average_confidence': avg_confidence,
                'review_count': review_count,
                'rejected_count': rejected_count,
                'geometry': geometry.value if geometry else 'unknown',
            }
        )
        
        # Save result manifest
        result.save_manifest(output_path / "stage_result.json")
        
        self.log(f"Masking complete: {processed} masks generated")
        if review_count > 0:
            self.log(f"Review needed: {review_count} low-confidence masks", "warning")
        
        return result
    
    def _detect_geometry(
        self,
        input_path: Path,
        override: Optional[str] = None
    ) -> ImageGeometry:
        """Detect geometry from manifests or override."""
        
        if override:
            geometry_map = {
                'pinhole': ImageGeometry.PINHOLE,
                'fisheye': ImageGeometry.FISHEYE,
                'dual_fisheye': ImageGeometry.DUAL_FISHEYE,
                'equirect': ImageGeometry.EQUIRECTANGULAR,
                'equirectangular': ImageGeometry.EQUIRECTANGULAR,
            }
            return geometry_map.get(override.lower(), ImageGeometry.PINHOLE)
        
        # Check reframe manifest first
        reframe_manifest = input_path / "reframe_manifest.json"
        if reframe_manifest.exists():
            try:
                with open(reframe_manifest) as f:
                    manifest = json.load(f)
                
                # If reframed, output is always pinhole
                if manifest.get('mode') != 'passthrough':
                    return ImageGeometry.PINHOLE
                
            except Exception as e:
                self.log(f"Error reading reframe manifest: {e}", "warning")
        
        # Check ingest manifest
        ingest_manifest = input_path / "ingest_manifest.json"
        if ingest_manifest.exists():
            try:
                with open(ingest_manifest) as f:
                    manifest = json.load(f)
                
                geometry_str = manifest.get('camera', {}).get('geometry', 'pinhole')
                
                geometry_map = {
                    'pinhole': ImageGeometry.PINHOLE,
                    'fisheye': ImageGeometry.FISHEYE,
                    'dual_fisheye': ImageGeometry.DUAL_FISHEYE,
                    'equirect': ImageGeometry.EQUIRECTANGULAR,
                    'equirectangular': ImageGeometry.EQUIRECTANGULAR,
                }
                
                return geometry_map.get(geometry_str.lower(), ImageGeometry.PINHOLE)
                
            except Exception as e:
                self.log(f"Error reading ingest manifest: {e}", "warning")
        
        # Default to pinhole
        return ImageGeometry.PINHOLE
    
    def _find_input_images(self, input_path: Path) -> List[Path]:
        """Find images to process (handles both frames and rig_views)."""
        
        # Check for rig_views directory (reframed output)
        rig_views_dir = input_path / "rig_views"
        if rig_views_dir.exists():
            # Process all camera directories
            all_images = []
            for camera_dir in sorted(rig_views_dir.iterdir()):
                if camera_dir.is_dir():
                    images = get_image_files(camera_dir)
                    all_images.extend(images)
            
            if all_images:
                self.log(f"Processing rig views from {len(list(rig_views_dir.iterdir()))} cameras")
                return sorted(all_images)
        
        # Check for frames directory
        frames_dir = input_path / "frames"
        if frames_dir.exists():
            return get_image_files(frames_dir)
        
        # Fallback to input directory itself
        return get_image_files(input_path)
    
    def _core_to_v2_geometry(self, geometry: ImageGeometry) -> V2ImageGeometry:
        """Convert core ImageGeometry to V2 ImageGeometry."""
        
        mapping = {
            ImageGeometry.PINHOLE: V2ImageGeometry.PINHOLE,
            ImageGeometry.FISHEYE: V2ImageGeometry.FISHEYE,
            ImageGeometry.DUAL_FISHEYE: V2ImageGeometry.DUAL_FISHEYE,
            ImageGeometry.EQUIRECTANGULAR: V2ImageGeometry.EQUIRECTANGULAR,
            ImageGeometry.CUBEMAP: V2ImageGeometry.CUBEMAP,
        }
        
        return mapping.get(geometry, V2ImageGeometry.PINHOLE)
    
    def _v2_to_core_quality(self, v2_quality: V2MaskQuality) -> QualityLevel:
        """Convert V2 MaskQuality to core QualityLevel."""
        
        mapping = {
            V2MaskQuality.EXCELLENT: QualityLevel.EXCELLENT,
            V2MaskQuality.GOOD: QualityLevel.GOOD,
            V2MaskQuality.REVIEW: QualityLevel.REVIEW,
            V2MaskQuality.POOR: QualityLevel.POOR,
            V2MaskQuality.REJECT: QualityLevel.REJECT,
        }
        
        return mapping.get(v2_quality, QualityLevel.GOOD)
    
    def _create_review_image(
        self,
        image,
        mask,
        alpha: float = 0.5
    ):
        """Create review image with mask overlay."""
        import cv2
        import numpy as np
        
        # Create colored mask (red)
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 2] = mask * 255
        
        # Blend
        review = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        return review


__all__ = [
    'MaskingConfig',
    'MaskingModule',
]
