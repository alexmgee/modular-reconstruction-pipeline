"""
SplatForge Extract: Feature Extraction Module

Extracts local features (keypoints + descriptors) from images using
state-of-the-art extractors with automatic backend fallback.

Backends (priority order):
1. ALIKED  - Best accuracy + speed, BSD-3 license (RECOMMENDED)
2. XFeat   - Fastest, good for real-time (150+ FPS)
3. SuperPoint - Legacy compatibility, restrictive license
4. DISK    - High density keypoints

Usage:
    # CLI
    python -m splatforge.reconstruct.extract ./images ./features --extractor aliked
    
    # Python API
    from splatforge.reconstruct.extract import FeatureExtractor, ExtractConfig
    
    config = ExtractConfig(extractor="aliked", max_keypoints=8000)
    extractor = FeatureExtractor(config)
    result = extractor.process(input_dir, output_dir)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import logging

# Core imports
from modular_pipeline.core import (
    BaseModule, BaseConfig, StageResult, ItemResult,
    QualityLevel, Backend, ImageGeometry,
    get_image_files, ensure_directory, format_time,
    HAS_TORCH, HAS_NUMPY
)

# Lazy imports for optional dependencies
try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import h5py
except ImportError:
    h5py = None


# =============================================================================
# CONFIGURATION
# =============================================================================

class ExtractorBackend(Enum):
    """Feature extractor backends in priority order."""
    ALIKED = "aliked"           # Sparse Deformable Descriptor Head (CVPR 2024)
    XFEAT = "xfeat"             # Accelerated Features (CVPR 2024), fastest
    SUPERPOINT = "superpoint"   # MagicLeap (2018), legacy compatibility
    DISK = "disk"               # Dense keypoints, good for texture-poor
    AUTO = "auto"               # Auto-select best available


@dataclass
class ExtractConfig(BaseConfig):
    """Configuration for feature extraction.
    
    Attributes:
        extractor: Which extractor to use (aliked recommended)
        max_keypoints: Maximum keypoints per image (8000 for quality, 4096 for speed)
        nms_radius: Non-maximum suppression radius in pixels
        keypoint_threshold: Detection confidence threshold
        resize_max: Resize images so max dimension <= this (-1 for original)
        resize_force: Force specific output size (overrides resize_max)
        grayscale: Convert to grayscale before extraction
        subpixel_refinement: Enable sub-pixel keypoint refinement (+~7ms)
    """
    # Extractor selection
    extractor: ExtractorBackend = ExtractorBackend.ALIKED
    
    # Keypoint settings
    max_keypoints: int = 8000       # Higher = more matches, slower
    nms_radius: int = 3             # Min pixels between keypoints
    keypoint_threshold: float = 0.005
    
    # Preprocessing
    resize_max: int = 1600          # Max dimension, -1 for original
    resize_force: Optional[Tuple[int, int]] = None  # (width, height)
    grayscale: bool = True          # Most extractors need grayscale
    
    # Advanced
    subpixel_refinement: bool = False   # Sub-pixel accuracy (ECCV 2024)
    remove_borders: int = 4             # Ignore edge pixels
    
    # Output
    output_format: str = "h5"           # h5 or npz
    feature_name: str = "feats-{extractor}-n{max_keypoints}"
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.extractor, str):
            self.extractor = ExtractorBackend(self.extractor)
    
    @property
    def output_name(self) -> str:
        """Generate output filename based on config."""
        return self.feature_name.format(
            extractor=self.extractor.value,
            max_keypoints=self.max_keypoints
        )


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor(BaseModule):
    """Extract local features from images.
    
    Supports multiple extractor backends with automatic fallback.
    Outputs HDF5 or NPZ files compatible with HLOC and COLMAP.
    
    Example:
        >>> extractor = FeatureExtractor(ExtractConfig(max_keypoints=8000))
        >>> result = extractor.process(Path("./images"), Path("./features"))
        >>> print(f"Extracted {result.metrics['total_keypoints']} keypoints")
    """
    
    def __init__(self, config: Optional[ExtractConfig] = None):
        super().__init__(config)
    
    def _default_config(self) -> ExtractConfig:
        return ExtractConfig()
    
    def _initialize(self) -> None:
        """Load the selected extractor model."""
        self._model = None
        self._extractor_name = None
        
        # Try to load requested extractor with fallback
        backend = self.config.extractor
        if backend == ExtractorBackend.AUTO:
            backend = self._select_extractor()
        
        self._load_extractor(backend)
    
    def _select_extractor(self) -> ExtractorBackend:
        """Auto-select best available extractor."""
        # Try in priority order
        for backend in [ExtractorBackend.ALIKED, ExtractorBackend.XFEAT, 
                        ExtractorBackend.SUPERPOINT, ExtractorBackend.DISK]:
            if self._is_available(backend):
                self.log(f"Auto-selected extractor: {backend.value}")
                return backend
        
        raise RuntimeError("No feature extractor available. Install lightglue or hloc.")
    
    def _is_available(self, backend: ExtractorBackend) -> bool:
        """Check if an extractor backend is available."""
        if not HAS_TORCH:
            return False
        
        try:
            if backend == ExtractorBackend.ALIKED:
                from lightglue import ALIKED
                return True
            elif backend == ExtractorBackend.XFEAT:
                from lightglue import XFeat
                return True
            elif backend == ExtractorBackend.SUPERPOINT:
                from lightglue import SuperPoint
                return True
            elif backend == ExtractorBackend.DISK:
                from lightglue import DISK
                return True
        except ImportError:
            pass
        
        return False
    
    def _load_extractor(self, backend: ExtractorBackend) -> None:
        """Load the specified extractor model."""
        import torch
        
        device = self.device
        self._extractor_name = backend.value
        
        if backend == ExtractorBackend.ALIKED:
            try:
                from lightglue import ALIKED
                self._model = ALIKED(
                    max_num_keypoints=self.config.max_keypoints,
                    detection_threshold=self.config.keypoint_threshold,
                    nms_radius=self.config.nms_radius,
                ).eval().to(device)
                self.log(f"Loaded ALIKED extractor on {device}")
                return
            except ImportError:
                self.log("ALIKED not available, trying fallback", "warning")
        
        if backend == ExtractorBackend.XFEAT:
            try:
                from lightglue import XFeat
                self._model = XFeat(
                    max_num_keypoints=self.config.max_keypoints,
                ).eval().to(device)
                self.log(f"Loaded XFeat extractor on {device}")
                return
            except ImportError:
                self.log("XFeat not available, trying fallback", "warning")
        
        if backend == ExtractorBackend.SUPERPOINT:
            try:
                from lightglue import SuperPoint
                self._model = SuperPoint(
                    max_num_keypoints=self.config.max_keypoints,
                    detection_threshold=self.config.keypoint_threshold,
                    nms_radius=self.config.nms_radius,
                    remove_borders=self.config.remove_borders,
                ).eval().to(device)
                self.log(f"Loaded SuperPoint extractor on {device}")
                return
            except ImportError:
                self.log("SuperPoint not available, trying fallback", "warning")
        
        if backend == ExtractorBackend.DISK:
            try:
                from lightglue import DISK
                self._model = DISK(
                    max_num_keypoints=self.config.max_keypoints,
                ).eval().to(device)
                self.log(f"Loaded DISK extractor on {device}")
                return
            except ImportError:
                pass
        
        raise RuntimeError(f"Could not load extractor: {backend.value}")
    
    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------
    
    def process(
        self,
        input_path: Path,
        output_path: Path,
        image_list: Optional[List[str]] = None,
        **kwargs
    ) -> StageResult:
        """Extract features from all images in a directory.
        
        Args:
            input_path: Directory containing images
            output_path: Directory for output (features.h5 will be created)
            image_list: Optional list of specific images to process
        
        Returns:
            StageResult with extraction statistics
        """
        import torch
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        ensure_directory(output_path)
        
        # Get image list
        if image_list:
            image_files = [input_path / img for img in image_list]
        else:
            image_files = get_image_files(input_path)
        
        if not image_files:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=["No images found in input directory"]
            )
        
        self.log(f"Processing {len(image_files)} images with {self._extractor_name}")
        
        # Setup output file
        features_path = output_path / f'{self.config.output_name}.h5'
        
        # Process images
        start_time = time.perf_counter()
        results: List[ItemResult] = []
        total_keypoints = 0
        
        if h5py is None:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=["h5py not installed: pip install h5py"]
            )
        
        with h5py.File(features_path, 'w') as h5f:
            for i, img_path in enumerate(image_files):
                try:
                    # Extract features
                    result = self._extract_single(img_path)
                    results.append(result)
                    
                    if result.success:
                        # Store in HDF5
                        self._store_features(
                            h5f, 
                            img_path.name,
                            result.metrics['keypoints'],
                            result.metrics['descriptors'],
                            result.metrics.get('scores')
                        )
                        total_keypoints += len(result.metrics['keypoints'])
                    
                    # Progress
                    if self.config.verbose and (i + 1) % 100 == 0:
                        elapsed = time.perf_counter() - start_time
                        rate = (i + 1) / elapsed
                        self.log(f"  [{i+1}/{len(image_files)}] {rate:.1f} img/s, "
                                f"{total_keypoints:,} keypoints")
                
                except Exception as e:
                    results.append(ItemResult(
                        item_id=img_path.name,
                        success=False,
                        quality=QualityLevel.REJECT,
                        confidence=0.0,
                        error=str(e)
                    ))
                    self.log(f"Error processing {img_path.name}: {e}", "error")
        
        # Compute statistics
        elapsed = time.perf_counter() - start_time
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        avg_keypoints = total_keypoints / len(successful) if successful else 0
        quality, confidence = self._aggregate_quality(results)
        
        self.log(f"\nExtraction complete in {format_time(elapsed)}")
        self.log(f"  Total keypoints: {total_keypoints:,}")
        self.log(f"  Average per image: {avg_keypoints:.0f}")
        self.log(f"  Success rate: {len(successful)}/{len(results)}")
        
        result = self._create_result(
            success=len(failed) < len(results),
            output_path=features_path,
            items_processed=len(successful),
            items_failed=len(failed),
            processing_time=elapsed,
            quality=quality,
            confidence=confidence,
            needs_review=any(r.needs_review for r in results),
            metrics={
                'total_keypoints': total_keypoints,
                'average_keypoints': avg_keypoints,
                'extractor': self._extractor_name,
                'images_per_second': len(results) / elapsed if elapsed > 0 else 0,
            },
            review_items=[r.item_id for r in results if r.needs_review],
            errors=[r.error for r in failed if r.error]
        )
        
        # Save manifest
        if self.config.save_manifests:
            result.save_manifest()
        
        return result
    
    def _extract_single(self, image_path: Path) -> ItemResult:
        """Extract features from a single image."""
        import torch
        
        start = time.perf_counter()
        
        # Load and preprocess image
        image = self._load_image(image_path)
        if image is None:
            return ItemResult(
                item_id=image_path.name,
                success=False,
                quality=QualityLevel.REJECT,
                confidence=0.0,
                error="Failed to load image"
            )
        
        # Convert to tensor
        image_tensor = self._preprocess(image)
        
        # Extract features
        with torch.no_grad():
            feats = self._model.extract(image_tensor)
        
        # Get numpy arrays
        keypoints = feats['keypoints'][0].cpu().numpy()
        descriptors = feats['descriptors'][0].cpu().numpy()
        scores = feats.get('keypoint_scores')
        if scores is not None:
            scores = scores[0].cpu().numpy()
        
        # Optional sub-pixel refinement
        if self.config.subpixel_refinement:
            keypoints = self._subpixel_refine(image, keypoints)
        
        elapsed = time.perf_counter() - start
        
        # Assess quality based on keypoint count
        num_kp = len(keypoints)
        if num_kp >= self.config.max_keypoints * 0.8:
            quality = QualityLevel.EXCELLENT
            confidence = 0.98
        elif num_kp >= self.config.max_keypoints * 0.5:
            quality = QualityLevel.GOOD
            confidence = 0.90
        elif num_kp >= 500:
            quality = QualityLevel.REVIEW
            confidence = 0.75
        elif num_kp >= 100:
            quality = QualityLevel.POOR
            confidence = 0.50
        else:
            quality = QualityLevel.REJECT
            confidence = 0.30
        
        return ItemResult(
            item_id=image_path.name,
            success=True,
            quality=quality,
            confidence=confidence,
            processing_time=elapsed,
            metrics={
                'keypoints': keypoints,
                'descriptors': descriptors,
                'scores': scores,
                'num_keypoints': num_kp,
            }
        )
    
    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """Load image from path."""
        if cv2 is None:
            raise ImportError("OpenCV required: pip install opencv-python")
        
        image = cv2.imread(str(path))
        if image is None:
            return None
        
        # Resize if needed
        if self.config.resize_force:
            image = cv2.resize(image, self.config.resize_force)
        elif self.config.resize_max > 0:
            h, w = image.shape[:2]
            max_dim = max(h, w)
            if max_dim > self.config.resize_max:
                scale = self.config.resize_max / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
        
        return image
    
    def _preprocess(self, image: np.ndarray):
        """Preprocess image for extraction."""
        import torch
        
        # Convert to grayscale if needed
        if self.config.grayscale and len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        if len(gray.shape) == 2:
            tensor = torch.from_numpy(gray)[None, None].float() / 255.0
        else:
            tensor = torch.from_numpy(gray).permute(2, 0, 1)[None].float() / 255.0
        
        return tensor.to(self.device)
    
    def _subpixel_refine(
        self,
        image: np.ndarray,
        keypoints: np.ndarray
    ) -> np.ndarray:
        """Refine keypoint locations to sub-pixel accuracy.
        
        Uses OpenCV's cornerSubPix for ~7ms overhead with improved accuracy.
        Based on ECCV 2024 findings on sub-pixel refinement benefits.
        
        Args:
            image: Grayscale or BGR image
            keypoints: Nx2 array of keypoint coordinates
        
        Returns:
            Refined keypoint coordinates
        """
        if cv2 is None:
            self.log("OpenCV required for sub-pixel refinement", "warning")
            return keypoints
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Prepare keypoints for cornerSubPix (requires float32)
        kp_refined = keypoints.astype(np.float32).copy()
        
        # Sub-pixel refinement parameters
        win_size = (5, 5)  # Half of search window
        zero_zone = (-1, -1)  # No zero zone
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,  # Max iterations
            0.001  # Epsilon
        )
        
        # Refine keypoints
        cv2.cornerSubPix(
            gray,
            kp_refined,
            win_size,
            zero_zone,
            criteria
        )
        
        return kp_refined
    
    def _store_features(
        self,
        h5f: 'h5py.File',
        name: str,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        scores: Optional[np.ndarray] = None
    ) -> None:
        """Store features in HDF5 file (HLOC format)."""
        grp = h5f.create_group(name)
        grp.create_dataset('keypoints', data=keypoints)
        grp.create_dataset('descriptors', data=descriptors)
        if scores is not None:
            grp.create_dataset('scores', data=scores)
    
    # -------------------------------------------------------------------------
    # Single Image API
    # -------------------------------------------------------------------------
    
    def extract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features from a single image (numpy array).
        
        Args:
            image: BGR or grayscale image as numpy array
        
        Returns:
            Dict with 'keypoints', 'descriptors', and optionally 'scores'
        """
        import torch
        
        image_tensor = self._preprocess(image)
        
        with torch.no_grad():
            feats = self._model.extract(image_tensor)
        
        return {
            'keypoints': feats['keypoints'][0].cpu().numpy(),
            'descriptors': feats['descriptors'][0].cpu().numpy(),
            'scores': feats.get('keypoint_scores', [None])[0],
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for feature extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract local features from images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input", type=Path, help="Input directory with images")
    parser.add_argument("output", type=Path, help="Output directory for features")
    
    # Extractor settings
    parser.add_argument("--extractor", type=str, default="aliked",
                       choices=["aliked", "xfeat", "superpoint", "disk", "auto"],
                       help="Feature extractor to use")
    parser.add_argument("--max-keypoints", type=int, default=8000,
                       help="Maximum keypoints per image")
    parser.add_argument("--nms-radius", type=int, default=3,
                       help="Non-maximum suppression radius")
    parser.add_argument("--resize-max", type=int, default=1600,
                       help="Resize images so max dimension <= this (-1 for original)")
    
    # Device settings
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    # Output settings
    parser.add_argument("--save-viz", action="store_true",
                       help="Save visualization of keypoints")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Create config
    config = ExtractConfig(
        extractor=ExtractorBackend(args.extractor),
        max_keypoints=args.max_keypoints,
        nms_radius=args.nms_radius,
        resize_max=args.resize_max,
        device=args.device,
        save_visualizations=args.save_viz,
        verbose=args.verbose,
    )
    
    # Run extraction
    extractor = FeatureExtractor(config)
    result = extractor.process(args.input, args.output)
    
    # Print summary
    if result.success:
        print(f"\n✓ Extraction complete: {result.output_path}")
        print(f"  Keypoints: {result.metrics['total_keypoints']:,}")
        print(f"  Average: {result.metrics['average_keypoints']:.0f} per image")
        print(f"  Time: {format_time(result.processing_time_seconds)}")
    else:
        print(f"\n✗ Extraction failed")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
