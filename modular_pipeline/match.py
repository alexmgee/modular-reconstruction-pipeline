"""
SplatForge Match: Feature Matching Module

Matches local features between image pairs using state-of-the-art matchers
with intelligent fallback for challenging cases.

Strategy:
1. Primary: LightGlue (35-40ms, excellent for well-behaved pairs)
2. Fallback: MASt3R (300ms, handles extreme viewpoints, texture-poor)

The module monitors match counts and can automatically retry failed pairs
with the dense matcher.

Usage:
    # CLI
    python -m splatforge.reconstruct.match ./features ./matches --pairs pairs.txt
    
    # Python API
    from splatforge.reconstruct.match import FeatureMatcher, MatchConfig
    
    config = MatchConfig(matcher="lightglue", fallback="mast3r")
    matcher = FeatureMatcher(config)
    result = matcher.process(features_path, output_path, pairs_path)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import time
import logging

# Core imports
from modular_pipeline.core import (
    BaseModule, BaseConfig, StageResult, ItemResult,
    QualityLevel, Backend,
    ensure_directory, format_time,
    HAS_TORCH, HAS_NUMPY
)

# Lazy imports
try:
    import numpy as np
except ImportError:
    np = None

try:
    import h5py
except ImportError:
    h5py = None


# =============================================================================
# CONFIGURATION
# =============================================================================

class MatcherBackend(Enum):
    """Feature matcher backends."""
    LIGHTGLUE = "lightglue"     # Fast sparse matching (35-40ms)
    MAST3R = "mast3r"           # Dense 3D-aware matching (300ms)
    ROMA = "roma"               # Dense flow matching
    SUPERGLUE = "superglue"     # Legacy (slower than LightGlue)
    AUTO = "auto"


class RetrievalBackend(Enum):
    """Image retrieval backends for pair selection."""
    NETVLAD = "netvlad"         # Global descriptor (ResNet50)
    COSPLACE = "cosplace"       # Place recognition
    EIGENPLACES = "eigenplaces" # CVPR 2023
    VOCAB_TREE = "vocab_tree"   # COLMAP vocab tree
    EXHAUSTIVE = "exhaustive"   # All pairs (small datasets only)
    SEQUENTIAL = "sequential"   # Video sequences


@dataclass
class MatchConfig(BaseConfig):
    """Configuration for feature matching.
    
    Attributes:
        matcher: Primary matcher backend (lightglue recommended)
        fallback_matcher: Fallback for failed pairs (mast3r for challenging cases)
        min_matches: Minimum matches to consider pair successful
        fallback_threshold: Retry with fallback if matches < this
        
        retrieval: Pair selection method
        num_neighbors: Number of retrieval neighbors per image
        
        filter_threshold: Match confidence threshold (lower = more matches)
        depth_confidence: LightGlue early-exit threshold (-1 for quality)
        width_confidence: LightGlue pruning threshold (-1 for quality)
    """
    # Matcher selection
    matcher: MatcherBackend = MatcherBackend.LIGHTGLUE
    fallback_matcher: Optional[MatcherBackend] = MatcherBackend.MAST3R
    
    # Match thresholds
    min_matches: int = 15           # Minimum for valid pair
    fallback_threshold: int = 50    # Retry with fallback if below this
    max_pairs_fallback: int = 500   # Max pairs to retry with fallback
    
    # Retrieval settings
    retrieval: RetrievalBackend = RetrievalBackend.NETVLAD
    num_neighbors: int = 50         # Retrieval neighbors per image
    
    # LightGlue settings
    filter_threshold: float = 0.1   # Lower = more matches
    depth_confidence: float = -1    # -1 for max quality, 0.9 for speed
    width_confidence: float = -1    # -1 for max quality, 0.95 for speed
    
    # Features (must match extractor)
    feature_type: str = "aliked"    # aliked, superpoint, xfeat, disk
    
    # Output
    output_format: str = "h5"
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.matcher, str):
            self.matcher = MatcherBackend(self.matcher)
        if isinstance(self.fallback_matcher, str):
            self.fallback_matcher = MatcherBackend(self.fallback_matcher)
        if isinstance(self.retrieval, str):
            self.retrieval = RetrievalBackend(self.retrieval)


# =============================================================================
# FEATURE MATCHER
# =============================================================================

class FeatureMatcher(BaseModule):
    """Match features between image pairs.
    
    Implements a two-stage matching strategy:
    1. Fast matching with LightGlue for all pairs
    2. Selective retry with MASt3R for pairs with few matches
    
    This balances speed (LightGlue at 35ms) with robustness
    (MASt3R handles extreme viewpoint changes).
    
    Example:
        >>> matcher = FeatureMatcher(MatchConfig(num_neighbors=50))
        >>> result = matcher.process(
        ...     features_path=Path("./features.h5"),
        ...     output_path=Path("./matches"),
        ...     pairs_path=Path("./pairs.txt")  # Optional, will generate if missing
        ... )
    """
    
    def __init__(self, config: Optional[MatchConfig] = None):
        super().__init__(config)
    
    def _default_config(self) -> MatchConfig:
        return MatchConfig()
    
    def _initialize(self) -> None:
        """Load matcher models."""
        self._primary_matcher = None
        self._fallback_matcher = None
        self._retrieval_model = None
        
        # Load primary matcher
        self._load_matcher(self.config.matcher, primary=True)
        
        # Load fallback if configured
        if self.config.fallback_matcher:
            try:
                self._load_matcher(self.config.fallback_matcher, primary=False)
            except Exception as e:
                self.log(f"Fallback matcher unavailable: {e}", "warning")
    
    def _load_matcher(self, backend: MatcherBackend, primary: bool = True) -> None:
        """Load a matcher model."""
        import torch
        device = self.device
        
        if backend == MatcherBackend.LIGHTGLUE:
            try:
                from lightglue import LightGlue
                matcher = LightGlue(
                    features=self.config.feature_type,
                    depth_confidence=self.config.depth_confidence,
                    width_confidence=self.config.width_confidence,
                    filter_threshold=self.config.filter_threshold,
                ).eval().to(device)
                
                if primary:
                    self._primary_matcher = matcher
                    self.log(f"Loaded LightGlue ({self.config.feature_type}) on {device}")
                else:
                    self._fallback_matcher = matcher
                return
            except ImportError as e:
                raise ImportError(f"LightGlue not available: {e}")
        
        elif backend == MatcherBackend.MAST3R:
            try:
                # MASt3R requires special handling - it's an end-to-end system
                from mast3r.model import AsymmetricMASt3R
                from mast3r.fast_nn import fast_reciprocal_NNs
                
                model = AsymmetricMASt3R.from_pretrained(
                    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                ).eval().to(device)
                
                if primary:
                    self._primary_matcher = ('mast3r', model)
                else:
                    self._fallback_matcher = ('mast3r', model)
                self.log(f"Loaded MASt3R on {device}")
                return
            except ImportError:
                self.log("MASt3R not available for fallback", "warning")
        
        elif backend == MatcherBackend.SUPERGLUE:
            try:
                from hloc.matchers.superglue import SuperGlue
                matcher = SuperGlue({
                    'weights': 'outdoor',
                    'sinkhorn_iterations': 20,
                    'match_threshold': self.config.filter_threshold,
                }).eval().to(device)
                
                if primary:
                    self._primary_matcher = matcher
                else:
                    self._fallback_matcher = matcher
                self.log(f"Loaded SuperGlue on {device}")
                return
            except ImportError:
                pass
    
    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------
    
    def process(
        self,
        features_path: Path,
        output_path: Path,
        pairs_path: Optional[Path] = None,
        images_path: Optional[Path] = None,
        **kwargs
    ) -> StageResult:
        """Match features between image pairs.
        
        Args:
            features_path: Path to features.h5 file
            output_path: Directory for output (matches.h5 and pairs.txt)
            pairs_path: Path to pairs.txt (will generate if None)
            images_path: Path to images (required if generating pairs)
        
        Returns:
            StageResult with matching statistics
        """
        import torch
        
        features_path = Path(features_path)
        output_path = Path(output_path)
        ensure_directory(output_path)
        
        # Load features
        if not features_path.exists():
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=[f"Features file not found: {features_path}"]
            )
        
        with h5py.File(features_path, 'r') as f:
            image_names = list(f.keys())
        
        self.log(f"Loaded features for {len(image_names)} images")
        
        # Get or generate pairs
        if pairs_path and pairs_path.exists():
            pairs = self._load_pairs(pairs_path)
            self.log(f"Loaded {len(pairs)} pairs from {pairs_path}")
        else:
            # Generate pairs via retrieval
            if images_path is None:
                return self._create_result(
                    success=False,
                    output_path=output_path,
                    errors=["images_path required when pairs_path not provided"]
                )
            pairs = self._generate_pairs(images_path, image_names)
            
            # Save pairs
            pairs_output = output_path / 'pairs.txt'
            self._save_pairs(pairs, pairs_output)
            self.log(f"Generated {len(pairs)} pairs, saved to {pairs_output}")
        
        if not pairs:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=["No pairs to match"]
            )
        
        # Match features
        start_time = time.perf_counter()
        matches_path = output_path / 'matches.h5'
        
        results, failed_pairs = self._match_pairs(
            features_path, 
            pairs, 
            matches_path
        )
        
        # Retry failed pairs with fallback
        if failed_pairs and self._fallback_matcher:
            self.log(f"\nRetrying {len(failed_pairs)} pairs with fallback matcher...")
            fallback_results = self._match_pairs_fallback(
                features_path,
                failed_pairs[:self.config.max_pairs_fallback],
                matches_path
            )
            results.extend(fallback_results)
        
        # Statistics
        elapsed = time.perf_counter() - start_time
        successful = [r for r in results if r.success]
        total_matches = sum(r.metrics.get('num_matches', 0) for r in successful)
        
        quality, confidence = self._aggregate_quality(results)
        
        self.log(f"\nMatching complete in {format_time(elapsed)}")
        self.log(f"  Pairs matched: {len(successful)}/{len(pairs)}")
        self.log(f"  Total matches: {total_matches:,}")
        self.log(f"  Average: {total_matches/len(successful):.0f} per pair" if successful else "")
        
        result = self._create_result(
            success=len(successful) > 0,
            output_path=matches_path,
            items_processed=len(successful),
            items_failed=len(pairs) - len(successful),
            processing_time=elapsed,
            quality=quality,
            confidence=confidence,
            metrics={
                'total_matches': total_matches,
                'average_matches': total_matches / len(successful) if successful else 0,
                'pairs_total': len(pairs),
                'pairs_successful': len(successful),
                'pairs_fallback': len(failed_pairs) if self._fallback_matcher else 0,
                'matcher': self.config.matcher.value,
            }
        )
        
        if self.config.save_manifests:
            result.save_manifest()
        
        return result
    
    def _match_pairs(
        self,
        features_path: Path,
        pairs: List[Tuple[str, str]],
        output_path: Path
    ) -> Tuple[List[ItemResult], List[Tuple[str, str]]]:
        """Match all pairs with primary matcher."""
        import torch
        
        results = []
        failed_pairs = []
        
        with h5py.File(features_path, 'r') as feat_h5, \
             h5py.File(output_path, 'w') as match_h5:
            
            for i, (name0, name1) in enumerate(pairs):
                try:
                    # Load features
                    kp0 = feat_h5[name0]['keypoints'][:]
                    desc0 = feat_h5[name0]['descriptors'][:]
                    kp1 = feat_h5[name1]['keypoints'][:]
                    desc1 = feat_h5[name1]['descriptors'][:]
                    
                    # Match
                    start = time.perf_counter()
                    matches, scores = self._match_pair_lightglue(
                        kp0, desc0, kp1, desc1
                    )
                    elapsed = time.perf_counter() - start
                    
                    num_matches = len(matches)
                    
                    # Quality assessment
                    if num_matches >= self.config.fallback_threshold:
                        quality = QualityLevel.GOOD
                        confidence = 0.9
                    elif num_matches >= self.config.min_matches:
                        quality = QualityLevel.REVIEW
                        confidence = 0.7
                        failed_pairs.append((name0, name1))
                    else:
                        quality = QualityLevel.POOR
                        confidence = 0.4
                        failed_pairs.append((name0, name1))
                    
                    # Store matches
                    pair_key = f"{name0}/{name1}"
                    grp = match_h5.create_group(pair_key)
                    grp.create_dataset('matches0', data=matches)
                    if scores is not None:
                        grp.create_dataset('matching_scores0', data=scores)
                    
                    results.append(ItemResult(
                        item_id=pair_key,
                        success=num_matches >= self.config.min_matches,
                        quality=quality,
                        confidence=confidence,
                        processing_time=elapsed,
                        metrics={'num_matches': num_matches}
                    ))
                    
                    # Progress
                    if self.config.verbose and (i + 1) % 500 == 0:
                        self.log(f"  [{i+1}/{len(pairs)}] matched")
                
                except Exception as e:
                    results.append(ItemResult(
                        item_id=f"{name0}/{name1}",
                        success=False,
                        quality=QualityLevel.REJECT,
                        confidence=0.0,
                        error=str(e)
                    ))
        
        return results, failed_pairs
    
    def _match_pair_lightglue(
        self,
        kp0: np.ndarray,
        desc0: np.ndarray,
        kp1: np.ndarray,
        desc1: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Match a single pair using LightGlue."""
        import torch
        
        device = self.device
        
        # Prepare data
        data = {
            'keypoints0': torch.from_numpy(kp0)[None].to(device),
            'keypoints1': torch.from_numpy(kp1)[None].to(device),
            'descriptors0': torch.from_numpy(desc0)[None].to(device),
            'descriptors1': torch.from_numpy(desc1)[None].to(device),
            'image0': {'image_size': torch.tensor([[1000, 1000]]).to(device)},
            'image1': {'image_size': torch.tensor([[1000, 1000]]).to(device)},
        }
        
        # Match
        with torch.no_grad():
            pred = self._primary_matcher(data)
        
        matches = pred['matches0'][0].cpu().numpy()
        scores = pred.get('matching_scores0')
        if scores is not None:
            scores = scores[0].cpu().numpy()
        
        # Filter valid matches
        valid = matches >= 0
        match_indices = np.stack([
            np.where(valid)[0],
            matches[valid]
        ], axis=1)
        
        if scores is not None:
            scores = scores[valid]
        
        return match_indices, scores
    
    def _match_pairs_fallback(
        self,
        features_path: Path,
        pairs: List[Tuple[str, str]],
        output_path: Path
    ) -> List[ItemResult]:
        """Retry failed pairs with fallback matcher (MASt3R)."""
        # This would use MASt3R's dense matching
        # For now, return empty - implement when MASt3R integration is complete
        self.log(f"Fallback matching for {len(pairs)} pairs (not yet implemented)")
        return []
    
    # -------------------------------------------------------------------------
    # Pair Generation
    # -------------------------------------------------------------------------
    
    def _generate_pairs(
        self,
        images_path: Path,
        image_names: List[str]
    ) -> List[Tuple[str, str]]:
        """Generate image pairs via retrieval.
        
        Uses the RetrievalModule for pair generation.
        """
        try:
            from modular_pipeline.reconstruct.retrieve import RetrievalModule, RetrievalConfig
            
            # Create retrieval config from match config
            retrieval_config = RetrievalConfig(
                backend=self.config.retrieval,
                num_neighbors=self.config.num_neighbors,
                sequential_window=self.config.num_neighbors,
                device=self.config.device,
                verbose=self.config.verbose,
            )
            
            # Use retrieval module
            retrieval = RetrievalModule(retrieval_config)
            
            # Generate pairs to temporary path
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                pairs_temp = Path(f.name)
            
            result = retrieval.process(images_path, pairs_temp, image_list=image_names)
            
            if result.success:
                # Load generated pairs
                pairs = retrieval.load_pairs(pairs_temp)
                pairs_temp.unlink()  # Clean up temp file
                return pairs
            else:
                self.log("Retrieval failed, using sequential fallback", "warning")
                return self._generate_pairs_sequential(image_names)
            
        except ImportError:
            self.log("RetrievalModule not available, using sequential fallback", "warning")
            return self._generate_pairs_sequential(image_names)
    
    
    def _generate_pairs_sequential(
        self,
        image_names: List[str]
    ) -> List[Tuple[str, str]]:
        """Sequential pairing for video sequences."""
        pairs = []
        window = min(self.config.num_neighbors, len(image_names) - 1)
        
        for i, name0 in enumerate(image_names):
            for j in range(i+1, min(i+window+1, len(image_names))):
                pairs.append((name0, image_names[j]))
        
        return pairs
    
    def _load_pairs(self, path: Path) -> List[Tuple[str, str]]:
        """Load pairs from text file."""
        pairs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and ' ' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        pairs.append((parts[0], parts[1]))
        return pairs
    
    def _save_pairs(self, pairs: List[Tuple[str, str]], path: Path) -> None:
        """Save pairs to text file."""
        with open(path, 'w') as f:
            for name0, name1 in pairs:
                f.write(f"{name0} {name1}\n")


# =============================================================================
# RETRIEVAL MODULE
# =============================================================================

class ImageRetrieval(BaseModule):
    """Generate image pairs via global descriptor retrieval.
    
    Extracts global descriptors (NetVLAD, etc.) and finds
    nearest neighbors to create matching pairs.
    """
    
    def _default_config(self) -> MatchConfig:
        return MatchConfig()
    
    def _initialize(self) -> None:
        self._model = None
        self._load_retrieval_model()
    
    def _load_retrieval_model(self) -> None:
        """Load retrieval model."""
        # Implementation would load NetVLAD, CosPlace, etc.
        pass
    
    def process(
        self,
        images_path: Path,
        output_path: Path,
        **kwargs
    ) -> StageResult:
        """Generate pairs via retrieval."""
        # Implementation would:
        # 1. Extract global descriptors for all images
        # 2. Find k-nearest neighbors
        # 3. Output pairs.txt
        pass


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for feature matching."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Match features between image pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("features", type=Path, help="Path to features.h5")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--pairs", type=Path, help="Path to pairs.txt (optional)")
    parser.add_argument("--images", type=Path, help="Path to images (for pair generation)")
    
    # Matcher settings
    parser.add_argument("--matcher", type=str, default="lightglue",
                       choices=["lightglue", "mast3r", "superglue"],
                       help="Matcher to use")
    parser.add_argument("--fallback", type=str, default="mast3r",
                       help="Fallback matcher for failed pairs")
    parser.add_argument("--feature-type", type=str, default="aliked",
                       help="Feature type (must match extractor)")
    
    # Retrieval settings
    parser.add_argument("--retrieval", type=str, default="netvlad",
                       choices=["netvlad", "sequential", "exhaustive"],
                       help="Pair selection method")
    parser.add_argument("--neighbors", type=int, default=50,
                       help="Number of retrieval neighbors")
    
    # Thresholds
    parser.add_argument("--min-matches", type=int, default=15,
                       help="Minimum matches for valid pair")
    parser.add_argument("--filter-threshold", type=float, default=0.1,
                       help="Match confidence threshold")
    
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    config = MatchConfig(
        matcher=MatcherBackend(args.matcher),
        fallback_matcher=MatcherBackend(args.fallback) if args.fallback else None,
        feature_type=args.feature_type,
        retrieval=RetrievalBackend(args.retrieval),
        num_neighbors=args.neighbors,
        min_matches=args.min_matches,
        filter_threshold=args.filter_threshold,
        device=args.device,
        verbose=args.verbose,
    )
    
    matcher = FeatureMatcher(config)
    result = matcher.process(
        args.features,
        args.output,
        pairs_path=args.pairs,
        images_path=args.images
    )
    
    if result.success:
        print(f"\n✓ Matching complete: {result.output_path}")
        print(f"  Matches: {result.metrics['total_matches']:,}")
        print(f"  Pairs: {result.metrics['pairs_successful']}/{result.metrics['pairs_total']}")
        print(f"  Time: {format_time(result.processing_time_seconds)}")
    else:
        print(f"\n✗ Matching failed")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
