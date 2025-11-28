"""
SplatForge Retrieve: Image Pair Generation Module

Generates image pairs for matching using global descriptors (NetVLAD, CosPlace)
or vocabulary trees. Essential for large datasets (6000+ images) to avoid O(n²) complexity.

Backends:
1. NetVLAD - ResNet50-based global descriptors (CVPR 2016)
2. CosPlace - Place recognition (CVPR 2022)
3. EigenPlaces - Eigenvalue-based descriptors (CVPR 2023)
4. VocabTree - COLMAP vocabulary tree
5. Sequential - For video sequences (fallback)
6. Exhaustive - All pairs (small datasets only)

Usage:
    # CLI
    python -m modular_pipeline.reconstruct.retrieve ./images ./pairs.txt --backend netvlad
    
    # Python API
    from modular_pipeline.reconstruct.retrieve import RetrievalModule, RetrievalConfig
    
    config = RetrievalConfig(backend="netvlad", num_neighbors=50)
    retrieval = RetrievalModule(config)
    result = retrieval.process(images_path, output_path)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
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

class RetrievalBackend(Enum):
    """Image retrieval backend options."""
    NETVLAD = "netvlad"           # ResNet50 + NetVLAD (recommended)
    COSPLACE = "cosplace"         # CosPlace descriptor
    EIGENPLACES = "eigenplaces"   # EigenPlaces (CVPR 2023)
    VOCAB_TREE = "vocab_tree"     # COLMAP vocabulary tree
    SEQUENTIAL = "sequential"     # Video sequences
    EXHAUSTIVE = "exhaustive"     # All pairs (O(n²), small datasets only)
    HYBRID = "hybrid"             # Sequential + NetVLAD (Video + Loop closure)


@dataclass
class RetrievalConfig(BaseConfig):
    """Configuration for image retrieval.
    
    Attributes:
        backend: Retrieval method (netvlad recommended for general use)
        num_neighbors: Number of nearest neighbors per image
        max_pairs: Maximum total pairs to generate (prevents O(n²) explosion)
        
        # Sequential settings (for video)
        sequential_window: Window size for sequential pairing
        sequential_overlap: Use overlapping windows
        
        # NetVLAD settings
        vlad_cluster_size: NetVLAD cluster centers (32 or 64)
        
        # Quality filtering
        min_score: Minimum similarity score to keep pair
        filter_duplicates: Remove duplicate pairs
    """
    # Backend selection
    backend: RetrievalBackend = RetrievalBackend.NETVLAD
    
    # Pairing settings
    num_neighbors: int = 50           # k-NN neighbors per image
    max_pairs: int = 1_000_000        # Safety limit for large datasets
    
    # Sequential mode (video)
    sequential_window: int = 50       # Frame window for sequential
    sequential_overlap: bool = True   # Overlapping windows
    
    # NetVLAD settings
    vlad_cluster_size: int = 64       # 32 or 64 clusters
    vlad_dim: int = 4096              # Output descriptor dimension
    
    # Quality filtering
    min_score: float = 0.0            # Minimum similarity (0-1)
    filter_duplicates: bool = True    # Remove (A,B) if (B,A) exists
    
    # Output
    output_format: str = "txt"        # txt or json
    
    def __post_init__(self):
        # Do NOT call super().__post_init__() because BaseConfig tries to
        # convert self.backend to core.Backend, but we use RetrievalBackend here.
        # This shadows the parent 'backend' field with a different Enum type.
        
        if isinstance(self.backend, str):
            self.backend = RetrievalBackend(self.backend)


# =============================================================================
# RETRIEVAL MODULE
# =============================================================================

class RetrievalModule(BaseModule):
    """Generate image pairs via global descriptor retrieval.
    
    Extracts global descriptors for all images and uses k-NN
    to find similar pairs, avoiding O(n²) exhaustive matching.
    
    For sequential video, uses sliding window approach.
    For general scenes, uses NetVLAD or vocabulary tree.
    
    Example:
        >>> retrieval = RetrievalModule(RetrievalConfig(num_neighbors=50))
        >>> result = retrieval.process(
        ...     images_path=Path("./images"),
        ...     output_path=Path("./pairs.txt")
        ... )
        >>> print(f"Generated {result.metrics['num_pairs']} pairs")
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__(config)
    
    def _default_config(self) -> RetrievalConfig:
        return RetrievalConfig()
    
    def _initialize(self) -> None:
        """Initialize retrieval backend."""
        self._model = None
        self._backend_name = self.config.backend.value
        
        if self.config.backend in [RetrievalBackend.NETVLAD, 
                                    RetrievalBackend.COSPLACE,
                                    RetrievalBackend.EIGENPLACES]:
            self._load_retrieval_model()
    
    def _load_retrieval_model(self) -> None:
        """Load global descriptor model."""
        backend = self.config.backend
        
        if backend in [RetrievalBackend.NETVLAD, RetrievalBackend.HYBRID]:
            try:
                # Try to use hloc's NetVLAD
                from hloc import extract_features
                self._model = 'netvlad'
                self.log(f"Will use HLOC NetVLAD extractor")
                return
            except ImportError:
                self.log("HLOC not available, will use sequential fallback", "warning")
                if backend == RetrievalBackend.HYBRID:
                    self.log("Hybrid mode degrading to Sequential only", "warning")
                    self.config.backend = RetrievalBackend.SEQUENTIAL
                else:
                    self.config.backend = RetrievalBackend.SEQUENTIAL
                return
        
        elif backend == RetrievalBackend.COSPLACE:
            try:
                # CosPlace model
                import torch
                self.log("CosPlace not yet implemented, using sequential", "warning")
                self.config.backend = RetrievalBackend.SEQUENTIAL
                return
            except ImportError:
                self.config.backend = RetrievalBackend.SEQUENTIAL
        
        # Fallback to sequential
        if backend != RetrievalBackend.SEQUENTIAL and backend != RetrievalBackend.EXHAUSTIVE:
            self.config.backend = RetrievalBackend.SEQUENTIAL
            self._backend_name = "sequential"
    
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
        """Generate image pairs for matching.
        
        Args:
            images_path: Directory containing images
            output_path: Path for output pairs file (pairs.txt or pairs.json)
            image_list: Optional list of specific images
        
        Returns:
            StageResult with pairing statistics
        """
        images_path = Path(images_path)
        output_path = Path(output_path)
        
        # Ensure output is a file path
        if output_path.is_dir() or output_path.suffix == '':
            output_path = output_path / 'pairs.txt'
        
        ensure_directory(output_path.parent)
        
        # Get image list
        if image_list:
            image_files = [images_path / img for img in image_list]
            image_names = image_list
        else:
            image_files = get_image_files(images_path)
            image_names = [f.name for f in image_files]
        
        if not image_files:
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=["No images found in input directory"]
            )
        
        self.log(f"Generating pairs for {len(image_files)} images using {self._backend_name}")
        
        start_time = time.perf_counter()
        
        # Generate pairs based on backend
        if self.config.backend == RetrievalBackend.EXHAUSTIVE:
            pairs = self._generate_exhaustive(image_names)
        elif self.config.backend == RetrievalBackend.SEQUENTIAL:
            pairs = self._generate_sequential(image_names)
        elif self.config.backend == RetrievalBackend.NETVLAD:
            pairs = self._generate_netvlad(images_path, image_names, output_path.parent)
        elif self.config.backend == RetrievalBackend.HYBRID:
            pairs = self._generate_hybrid(images_path, image_names, output_path.parent)
        elif self.config.backend == RetrievalBackend.VOCAB_TREE:
            pairs = self._generate_vocab_tree(images_path, image_names)
        else:
            # Fallback to sequential
            self.log(f"Backend {self.config.backend.value} not implemented, using sequential", "warning")
            pairs = self._generate_sequential(image_names)
        
        # Safety limit
        if len(pairs) > self.config.max_pairs:
            self.log(f"Limiting pairs from {len(pairs)} to {self.config.max_pairs}", "warning")
            pairs = pairs[:self.config.max_pairs]
        
        # Save pairs
        self._save_pairs(pairs, output_path)
        
        elapsed = time.perf_counter() - start_time
        
        # Compute statistics
        unique_images = len(set([p[0] for p in pairs] + [p[1] for p in pairs]))
        avg_pairs_per_image = len(pairs) * 2 / len(image_names)  # Each pair connects 2 images
        
        self.log(f"\nPair generation complete in {format_time(elapsed)}")
        self.log(f"  Total pairs: {len(pairs):,}")
        self.log(f"  Images in pairs: {unique_images}/{len(image_names)}")
        self.log(f"  Average connections per image: {avg_pairs_per_image:.1f}")
        
        # Quality assessment
        if unique_images == len(image_names):
            quality = QualityLevel.EXCELLENT
            confidence = 0.95
        elif unique_images >= len(image_names) * 0.9:
            quality = QualityLevel.GOOD
            confidence = 0.85
        else:
            quality = QualityLevel.REVIEW
            confidence = 0.70
        
        result = self._create_result(
            success=True,
            output_path=output_path,
            items_processed=len(pairs),
            processing_time=elapsed,
            quality=quality,
            confidence=confidence,
            metrics={
                'num_pairs': len(pairs),
                'num_images': len(image_names),
                'images_in_pairs': unique_images,
                'avg_connections_per_image': avg_pairs_per_image,
                'backend': self._backend_name,
            }
        )
        
        if self.config.save_manifests:
            result.save_manifest()
        
        return result
    
    # -------------------------------------------------------------------------
    # Pair Generation Strategies
    # -------------------------------------------------------------------------
    
    def _generate_exhaustive(self, image_names: List[str]) -> List[Tuple[str, str]]:
        """Generate all possible pairs (O(n²)).
        
        Only suitable for small datasets (<500 images).
        """
        if len(image_names) > 500:
            self.log(f"WARNING: Exhaustive pairing with {len(image_names)} images = {len(image_names)*(len(image_names)-1)//2} pairs!", "warning")
        
        pairs = []
        for i, name0 in enumerate(image_names):
            for name1 in image_names[i+1:]:
                pairs.append((name0, name1))
        
        return pairs
    
    def _generate_sequential(self, image_names: List[str]) -> List[Tuple[str, str]]:
        """Generate pairs using sliding window (for video sequences)."""
        pairs = []
        window = self.config.sequential_window
        
        for i, name0 in enumerate(image_names):
            # Connect to next N frames
            for j in range(i + 1, min(i + window + 1, len(image_names))):
                pairs.append((name0, image_names[j]))
        
        self.log(f"Sequential pairing with window={window}")
        return pairs
    
    def _generate_netvlad(
        self,
        images_path: Path,
        image_names: List[str],
        export_dir: Path
    ) -> List[Tuple[str, str]]:
        """Generate pairs using NetVLAD global descriptors."""
        try:
            from hloc import extract_features, pairs_from_retrieval
            
            # Path for global descriptors
            # We place it in /features/global-feats-netvlad.h5 if possible, or alongside output
            # Try to infer "features" dir from export_dir (which is matches dir)
            if export_dir.name == "matches":
                features_dir = export_dir.parent / "features"
            else:
                features_dir = export_dir
            
            ensure_directory(features_dir)
            global_descriptors_path = features_dir / 'global-feats-netvlad.h5'
            
            # 1. Extract NetVLAD descriptors
            conf = extract_features.confs['netvlad']
            self.log(f"Extracting NetVLAD descriptors to {global_descriptors_path}...")
            
            # We need to pass list of images relative to images_path if image_list provided
            # hloc.extract_features expects image_list as list of paths relative to image_dir
            # image_names are already basenames relative to images_path in our pipeline usually
            
            # Note: hloc modifies logger, we might want to capture output
            extract_features.main(
                conf=conf,
                image_dir=images_path,
                export_dir=features_dir,
                image_list=image_names if image_names else None,
                feature_path=global_descriptors_path,
                overwrite=False # Cache if exists
            )
            
            # 2. Generate pairs from retrieval
            self.log(f"Generating NetVLAD pairs (k={self.config.num_neighbors})...")
            
            # We use a temp file for hloc output then parse it
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                temp_pairs_path = Path(f.name)
            
            pairs_from_retrieval.main(
                global_descriptors_path,
                temp_pairs_path,
                num_matched=self.config.num_neighbors,
                db_prefix="", # image_dir prefix not needed for standard usage
            )
            
            # 3. Load pairs
            pairs = self.load_pairs(temp_pairs_path)
            temp_pairs_path.unlink()
            
            return pairs
            
        except Exception as e:
            self.log(f"NetVLAD retrieval failed: {e}", "error")
            import traceback
            traceback.print_exc()
            self.log("Falling back to sequential pairing", "warning")
            return self._generate_sequential(image_names)

    def _generate_hybrid(
        self,
        images_path: Path,
        image_names: List[str],
        export_dir: Path
    ) -> List[Tuple[str, str]]:
        """Generate both sequential and retrieval pairs."""
        
        # 1. Sequential pairs (video continuity)
        seq_pairs = self._generate_sequential(image_names)
        self.log(f"Generated {len(seq_pairs)} sequential pairs")
        
        # 2. NetVLAD pairs (loop closure)
        # Use half the neighbors for retrieval to balance count
        original_neighbors = self.config.num_neighbors
        self.config.num_neighbors = max(10, original_neighbors // 2)
        
        ret_pairs = self._generate_netvlad(images_path, image_names, export_dir)
        
        # Restore config
        self.config.num_neighbors = original_neighbors
        
        self.log(f"Generated {len(ret_pairs)} retrieval pairs")
        
        # 3. Merge and deduplicate
        all_pairs_set = set(seq_pairs)
        
        # Normalize order for retrieval pairs to check dups
        for p1, p2 in ret_pairs:
            if (p1, p2) not in all_pairs_set and (p2, p1) not in all_pairs_set:
                all_pairs_set.add((p1, p2))
        
        merged_pairs = list(all_pairs_set)
        self.log(f"Merged hybrid pairs: {len(merged_pairs)} (deduplicated)")
        
        return merged_pairs
    
    def _generate_vocab_tree(
        self,
        images_path: Path,
        image_names: List[str]
    ) -> List[Tuple[str, str]]:
        """Generate pairs using COLMAP vocabulary tree.
        
        Requires pre-built vocabulary tree and SIFT features.
        """
        self.log("Vocabulary tree retrieval not yet implemented", "warning")
        self.log("Falling back to sequential pairing", "warning")
        return self._generate_sequential(image_names)
    
    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------
    
    def _save_pairs(self, pairs: List[Tuple[str, str]], path: Path) -> None:
        """Save pairs to text file."""
        with open(path, 'w') as f:
            for name0, name1 in pairs:
                f.write(f"{name0} {name1}\n")
        
        self.log(f"Saved {len(pairs)} pairs to {path}")
    
    @staticmethod
    def load_pairs(path: Path) -> List[Tuple[str, str]]:
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


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for pair generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate image pairs for feature matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("images", type=Path, help="Directory containing images")
    parser.add_argument("output", type=Path, help="Output pairs file (pairs.txt)")
    
    # Backend settings
    parser.add_argument("--backend", type=str, default="sequential",
                       choices=["netvlad", "hybrid", "cosplace", "vocab_tree", "sequential", "exhaustive"],
                       help="Retrieval backend (hybrid = sequential + netvlad)")
    parser.add_argument("--neighbors", type=int, default=50,
                       help="Number of neighbors per image (k-NN)")
    parser.add_argument("--max-pairs", type=int, default=1_000_000,
                       help="Maximum total pairs")
    
    # Sequential settings
    parser.add_argument("--window", type=int, default=50,
                       help="Window size for sequential pairing")
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = RetrievalConfig(
        backend=RetrievalBackend(args.backend),
        num_neighbors=args.neighbors,
        max_pairs=args.max_pairs,
        sequential_window=args.window,
        verbose=args.verbose,
    )
    
    retrieval = RetrievalModule(config)
    result = retrieval.process(args.images, args.output)
    
    if result.success:
        print(f"\n✓ Pair generation complete: {result.output_path}")
        print(f"  Pairs: {result.metrics['num_pairs']:,}")
        print(f"  Images: {result.metrics['images_in_pairs']}/{result.metrics['num_images']}")
        print(f"  Time: {format_time(result.processing_time_seconds)}")
    else:
        print(f"\n✗ Pair generation failed")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
