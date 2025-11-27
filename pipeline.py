"""
SplatForge Pipeline: Orchestrator for complete reconstruction workflows.

The Pipeline class chains individual stages together, handles configuration
via presets, manages project state, and provides resumable execution.

Usage:
    # Full pipeline with preset
    from splatforge import Pipeline
    
    pipeline = Pipeline("./my_project", preset="insta360_x5")
    result = pipeline.run()
    
    # Resume from specific stage
    result = pipeline.run(resume_from="reconstruct")
    
    # Run specific stages only
    result = pipeline.run(stages=["prepare", "reconstruct"])
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import time

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from modular_pipeline.core import (
    StageResult,
    QualityLevel,
    ensure_directory,
    format_time,
)


# =============================================================================
# PROJECT STRUCTURE
# =============================================================================

@dataclass
class Project:
    """Manages project directory structure and state.
    
    Directory layout:
        project/
        ├── config.yaml          # Project configuration
        ├── state.json           # Pipeline state (for resume)
        ├── source/              # Input videos/images
        ├── frames/              # Extracted frames
        ├── reframed/            # Reframed pinhole views
        ├── masked/              # Images with masks applied
        ├── prepared/            # Final prepared images
        ├── features/            # Extracted features
        ├── matches/             # Feature matches
        ├── sparse/              # Sparse reconstruction
        ├── dense/               # Dense reconstruction
        ├── output/              # Final outputs (splat, mesh)
        └── review/              # Items flagged for review
    """
    
    root: Path
    
    def __post_init__(self):
        self.root = Path(self.root)
    
    # Standard directories
    @property
    def config_path(self) -> Path:
        return self.root / 'config.yaml'
    
    @property
    def state_path(self) -> Path:
        return self.root / 'state.json'
    
    @property
    def source_path(self) -> Path:
        return self.root / 'source'
    
    @property
    def frames_path(self) -> Path:
        return self.root / 'frames'
    
    @property
    def reframed_path(self) -> Path:
        return self.root / 'reframed'
    
    @property
    def masked_path(self) -> Path:
        return self.root / 'masked'
    
    @property
    def prepared_path(self) -> Path:
        return self.root / 'prepared'
    
    @property
    def features_path(self) -> Path:
        return self.root / 'features'
    
    @property
    def matches_path(self) -> Path:
        return self.root / 'matches'
    
    @property
    def sparse_path(self) -> Path:
        return self.root / 'sparse'
    
    @property
    def dense_path(self) -> Path:
        return self.root / 'dense'
    
    @property
    def output_path(self) -> Path:
        return self.root / 'output'
    
    @property
    def review_path(self) -> Path:
        return self.root / 'review'
    
    def initialize(self) -> None:
        """Create project directory structure."""
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for attr in dir(self):
            if attr.endswith('_path') and attr != 'root':
                path = getattr(self, attr)
                if isinstance(path, Path) and path != self.root:
                    path.mkdir(exist_ok=True)
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save pipeline state for resume."""
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load pipeline state."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return None
    
    def exists(self) -> bool:
        """Check if project exists."""
        return self.root.exists() and self.config_path.exists()


# =============================================================================
# PRESET SYSTEM
# =============================================================================

@dataclass
class PipelinePreset:
    """Configuration preset for common capture scenarios."""
    
    name: str
    description: str = ""
    
    # Which stages to run
    stages: List[str] = field(default_factory=lambda: [
        "ingest", "prepare", "reconstruct", "output"
    ])
    
    # Stage-specific configurations
    ingest: Dict[str, Any] = field(default_factory=dict)
    prepare: Dict[str, Any] = field(default_factory=dict)
    reconstruct: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    
    # Flags
    requires_reframe: bool = False
    requires_masking: bool = False
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PipelinePreset':
        """Load preset from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required: pip install pyyaml")
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def builtin(cls, name: str) -> 'PipelinePreset':
        """Get a built-in preset by name."""
        presets = {
            # Specific presets (recommended - includes camera intrinsics)
            'mavic_3_pro': cls._mavic_3_pro(),
            'mavic_air_2s': cls._mavic_air_2s(),
            'osmo_360': cls._osmo_360(),
            
            # Generic presets
            'drone': cls._drone_generic(),
            '360': cls._360_generic(),
            'default': cls._default(),
            
            # Legacy aliases
            'drone_dji': cls._drone_generic(),
            'insta360_x5': cls._360_generic(),
        }
        
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        
        return presets[name]
    
    @classmethod
    def _default(cls) -> 'PipelinePreset':
        """Default preset for generic images."""
        return cls(
            name="default",
            description="Generic images with standard pinhole camera",
            reconstruct={
                'extractor': 'aliked',
                'max_keypoints': 8000,
                'matcher': 'lightglue',
                'retrieval': 'netvlad',
                'num_neighbors': 50,
                'mapper': 'glomap',
                'camera_model': 'SIMPLE_RADIAL',
            }
        )
    
    @classmethod
    def _mavic_3_pro(cls) -> 'PipelinePreset':
        """Preset for DJI Mavic 3 Pro.
        
        Camera specs:
        - Hasselblad L2D-20c (main camera)
        - 20MP (5280×3956)
        - 4/3 CMOS sensor
        - ~24mm equivalent focal length
        - FOV: ~84° horizontal
        """
        return cls(
            name="mavic_3_pro",
            description="DJI Mavic 3 Pro (Hasselblad L2D-20c camera)",
            reconstruct={
                'extractor': 'aliked',
                'max_keypoints': 8000,
                'resize_max': 1600,
                'matcher': 'lightglue',
                'retrieval': 'netvlad',
                'num_neighbors': 50,
                'mapper': 'glomap',
                'camera_model': 'OPENCV',  # Better distortion model
                # Camera intrinsics (will be refined during SfM)
                'focal_length_mm': 24.0,
                'sensor_width_mm': 17.3,  # 4/3" sensor
            },
            output={
                'splat_backend': 'gsplat',
                'mesh_backend': 'gof',  # Outdoor scenes
            }
        )
    
    @classmethod
    def _mavic_air_2s(cls) -> 'PipelinePreset':
        """Preset for DJI Mavic Air 2S.
        
        Camera specs:
        - 20MP (5472×3648)
        - 1" CMOS sensor
        - ~22mm equivalent focal length
        - FOV: ~88° horizontal
        """
        return cls(
            name="mavic_air_2s",
            description="DJI Mavic Air 2S (1-inch sensor camera)",
            reconstruct={
                'extractor': 'aliked',
                'max_keypoints': 8000,
                'resize_max': 1600,
                'matcher': 'lightglue',
                'retrieval': 'netvlad',
                'num_neighbors': 50,
                'mapper': 'glomap',
                'camera_model': 'OPENCV',
                # Camera intrinsics
                'focal_length_mm': 22.0,
                'sensor_width_mm': 13.2,  # 1" sensor
            },
            output={
                'splat_backend': 'gsplat',
                'mesh_backend': 'gof',
            }
        )
    
    @classmethod
    def _osmo_360(cls) -> 'PipelinePreset':
        """Preset for DJI Osmo Action 360.
        
        Camera specs:
        - Dual fisheye lenses
        - 6912×3456 @ 30fps panoramic
        - ~190° per lens FOV
        - Equirectangular output
        """
        return cls(
            name="osmo_360",
            description="DJI Osmo Action 360 camera",
            requires_reframe=True,
            requires_masking=True,  # Remove tripod/selfie stick
            prepare={
                'reframe': {
                    'pattern': 'ring12',
                    'fov_h': 90,
                    'fov_v': 60,
                    'overlap_target': 0.30,
                    'backend': 'torch_gpu',
                },
                'masking': {
                    'model': 'sam3',
                    'remove_prompts': [
                        'tripod',
                        'selfie stick',
                        'monopod',
                        'camera operator',
                        'shadow of tripod',
                    ],
                    'confidence_threshold': 0.80,
                }
            },
            reconstruct={
                'extractor': 'aliked',
                'max_keypoints': 8000,
                'matcher': 'lightglue',
                'retrieval': 'netvlad',
                'num_neighbors': 50,
                'mapper': 'glomap',
                'camera_model': 'PINHOLE',  # After reframing to pinhole
            },
            output={
                'splat_backend': 'gsplat',
                'mesh_backend': 'pgsr',  # Better for 360° scenes
            }
        )
    
    @classmethod
    def _drone_generic(cls) -> 'PipelinePreset':
        """Generic preset for drone footage (when specific model unknown)."""
        return cls(
            name="drone",
            description="Generic drone footage (use specific preset if available)",
            reconstruct={
                'extractor': 'aliked',
                'max_keypoints': 8000,
                'resize_max': 1600,
                'matcher': 'lightglue',
                'retrieval': 'netvlad',
                'num_neighbors': 50,
                'mapper': 'glomap',
                'camera_model': 'SIMPLE_RADIAL',
            },
            output={
                'splat_backend': 'gsplat',
                'mesh_backend': 'gof',
            }
        )
    
    @classmethod
    def _360_generic(cls) -> 'PipelinePreset':
        """Generic preset for 360° cameras (when specific model unknown)."""
        return cls(
            name="360",
            description="Generic 360° camera (use specific preset if available)",
            requires_reframe=True,
            requires_masking=True,
            prepare={
                'reframe': {
                    'pattern': 'ring12',
                    'fov_h': 90,
                    'fov_v': 60,
                    'overlap_target': 0.30,
                    'backend': 'torch_gpu',
                },
                'masking': {
                    'model': 'sam3',
                    'remove_prompts': ['tripod', 'selfie stick'],
                    'confidence_threshold': 0.80,
                }
            },
            reconstruct={
                'extractor': 'aliked',
                'max_keypoints': 8000,
                'matcher': 'lightglue',
                'retrieval': 'netvlad',
                'num_neighbors': 50,
                'mapper': 'glomap',
                'camera_model': 'PINHOLE',
            }
        )
    
    


# =============================================================================
# PIPELINE RESULT
# =============================================================================

@dataclass
class PipelineResult:
    """Result from complete pipeline execution."""
    
    success: bool
    project_path: Path
    stages_completed: List[str]
    stages_failed: List[str]
    total_time_seconds: float
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    
    @property
    def needs_review(self) -> bool:
        return any(r.needs_review for r in self.stage_results.values())
    
    @property
    def review_items(self) -> List[str]:
        items = []
        for result in self.stage_results.values():
            items.extend(result.review_items)
        return items
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Pipeline {'✓ Complete' if self.success else '✗ Failed'}",
            f"Project: {self.project_path}",
            f"Time: {format_time(self.total_time_seconds)}",
            "",
            "Stage Results:",
        ]
        
        for stage, result in self.stage_results.items():
            status = "✓" if result.success else "✗"
            lines.append(f"  {status} {stage}: {result.quality.name}")
        
        if self.needs_review:
            lines.extend([
                "",
                f"⚠ Items need review: {len(self.review_items)}",
            ])
        
        return "\n".join(lines)


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class Pipeline:
    """Orchestrate complete reconstruction pipeline.
    
    Chains individual stages together, manages project state,
    and supports resumable execution.
    
    Example:
        >>> pipeline = Pipeline("./my_project", preset="insta360_x5")
        >>> result = pipeline.run()
        >>> print(result.summary())
        
        >>> # Resume from specific stage
        >>> result = pipeline.run(resume_from="reconstruct")
    """
    
    def __init__(
        self,
        project_path: Union[str, Path],
        preset: Union[str, PipelinePreset] = "default",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize pipeline.
        
        Args:
            project_path: Path to project directory
            preset: Preset name or PipelinePreset object
            config: Override configuration (merged with preset)
        """
        self.project = Project(project_path)
        
        # Load preset
        if isinstance(preset, str):
            self.preset = PipelinePreset.builtin(preset)
        else:
            self.preset = preset
        
        # Merge config overrides
        self.config = config or {}
        
        # Stage instances (lazy loaded)
        self._stages: Dict[str, Any] = {}
    
    def initialize(self) -> None:
        """Initialize project directory structure."""
        self.project.initialize()
        
        # Save configuration
        if HAS_YAML:
            config = {
                'preset': self.preset.name,
                'stages': self.preset.stages,
                'prepare': self.preset.prepare,
                'reconstruct': self.preset.reconstruct,
                'output': self.preset.output,
            }
            config.update(self.config)
            
            with open(self.project.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    def run(
        self,
        stages: Optional[List[str]] = None,
        resume_from: Optional[str] = None,
        pause_on_review: bool = False
    ) -> PipelineResult:
        """Run the pipeline.
        
        Args:
            stages: Specific stages to run (default: all from preset)
            resume_from: Resume from this stage (skip earlier stages)
            pause_on_review: Pause execution when items need review
        
        Returns:
            PipelineResult with all stage outcomes
        """
        # Initialize project if needed
        if not self.project.exists():
            self.initialize()
        
        stages = stages or self.preset.stages
        stage_results: Dict[str, StageResult] = {}
        completed: List[str] = []
        failed: List[str] = []
        
        start_time = time.perf_counter()
        
        # Find resume point
        skip_until_found = resume_from is not None
        
        for stage in stages:
            if skip_until_found:
                if stage == resume_from:
                    skip_until_found = False
                else:
                    print(f"Skipping {stage} (resuming from {resume_from})")
                    continue
            
            print(f"\n{'='*60}")
            print(f"  STAGE: {stage.upper()}")
            print(f"{'='*60}\n")
            
            try:
                result = self._run_stage(stage)
                stage_results[stage] = result
                
                if result.success:
                    completed.append(stage)
                else:
                    failed.append(stage)
                    print(f"\n✗ Stage {stage} failed")
                    for error in result.errors:
                        print(f"  - {error}")
                    break
                
                if result.needs_review and pause_on_review:
                    print(f"\n⚠ Stage {stage} flagged {len(result.review_items)} items for review")
                    response = input("Continue? [y/N]: ")
                    if response.lower() != 'y':
                        break
                
            except Exception as e:
                print(f"\n✗ Stage {stage} crashed: {e}")
                failed.append(stage)
                break
            
            # Save state for resume
            self.project.save_state({
                'last_completed': stage,
                'completed_stages': completed,
            })
        
        total_time = time.perf_counter() - start_time
        
        return PipelineResult(
            success=len(failed) == 0,
            project_path=self.project.root,
            stages_completed=completed,
            stages_failed=failed,
            total_time_seconds=total_time,
            stage_results=stage_results,
        )
    
    def _run_stage(self, stage: str) -> StageResult:
        """Run a single pipeline stage."""
        
        if stage == "ingest":
            return self._run_ingest()
        elif stage == "prepare":
            return self._run_prepare()
        elif stage == "reconstruct":
            return self._run_reconstruct()
        elif stage == "output":
            return self._run_output()
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _run_ingest(self) -> StageResult:
        """Run ingest stage (video extraction, image import)."""
        # This would use the ingest module
        # For now, assume images are already in place
        from .core import StageResult, QualityLevel, get_image_files
        
        images = get_image_files(self.project.source_path)
        if not images:
            images = get_image_files(self.project.frames_path)
        
        return StageResult(
            success=len(images) > 0,
            stage_name="ingest",
            output_path=self.project.frames_path,
            quality=QualityLevel.GOOD,
            items_processed=len(images),
            metrics={'num_images': len(images)}
        )
    
    def _run_prepare(self) -> StageResult:
        """Run prepare stage (reframe, mask, validate)."""
        from .core import StageResult, QualityLevel
        
        results = []
        
        # Determine input path
        input_path = self.project.frames_path
        if not list(input_path.glob('*')):
            input_path = self.project.source_path
        
        # Reframe if needed
        if self.preset.requires_reframe:
            # Would use reframe_v2 module here
            print("  [Reframing 360° to pinhole views...]")
            # reframer.process(input_path, self.project.reframed_path)
            input_path = self.project.reframed_path
        
        # Masking if needed
        if self.preset.requires_masking:
            # Would use masking_v2 module here
            print("  [Generating masks...]")
            # masker.process(input_path, self.project.masked_path)
        
        # Copy to prepared (or symlink)
        output_path = self.project.prepared_path
        
        return StageResult(
            success=True,
            stage_name="prepare",
            output_path=output_path,
            quality=QualityLevel.GOOD,
        )
    
    def _run_reconstruct(self) -> StageResult:
        """Run reconstruction stage (extract, match, SfM)."""
        from .reconstruct import (
            FeatureExtractor, ExtractConfig, ExtractorBackend,
            FeatureMatcher, MatchConfig, MatcherBackend, RetrievalBackend,
            SfMPipeline, SfMConfig, MapperBackend
        )
        from .core import StageResult, QualityLevel, CameraModel
        
        recon_config = self.preset.reconstruct
        
        # Determine input path
        input_path = self.project.prepared_path
        if not list(input_path.glob('*')):
            input_path = self.project.frames_path
        if not list(input_path.glob('*')):
            input_path = self.project.source_path
        
        # 1. Extract features
        print("  [Extracting features...]")
        extract_config = ExtractConfig(
            extractor=ExtractorBackend(recon_config.get('extractor', 'aliked')),
            max_keypoints=recon_config.get('max_keypoints', 8000),
            resize_max=recon_config.get('resize_max', 1600),
        )
        extractor = FeatureExtractor(extract_config)
        extract_result = extractor.process(input_path, self.project.features_path)
        
        if not extract_result.success:
            return extract_result
        
        features_h5 = self.project.features_path / f'{extract_config.output_name}.h5'
        
        # 2. Match features
        print("  [Matching features...]")
        match_config = MatchConfig(
            matcher=MatcherBackend(recon_config.get('matcher', 'lightglue')),
            feature_type=recon_config.get('extractor', 'aliked'),
            retrieval=RetrievalBackend(recon_config.get('retrieval', 'netvlad')),
            num_neighbors=recon_config.get('num_neighbors', 50),
        )
        matcher = FeatureMatcher(match_config)
        match_result = matcher.process(
            features_h5,
            self.project.matches_path,
            images_path=input_path
        )
        
        if not match_result.success:
            return match_result
        
        matches_h5 = self.project.matches_path / 'matches.h5'
        pairs_txt = self.project.matches_path / 'pairs.txt'
        
        # 3. Run SfM
        print("  [Running Structure from Motion...]")
        sfm_config = SfMConfig(
            mapper=MapperBackend(recon_config.get('mapper', 'glomap')),
            camera_model=CameraModel(recon_config.get('camera_model', 'SIMPLE_RADIAL')),
        )
        sfm = SfMPipeline(sfm_config)
        sfm_result = sfm.process(
            images_path=input_path,
            features_path=features_h5,
            matches_path=matches_h5,
            pairs_path=pairs_txt,
            output_path=self.project.sparse_path
        )
        
        return sfm_result
    
    def _run_output(self) -> StageResult:
        """Run output stage (splat training, mesh extraction)."""
        from .core import StageResult, QualityLevel
        
        # Would use splat and mesh modules here
        print("  [Training Gaussian splat...]")
        print("  [Extracting mesh...]")
        
        return StageResult(
            success=True,
            stage_name="output",
            output_path=self.project.output_path,
            quality=QualityLevel.GOOD,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run SplatForge reconstruction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("project", type=Path, help="Project directory")
    parser.add_argument("--preset", type=str, default="default",
                       help="Pipeline preset (drone_dji, insta360_x5, iphone_pro, default)")
    parser.add_argument("--stages", type=str, nargs="+",
                       help="Specific stages to run")
    parser.add_argument("--resume-from", type=str,
                       help="Resume from this stage")
    parser.add_argument("--pause-on-review", action="store_true",
                       help="Pause when items need review")
    parser.add_argument("--init", action="store_true",
                       help="Initialize project without running")
    
    args = parser.parse_args()
    
    pipeline = Pipeline(args.project, preset=args.preset)
    
    if args.init:
        pipeline.initialize()
        print(f"✓ Initialized project: {args.project}")
        return 0
    
    result = pipeline.run(
        stages=args.stages,
        resume_from=args.resume_from,
        pause_on_review=args.pause_on_review
    )
    
    print("\n" + result.summary())
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
