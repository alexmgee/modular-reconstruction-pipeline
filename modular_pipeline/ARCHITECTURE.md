# SplatForge: Modular 3D Reconstruction Pipeline

## Vision

A cohesive, modular reconstruction system where each stage is:
- **Standalone**: Runs independently with its own CLI and Python API
- **Composable**: Chains seamlessly with other stages via standardized interfaces
- **Configurable**: YAML/JSON configs with sensible defaults and full override capability
- **Observable**: Quality metrics, progress tracking, and review flagging at every stage
- **Resilient**: Multiple backends with automatic fallback, graceful degradation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SPLATFORGE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   INGEST    │───▶│   PREPARE   │───▶│  RECONSTRUCT│───▶│   OUTPUT    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SHARED FOUNDATION                             │   │
│  │  • BaseModule (interface)  • StageResult (dataclass)                │   │
│  │  • BaseConfig (dataclass)  • QualityLevel (enum)                    │   │
│  │  • Backend (enum)          • DeviceManager (GPU/CPU)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

### Stage 1: INGEST (Source → Frames)
```
ingest_v2/
├── video_extract.py      # Video → frames (ffmpeg, OpenCV)
├── image_import.py       # Copy/organize existing images
├── metadata_extract.py   # EXIF, GPS, timestamps
└── source_catalog.py     # Track provenance, create manifest
```

**Inputs**: Video files (MP4, MOV), image directories, mixed sources
**Outputs**: Organized frame directory + `manifest.json`

### Stage 2: PREPARE (Frames → Pipeline-Ready Images)
```
prepare_v2/
├── reframe_v2.py         # 360° → pinhole rig (YOUR MODULE)
├── masking_v2.py         # Remove tripods/operators (YOUR MODULE)
├── undistort.py          # Fisheye → pinhole correction
├── enhance.py            # Exposure, color, sharpening
└── validate.py           # Quality checks, blur detection
```

**Inputs**: Raw frames from ingest
**Outputs**: Clean, masked, pinhole-ready images + `prepare_manifest.json`

### Stage 3: RECONSTRUCT (Images → 3D)
```
reconstruct_v2/
├── extract_v2.py         # Feature extraction (ALIKED, SuperPoint, XFeat)
├── match_v2.py           # Feature matching (LightGlue + MASt3R fallback)
├── retrieve_v2.py        # Pair selection (NetVLAD, vocab tree)
├── sfm_v2.py             # Structure from Motion (GLOMAP, COLMAP)
├── densify_v2.py         # Point cloud densification
└── database.py           # COLMAP DB operations
```

**Inputs**: Prepared images + masks
**Outputs**: Sparse reconstruction, camera poses, dense point cloud

### Stage 4: OUTPUT (3D → Deliverables)
```
output_v2/
├── splat_v2.py           # Gaussian splatting training (gsplat, 3DGS)
├── mesh_v2.py            # Mesh extraction (GOF, PGSR, 2DGS)
├── export.py             # Format conversion (PLY, OBJ, GLB)
└── viewer.py             # Local preview server
```

**Inputs**: Sparse/dense reconstruction
**Outputs**: Trained Gaussian splat, extracted mesh, viewer-ready exports

---

## Shared Foundation

### Base Interfaces

All modules inherit from `BaseModule` and use standardized types:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml

class QualityLevel(Enum):
    """Universal quality assessment."""
    EXCELLENT = auto()  # >0.95 confidence - auto-accept
    GOOD = auto()       # 0.85-0.95 - accept with log
    REVIEW = auto()     # 0.70-0.85 - flag for review
    POOR = auto()       # 0.50-0.70 - manual required
    REJECT = auto()     # <0.50 - discard

class Backend(Enum):
    """Processing backend with auto-fallback."""
    AUTO = "auto"           # Auto-select best available
    TORCH_GPU = "torch_gpu" # CUDA acceleration
    TORCH_CPU = "torch_cpu" # PyTorch CPU
    NUMPY = "numpy"         # Pure NumPy
    OPENCV = "opencv"       # OpenCV (always available)

@dataclass
class StageResult:
    """Standardized result from any pipeline stage."""
    success: bool
    stage_name: str
    output_path: Path
    
    # Quality metrics
    quality: QualityLevel = QualityLevel.GOOD
    confidence: float = 1.0
    needs_review: bool = False
    
    # Statistics
    items_processed: int = 0
    items_failed: int = 0
    processing_time: float = 0.0
    
    # Detailed results
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Provenance
    config_used: Dict[str, Any] = field(default_factory=dict)
    backend_used: str = ""

@dataclass
class BaseConfig:
    """Base configuration all modules extend."""
    # Device settings
    device: str = "auto"  # auto, cuda, cuda:0, cpu
    backend: Backend = Backend.AUTO
    
    # Processing settings
    num_workers: int = 8
    batch_size: int = 4
    
    # Quality control
    confidence_threshold: float = 0.70
    review_threshold: float = 0.85
    
    # Output settings
    save_intermediates: bool = False
    save_visualizations: bool = False
    verbose: bool = True
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseConfig':
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

class BaseModule(ABC):
    """Abstract base for all pipeline modules."""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        self.config = config or self._default_config()
        self._backend = None
        self._initialize()
    
    @abstractmethod
    def _default_config(self) -> BaseConfig:
        """Return default configuration."""
        pass
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize module (load models, setup backend)."""
        pass
    
    @abstractmethod
    def process(self, input_path: Path, output_path: Path, **kwargs) -> StageResult:
        """Main processing entry point."""
        pass
    
    def _select_backend(self) -> Backend:
        """Auto-select best available backend."""
        if self.config.backend != Backend.AUTO:
            return self.config.backend
        
        # Try backends in priority order
        try:
            import torch
            if torch.cuda.is_available():
                return Backend.TORCH_GPU
            return Backend.TORCH_CPU
        except ImportError:
            pass
        
        return Backend.OPENCV  # Always available fallback
    
    def _get_device(self) -> str:
        """Get torch device string."""
        if self.config.device != "auto":
            return self.config.device
        
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
```

---

## Module Design Patterns

### Pattern 1: Backend Abstraction
Each module supports multiple backends with automatic fallback:

```python
class ExtractorBackend(Enum):
    ALIKED = "aliked"           # Recommended (fast + accurate)
    SUPERPOINT = "superpoint"   # Legacy compatibility
    XFEAT = "xfeat"             # Real-time scenarios
    DISK = "disk"               # Dense keypoints

class ExtractConfig(BaseConfig):
    extractor: ExtractorBackend = ExtractorBackend.ALIKED
    max_keypoints: int = 8000
    nms_radius: int = 3
    resize_max: int = 1600
    
class FeatureExtractor(BaseModule):
    def _initialize(self):
        self._load_model(self.config.extractor)
    
    def _load_model(self, backend: ExtractorBackend):
        if backend == ExtractorBackend.ALIKED:
            try:
                from lightglue import ALIKED
                self._model = ALIKED(max_num_keypoints=self.config.max_keypoints)
                return
            except ImportError:
                print("ALIKED unavailable, falling back to SuperPoint")
        
        # Fallback chain continues...
```

### Pattern 2: Quality Control Pipeline
Every stage produces quality metrics and can flag items for review:

```python
def process_image(self, image: np.ndarray) -> ImageResult:
    # Process
    keypoints, descriptors = self._extract(image)
    
    # Quality assessment
    confidence = self._assess_quality(keypoints, descriptors, image)
    quality = self._confidence_to_quality(confidence)
    needs_review = quality in [QualityLevel.REVIEW, QualityLevel.POOR]
    
    return ImageResult(
        keypoints=keypoints,
        descriptors=descriptors,
        confidence=confidence,
        quality=quality,
        needs_review=needs_review
    )
```

### Pattern 3: CLI + Python API
Every module has both interfaces:

```python
# Python API
from reconstruct_v2 import FeatureExtractor, ExtractConfig

config = ExtractConfig(extractor="aliked", max_keypoints=8000)
extractor = FeatureExtractor(config)
result = extractor.process(input_dir, output_dir)

# CLI (auto-generated from config dataclass)
# python -m reconstruct_v2.extract input/ output/ --extractor aliked --max-keypoints 8000
```

### Pattern 4: Manifest Tracking
Each stage produces a manifest for provenance:

```json
{
  "stage": "extract_v2",
  "version": "2.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "config": {
    "extractor": "aliked",
    "max_keypoints": 8000
  },
  "input": {
    "path": "/data/prepared",
    "manifest": "prepare_manifest.json",
    "image_count": 6234
  },
  "output": {
    "features_path": "features.h5",
    "total_keypoints": 42851234,
    "average_per_image": 6876
  },
  "quality": {
    "excellent": 5890,
    "good": 312,
    "review": 28,
    "poor": 4
  }
}
```

---

## Camera Preset System

Standardized presets for common capture scenarios:

```yaml
# presets/drone_dji.yaml
name: "DJI Drone (Mavic/Air/Mini)"
geometry: pinhole
camera_model: SIMPLE_RADIAL
extractor:
  backend: aliked
  max_keypoints: 8000
  resize_max: 1600
matcher:
  backend: lightglue
  filter_threshold: 0.1
sfm:
  backend: glomap
  mapper_options:
    max_num_tracks: 6000000

# presets/insta360_x5.yaml
name: "Insta360 X5 (360° Video)"
geometry: equirectangular
requires_reframe: true
reframe:
  pattern: ring
  num_cameras: 12
  fov_h: 90
  fov_v: 60
  overlap_target: 0.30
masking:
  model: sam3
  remove_prompts:
    - "tripod"
    - "selfie stick"
    - "camera operator"
camera_model: PINHOLE  # After reframing
extractor:
  backend: aliked
  max_keypoints: 8000

# presets/iphone_pro.yaml
name: "iPhone Pro (with LiDAR)"
geometry: pinhole
camera_model: SIMPLE_RADIAL
capture_tips:
  - "Use standard 12MP JPEG, not 48MP ProRAW"
  - "Lock exposure and focus (Blackmagic Cam or Halide)"
extractor:
  backend: xfeat  # CPU-friendly
  max_keypoints: 4096
```

---

## Orchestrator

The runner that chains stages together:

```python
class SplatForgePipeline:
    """Orchestrates the complete reconstruction pipeline."""
    
    def __init__(self, project_path: Path, preset: str = "auto"):
        self.project = Project(project_path)
        self.preset = self._load_preset(preset)
        self._stages = {}
    
    def run(
        self,
        stages: List[str] = ["ingest", "prepare", "reconstruct", "output"],
        resume_from: Optional[str] = None
    ) -> PipelineResult:
        """Run specified stages in sequence."""
        
        results = []
        for stage in stages:
            if resume_from and stage != resume_from:
                continue
            resume_from = None  # Found resume point
            
            print(f"\n{'='*60}")
            print(f"  STAGE: {stage.upper()}")
            print(f"{'='*60}\n")
            
            result = self._run_stage(stage)
            results.append(result)
            
            if not result.success:
                print(f"Stage {stage} failed. Stopping pipeline.")
                break
            
            if result.needs_review:
                print(f"Stage {stage} flagged items for review.")
                if self.config.pause_on_review:
                    input("Press Enter to continue after review...")
        
        return PipelineResult(stages=results)
    
    def _run_stage(self, stage: str) -> StageResult:
        """Run a single pipeline stage."""
        
        if stage == "ingest":
            from ingest_v2 import VideoExtractor
            module = VideoExtractor(self.preset.ingest)
            return module.process(self.project.source_path, self.project.frames_path)
        
        elif stage == "prepare":
            # Chain: reframe → mask → validate
            results = []
            
            if self.preset.requires_reframe:
                from prepare_v2 import EquirectToRig
                reframer = EquirectToRig(self.preset.reframe)
                results.append(reframer.process(
                    self.project.frames_path,
                    self.project.reframed_path
                ))
            
            from prepare_v2 import MaskingPipeline
            masker = MaskingPipeline(self.preset.masking)
            results.append(masker.process(
                self.project.reframed_path or self.project.frames_path,
                self.project.masked_path
            ))
            
            return self._merge_results("prepare", results)
        
        # ... other stages
```

---

## Directory Structure

```
splatforge/
├── pyproject.toml              # Package configuration
├── README.md                   # Getting started guide
├── ARCHITECTURE.md             # This document
│
├── splatforge/
│   ├── __init__.py
│   ├── core/                   # Shared foundation
│   │   ├── __init__.py
│   │   ├── base.py             # BaseModule, BaseConfig, StageResult
│   │   ├── device.py           # GPU/CPU management
│   │   ├── quality.py          # Quality assessment utilities
│   │   └── manifest.py         # Provenance tracking
│   │
│   ├── ingest/                 # Stage 1: Source ingestion
│   │   ├── __init__.py
│   │   ├── video.py
│   │   ├── images.py
│   │   └── metadata.py
│   │
│   ├── prepare/                # Stage 2: Data preparation
│   │   ├── __init__.py
│   │   ├── reframe.py          # YOUR reframe_v2
│   │   ├── masking.py          # YOUR masking_v2
│   │   ├── undistort.py
│   │   └── validate.py
│   │
│   ├── reconstruct/            # Stage 3: 3D reconstruction
│   │   ├── __init__.py
│   │   ├── extract.py          # Feature extraction
│   │   ├── match.py            # Feature matching
│   │   ├── retrieve.py         # Pair selection
│   │   ├── sfm.py              # Structure from Motion
│   │   └── database.py         # COLMAP DB operations
│   │
│   ├── output/                 # Stage 4: Output generation
│   │   ├── __init__.py
│   │   ├── splat.py            # Gaussian splatting
│   │   ├── mesh.py             # Mesh extraction
│   │   └── export.py           # Format conversion
│   │
│   ├── presets/                # Camera/scenario presets
│   │   ├── drone_dji.yaml
│   │   ├── insta360_x5.yaml
│   │   ├── iphone_pro.yaml
│   │   └── mixed_capture.yaml
│   │
│   └── cli/                    # Command-line interface
│       ├── __init__.py
│       ├── main.py             # splatforge command
│       └── commands/           # Subcommands
│
├── tests/
│   ├── test_core/
│   ├── test_ingest/
│   ├── test_prepare/
│   ├── test_reconstruct/
│   └── test_output/
│
└── examples/
    ├── drone_workflow.py
    ├── insta360_workflow.py
    └── mixed_sources_workflow.py
```

---

## CLI Interface

```bash
# Full pipeline with preset
splatforge run ./my_project --preset insta360_x5

# Individual stages
splatforge ingest ./video.mp4 ./frames --skip-frames 2
splatforge prepare ./frames ./prepared --reframe ring-12 --mask sam3
splatforge reconstruct ./prepared ./sparse --extractor aliked --sfm glomap
splatforge output ./sparse ./final --splat gsplat --mesh gof

# Stage-specific options
splatforge extract ./images ./features \
    --extractor aliked \
    --max-keypoints 8000 \
    --resize-max 1600 \
    --save-visualizations

splatforge match ./features ./matches \
    --matcher lightglue \
    --fallback mast3r \
    --retrieval netvlad \
    --neighbors 50

# Project management
splatforge init ./new_project --preset drone_dji
splatforge status ./my_project
splatforge review ./my_project  # Interactive review of flagged items
splatforge resume ./my_project --from reconstruct
```

---

## Next Steps

1. **Core Foundation** (this session): Implement `core/` module with base classes
2. **Integrate Your Modules**: Adapt reframe_v2 and masking_v2 to use shared base
3. **Reconstruction Stages**: Build extract, match, sfm modules
4. **Output Stages**: Build splat and mesh modules
5. **CLI & Presets**: Create command-line interface and preset system
6. **Testing & Examples**: Comprehensive test suite and workflow examples
