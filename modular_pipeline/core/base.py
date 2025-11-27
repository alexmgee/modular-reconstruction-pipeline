"""
SplatForge Core: Shared foundation for all pipeline modules.

This module provides:
- BaseModule: Abstract base class all stages inherit from
- BaseConfig: Configuration dataclass with YAML serialization
- StageResult: Standardized result type for pipeline stages
- QualityLevel: Universal quality assessment enum
- Backend: Processing backend with auto-fallback
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar
import json
import logging
import time

# Optional imports with fallback
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# ENUMS
# =============================================================================

class QualityLevel(Enum):
    """Universal quality assessment levels.
    
    Used across all pipeline stages to flag items for review or rejection.
    Thresholds are configurable per-module but semantic meaning is consistent.
    """
    EXCELLENT = auto()  # >0.95 confidence - auto-accept, no review needed
    GOOD = auto()       # 0.85-0.95 - accept with log entry
    REVIEW = auto()     # 0.70-0.85 - flag for manual review
    POOR = auto()       # 0.50-0.70 - likely needs manual intervention
    REJECT = auto()     # <0.50 - discard, do not use in pipeline

    @classmethod
    def from_confidence(
        cls,
        confidence: float,
        excellent: float = 0.95,
        good: float = 0.85,
        review: float = 0.70,
        poor: float = 0.50
    ) -> 'QualityLevel':
        """Convert confidence score to quality level."""
        if confidence >= excellent:
            return cls.EXCELLENT
        elif confidence >= good:
            return cls.GOOD
        elif confidence >= review:
            return cls.REVIEW
        elif confidence >= poor:
            return cls.POOR
        else:
            return cls.REJECT


class Backend(Enum):
    """Processing backend selection with automatic fallback.
    
    Modules should attempt backends in priority order:
    TORCH_GPU -> TORCH_CPU -> NUMPY -> OPENCV
    """
    AUTO = "auto"           # Auto-select best available
    TORCH_GPU = "torch_gpu" # CUDA/MPS acceleration (fastest)
    TORCH_CPU = "torch_cpu" # PyTorch CPU (good for batching)
    NUMPY = "numpy"         # Pure NumPy (portable)
    OPENCV = "opencv"       # OpenCV (always available baseline)


class ImageGeometry(Enum):
    """Image projection geometry types."""
    PINHOLE = "pinhole"             # Standard perspective camera
    FISHEYE = "fisheye"             # Single fisheye lens
    DUAL_FISHEYE = "dual_fisheye"   # 360° camera (two fisheyes)
    EQUIRECTANGULAR = "equirect"    # Full 360° panorama
    CUBEMAP = "cubemap"             # Cube map faces


class CameraModel(Enum):
    """COLMAP camera model types."""
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"       # f, cx, cy
    PINHOLE = "PINHOLE"                     # fx, fy, cx, cy
    SIMPLE_RADIAL = "SIMPLE_RADIAL"         # f, cx, cy, k1
    RADIAL = "RADIAL"                       # f, cx, cy, k1, k2
    OPENCV = "OPENCV"                       # fx, fy, cx, cy, k1, k2, p1, p2
    OPENCV_FISHEYE = "OPENCV_FISHEYE"       # fx, fy, cx, cy, k1, k2, k3, k4
    FULL_OPENCV = "FULL_OPENCV"             # All distortion params
    RADIAL_FISHEYE = "RADIAL_FISHEYE"       # For auto-calibration
    AUTO = "AUTO"                           # Let COLMAP decide


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class StageResult:
    """Standardized result from any pipeline stage.
    
    Every module returns this type, enabling:
    - Consistent success/failure checking
    - Quality metrics aggregation
    - Provenance tracking through manifests
    - Pipeline flow control (pause on review, halt on failure)
    """
    success: bool
    stage_name: str
    output_path: Path
    
    # Quality metrics
    quality: QualityLevel = QualityLevel.GOOD
    confidence: float = 1.0
    needs_review: bool = False
    
    # Processing statistics
    items_processed: int = 0
    items_failed: int = 0
    items_skipped: int = 0
    processing_time_seconds: float = 0.0
    
    # Detailed metrics (stage-specific)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Issues encountered
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    review_items: List[str] = field(default_factory=list)
    
    # Provenance
    config_used: Dict[str, Any] = field(default_factory=dict)
    backend_used: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['output_path'] = str(d['output_path'])
        d['quality'] = d['quality'].name
        return d
    
    def to_manifest(self) -> Dict[str, Any]:
        """Generate manifest entry for this stage."""
        return {
            'stage': self.stage_name,
            'timestamp': self.timestamp,
            'success': self.success,
            'output_path': str(self.output_path),
            'config': self.config_used,
            'backend': self.backend_used,
            'statistics': {
                'processed': self.items_processed,
                'failed': self.items_failed,
                'skipped': self.items_skipped,
                'time_seconds': self.processing_time_seconds,
            },
            'quality': {
                'level': self.quality.name,
                'confidence': self.confidence,
                'needs_review': self.needs_review,
                'review_items': self.review_items,
            },
            'metrics': self.metrics,
            'warnings': self.warnings,
            'errors': self.errors,
        }
    
    def save_manifest(self, path: Optional[Path] = None) -> Path:
        """Save manifest to JSON file."""
        if path is None:
            path = self.output_path / f'{self.stage_name}_manifest.json'
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_manifest(), f, indent=2)
        
        return path


@dataclass
class ItemResult:
    """Result for a single processed item (image, frame, etc.)."""
    item_id: str                    # Filename or unique identifier
    success: bool
    quality: QualityLevel
    confidence: float
    processing_time: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def needs_review(self) -> bool:
        return self.quality in [QualityLevel.REVIEW, QualityLevel.POOR]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BaseConfig:
    """Base configuration that all module configs extend.
    
    Provides:
    - Device/backend selection with auto-detection
    - Processing parallelism settings
    - Quality control thresholds
    - Output and logging options
    - YAML/JSON serialization
    """
    # Device settings
    device: str = "auto"        # auto, cuda, cuda:0, cpu, mps
    backend: Backend = Backend.AUTO
    
    # Processing settings
    num_workers: int = 8        # CPU workers for parallel processing
    batch_size: int = 4         # Items per batch (GPU memory dependent)
    
    # Quality control thresholds
    excellent_threshold: float = 0.95
    good_threshold: float = 0.85
    review_threshold: float = 0.70
    reject_threshold: float = 0.50
    
    # Output settings
    save_intermediates: bool = False    # Save intermediate results
    save_visualizations: bool = False   # Save debug visualizations
    save_manifests: bool = True         # Save provenance manifests
    
    # Logging
    verbose: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        if isinstance(self.backend, str):
            self.backend = Backend(self.backend)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        
        with open(path) as f:
            if path.suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Handle nested Backend enum
        if 'backend' in data and isinstance(data['backend'], str):
            data['backend'] = Backend(data['backend'])
        
        return cls(**data)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML or JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = asdict(self)
        # Convert enums to strings
        data['backend'] = data['backend'].value if isinstance(data['backend'], Backend) else data['backend']
        
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['backend'] = d['backend'].value if isinstance(d['backend'], Backend) else d['backend']
        return d
    
    def confidence_to_quality(self, confidence: float) -> QualityLevel:
        """Convert confidence score to quality level using this config's thresholds."""
        return QualityLevel.from_confidence(
            confidence,
            excellent=self.excellent_threshold,
            good=self.good_threshold,
            review=self.review_threshold,
            poor=self.reject_threshold
        )


# =============================================================================
# BASE MODULE
# =============================================================================

class BaseModule(ABC):
    """Abstract base class for all pipeline modules.
    
    Provides:
    - Backend selection with automatic fallback
    - Device management (GPU/CPU)
    - Logging setup
    - Progress tracking
    - Quality assessment utilities
    
    Subclasses must implement:
    - _default_config(): Return default configuration
    - _initialize(): Load models, setup backends
    - process(): Main processing entry point
    """
    
    def __init__(self, config: Optional[BaseConfig] = None):
        """Initialize module with optional configuration.
        
        Args:
            config: Configuration object. If None, uses default.
        """
        self.config = config if config is not None else self._default_config()
        self._backend: Optional[Backend] = None
        self._device: Optional[str] = None
        self._logger: Optional[logging.Logger] = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize (load models, etc.)
        self._initialize()
    
    @abstractmethod
    def _default_config(self) -> BaseConfig:
        """Return default configuration for this module.
        
        Must be implemented by subclasses to provide sensible defaults.
        """
        pass
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize module resources.
        
        Called after config is set. Should:
        - Select and initialize backend
        - Load any required models
        - Setup processing resources
        """
        pass
    
    @abstractmethod
    def process(self, input_path: Path, output_path: Path, **kwargs) -> StageResult:
        """Process input and produce output.
        
        Args:
            input_path: Path to input (file or directory)
            output_path: Path for output (file or directory)
            **kwargs: Stage-specific options
        
        Returns:
            StageResult with processing outcome
        """
        pass
    
    # -------------------------------------------------------------------------
    # Backend & Device Management
    # -------------------------------------------------------------------------
    
    def _select_backend(self) -> Backend:
        """Select best available backend with fallback.
        
        Checks backend availability in priority order and returns
        the first available option.
        """
        if self.config.backend != Backend.AUTO:
            return self.config.backend
        
        # Try backends in priority order
        if HAS_TORCH:
            import torch
            if torch.cuda.is_available():
                return Backend.TORCH_GPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return Backend.TORCH_GPU  # MPS uses same code path
            return Backend.TORCH_CPU
        
        if HAS_NUMPY:
            return Backend.NUMPY
        
        return Backend.OPENCV
    
    def _get_device(self) -> str:
        """Get device string for torch operations."""
        if self.config.device != "auto":
            return self.config.device
        
        if not HAS_TORCH:
            return "cpu"
        
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    @property
    def backend(self) -> Backend:
        """Currently selected backend."""
        if self._backend is None:
            self._backend = self._select_backend()
        return self._backend
    
    @property
    def device(self) -> str:
        """Currently selected device."""
        if self._device is None:
            self._device = self._get_device()
        return self._device
    
    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    
    def _setup_logging(self) -> None:
        """Setup logging for this module."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            getattr(self._logger, level.lower())(message)
    
    # -------------------------------------------------------------------------
    # Processing Utilities
    # -------------------------------------------------------------------------
    
    def _create_result(
        self,
        success: bool,
        output_path: Path,
        items_processed: int = 0,
        items_failed: int = 0,
        processing_time: float = 0.0,
        **kwargs
    ) -> StageResult:
        """Create a StageResult with common fields populated."""
        return StageResult(
            success=success,
            stage_name=self.__class__.__name__,
            output_path=output_path,
            items_processed=items_processed,
            items_failed=items_failed,
            processing_time_seconds=processing_time,
            config_used=self.config.to_dict(),
            backend_used=self.backend.value,
            **kwargs
        )
    
    def _timed_process(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> tuple[Any, float]:
        """Execute a function and return result with elapsed time."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    
    def _aggregate_quality(self, results: List[ItemResult]) -> tuple[QualityLevel, float]:
        """Aggregate quality from multiple item results."""
        if not results:
            return QualityLevel.GOOD, 1.0
        
        confidences = [r.confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Overall quality is the worst individual quality
        qualities = [r.quality for r in results]
        worst_quality = max(qualities, key=lambda q: q.value)
        
        return worst_quality, avg_confidence


# =============================================================================
# DEVICE MANAGER
# =============================================================================

class DeviceManager:
    """Manages GPU/CPU device allocation and memory.
    
    Provides utilities for:
    - Multi-GPU selection
    - Memory tracking
    - Automatic garbage collection
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._devices: List[str] = []
        self._detect_devices()
    
    def _detect_devices(self) -> None:
        """Detect available compute devices."""
        self._devices = ["cpu"]
        
        if HAS_TORCH:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    self._devices.append(f"cuda:{i}")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._devices.append("mps")
    
    @property
    def available_devices(self) -> List[str]:
        """List of available compute devices."""
        return self._devices.copy()
    
    @property
    def best_device(self) -> str:
        """Best available device (prefers GPU)."""
        for device in self._devices:
            if device.startswith("cuda") or device == "mps":
                return device
        return "cpu"
    
    @property
    def gpu_count(self) -> int:
        """Number of available GPUs."""
        return len([d for d in self._devices if d.startswith("cuda")])
    
    def get_memory_info(self, device: str = "cuda:0") -> Dict[str, int]:
        """Get memory info for a device (in bytes)."""
        if not HAS_TORCH or not device.startswith("cuda"):
            return {"total": 0, "allocated": 0, "free": 0}
        
        import torch
        idx = int(device.split(":")[1]) if ":" in device else 0
        props = torch.cuda.get_device_properties(idx)
        allocated = torch.cuda.memory_allocated(idx)
        
        return {
            "total": props.total_memory,
            "allocated": allocated,
            "free": props.total_memory - allocated
        }
    
    def clear_cache(self, device: Optional[str] = None) -> None:
        """Clear GPU memory cache."""
        if not HAS_TORCH:
            return
        
        import torch
        if device is None or device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_image_files(
    directory: Path,
    extensions: tuple = ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'),
    recursive: bool = False
) -> List[Path]:
    """Get all image files in a directory."""
    directory = Path(directory)
    
    if recursive:
        files = []
        for ext in extensions:
            files.extend(directory.rglob(f'*{ext}'))
            files.extend(directory.rglob(f'*{ext.upper()}'))
    else:
        files = []
        for ext in extensions:
            files.extend(directory.glob(f'*{ext}'))
            files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(files)


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_size(bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} PB"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'QualityLevel',
    'Backend',
    'ImageGeometry',
    'CameraModel',
    
    # Result types
    'StageResult',
    'ItemResult',
    
    # Configuration
    'BaseConfig',
    
    # Base class
    'BaseModule',
    
    # Device management
    'DeviceManager',
    
    # Utilities
    'get_image_files',
    'ensure_directory',
    'format_time',
    'format_size',
    
    # Feature flags
    'HAS_TORCH',
    'HAS_NUMPY',
    'HAS_YAML',
]
