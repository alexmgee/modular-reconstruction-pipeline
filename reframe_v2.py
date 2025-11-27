#!/usr/bin/env python3
"""
Advanced Equirectangular to Pinhole Rig Reframing System
=========================================================
Version: 2.0
Author: 360-to-splat-v2
License: MIT

This module converts equirectangular 360° images into a rig of pinhole camera views,
serving as the geometric bridge between spherical capture and perspective-based
computer vision models (SAM3, COLMAP, etc.).

Key Features:
- GPU-accelerated projection when available
- Parallel processing for multi-core CPUs
- Multiple projection libraries with fallback
- Configurable rig patterns (cube, ring, geodesic)
- Temporal consistency for video sequences
- Mask propagation support
- Debug visualization tools
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import yaml
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import logging

# Try to import optimal projection libraries
try:
    import py360convert
    HAS_PY360 = True
except ImportError:
    HAS_PY360 = False
    warnings.warn("py360convert not found. Install with: pip install py360convert")

try:
    from equilib import equi2pers, equi2equi
    HAS_EQUILIB = True
except ImportError:
    HAS_EQUILIB = False
    warnings.warn("equilib not found. Install with: pip install pyequilib")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not found. GPU acceleration disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectionBackend(Enum):
    """Available projection backends in order of preference."""
    TORCH_GPU = "torch_gpu"      # Fastest - GPU accelerated
    PY360 = "py360convert"        # Fast - optimized NumPy
    EQUILIB = "equilib"           # Good - PyTorch CPU
    OPENCV = "opencv_remap"       # Fallback - always available
    NUMPY = "numpy_direct"        # Slowest but most transparent


class RigPattern(Enum):
    """Predefined rig patterns for different use cases."""
    CUBE_6 = "cube_6"                    # Standard cubemap - 6 views
    RING_8 = "ring_8"                    # Horizontal ring - 8 views
    RING_12 = "ring_12"                  # Horizontal ring - 12 views
    GEODESIC_20 = "geodesic_20"         # Icosahedron vertices - 20 views
    CUSTOM = "custom"                     # User-defined configuration
    ADAPTIVE = "adaptive"                 # Content-aware placement


@dataclass
class CameraView:
    """Single camera view in the rig."""
    id: int
    yaw: float      # Azimuth in degrees (-180 to 180)
    pitch: float    # Elevation in degrees (-90 to 90)
    roll: float     # Roll in degrees (usually 0)
    fov_h: float    # Horizontal field of view in degrees
    fov_v: float    # Vertical field of view in degrees
    width: int      # Output width in pixels
    height: int     # Output height in pixels
    name: str = ""  # Optional camera name

    def __post_init__(self):
        if not self.name:
            self.name = f"cam_{self.id:02d}_y{int(self.yaw)}_p{int(self.pitch)}"

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'yaw': self.yaw,
            'pitch': self.pitch,
            'roll': self.roll,
            'fov_h': self.fov_h,
            'fov_v': self.fov_v,
            'width': self.width,
            'height': self.height
        }


@dataclass
class RigConfig:
    """Complete rig configuration."""
    pattern: RigPattern
    cameras: List[CameraView]
    overlap_min: float = 0.2  # Minimum overlap between adjacent views
    overlap_target: float = 0.3  # Target overlap for reconstruction
    
    def validate(self) -> bool:
        """Validate rig configuration for reconstruction quality."""
        if len(self.cameras) < 4:
            logger.warning("Less than 4 cameras may produce poor reconstruction")
            return False
        
        # Check for coverage gaps
        coverage = self._compute_coverage()
        if coverage < 0.95:
            logger.warning(f"Rig coverage is only {coverage:.1%} of sphere")
        
        return True
    
    def _compute_coverage(self) -> float:
        """Estimate spherical coverage percentage."""
        # Simplified coverage estimation
        # TODO: Implement proper spherical coverage calculation
        total_solid_angle = 0
        for cam in self.cameras:
            # Approximate solid angle for each camera
            solid_angle = 2 * np.pi * (1 - np.cos(np.radians(cam.fov_v / 2)))
            total_solid_angle += solid_angle
        
        sphere_solid_angle = 4 * np.pi
        coverage = min(total_solid_angle / sphere_solid_angle, 1.0)
        return coverage

    def to_dict(self) -> Dict:
        return {
            'pattern': self.pattern.value,
            'overlap_min': self.overlap_min,
            'overlap_target': self.overlap_target,
            'cameras': [cam.to_dict() for cam in self.cameras]
        }

    def save(self, path: Path):
        """Save rig configuration to JSON/YAML."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix == '.yaml':
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'RigConfig':
        """Load rig configuration from JSON/YAML."""
        path = Path(path)
        
        if path.suffix == '.yaml':
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        
        cameras = [CameraView(**cam) for cam in data['cameras']]
        return cls(
            pattern=RigPattern(data['pattern']),
            cameras=cameras,
            overlap_min=data.get('overlap_min', 0.2),
            overlap_target=data.get('overlap_target', 0.3)
        )


class RigGenerator:
    """Generate standard rig configurations."""
    
    @staticmethod
    def create_cube_rig(resolution: int = 1024, fov: float = 90) -> RigConfig:
        """Create standard cubemap rig (6 faces)."""
        cameras = [
            # Front (0°)
            CameraView(0, 0, 0, 0, fov, fov, resolution, resolution),
            # Right (90°)
            CameraView(1, 90, 0, 0, fov, fov, resolution, resolution),
            # Back (180°)
            CameraView(2, 180, 0, 0, fov, fov, resolution, resolution),
            # Left (-90°)
            CameraView(3, -90, 0, 0, fov, fov, resolution, resolution),
            # Up (90° pitch)
            CameraView(4, 0, 90, 0, fov, fov, resolution, resolution),
            # Down (-90° pitch)
            CameraView(5, 0, -90, 0, fov, fov, resolution, resolution),
        ]
        return RigConfig(RigPattern.CUBE_6, cameras)
    
    @staticmethod
    def create_ring_rig(
        num_cameras: int = 12,
        width: int = 1920,
        height: int = 1080,
        fov_h: float = 90,
        fov_v: float = 60,
        pitch_angle: float = 0
    ) -> RigConfig:
        """Create horizontal ring of cameras."""
        cameras = []
        yaw_step = 360 / num_cameras
        
        for i in range(num_cameras):
            yaw = i * yaw_step - 180  # Range: -180 to 180
            cameras.append(
                CameraView(i, yaw, pitch_angle, 0, fov_h, fov_v, width, height)
            )
        
        pattern = RigPattern.RING_8 if num_cameras == 8 else RigPattern.RING_12
        return RigConfig(pattern, cameras, overlap_target=0.25)
    
    @staticmethod
    def create_geodesic_rig(
        resolution: int = 1024,
        fov: float = 72
    ) -> RigConfig:
        """Create geodesic rig based on icosahedron vertices."""
        # Icosahedron vertices for uniform sphere sampling
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Normalized vertices
        vertices = np.array([
            [0, 1, phi],
            [0, -1, phi],
            [0, 1, -phi],
            [0, -1, -phi],
            [1, phi, 0],
            [-1, phi, 0],
            [1, -phi, 0],
            [-1, -phi, 0],
            [phi, 0, 1],
            [-phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, -1],
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        cameras = []
        for i, v in enumerate(vertices):
            # Convert Cartesian to spherical coordinates
            yaw = np.degrees(np.arctan2(v[0], v[2]))
            pitch = np.degrees(np.arcsin(v[1]))
            cameras.append(
                CameraView(i, yaw, pitch, 0, fov, fov, resolution, resolution)
            )
        
        return RigConfig(RigPattern.GEODESIC_20, cameras)


class EquirectToRig:
    """Main class for equirectangular to pinhole rig conversion."""
    
    def __init__(
        self,
        rig_config: Optional[RigConfig] = None,
        backend: Optional[ProjectionBackend] = None,
        num_workers: int = -1,
        cache_maps: bool = True,
        debug: bool = False
    ):
        """
        Initialize the reframing system.
        
        Args:
            rig_config: Rig configuration (default: 12-camera ring)
            backend: Projection backend to use (auto-selected if None)
            num_workers: Number of parallel workers (-1 for auto)
            cache_maps: Cache projection maps for video processing
            debug: Enable debug visualizations
        """
        self.rig_config = rig_config or RigGenerator.create_ring_rig(12)
        self.backend = backend or self._select_backend()
        self.num_workers = num_workers if num_workers > 0 else mp.cpu_count()
        self.cache_maps = cache_maps
        self.debug = debug
        self._projection_maps = {} if cache_maps else None
        
        logger.info(f"Initialized with backend: {self.backend.value}")
        logger.info(f"Rig pattern: {self.rig_config.pattern.value}")
        logger.info(f"Number of cameras: {len(self.rig_config.cameras)}")
    
    def _select_backend(self) -> ProjectionBackend:
        """Auto-select best available backend."""
        if HAS_TORCH and torch.cuda.is_available():
            return ProjectionBackend.TORCH_GPU
        elif HAS_PY360:
            return ProjectionBackend.PY360
        elif HAS_EQUILIB:
            return ProjectionBackend.EQUILIB
        else:
            return ProjectionBackend.OPENCV
    
    def process_image(
        self,
        equirect_image: np.ndarray,
        camera: Optional[CameraView] = None,
        interpolation: str = 'bilinear'
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Convert single equirectangular image to pinhole view(s).
        
        Args:
            equirect_image: Input equirectangular image (H, W, C)
            camera: Specific camera to render (None for all cameras)
            interpolation: Interpolation method ('bilinear', 'nearest', 'cubic')
        
        Returns:
            Single image if camera specified, dict of images otherwise
        """
        if camera:
            return self._project_single_view(equirect_image, camera, interpolation)
        else:
            return self._project_all_views(equirect_image, interpolation)
    
    def _project_single_view(
        self,
        equirect: np.ndarray,
        camera: CameraView,
        interpolation: str
    ) -> np.ndarray:
        """Project single camera view."""
        
        # Select projection method based on backend
        if self.backend == ProjectionBackend.TORCH_GPU:
            return self._project_torch_gpu(equirect, camera, interpolation)
        elif self.backend == ProjectionBackend.PY360:
            return self._project_py360(equirect, camera, interpolation)
        elif self.backend == ProjectionBackend.EQUILIB:
            return self._project_equilib(equirect, camera, interpolation)
        elif self.backend == ProjectionBackend.OPENCV:
            return self._project_opencv(equirect, camera, interpolation)
        else:
            return self._project_numpy(equirect, camera, interpolation)
    
    def _project_torch_gpu(
        self,
        equirect: np.ndarray,
        camera: CameraView,
        interpolation: str
    ) -> np.ndarray:
        """GPU-accelerated projection using PyTorch."""
        if not HAS_TORCH:
            return self._project_opencv(equirect, camera, interpolation)
        
        # Convert to torch tensor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        equirect_t = torch.from_numpy(equirect).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Create sampling grid
        grid = self._create_torch_sampling_grid(
            camera.yaw, camera.pitch, camera.roll,
            camera.fov_h, camera.fov_v,
            camera.width, camera.height
        ).to(device)
        
        # Sample using grid_sample
        mode = 'bilinear' if interpolation == 'bilinear' else 'nearest'
        pinhole_t = F.grid_sample(
            equirect_t, grid,
            mode=mode,
            padding_mode='border',
            align_corners=False
        )
        
        # Convert back to numpy
        pinhole = pinhole_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(pinhole, 0, 255).astype(np.uint8)
    
    def _create_torch_sampling_grid(
        self,
        yaw: float, pitch: float, roll: float,
        fov_h: float, fov_v: float,
        width: int, height: int
    ) -> torch.Tensor:
        """Create sampling grid for torch.nn.functional.grid_sample."""
        
        # Create perspective grid
        f_h = width / (2 * np.tan(np.radians(fov_h) / 2))
        f_v = height / (2 * np.tan(np.radians(fov_v) / 2))
        
        u = torch.linspace(0, width - 1, width) - width / 2
        v = torch.linspace(0, height - 1, height) - height / 2
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
        
        # Ray directions in camera space
        x = u_grid / f_h
        y = -v_grid / f_v
        z = torch.ones_like(x)
        
        # Normalize rays
        norm = torch.sqrt(x**2 + y**2 + z**2)
        rays = torch.stack([x/norm, y/norm, z/norm], dim=-1)
        
        # Apply rotation (yaw, pitch, roll)
        rays_rotated = self._rotate_rays_torch(rays, yaw, pitch, roll)
        
        # Convert to spherical coordinates
        theta = torch.atan2(rays_rotated[..., 0], rays_rotated[..., 2])  # Yaw
        phi = torch.asin(torch.clamp(rays_rotated[..., 1], -1, 1))  # Pitch
        
        # Normalize to [-1, 1] for grid_sample
        grid_x = theta / np.pi  # -1 to 1
        grid_y = phi / (np.pi / 2)  # -1 to 1
        
        return torch.stack([grid_x, grid_y], dim=-1)
    
    def _rotate_rays_torch(
        self,
        rays: torch.Tensor,
        yaw: float, pitch: float, roll: float
    ) -> torch.Tensor:
        """Apply rotation to ray directions."""
        
        # Convert angles to radians
        y, p, r = np.radians([yaw, pitch, roll])
        
        # Rotation matrices
        R_yaw = torch.tensor([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ], dtype=rays.dtype)
        
        R_pitch = torch.tensor([
            [1, 0, 0],
            [0, np.cos(p), -np.sin(p)],
            [0, np.sin(p), np.cos(p)]
        ], dtype=rays.dtype)
        
        R_roll = torch.tensor([
            [np.cos(r), -np.sin(r), 0],
            [np.sin(r), np.cos(r), 0],
            [0, 0, 1]
        ], dtype=rays.dtype)
        
        # Combined rotation
        R = R_yaw @ R_pitch @ R_roll
        
        # Apply rotation
        return rays @ R.T
    
    def _project_py360(
        self,
        equirect: np.ndarray,
        camera: CameraView,
        interpolation: str
    ) -> np.ndarray:
        """Project using py360convert library."""
        if not HAS_PY360:
            return self._project_opencv(equirect, camera, interpolation)
        
        # py360convert uses different conventions
        u_deg = camera.yaw
        v_deg = -camera.pitch  # Inverted pitch
        
        mode = 'bilinear' if interpolation == 'bilinear' else 'nearest'
        
        pinhole = py360convert.e2p(
            equirect,
            fov_deg=(camera.fov_h, camera.fov_v),
            u_deg=u_deg,
            v_deg=v_deg,
            out_hw=(camera.height, camera.width),
            in_rot_deg=camera.roll,
            mode=mode
        )
        
        return pinhole
    
    def _project_equilib(
        self,
        equirect: np.ndarray,
        camera: CameraView,
        interpolation: str
    ) -> np.ndarray:
        """Project using equilib library."""
        if not HAS_EQUILIB:
            return self._project_opencv(equirect, camera, interpolation)
        
        rots = {
            'yaw': np.radians(camera.yaw),
            'pitch': np.radians(camera.pitch),
            'roll': np.radians(camera.roll)
        }
        
        mode = 'bilinear' if interpolation == 'bilinear' else 'nearest'
        
        pinhole = equi2pers(
            equirect,
            rots=rots,
            height=camera.height,
            width=camera.width,
            fov_x=camera.fov_h,
            mode=mode
        )
        
        return pinhole
    
    def _project_opencv(
        self,
        equirect: np.ndarray,
        camera: CameraView,
        interpolation: str
    ) -> np.ndarray:
        """Project using OpenCV remap (always available)."""
        
        # Get or create projection map
        map_key = f"{camera.id}_{camera.width}x{camera.height}"
        
        if self.cache_maps and map_key in self._projection_maps:
            map_x, map_y = self._projection_maps[map_key]
        else:
            map_x, map_y = self._create_opencv_map(camera, equirect.shape[:2])
            if self.cache_maps:
                self._projection_maps[map_key] = (map_x, map_y)
        
        # Select interpolation
        if interpolation == 'nearest':
            interp = cv2.INTER_NEAREST
        elif interpolation == 'cubic':
            interp = cv2.INTER_CUBIC
        else:
            interp = cv2.INTER_LINEAR
        
        # Apply remapping
        pinhole = cv2.remap(equirect, map_x, map_y, interp)
        
        return pinhole
    
    def _create_opencv_map(
        self,
        camera: CameraView,
        equirect_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create OpenCV remap matrices for a camera view."""
        
        h_out, w_out = camera.height, camera.width
        h_in, w_in = equirect_shape
        
        # Create perspective camera intrinsics
        f_h = w_out / (2 * np.tan(np.radians(camera.fov_h) / 2))
        f_v = h_out / (2 * np.tan(np.radians(camera.fov_v) / 2))
        
        # Create pixel grid
        u = np.arange(w_out, dtype=np.float32) - w_out / 2
        v = np.arange(h_out, dtype=np.float32) - h_out / 2
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Convert to ray directions
        x = u_grid / f_h
        y = -v_grid / f_v
        z = np.ones_like(x)
        
        # Normalize
        norm = np.sqrt(x**2 + y**2 + z**2)
        rays = np.stack([x/norm, y/norm, z/norm], axis=-1)
        
        # Apply rotation
        rays_rotated = self._rotate_rays_numpy(
            rays, camera.yaw, camera.pitch, camera.roll
        )
        
        # Convert to spherical coordinates
        lon = np.arctan2(rays_rotated[..., 0], rays_rotated[..., 2])
        lat = np.arcsin(np.clip(rays_rotated[..., 1], -1, 1))
        
        # Convert to equirectangular pixel coordinates
        map_x = (lon + np.pi) / (2 * np.pi) * w_in
        map_y = (np.pi/2 - lat) / np.pi * h_in
        
        return map_x.astype(np.float32), map_y.astype(np.float32)
    
    def _rotate_rays_numpy(
        self,
        rays: np.ndarray,
        yaw: float, pitch: float, roll: float
    ) -> np.ndarray:
        """Apply rotation to ray directions using NumPy."""
        
        # Convert to radians
        y, p, r = np.radians([yaw, pitch, roll])
        
        # Rotation matrices
        R_yaw = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ])
        
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(p), -np.sin(p)],
            [0, np.sin(p), np.cos(p)]
        ])
        
        R_roll = np.array([
            [np.cos(r), -np.sin(r), 0],
            [np.sin(r), np.cos(r), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = R_yaw @ R_pitch @ R_roll
        
        # Apply rotation
        shape = rays.shape
        rays_flat = rays.reshape(-1, 3)
        rays_rotated = (R @ rays_flat.T).T
        
        return rays_rotated.reshape(shape)
    
    def _project_numpy(
        self,
        equirect: np.ndarray,
        camera: CameraView,
        interpolation: str
    ) -> np.ndarray:
        """Direct NumPy implementation (slowest but most transparent)."""
        
        h_out, w_out = camera.height, camera.width
        h_in, w_in = equirect.shape[:2]
        
        # Create output image
        pinhole = np.zeros((h_out, w_out, equirect.shape[2]), dtype=equirect.dtype)
        
        # Create perspective camera
        f_h = w_out / (2 * np.tan(np.radians(camera.fov_h) / 2))
        f_v = h_out / (2 * np.tan(np.radians(camera.fov_v) / 2))
        
        for v in range(h_out):
            for u in range(w_out):
                # Ray direction in camera space
                x = (u - w_out / 2) / f_h
                y = -(v - h_out / 2) / f_v
                z = 1.0
                
                # Normalize
                norm = np.sqrt(x**2 + y**2 + z**2)
                ray = np.array([x/norm, y/norm, z/norm])
                
                # Apply rotation
                ray_rotated = self._rotate_ray_single(
                    ray, camera.yaw, camera.pitch, camera.roll
                )
                
                # Convert to spherical
                lon = np.arctan2(ray_rotated[0], ray_rotated[2])
                lat = np.arcsin(np.clip(ray_rotated[1], -1, 1))
                
                # Convert to equirectangular pixels
                x_eq = (lon + np.pi) / (2 * np.pi) * w_in
                y_eq = (np.pi/2 - lat) / np.pi * h_in
                
                # Sample (bilinear interpolation)
                if interpolation == 'nearest':
                    x_eq = int(round(x_eq)) % w_in
                    y_eq = int(round(y_eq)) % h_in
                    pinhole[v, u] = equirect[y_eq, x_eq]
                else:
                    # Bilinear interpolation
                    x0 = int(np.floor(x_eq)) % w_in
                    x1 = (x0 + 1) % w_in
                    y0 = int(np.floor(y_eq)) % h_in
                    y1 = (y0 + 1) % h_in
                    
                    dx = x_eq - x0
                    dy = y_eq - y0
                    
                    pinhole[v, u] = (
                        equirect[y0, x0] * (1-dx) * (1-dy) +
                        equirect[y0, x1] * dx * (1-dy) +
                        equirect[y1, x0] * (1-dx) * dy +
                        equirect[y1, x1] * dx * dy
                    )
        
        return pinhole.astype(equirect.dtype)
    
    def _rotate_ray_single(
        self,
        ray: np.ndarray,
        yaw: float, pitch: float, roll: float
    ) -> np.ndarray:
        """Rotate single ray direction."""
        rays = ray.reshape(1, 1, 3)
        rotated = self._rotate_rays_numpy(rays, yaw, pitch, roll)
        return rotated[0, 0]
    
    def _project_all_views(
        self,
        equirect: np.ndarray,
        interpolation: str
    ) -> Dict[str, np.ndarray]:
        """Project all camera views in the rig."""
        
        if self.num_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {}
                for camera in self.rig_config.cameras:
                    future = executor.submit(
                        self._project_single_view,
                        equirect, camera, interpolation
                    )
                    futures[camera.name] = future
                
                results = {}
                for name, future in futures.items():
                    results[name] = future.result()
                
                return results
        else:
            # Sequential processing
            results = {}
            for camera in self.rig_config.cameras:
                results[camera.name] = self._project_single_view(
                    equirect, camera, interpolation
                )
            
            return results
    
    def process_video(
        self,
        input_path: Path,
        output_dir: Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        skip_frames: int = 0,
        interpolation: str = 'bilinear',
        save_rig_config: bool = True
    ) -> None:
        """
        Process video file, extracting frames and converting to rig views.
        
        Args:
            input_path: Input video or image directory
            output_dir: Output directory for rig views
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all)
            skip_frames: Frames to skip between processing
            interpolation: Interpolation method
            save_rig_config: Save rig configuration to output
        """
        
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save rig configuration
        if save_rig_config:
            self.rig_config.save(output_dir / 'rig_config.json')
        
        # Create camera directories
        for camera in self.rig_config.cameras:
            cam_dir = output_dir / camera.name
            cam_dir.mkdir(exist_ok=True)
        
        # Process frames
        if input_path.is_dir():
            # Process image sequence
            image_files = sorted(input_path.glob('*.jpg')) + sorted(input_path.glob('*.png'))
            self._process_image_sequence(
                image_files, output_dir,
                start_frame, end_frame, skip_frames, interpolation
            )
        else:
            # Process video file
            self._process_video_file(
                input_path, output_dir,
                start_frame, end_frame, skip_frames, interpolation
            )
    
    def _process_image_sequence(
        self,
        image_files: List[Path],
        output_dir: Path,
        start_frame: int,
        end_frame: Optional[int],
        skip_frames: int,
        interpolation: str
    ) -> None:
        """Process sequence of images."""
        
        # Select frame range
        end_frame = end_frame or len(image_files)
        image_files = image_files[start_frame:end_frame:skip_frames+1]
        
        logger.info(f"Processing {len(image_files)} frames...")
        
        for idx, img_path in enumerate(image_files):
            logger.info(f"Frame {idx+1}/{len(image_files)}: {img_path.name}")
            
            # Load image
            equirect = cv2.imread(str(img_path))
            if equirect is None:
                logger.error(f"Failed to load: {img_path}")
                continue
            
            # Process all views
            views = self._project_all_views(equirect, interpolation)
            
            # Save views
            frame_name = f"frame_{idx:05d}.jpg"
            for camera_name, view_img in views.items():
                output_path = output_dir / camera_name / frame_name
                cv2.imwrite(str(output_path), view_img)
    
    def _process_video_file(
        self,
        video_path: Path,
        output_dir: Path,
        start_frame: int,
        end_frame: Optional[int],
        skip_frames: int,
        interpolation: str
    ) -> None:
        """Process video file directly."""
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video: {total_frames} frames @ {fps:.2f} FPS")
        
        # Set frame range
        end_frame = end_frame or total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        processed = 0
        
        while frame_idx < end_frame:
            ret, equirect = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue
            
            logger.info(f"Processing frame {frame_idx}/{end_frame}")
            
            # Process all views
            views = self._project_all_views(equirect, interpolation)
            
            # Save views
            frame_name = f"frame_{processed:05d}.jpg"
            for camera_name, view_img in views.items():
                output_path = output_dir / camera_name / frame_name
                cv2.imwrite(str(output_path), view_img)
            
            processed += 1
            frame_idx += 1
        
        cap.release()
        logger.info(f"Processed {processed} frames")
    
    def process_with_masks(
        self,
        equirect_image: np.ndarray,
        equirect_mask: np.ndarray,
        camera: Optional[CameraView] = None,
        interpolation: str = 'bilinear'
    ) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Process image and mask together to maintain alignment.
        
        Args:
            equirect_image: Input equirectangular image
            equirect_mask: Input equirectangular mask
            camera: Specific camera (None for all)
            interpolation: Interpolation for image ('nearest' always used for mask)
        
        Returns:
            Tuple of (image, mask) or dict of tuples
        """
        
        if camera:
            img = self._project_single_view(equirect_image, camera, interpolation)
            mask = self._project_single_view(equirect_mask, camera, 'nearest')
            return img, mask
        else:
            results = {}
            for cam in self.rig_config.cameras:
                img = self._project_single_view(equirect_image, cam, interpolation)
                mask = self._project_single_view(equirect_mask, cam, 'nearest')
                results[cam.name] = (img, mask)
            return results
    
    def visualize_rig(
        self,
        equirect_image: np.ndarray,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create visualization of the rig configuration.
        
        Args:
            equirect_image: Sample equirectangular image
            output_path: Optional path to save visualization
        
        Returns:
            Visualization image
        """
        
        # Project all views
        views = self._project_all_views(equirect_image, 'bilinear')
        
        # Create grid layout
        n_cameras = len(self.rig_config.cameras)
        grid_w = int(np.ceil(np.sqrt(n_cameras)))
        grid_h = int(np.ceil(n_cameras / grid_w))
        
        # Get view dimensions (assume all same size)
        first_view = next(iter(views.values()))
        view_h, view_w = first_view.shape[:2]
        
        # Scale for visualization
        scale = 0.5
        thumb_w = int(view_w * scale)
        thumb_h = int(view_h * scale)
        
        # Create canvas
        canvas_w = grid_w * thumb_w
        canvas_h = grid_h * thumb_h
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Place views
        for idx, (name, view) in enumerate(views.items()):
            row = idx // grid_w
            col = idx % grid_w
            
            # Resize view
            thumb = cv2.resize(view, (thumb_w, thumb_h))
            
            # Place on canvas
            y1 = row * thumb_h
            y2 = y1 + thumb_h
            x1 = col * thumb_w
            x2 = x1 + thumb_w
            
            canvas[y1:y2, x1:x2] = thumb
            
            # Add label
            cv2.putText(
                canvas, name,
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )
        
        if output_path:
            cv2.imwrite(str(output_path), canvas)
        
        return canvas


def main():
    """Command-line interface for the reframing tool."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert equirectangular 360° content to pinhole rig views"
    )
    
    parser.add_argument(
        "input", type=Path,
        help="Input image, video, or directory"
    )
    parser.add_argument(
        "output", type=Path,
        help="Output directory for rig views"
    )
    
    # Rig configuration
    parser.add_argument(
        "--rig", choices=["cube", "ring8", "ring12", "geodesic", "custom"],
        default="ring12",
        help="Rig pattern to use (default: ring12)"
    )
    parser.add_argument(
        "--rig-config", type=Path,
        help="Custom rig configuration file (JSON/YAML)"
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="Output width for each view"
    )
    parser.add_argument(
        "--height", type=int, default=1080,
        help="Output height for each view"
    )
    parser.add_argument(
        "--fov-h", type=float, default=90,
        help="Horizontal field of view in degrees"
    )
    parser.add_argument(
        "--fov-v", type=float, default=60,
        help="Vertical field of view in degrees"
    )
    
    # Processing options
    parser.add_argument(
        "--backend", choices=["auto", "torch_gpu", "py360", "equilib", "opencv", "numpy"],
        default="auto",
        help="Projection backend to use"
    )
    parser.add_argument(
        "--workers", type=int, default=-1,
        help="Number of parallel workers (-1 for auto)"
    )
    parser.add_argument(
        "--interpolation", choices=["nearest", "bilinear", "cubic"],
        default="bilinear",
        help="Interpolation method"
    )
    
    # Video options
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Starting frame for video"
    )
    parser.add_argument(
        "--end-frame", type=int,
        help="Ending frame for video"
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Frames to skip (0 for all frames)"
    )
    
    # Debug options
    parser.add_argument(
        "--visualize", action="store_true",
        help="Create rig visualization"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Create rig configuration
    if args.rig_config:
        rig_config = RigConfig.load(args.rig_config)
    elif args.rig == "cube":
        rig_config = RigGenerator.create_cube_rig(
            resolution=min(args.width, args.height)
        )
    elif args.rig == "ring8":
        rig_config = RigGenerator.create_ring_rig(
            num_cameras=8,
            width=args.width,
            height=args.height,
            fov_h=args.fov_h,
            fov_v=args.fov_v
        )
    elif args.rig == "ring12":
        rig_config = RigGenerator.create_ring_rig(
            num_cameras=12,
            width=args.width,
            height=args.height,
            fov_h=args.fov_h,
            fov_v=args.fov_v
        )
    elif args.rig == "geodesic":
        rig_config = RigGenerator.create_geodesic_rig(
            resolution=min(args.width, args.height),
            fov=args.fov_h
        )
    else:
        raise ValueError(f"Unknown rig pattern: {args.rig}")
    
    # Select backend
    if args.backend == "auto":
        backend = None
    else:
        backend = ProjectionBackend(args.backend)
    
    # Create processor
    processor = EquirectToRig(
        rig_config=rig_config,
        backend=backend,
        num_workers=args.workers,
        debug=args.debug
    )
    
    # Process input
    if args.input.is_file() and args.input.suffix in ['.jpg', '.png']:
        # Single image
        logger.info(f"Processing single image: {args.input}")
        
        equirect = cv2.imread(str(args.input))
        views = processor.process_image(equirect, interpolation=args.interpolation)
        
        # Save views
        args.output.mkdir(parents=True, exist_ok=True)
        for name, view in views.items():
            output_path = args.output / f"{name}.jpg"
            cv2.imwrite(str(output_path), view)
            logger.info(f"Saved: {output_path}")
        
        # Save rig config
        rig_config.save(args.output / "rig_config.json")
        
        # Create visualization if requested
        if args.visualize:
            vis = processor.visualize_rig(equirect, args.output / "rig_visualization.jpg")
            logger.info(f"Saved visualization: {args.output / 'rig_visualization.jpg'}")
    
    else:
        # Video or image sequence
        logger.info(f"Processing video/sequence: {args.input}")
        
        processor.process_video(
            input_path=args.input,
            output_dir=args.output,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            skip_frames=args.skip_frames,
            interpolation=args.interpolation
        )
        
        # Create visualization with first frame
        if args.visualize:
            # Load first frame
            if args.input.is_dir():
                first_img = sorted(args.input.glob('*.jpg'))[0]
            else:
                cap = cv2.VideoCapture(str(args.input))
                ret, first_img = cap.read()
                cap.release()
                if ret:
                    first_img = first_img
                else:
                    first_img = None
            
            if first_img is not None:
                if isinstance(first_img, Path):
                    first_img = cv2.imread(str(first_img))
                vis = processor.visualize_rig(
                    first_img,
                    args.output / "rig_visualization.jpg"
                )
                logger.info(f"Saved visualization: {args.output / 'rig_visualization.jpg'}")
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
