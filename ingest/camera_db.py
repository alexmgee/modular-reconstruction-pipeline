"""
Camera Profile Database
=======================

Maintains a database of known camera models with their geometric properties,
sensor specifications, and reconstruction hints. Used for auto-detection and
configuration during the ingest phase.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import re

from modular_pipeline.core import ImageGeometry, CameraModel


@dataclass
class CameraProfile:
    """Profile for a specific camera model."""
    
    # Identification
    make: str                           # Camera manufacturer
    model: str                          # Camera model name
    aliases: List[str] = field(default_factory=list)  # Alternative names
    
    # Geometric properties
    geometry: ImageGeometry = ImageGeometry.PINHOLE
    camera_model: CameraModel = CameraModel.SIMPLE_RADIAL
    
    # Sensor specifications
    sensor_width: Optional[float] = None     # mm
    sensor_height: Optional[float] = None    # mm
    resolution: Optional[Tuple[int, int]] = None  # (width, height)
    
    # Lens properties
    fov_horizontal: Optional[float] = None   # degrees
    fov_vertical: Optional[float] = None     # degrees
    focal_length: Optional[float] = None     # mm
    
    # Reconstruction hints
    requires_reframe: bool = False
    requires_undistort: bool = False
    supports_lidar: bool = False
    typical_masks: List[str] = field(default_factory=list)
    
    # Quality expectations
    typical_blur_threshold: float = 100.0
    typical_exposure_range: Tuple[float, float] = (0.1, 0.9)
    
    # Metadata
    notes: str = ""
    
    def matches(self, make: str, model: str) -> bool:
        """Check if this profile matches the given make/model."""
        # Normalize strings for comparison
        make_norm = make.lower().strip()
        model_norm = model.lower().strip()
        
        # Check exact match
        if self.make.lower() == make_norm and self.model.lower() == model_norm:
            return True
        
        # Check aliases
        for alias in self.aliases:
            if alias.lower() in model_norm or model_norm in alias.lower():
                return True
        
        return False


class CameraDatabase:
    """Database of known camera profiles."""
    
    # Built-in camera profiles
    PROFILES: Dict[str, CameraProfile] = {
        # DJI 360° Cameras
        'dji_osmo_action_5_pro': CameraProfile(
            make='DJI',
            model='Osmo Action 5 Pro',
            aliases=['DFC7115', 'Action 5 Pro'],
            geometry=ImageGeometry.EQUIRECTANGULAR,
            camera_model=CameraModel.PINHOLE,  # After reframing
            resolution=(5312, 2656),  # 360° mode
            fov_horizontal=360.0,
            fov_vertical=180.0,
            requires_reframe=True,
            typical_masks=['tripod', 'selfie stick', 'camera operator'],
            notes='360° action camera with dual fisheye lenses, stitched to equirect'
        ),
        
        'dji_osmo_action_4': CameraProfile(
            make='DJI',
            model='Osmo Action 4',
            aliases=['DFC3112', 'Action 4'],
            geometry=ImageGeometry.EQUIRECTANGULAR,
            camera_model=CameraModel.PINHOLE,
            resolution=(3840, 1920),
            fov_horizontal=360.0,
            fov_vertical=180.0,
            requires_reframe=True,
            typical_masks=['tripod', 'selfie stick'],
            notes='360° mode available with horizontal mount'
        ),
        
        # DJI Drones - Mavic Series
        'dji_mavic_3_pro': CameraProfile(
            make='Hasselblad',
            model='L2D-20c',
            aliases=['Mavic 3 Pro', 'M3P', 'Hasselblad'],
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.SIMPLE_RADIAL,
            sensor_width=17.3,
            sensor_height=13.0,
            resolution=(5280, 3956),
            focal_length=24.0,  # equivalent
            fov_horizontal=84.0,
            notes='Mavic 3 Pro main camera, 4/3 CMOS Hasselblad'
        ),
        
        'dji_mavic_3_classic': CameraProfile(
            make='Hasselblad',
            model='L2D-20c',
            aliases=['Mavic 3 Classic', 'M3C'],
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.SIMPLE_RADIAL,
            sensor_width=17.3,
            sensor_height=13.0,
            resolution=(5280, 3956),
            focal_length=24.0,
            fov_horizontal=84.0,
            notes='Same camera as Mavic 3 Pro'
        ),
        
        'dji_mavic_air_3': CameraProfile(
            make='DJI',
            model='Mavic Air 3',
            aliases=['MA3', 'Air 3'],
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.SIMPLE_RADIAL,
            sensor_width=9.7,
            sensor_height=7.3,
            resolution=(8064, 6048),
            focal_length=24.0,
            fov_horizontal=82.1,
            notes='Wide camera, 48MP 1/1.3" CMOS'
        ),
        
        'dji_mini_4_pro': CameraProfile(
            make='DJI',
            model='Mini 4 Pro',
            aliases=['M4P', 'Mini4Pro'],
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.SIMPLE_RADIAL,
            sensor_width=9.7,
            sensor_height=7.3,
            resolution=(8064, 6048),
            focal_length=24.0,
            fov_horizontal=82.1,
            notes='Same sensor as Air 3'
        ),
        
        # Insta360 Cameras
        'insta360_x4': CameraProfile(
            make='Insta360',
            model='X4',
            aliases=['Insta360 X4'],
            geometry=ImageGeometry.EQUIRECTANGULAR,
            camera_model=CameraModel.PINHOLE,
            resolution=(8192, 4096),
            fov_horizontal=360.0,
            fov_vertical=180.0,
            requires_reframe=True,
            typical_masks=['selfie stick', 'tripod'],
            notes='8K 360° camera with invisible selfie stick'
        ),
        
        'insta360_x3': CameraProfile(
            make='Insta360',
            model='X3',
            aliases=['Insta360 X3'],
            geometry=ImageGeometry.EQUIRECTANGULAR,
            camera_model=CameraModel.PINHOLE,
            resolution=(6080, 3040),
            fov_horizontal=360.0,
            fov_vertical=180.0,
            requires_reframe=True,
            typical_masks=['selfie stick', 'tripod'],
            notes='5.7K 360° camera'
        ),
        
        'insta360_one_x2': CameraProfile(
            make='Insta360',
            model='ONE X2',
            aliases=['Insta360 ONE X2', 'One X2'],
            geometry=ImageGeometry.EQUIRECTANGULAR,
            camera_model=CameraModel.PINHOLE,
            resolution=(6080, 3040),
            fov_horizontal=360.0,
            fov_vertical=180.0,
            requires_reframe=True,
            typical_masks=['selfie stick'],
            notes='Popular 360° action camera'
        ),
        
        # GoPro Cameras
        'gopro_max': CameraProfile(
            make='GoPro',
            model='MAX',
            aliases=['GoPro MAX', 'HERO MAX'],
            geometry=ImageGeometry.EQUIRECTANGULAR,
            camera_model=CameraModel.PINHOLE,
            resolution=(5760, 2880),
            fov_horizontal=360.0,
            fov_vertical=180.0,
            requires_reframe=True,
            typical_masks=['selfie stick', 'mount'],
            notes='GoPro 360° camera'
        ),
        
        'gopro_hero12': CameraProfile(
            make='GoPro',
            model='HERO12 Black',
            aliases=['Hero 12', 'HERO12'],
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.OPENCV_FISHEYE,  # Wide mode
            resolution=(5312, 2988),
            fov_horizontal=156.0,
            requires_undistort=True,
            notes='Action camera with HyperView (ultra-wide fisheye)'
        ),
        
        'gopro_hero11': CameraProfile(
            make='GoPro',
            model='HERO11 Black',
            aliases=['Hero 11', 'HERO11'],
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.OPENCV_FISHEYE,
            resolution=(5312, 2988),
            fov_horizontal=156.0,
            requires_undistort=True,
            notes='Action camera with HyperView'
        ),
        
        # Apple iPhone (with LiDAR)
        'iphone_16_pro_max': CameraProfile(
            make='Apple',
            model='iPhone 16 Pro Max',
            aliases=['iPhone16,2', '16 Pro Max'],
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.SIMPLE_RADIAL,
            sensor_width=9.8,
            sensor_height=7.3,
            resolution=(8064, 6048),  # 48MP mode
            focal_length=24.0,
            fov_horizontal=82.0,
            supports_lidar=True,
            notes='Main camera, 48MP with LiDAR support'
        ),
        
        'iphone_16_pro': CameraProfile(
            make='Apple',
            model='iPhone 16 Pro',
            aliases=['iPhone16,1', '16 Pro'],
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.SIMPLE_RADIAL,
            sensor_width=9.8,
            sensor_height=7.3,
            resolution=(8064, 6048),
            focal_length=24.0,
            fov_horizontal=82.0,
            supports_lidar=True,
            notes='Same camera as 16 Pro Max'
        ),
        
        'iphone_15_pro_max': CameraProfile(
            make='Apple',
            model='iPhone 15 Pro Max',
            aliases=['iPhone15,3', '15 Pro Max'],
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.SIMPLE_RADIAL,
            sensor_width=9.8,
            sensor_height=7.3,
            resolution=(8064, 6048),
            focal_length=24.0,
            supports_lidar=True,
            notes='48MP main camera with LiDAR'
        ),
        
        'iphone_15_pro': CameraProfile(
            make='Apple',
            model='iPhone 15 Pro',
            aliases=['iPhone15,2', '15 Pro'],
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.SIMPLE_RADIAL,
            sensor_width=9.8,
            sensor_height=7.3,
            resolution=(8064, 6048),
            focal_length=24.0,
            supports_lidar=True,
            notes='48MP main camera with LiDAR'
        ),
        
        # Generic fallback profiles
        'generic_360': CameraProfile(
            make='Generic',
            model='360° Camera',
            geometry=ImageGeometry.EQUIRECTANGULAR,
            camera_model=CameraModel.PINHOLE,
            requires_reframe=True,
            typical_masks=['tripod', 'selfie stick', 'operator'],
            notes='Fallback for unrecognized 360° cameras'
        ),
        
        'generic_fisheye': CameraProfile(
            make='Generic',
            model='Fisheye Camera',
            geometry=ImageGeometry.FISHEYE,
            camera_model=CameraModel.OPENCV_FISHEYE,
            requires_undistort=True,
            notes='Fallback for fisheye lenses'
        ),
        
        'generic_pinhole': CameraProfile(
            make='Generic',
            model='Standard Camera',
            geometry=ImageGeometry.PINHOLE,
            camera_model=CameraModel.SIMPLE_RADIAL,
            notes='Fallback for standard cameras'
        ),
    }
    
    def __init__(self, custom_db_path: Optional[Path] = None):
        """Initialize camera database.
        
        Args:
            custom_db_path: Optional path to custom camera database JSON
        """
        self.profiles = self.PROFILES.copy()
        
        # Load custom profiles if provided
        if custom_db_path and custom_db_path.exists():
            self._load_custom_profiles(custom_db_path)
    
    def _load_custom_profiles(self, path: Path) -> None:
        """Load custom camera profiles from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        for key, profile_dict in data.items():
            # Convert string enums back to Enum types
            if 'geometry' in profile_dict:
                profile_dict['geometry'] = ImageGeometry(profile_dict['geometry'])
            if 'camera_model' in profile_dict:
                profile_dict['camera_model'] = CameraModel(profile_dict['camera_model'])
            
            self.profiles[key] = CameraProfile(**profile_dict)
    
    def lookup(self, make: str, model: str) -> Optional[CameraProfile]:
        """Look up camera profile by make and model.
        
        Args:
            make: Camera manufacturer
            model: Camera model
            
        Returns:
            CameraProfile if found, None otherwise
        """
        # Try exact match first
        for profile in self.profiles.values():
            if profile.matches(make, model):
                return profile
        
        return None
    
    def auto_detect_geometry(
        self,
        make: str,
        model: str,
        resolution: Tuple[int, int]
    ) -> ImageGeometry:
        """Auto-detect image geometry from camera info.
        
        Args:
            make: Camera manufacturer
            model: Camera model
            resolution: Image resolution (width, height)
            
        Returns:
            Detected ImageGeometry
        """
        # First try database lookup
        profile = self.lookup(make, model)
        if profile:
            return profile.geometry
        
        # Heuristic detection based on resolution
        width, height = resolution
        aspect_ratio = width / height
        
        # 2:1 aspect ratio typically indicates equirectangular
        if 1.9 < aspect_ratio < 2.1:
            return ImageGeometry.EQUIRECTANGULAR
        
        # Very wide FOV might be fisheye
        if aspect_ratio > 1.5:
            # Could be ultra-wide or fisheye
            return ImageGeometry.PINHOLE
        
        # Default to pinhole for standard cameras
        return ImageGeometry.PINHOLE
    
    def get_reconstruction_hints(
        self,
        make: str,
        model: str
    ) -> Dict[str, any]:
        """Get reconstruction hints for a camera.
        
        Args:
            make: Camera manufacturer
            model: Camera model
            
        Returns:
            Dictionary of reconstruction hints
        """
        profile = self.lookup(make, model)
        
        if not profile:
            return {
                'requires_reframe': False,
                'requires_undistort': False,
                'typical_masks': [],
                'supports_lidar': False,
                'camera_model': CameraModel.SIMPLE_RADIAL,
            }
        
        return {
            'requires_reframe': profile.requires_reframe,
            'requires_undistort': profile.requires_undistort,
            'typical_masks': profile.typical_masks,
            'supports_lidar': profile.supports_lidar,
            'camera_model': profile.camera_model,
            'geometry': profile.geometry,
        }
    
    def suggest_mask_prompts(self, make: str, model: str) -> List[str]:
        """Suggest mask removal prompts for a camera.
        
        Args:
            make: Camera manufacturer
            model: Camera model
            
        Returns:
            List of suggested mask prompts
        """
        profile = self.lookup(make, model)
        
        if profile and profile.typical_masks:
            return profile.typical_masks.copy()
        
        # Default prompts
        return ['tripod', 'camera operator', 'equipment']
    
    def list_cameras(
        self,
        geometry: Optional[ImageGeometry] = None,
        supports_lidar: Optional[bool] = None
    ) -> List[CameraProfile]:
        """List cameras matching criteria.
        
        Args:
            geometry: Filter by geometry type
            supports_lidar: Filter by LiDAR support
            
        Returns:
            List of matching camera profiles
        """
        results = []
        
        for profile in self.profiles.values():
            # Skip generic profiles
            if profile.make == 'Generic':
                continue
            
            # Apply filters
            if geometry and profile.geometry != geometry:
                continue
            if supports_lidar is not None and profile.supports_lidar != supports_lidar:
                continue
            
            results.append(profile)
        
        return results


# Global instance
camera_db = CameraDatabase()


# Utility functions
def detect_camera_from_exif(exif_data: Dict[str, any]) -> Optional[CameraProfile]:
    """Detect camera profile from EXIF data.
    
    Args:
        exif_data: EXIF dictionary
        
    Returns:
        CameraProfile if detected, None otherwise
    """
    make = exif_data.get('Make', '').strip()
    model = exif_data.get('Model', '').strip()
    
    if not make or not model:
        return None
    
    return camera_db.lookup(make, model)


def get_camera_hints(make: str, model: str) -> Dict[str, any]:
    """Convenience function to get reconstruction hints.
    
    Args:
        make: Camera manufacturer
        model: Camera model
        
    Returns:
        Dictionary of hints
    """
    return camera_db.get_reconstruction_hints(make, model)


__all__ = [
    'CameraProfile',
    'CameraDatabase',
    'camera_db',
    'detect_camera_from_exif',
    'get_camera_hints',
]
