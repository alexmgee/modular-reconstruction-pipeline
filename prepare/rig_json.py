"""
COLMAP Rig JSON Generator
=========================

Generates COLMAP-compatible rig configuration JSON for 360° camera bundle adjustment.
This enables COLMAP to understand that multiple pinhole views belong to the same
rigid camera rig, significantly improving reconstruction accuracy.

References:
- COLMAP rig documentation: https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses
- Rig bundle adjustment: https://colmap.github.io/cameras.html#camera-rigs
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np


class RigJSONGenerator:
    """Generate COLMAP rig JSON from reframe rig configuration."""
    
    @staticmethod
    def from_rig_config(
        rig_config: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Convert reframe rig configuration to COLMAP rig JSON.
        
        Args:
            rig_config: Rig configuration from reframe module
            output_path: Optional path to save JSON
            
        Returns:
            COLMAP rig JSON dictionary
        """
        
        cameras_list = rig_config.get('cameras', [])
        pattern = rig_config.get('pattern', 'unknown')
        
        # Generate COLMAP camera definitions
        colmap_cameras = []
        rig_cameras = []
        
        for cam in cameras_list:
            camera_id = cam['id'] + 1  # COLMAP uses 1-based indexing
            
            # Camera intrinsics (OPENCV model for distortion-free pinhole)
            # params = [fx, fy, cx, cy, k1, k2, p1, p2]
            width = cam['width']
            height = cam['height']
            fov_h = cam['fov_h']
            
            # Calculate focal length from FOV
            # f = width / (2 * tan(fov/2))
            fx = width / (2 * np.tan(np.radians(fov_h) / 2))
            fy = fx  # Assume square pixels
            cx = width / 2
            cy = height / 2
            
            colmap_cameras.append({
                'camera_id': camera_id,
                'model': 'OPENCV',
                'width': width,
                'height': height,
                'params': [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0]  # No distortion
            })
            
            # Rig camera relative pose
            # Convert yaw/pitch/roll to quaternion and translation
            yaw = np.radians(cam['yaw'])
            pitch = np.radians(cam['pitch'])
            roll = np.radians(cam['roll'])
            
            qvec = RigJSONGenerator._euler_to_quaternion(yaw, pitch, roll)
            tvec = [0.0, 0.0, 0.0]  # All cameras at origin (virtual rig)
            
            rig_cameras.append({
                'camera_id': camera_id,
                'image_prefix': cam['name'],
                'rel_tvec': tvec,
                'rel_qvec': qvec.tolist()
            })
        
        # Reference camera (first camera)
        ref_camera_id = 1
        
        # Build COLMAP rig JSON
        colmap_rig = {
            'cameras': colmap_cameras,
            'rigs': [{
                'ref_camera_id': ref_camera_id,
                'cameras': rig_cameras
            }]
        }
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(colmap_rig, f, indent=2)
        
        return colmap_rig
    
    @staticmethod
    def _euler_to_quaternion(
        yaw: float,
        pitch: float,
        roll: float
    ) -> np.ndarray:
        """
        Convert Euler angles (yaw, pitch, roll) to quaternion.
        
        Returns quaternion as [w, x, y, z] (COLMAP format).
        """
        
        # Compute half angles
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        # Compute quaternion components
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def generate_simple_cube_rig(
        width: int = 1024,
        height: int = 1024,
        fov: float = 90.0,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate a simple cubemap rig JSON (6 faces).
        
        Convenience function for testing.
        """
        
        # Define cube faces
        cameras = [
            {'id': 0, 'name': 'cam_00_front', 'yaw': 0, 'pitch': 0, 'roll': 0,
             'fov_h': fov, 'fov_v': fov, 'width': width, 'height': height},
            {'id': 1, 'name': 'cam_01_right', 'yaw': 90, 'pitch': 0, 'roll': 0,
             'fov_h': fov, 'fov_v': fov, 'width': width, 'height': height},
            {'id': 2, 'name': 'cam_02_back', 'yaw': 180, 'pitch': 0, 'roll': 0,
             'fov_h': fov, 'fov_v': fov, 'width': width, 'height': height},
            {'id': 3, 'name': 'cam_03_left', 'yaw': -90, 'pitch': 0, 'roll': 0,
             'fov_h': fov, 'fov_v': fov, 'width': width, 'height': height},
            {'id': 4, 'name': 'cam_04_up', 'yaw': 0, 'pitch': 90, 'roll': 0,
             'fov_h': fov, 'fov_v': fov, 'width': width, 'height': height},
            {'id': 5, 'name': 'cam_05_down', 'yaw': 0, 'pitch': -90, 'roll': 0,
             'fov_h': fov, 'fov_v': fov, 'width': width, 'height': height},
        ]
        
        rig_config = {
            'pattern': 'cube_6',
            'cameras': cameras
        }
        
        return RigJSONGenerator.from_rig_config(rig_config, output_path)


__all__ = [
    'RigJSONGenerator',
]
