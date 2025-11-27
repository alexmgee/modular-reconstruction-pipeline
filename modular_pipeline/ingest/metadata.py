"""
Metadata Extraction
===================

Extracts and processes EXIF metadata, GPS tracks, and timestamps from images
and videos. Supports multiple extraction backends with fallback.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import warnings
import subprocess
import re

import numpy as np
import cv2

# Try importing EXIF libraries in order of preference
try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False
    warnings.warn("piexif not found. Install with: pip install piexif")

try:
    from PIL import Image, ExifTags
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    warnings.warn("PIL not found. Install with: pip install pillow")

# Check for exiftool (command-line tool)
def check_exiftool() -> bool:
    """Check if exiftool is available."""
    try:
        subprocess.run(['exiftool', '-ver'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

HAS_EXIFTOOL = check_exiftool()


@dataclass
class GPSPoint:
    """GPS coordinate with optional altitude and timestamp."""
    latitude: float         # Decimal degrees
    longitude: float        # Decimal degrees
    altitude: Optional[float] = None  # Meters
    timestamp: Optional[datetime] = None
    accuracy: Optional[float] = None  # Meters
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'accuracy': self.accuracy
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GPSPoint':
        """Create from dictionary."""
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ImageMetadata:
    """Comprehensive image metadata."""
    
    # File information
    filename: str
    filepath: Path
    file_size: int = 0
    
    # Image properties
    width: int = 0
    height: int = 0
    format: str = ""
    
    # Camera information
    make: str = ""
    model: str = ""
    lens_make: str = ""
    lens_model: str = ""
    
    # Capture settings
    iso: Optional[int] = None
    shutter_speed: Optional[float] = None
    aperture: Optional[float] = None
    focal_length: Optional[float] = None  # mm
    focal_length_35mm: Optional[float] = None  # 35mm equivalent
    exposure_bias: Optional[float] = None
    
    # Timestamps
    capture_time: Optional[datetime] = None
    modification_time: Optional[datetime] = None
    
    # GPS
    gps: Optional[GPSPoint] = None
    
    # Additional data
    orientation: int = 1  # EXIF orientation
    flash: bool = False
    white_balance: str = ""
    
    # Raw EXIF
    raw_exif: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = {
            'filename': self.filename,
            'filepath': str(self.filepath),
            'file_size': self.file_size,
            'width': self.width,
            'height': self.height,
            'format': self.format,
            'make': self.make,
            'model': self.model,
            'lens_make': self.lens_make,
            'lens_model': self.lens_model,
            'iso': self.iso,
            'shutter_speed': self.shutter_speed,
            'aperture': self.aperture,
            'focal_length': self.focal_length,
            'focal_length_35mm': self.focal_length_35mm,
            'exposure_bias': self.exposure_bias,
            'capture_time': self.capture_time.isoformat() if self.capture_time else None,
            'modification_time': self.modification_time.isoformat() if self.modification_time else None,
            'gps': self.gps.to_dict() if self.gps else None,
            'orientation': self.orientation,
            'flash': self.flash,
            'white_balance': self.white_balance,
        }
        return d
    
    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class MetadataExtractor:
    """Extract metadata from images using available backends."""
    
    def __init__(self, prefer_exiftool: bool = False):
        """Initialize metadata extractor.
        
        Args:
            prefer_exiftool: Prefer exiftool over Python libraries
        """
        self.prefer_exiftool = prefer_exiftool and HAS_EXIFTOOL
        
        # Determine available backends
        self.backends = []
        if self.prefer_exiftool:
            if HAS_EXIFTOOL:
                self.backends.append('exiftool')
        if HAS_PIEXIF:
            self.backends.append('piexif')
        if HAS_PIL:
            self.backends.append('pil')
        if HAS_EXIFTOOL and not self.prefer_exiftool:
            self.backends.append('exiftool')
        
        if not self.backends:
            warnings.warn("No EXIF extraction backends available")
    
    def extract(self, image_path: Path) -> ImageMetadata:
        """Extract metadata from an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ImageMetadata object
        """
        image_path = Path(image_path)
        
        # Initialize metadata with basic file info
        metadata = ImageMetadata(
            filename=image_path.name,
            filepath=image_path,
            file_size=image_path.stat().st_size
        )
        
        # Get image dimensions
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                metadata.height, metadata.width = img.shape[:2]
                metadata.format = image_path.suffix[1:].upper()
        except Exception:
            pass
        
        # Try backends in order
        for backend in self.backends:
            try:
                if backend == 'exiftool':
                    self._extract_exiftool(image_path, metadata)
                elif backend == 'piexif':
                    self._extract_piexif(image_path, metadata)
                elif backend == 'pil':
                    self._extract_pil(image_path, metadata)
                
                # If we got camera info, consider it successful
                if metadata.make or metadata.model:
                    break
            except Exception as e:
                warnings.warn(f"Backend {backend} failed: {e}")
                continue
        
        return metadata
    
    def _extract_exiftool(self, image_path: Path, metadata: ImageMetadata) -> None:
        """Extract metadata using exiftool command-line tool."""
        try:
            result = subprocess.run(
                ['exiftool', '-j', str(image_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            data = json.loads(result.stdout)[0]
            
            # Camera info
            metadata.make = data.get('Make', '').strip()
            metadata.model = data.get('Model', '').strip()
            metadata.lens_make = data.get('LensMake', '').strip()
            metadata.lens_model = data.get('LensModel', '').strip()
            
            # Capture settings
            metadata.iso = data.get('ISO')
            
            # Shutter speed (handle fractional notation)
            shutter = data.get('ShutterSpeed')
            if shutter:
                metadata.shutter_speed = self._parse_shutter_speed(shutter)
            
            metadata.aperture = data.get('Aperture') or data.get('FNumber')
            metadata.focal_length = data.get('FocalLength')
            metadata.focal_length_35mm = data.get('FocalLengthIn35mmFormat')
            metadata.exposure_bias = data.get('ExposureCompensation')
            
            # Timestamps
            capture_time = data.get('DateTimeOriginal') or data.get('CreateDate')
            if capture_time:
                metadata.capture_time = self._parse_datetime(capture_time)
            
            # GPS
            lat = data.get('GPSLatitude')
            lon = data.get('GPSLongitude')
            if lat is not None and lon is not None:
                metadata.gps = GPSPoint(
                    latitude=lat,
                    longitude=lon,
                    altitude=data.get('GPSAltitude'),
                )
            
            # Additional
            metadata.orientation = data.get('Orientation', 1)
            metadata.flash = bool(data.get('Flash'))
            metadata.white_balance = data.get('WhiteBalance', '')
            
            metadata.raw_exif = data
            
        except Exception as e:
            raise RuntimeError(f"exiftool extraction failed: {e}")
    
    def _extract_piexif(self, image_path: Path, metadata: ImageMetadata) -> None:
        """Extract metadata using piexif library."""
        exif_dict = piexif.load(str(image_path))
        
        # Helper to get tag value
        def get_tag(ifd, tag):
            return exif_dict.get(ifd, {}).get(tag)
        
        # Image info
        ifd0 = exif_dict.get('0th', {})
        metadata.make = ifd0.get(piexif.ImageIFD.Make, b'').decode('utf-8', errors='ignore').strip()
        metadata.model = ifd0.get(piexif.ImageIFD.Model, b'').decode('utf-8', errors='ignore').strip()
        metadata.orientation = ifd0.get(piexif.ImageIFD.Orientation, 1)
        
        # EXIF info
        exif = exif_dict.get('Exif', {})
        
        # ISO
        iso = exif.get(piexif.ExifIFD.ISOSpeedRatings)
        if iso:
            metadata.iso = iso
        
        # Aperture
        fnumber = exif.get(piexif.ExifIFD.FNumber)
        if fnumber:
            metadata.aperture = fnumber[0] / fnumber[1]
        
        # Shutter speed
        exposure_time = exif.get(piexif.ExifIFD.ExposureTime)
        if exposure_time:
            metadata.shutter_speed = exposure_time[0] / exposure_time[1]
        
        # Focal length
        focal = exif.get(piexif.ExifIFD.FocalLength)
        if focal:
            metadata.focal_length = focal[0] / focal[1]
        
        focal_35 = exif.get(piexif.ExifIFD.FocalLengthIn35mmFilm)
        if focal_35:
            metadata.focal_length_35mm = float(focal_35)
        
        # Exposure bias
        exp_bias = exif.get(piexif.ExifIFD.ExposureBiasValue)
        if exp_bias:
            metadata.exposure_bias = exp_bias[0] / exp_bias[1]
        
        # Flash
        flash = exif.get(piexif.ExifIFD.Flash)
        if flash:
            metadata.flash = bool(flash & 0x01)
        
        # Timestamps
        datetime_orig = exif.get(piexif.ExifIFD.DateTimeOriginal)
        if datetime_orig:
            metadata.capture_time = self._parse_datetime(
                datetime_orig.decode('utf-8', errors='ignore')
            )
        
        # GPS
        gps = exif_dict.get('GPS', {})
        if gps:
            lat = self._parse_gps_coord(
                gps.get(piexif.GPSIFD.GPSLatitude),
                gps.get(piexif.GPSIFD.GPSLatitudeRef, b'N').decode()
            )
            lon = self._parse_gps_coord(
                gps.get(piexif.GPSIFD.GPSLongitude),
                gps.get(piexif.GPSIFD.GPSLongitudeRef, b'E').decode()
            )
            
            if lat is not None and lon is not None:
                alt = gps.get(piexif.GPSIFD.GPSAltitude)
                if alt:
                    alt = alt[0] / alt[1]
                
                metadata.gps = GPSPoint(latitude=lat, longitude=lon, altitude=alt)
    
    def _extract_pil(self, image_path: Path, metadata: ImageMetadata) -> None:
        """Extract metadata using PIL/Pillow."""
        img = Image.open(image_path)
        exif_data = img.getexif()
        
        if not exif_data:
            return
        
        # Create reverse mapping for tag names
        tag_names = {v: k for k, v in ExifTags.TAGS.items()}
        
        # Helper to get tag value
        def get_tag(tag_name):
            tag_id = tag_names.get(tag_name)
            return exif_data.get(tag_id) if tag_id else None
        
        # Camera info
        make = get_tag('Make')
        if make:
            metadata.make = make.strip()
        
        model = get_tag('Model')
        if model:
            metadata.model = model.strip()
        
        lens_model = get_tag('LensModel')
        if lens_model:
            metadata.lens_model = lens_model.strip()
        
        # Capture settings
        iso = get_tag('ISOSpeedRatings')
        if iso:
            metadata.iso = iso
        
        fnumber = get_tag('FNumber')
        if fnumber:
            metadata.aperture = float(fnumber)
        
        exposure_time = get_tag('ExposureTime')
        if exposure_time:
            metadata.shutter_speed = float(exposure_time)
        
        focal_length = get_tag('FocalLength')
        if focal_length:
            metadata.focal_length = float(focal_length)
        
        # Timestamps
        datetime_orig = get_tag('DateTimeOriginal')
        if datetime_orig:
            metadata.capture_time = self._parse_datetime(datetime_orig)
        
        # Orientation
        orientation = get_tag('Orientation')
        if orientation:
            metadata.orientation = orientation
        
        # GPS (PIL handles GPS differently)
        gps_info = exif_data.get_ifd(0x8825)  # GPS IFD
        if gps_info:
            lat = self._parse_gps_coord(
                gps_info.get(2),  # GPSLatitude
                gps_info.get(1, 'N')  # GPSLatitudeRef
            )
            lon = self._parse_gps_coord(
                gps_info.get(4),  # GPSLongitude
                gps_info.get(3, 'E')  # GPSLongitudeRef
            )
            
            if lat is not None and lon is not None:
                alt = gps_info.get(6)  # GPSAltitude
                metadata.gps = GPSPoint(latitude=lat, longitude=lon, altitude=alt)
    
    def _parse_gps_coord(self, coord_tuple, ref) -> Optional[float]:
        """Parse GPS coordinate from EXIF format."""
        if not coord_tuple:
            return None
        
        try:
            # coord_tuple is typically ((degrees, 1), (minutes, 1), (seconds, 100))
            if isinstance(coord_tuple, (tuple, list)) and len(coord_tuple) >= 3:
                degrees = coord_tuple[0][0] / coord_tuple[0][1]
                minutes = coord_tuple[1][0] / coord_tuple[1][1]
                seconds = coord_tuple[2][0] / coord_tuple[2][1]
                
                decimal = degrees + minutes / 60 + seconds / 3600
                
                # Apply hemisphere
                if ref in ['S', 'W']:
                    decimal = -decimal
                
                return decimal
        except Exception:
            return None
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse EXIF datetime string."""
        try:
            # EXIF format: "YYYY:MM:DD HH:MM:SS"
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            # Try alternative formats
            try:
                return datetime.fromisoformat(dt_str)
            except ValueError:
                return None
    
    def _parse_shutter_speed(self, value) -> Optional[float]:
        """Parse shutter speed from various formats."""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Handle fractional notation like "1/1000"
            if '/' in value:
                parts = value.split('/')
                return float(parts[0]) / float(parts[1])
            
            try:
                return float(value)
            except ValueError:
                return None
        
        return None


class GPSTrack:
    """GPS track with interpolation capabilities."""
    
    def __init__(self, points: List[GPSPoint]):
        """Initialize GPS track.
        
        Args:
            points: List of GPS points (must have timestamps)
        """
        # Sort by timestamp
        self.points = sorted(
            [p for p in points if p.timestamp is not None],
            key=lambda p: p.timestamp
        )
    
    def interpolate(self, timestamp: datetime) -> Optional[GPSPoint]:
        """Interpolate GPS position at a given timestamp.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            Interpolated GPSPoint or None if out of range
        """
        if not self.points:
            return None
        
        # Check bounds
        if timestamp < self.points[0].timestamp:
            return None
        if timestamp > self.points[-1].timestamp:
            return None
        
        # Find bracketing points
        for i in range(len(self.points) - 1):
            t1 = self.points[i].timestamp
            t2 = self.points[i + 1].timestamp
            
            if t1 <= timestamp <= t2:
                # Linear interpolation
                dt = (t2 - t1).total_seconds()
                if dt == 0:
                    return self.points[i]
                
                alpha = (timestamp - t1).total_seconds() / dt
                
                lat = self.points[i].latitude + alpha * (
                    self.points[i + 1].latitude - self.points[i].latitude
                )
                lon = self.points[i].longitude + alpha * (
                    self.points[i + 1].longitude - self.points[i].longitude
                )
                
                alt = None
                if self.points[i].altitude and self.points[i + 1].altitude:
                    alt = self.points[i].altitude + alpha * (
                        self.points[i + 1].altitude - self.points[i].altitude
                    )
                
                return GPSPoint(
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    timestamp=timestamp
                )
        
        return None
    
    def nearest(self, timestamp: datetime, max_delta_seconds: float = 10.0) -> Optional[GPSPoint]:
        """Find nearest GPS point to timestamp.
        
        Args:
            timestamp: Target timestamp
            max_delta_seconds: Maximum time difference to accept
            
        Returns:
            Nearest GPSPoint or None if too far
        """
        if not self.points:
            return None
        
        min_delta = float('inf')
        nearest_point = None
        
        for point in self.points:
            delta = abs((point.timestamp - timestamp).total_seconds())
            if delta < min_delta:
                min_delta = delta
                nearest_point = point
        
        if min_delta <= max_delta_seconds:
            return nearest_point
        
        return None
    
    def to_json(self, path: Path) -> None:
        """Save GPS track to JSON file."""
        data = {
            'points': [p.to_dict() for p in self.points],
            'count': len(self.points),
            'start': self.points[0].timestamp.isoformat() if self.points else None,
            'end': self.points[-1].timestamp.isoformat() if self.points else None,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> 'GPSTrack':
        """Load GPS track from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        points = [GPSPoint.from_dict(p) for p in data['points']]
        return cls(points)


__all__ = [
    'GPSPoint',
    'ImageMetadata',
    'MetadataExtractor',
    'GPSTrack',
]
