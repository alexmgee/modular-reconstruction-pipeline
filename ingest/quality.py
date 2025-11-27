"""
Image Quality Analysis
======================

Analyzes image quality metrics including blur detection, exposure analysis,
and motion estimation. Used for filtering low-quality frames during ingest.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

import numpy as np
import cv2

try:
    from skimage import exposure as sk_exposure
    from skimage.metrics import structural_similarity
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not found. Some quality metrics unavailable. Install with: pip install scikit-image")


@dataclass
class QualityMetrics:
    """Quality metrics for an image."""
    
    # Blur metrics
    blur_laplacian: float = 0.0      # Laplacian variance (higher = sharper)
    blur_gradient: float = 0.0        # Average gradient magnitude
    blur_fft: float = 0.0             # FFT-based sharpness
    
    # Exposure metrics
    exposure_mean: float = 0.0        # Mean brightness (0-1)
    exposure_std: float = 0.0         # Contrast (std deviation)
    histogram_spread: float = 0.0     # Histogram distribution
    clipped_highlights: float = 0.0   # Fraction of overexposed pixels
    clipped_shadows: float = 0.0      # Fraction of underexposed pixels
    
    # Motion metrics
    motion_score: float = 0.0         # Motion blur indicator
    
    # Color metrics
    saturation: float = 0.0           # Average saturation
    color_cast: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # RGB bias
    
    # Overall quality
    is_blurry: bool = False
    is_overexposed: bool = False
    is_underexposed: bool = False
    is_acceptable: bool = True
    quality_score: float = 1.0        # 0-1 overall quality
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'blur': {
                'laplacian': self.blur_laplacian,
                'gradient': self.blur_gradient,
                'fft': self.blur_fft,
            },
            'exposure': {
                'mean': self.exposure_mean,
                'std': self.exposure_std,
                'histogram_spread': self.histogram_spread,
                'clipped_highlights': self.clipped_highlights,
                'clipped_shadows': self.clipped_shadows,
            },
            'motion': {
                'score': self.motion_score,
            },
            'color': {
                'saturation': self.saturation,
                'cast': self.color_cast,
            },
            'flags': {
                'is_blurry': self.is_blurry,
                'is_overexposed': self.is_overexposed,
                'is_underexposed': self.is_underexposed,
                'is_acceptable': self.is_acceptable,
            },
            'quality_score': self.quality_score,
        }


class QualityAnalyzer:
    """Analyze image quality metrics."""
    
    def __init__(
        self,
        blur_threshold: float = 100.0,
        exposure_range: Tuple[float, float] = (0.1, 0.9),
        clip_threshold: float = 0.02,  # 2% clipping allowed
    ):
        """Initialize quality analyzer.
        
        Args:
            blur_threshold: Laplacian variance threshold (lower = blurrier)
            exposure_range: Acceptable mean brightness range
            clip_threshold: Maximum fraction of clipped pixels
        """
        self.blur_threshold = blur_threshold
        self.exposure_range = exposure_range
        self.clip_threshold = clip_threshold
    
    def analyze(self, image: np.ndarray) -> QualityMetrics:
        """Analyze image quality.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            QualityMetrics object
        """
        metrics = QualityMetrics()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Blur detection
        metrics.blur_laplacian = self._blur_laplacian(gray)
        metrics.blur_gradient = self._blur_gradient(gray)
        metrics.blur_fft = self._blur_fft(gray)
        
        # Exposure analysis
        metrics.exposure_mean = self._exposure_mean(gray)
        metrics.exposure_std = self._exposure_std(gray)
        metrics.histogram_spread = self._histogram_spread(gray)
        metrics.clipped_highlights = self._clipped_highlights(gray)
        metrics.clipped_shadows = self._clipped_shadows(gray)
        
        # Color analysis (if color image)
        if len(image.shape) == 3:
            metrics.saturation = self._saturation(image)
            metrics.color_cast = self._color_cast(image)
        
        # Set flags
        metrics.is_blurry = metrics.blur_laplacian < self.blur_threshold
        metrics.is_overexposed = (
            metrics.exposure_mean > self.exposure_range[1] or
            metrics.clipped_highlights > self.clip_threshold
        )
        metrics.is_underexposed = (
            metrics.exposure_mean < self.exposure_range[0] or
            metrics.clipped_shadows > self.clip_threshold
        )
        
        # Overall acceptability
        metrics.is_acceptable = not (
            metrics.is_blurry or
            metrics.is_overexposed or
            metrics.is_underexposed
        )
        
        # Compute overall quality score
        metrics.quality_score = self._compute_quality_score(metrics)
        
        return metrics
    
    def _blur_laplacian(self, gray: np.ndarray) -> float:
        """Compute blur metric using Laplacian variance.
        
        Higher variance indicates sharper image.
        Typical values: <100 blurry, 100-500 acceptable, >500 sharp
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)
    
    def _blur_gradient(self, gray: np.ndarray) -> float:
        """Compute blur metric using gradient magnitude."""
        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Average magnitude
        return float(magnitude.mean())
    
    def _blur_fft(self, gray: np.ndarray) -> float:
        """Compute blur metric using FFT.
        
        Sharp images have more high-frequency content.
        """
        # Compute FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Get high-frequency energy
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        
        # Create high-pass filter (exclude center frequencies)
        mask = np.ones((h, w), dtype=np.float32)
        r = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask_area = (x - center_w)**2 + (y - center_h)**2 <= r**2
        mask[mask_area] = 0
        
        # High-frequency energy
        high_freq_energy = (magnitude * mask).sum()
        total_energy = magnitude.sum()
        
        if total_energy > 0:
            return float(high_freq_energy / total_energy)
        return 0.0
    
    def _exposure_mean(self, gray: np.ndarray) -> float:
        """Compute mean brightness (0-1)."""
        return float(gray.mean() / 255.0)
    
    def _exposure_std(self, gray: np.ndarray) -> float:
        """Compute exposure standard deviation (contrast)."""
        return float(gray.std() / 255.0)
    
    def _histogram_spread(self, gray: np.ndarray) -> float:
        """Compute histogram spread metric.
        
        Well-exposed images use more of the dynamic range.
        Returns percentage of histogram bins that are used.
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Count non-empty bins
        non_empty = (hist > 0).sum()
        spread = non_empty / 256.0
        
        return float(spread)
    
    def _clipped_highlights(self, gray: np.ndarray) -> float:
        """Compute fraction of overexposed (clipped) pixels."""
        total_pixels = gray.size
        clipped = (gray >= 250).sum()  # Near-white pixels
        return float(clipped / total_pixels)
    
    def _clipped_shadows(self, gray: np.ndarray) -> float:
        """Compute fraction of underexposed (clipped) pixels."""
        total_pixels = gray.size
        clipped = (gray <= 5).sum()  # Near-black pixels
        return float(clipped / total_pixels)
    
    def _saturation(self, image: np.ndarray) -> float:
        """Compute average saturation.
        
        Args:
            image: BGR image
            
        Returns:
            Average saturation (0-1)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        return float(saturation.mean() / 255.0)
    
    def _color_cast(self, image: np.ndarray) -> Tuple[float, float, float]:
        """Detect color cast (RGB bias).
        
        Args:
            image: BGR image
            
        Returns:
            (B, G, R) bias values (0-1)
        """
        # Compute mean for each channel
        b_mean = image[:, :, 0].mean() / 255.0
        g_mean = image[:, :, 1].mean() / 255.0
        r_mean = image[:, :, 2].mean() / 255.0
        
        # Normalize by overall brightness
        total = b_mean + g_mean + r_mean
        if total > 0:
            b_norm = b_mean / total
            g_norm = g_mean / total
            r_norm = r_mean / total
            
            # Compute deviation from neutral gray (0.333 for each channel)
            b_bias = abs(b_norm - 0.333)
            g_bias = abs(g_norm - 0.333)
            r_bias = abs(r_norm - 0.333)
            
            return (b_bias, g_bias, r_bias)
        
        return (0.0, 0.0, 0.0)
    
    def _compute_quality_score(self, metrics: QualityMetrics) -> float:
        """Compute overall quality score (0-1).
        
        Combines multiple metrics into a single score.
        """
        score = 1.0
        
        # Blur penalty
        if metrics.blur_laplacian < self.blur_threshold:
            blur_ratio = metrics.blur_laplacian / self.blur_threshold
            score *= blur_ratio
        
        # Exposure penalty
        if metrics.exposure_mean < self.exposure_range[0]:
            exp_ratio = metrics.exposure_mean / self.exposure_range[0]
            score *= exp_ratio
        elif metrics.exposure_mean > self.exposure_range[1]:
            exp_ratio = (1.0 - metrics.exposure_mean) / (1.0 - self.exposure_range[1])
            score *= exp_ratio
        
        # Clipping penalty
        total_clipping = metrics.clipped_highlights + metrics.clipped_shadows
        if total_clipping > self.clip_threshold:
            clip_ratio = self.clip_threshold / total_clipping
            score *= clip_ratio
        
        # Contrast penalty (too low contrast)
        if metrics.exposure_std < 0.1:
            contrast_ratio = metrics.exposure_std / 0.1
            score *= contrast_ratio
        
        return max(0.0, min(1.0, score))


class MotionDetector:
    """Detect motion between consecutive frames."""
    
    def __init__(self):
        """Initialize motion detector."""
        self.prev_frame = None
    
    def detect(self, frame: np.ndarray) -> float:
        """Detect motion in frame compared to previous frame.
        
        Args:
            frame: Current frame (BGR or grayscale)
            
        Returns:
            Motion score (0-1, higher = more motion)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0
        
        # Compute frame difference
        diff = cv2.absdiff(gray, self.prev_frame)
        
        # Threshold to get motion regions
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Motion score is fraction of pixels in motion
        motion_score = motion_mask.sum() / (255.0 * motion_mask.size)
        
        # Update previous frame
        self.prev_frame = gray
        
        return float(motion_score)
    
    def reset(self):
        """Reset detector (for new sequence)."""
        self.prev_frame = None


def analyze_image_quality(
    image_path: Path,
    blur_threshold: float = 100.0,
    exposure_range: Tuple[float, float] = (0.1, 0.9),
) -> QualityMetrics:
    """Convenience function to analyze image quality.
    
    Args:
        image_path: Path to image
        blur_threshold: Blur detection threshold
        exposure_range: Acceptable exposure range
        
    Returns:
        QualityMetrics object
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    analyzer = QualityAnalyzer(
        blur_threshold=blur_threshold,
        exposure_range=exposure_range
    )
    
    return analyzer.analyze(image)


def filter_quality_images(
    image_paths: list,
    blur_threshold: float = 100.0,
    exposure_range: Tuple[float, float] = (0.1, 0.9),
    min_quality_score: float = 0.5,
) -> Tuple[list, list, Dict]:
    """Filter images by quality.
    
    Args:
        image_paths: List of image paths
        blur_threshold: Blur detection threshold
        exposure_range: Acceptable exposure range
        min_quality_score: Minimum acceptable quality score
        
    Returns:
        (accepted_paths, rejected_paths, quality_stats)
    """
    analyzer = QualityAnalyzer(
        blur_threshold=blur_threshold,
        exposure_range=exposure_range
    )
    
    accepted = []
    rejected = []
    stats = {
        'total': len(image_paths),
        'accepted': 0,
        'rejected': 0,
        'blurry': 0,
        'overexposed': 0,
        'underexposed': 0,
        'quality_scores': [],
    }
    
    for img_path in image_paths:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                rejected.append(img_path)
                stats['rejected'] += 1
                continue
            
            metrics = analyzer.analyze(image)
            stats['quality_scores'].append(metrics.quality_score)
            
            # Check quality
            if metrics.quality_score >= min_quality_score and metrics.is_acceptable:
                accepted.append(img_path)
                stats['accepted'] += 1
            else:
                rejected.append(img_path)
                stats['rejected'] += 1
                
                if metrics.is_blurry:
                    stats['blurry'] += 1
                if metrics.is_overexposed:
                    stats['overexposed'] += 1
                if metrics.is_underexposed:
                    stats['underexposed'] += 1
        
        except Exception as e:
            rejected.append(img_path)
            stats['rejected'] += 1
    
    # Compute statistics
    if stats['quality_scores']:
        stats['quality_mean'] = np.mean(stats['quality_scores'])
        stats['quality_std'] = np.std(stats['quality_scores'])
        stats['quality_min'] = np.min(stats['quality_scores'])
        stats['quality_max'] = np.max(stats['quality_scores'])
    
    return accepted, rejected, stats


__all__ = [
    'QualityMetrics',
    'QualityAnalyzer',
    'MotionDetector',
    'analyze_image_quality',
    'filter_quality_images',
]
