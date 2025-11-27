"""
Image Directory Handler
=======================

Import and organize existing image directories with quality filtering
and metadata extraction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
import shutil
from datetime import datetime

import numpy as np
import cv2
from tqdm import tqdm

from modular_pipeline.ingest.quality import QualityAnalyzer
from modular_pipeline.ingest.metadata import MetadataExtractor, ImageMetadata


# Supported image extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.tiff', '.tif',
    '.bmp', '.webp', '.heic', '.heif',
    '.JPG', '.JPEG', '.PNG', '.TIFF', '.TIF',
    '.BMP', '.WEBP', '.HEIC', '.HEIF',
}


class ImageImporter:
    """Import and organize image directories."""
    
    def __init__(
        self,
        quality_threshold: float = 0.5,
        analyze_quality: bool = True,
        extract_metadata: bool = True,
        copy_files: bool = True,
    ):
        """Initialize image importer.
        
        Args:
            quality_threshold: Minimum quality score
            analyze_quality: Perform quality analysis
            extract_metadata: Extract EXIF metadata
            copy_files: Copy files to output (vs symlink)
        """
        self.quality_threshold = quality_threshold
        self.analyze_quality = analyze_quality
        self.extract_metadata = extract_metadata
        self.copy_files = copy_files
    
    def find_images(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: Optional[Set[str]] = None,
    ) -> List[Path]:
        """Find all images in a directory.
        
        Args:
            directory: Directory to search
            recursive: Search subdirectories
            extensions: Set of extensions to include (None = all)
            
        Returns:
            List of image paths (sorted)
        """
        directory = Path(directory)
        
        if extensions is None:
            extensions = IMAGE_EXTENSIONS
        
        image_paths = []
        
        if recursive:
            for ext in extensions:
                image_paths.extend(directory.rglob(f'*{ext}'))
        else:
            for ext in extensions:
                image_paths.extend(directory.glob(f'*{ext}'))
        
        return sorted(image_paths)
    
    def import_images(
        self,
        input_dir: Path,
        output_dir: Path,
        recursive: bool = True,
        sort_by: str = 'name',  # name, date, or exif
        rename_pattern: Optional[str] = None,
        max_images: Optional[int] = None,
    ) -> Tuple[List[Path], Dict]:
        """Import images from directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            recursive: Search subdirectories
            sort_by: Sorting method (name, date, exif)
            rename_pattern: Rename pattern (e.g., 'frame_{:06d}.jpg')
            max_images: Maximum images to import
            
        Returns:
            (imported_paths, statistics)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find images
        image_paths = self.find_images(input_dir, recursive)
        
        if not image_paths:
            raise ValueError(f"No images found in {input_dir}")
        
        # Sort images
        image_paths = self._sort_images(image_paths, sort_by)
        
        # Limit if requested
        if max_images:
            image_paths = image_paths[:max_images]
        
        # Initialize analyzers
        quality_analyzer = QualityAnalyzer() if self.analyze_quality else None
        metadata_extractor = MetadataExtractor() if self.extract_metadata else None
        
        # Statistics
        stats = {
            'total_found': len(image_paths),
            'total_imported': 0,
            'total_rejected': 0,
            'quality_rejected': 0,
            'load_failed': 0,
            'quality_scores': [],
            'metadata_extracted': 0,
        }
        
        imported_paths = []
        metadata_list = []
        
        # Process images
        for idx, img_path in enumerate(tqdm(image_paths, desc="Importing images")):
            try:
                # Load image for quality check
                image = cv2.imread(str(img_path))
                if image is None:
                    stats['load_failed'] += 1
                    stats['total_rejected'] += 1
                    continue
                
                # Quality analysis
                if quality_analyzer:
                    metrics = quality_analyzer.analyze(image)
                    stats['quality_scores'].append(metrics.quality_score)
                    
                    if metrics.quality_score < self.quality_threshold:
                        stats['quality_rejected'] += 1
                        stats['total_rejected'] += 1
                        continue
                
                # Extract metadata
                metadata = None
                if metadata_extractor:
                    try:
                        metadata = metadata_extractor.extract(img_path)
                        metadata_list.append(metadata)
                        stats['metadata_extracted'] += 1
                    except Exception:
                        pass
                
                # Determine output filename
                if rename_pattern:
                    output_name = rename_pattern.format(idx)
                else:
                    output_name = img_path.name
                
                output_path = output_dir / output_name
                
                # Copy or symlink
                if self.copy_files:
                    shutil.copy2(img_path, output_path)
                else:
                    if output_path.exists():
                        output_path.unlink()
                    output_path.symlink_to(img_path.resolve())
                
                imported_paths.append(output_path)
                stats['total_imported'] += 1
                
            except Exception as e:
                stats['total_rejected'] += 1
                continue
        
        # Compute quality statistics
        if stats['quality_scores']:
            stats['quality_mean'] = np.mean(stats['quality_scores'])
            stats['quality_std'] = np.std(stats['quality_scores'])
            stats['quality_min'] = np.min(stats['quality_scores'])
            stats['quality_max'] = np.max(stats['quality_scores'])
        
        # Save metadata if extracted
        if metadata_list:
            self._save_metadata_manifest(output_dir, metadata_list)
        
        return imported_paths, stats
    
    def _sort_images(self, image_paths: List[Path], sort_by: str) -> List[Path]:
        """Sort images by specified method."""
        
        if sort_by == 'name':
            # Already sorted by find_images
            return image_paths
        
        elif sort_by == 'date':
            # Sort by modification time
            return sorted(image_paths, key=lambda p: p.stat().st_mtime)
        
        elif sort_by == 'exif':
            # Sort by EXIF timestamp
            if not self.extract_metadata:
                # Fall back to name sort
                return image_paths
            
            extractor = MetadataExtractor()
            
            # Extract timestamps
            timestamps = []
            for img_path in image_paths:
                try:
                    metadata = extractor.extract(img_path)
                    if metadata.capture_time:
                        timestamps.append((metadata.capture_time, img_path))
                    else:
                        # No timestamp, use modification time
                        mod_time = datetime.fromtimestamp(img_path.stat().st_mtime)
                        timestamps.append((mod_time, img_path))
                except Exception:
                    # Use modification time as fallback
                    mod_time = datetime.fromtimestamp(img_path.stat().st_mtime)
                    timestamps.append((mod_time, img_path))
            
            # Sort by timestamp
            timestamps.sort(key=lambda x: x[0])
            return [path for _, path in timestamps]
        
        else:
            raise ValueError(f"Unknown sort method: {sort_by}")
    
    def _save_metadata_manifest(
        self,
        output_dir: Path,
        metadata_list: List[ImageMetadata]
    ) -> None:
        """Save metadata manifest."""
        import json
        
        manifest = {
            'images': [m.to_dict() for m in metadata_list],
            'count': len(metadata_list),
            'timestamp': datetime.now().isoformat(),
        }
        
        manifest_path = output_dir / 'metadata_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def organize_by_camera(
        self,
        input_dir: Path,
        output_dir: Path,
        create_subdirs: bool = True,
    ) -> Dict[str, List[Path]]:
        """Organize images by camera model.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            create_subdirs: Create subdirectories per camera
            
        Returns:
            Dictionary mapping camera models to image paths
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find images
        image_paths = self.find_images(input_dir, recursive=True)
        
        # Extract metadata
        extractor = MetadataExtractor()
        
        # Group by camera
        camera_groups = {}
        
        for img_path in tqdm(image_paths, desc="Organizing by camera"):
            try:
                metadata = extractor.extract(img_path)
                
                # Create camera key
                if metadata.make and metadata.model:
                    camera_key = f"{metadata.make}_{metadata.model}".replace(' ', '_')
                else:
                    camera_key = "unknown"
                
                # Add to group
                if camera_key not in camera_groups:
                    camera_groups[camera_key] = []
                
                # Copy to output
                if create_subdirs:
                    camera_dir = output_dir / camera_key
                    camera_dir.mkdir(exist_ok=True)
                    output_path = camera_dir / img_path.name
                else:
                    output_path = output_dir / f"{camera_key}_{img_path.name}"
                
                if self.copy_files:
                    shutil.copy2(img_path, output_path)
                else:
                    if output_path.exists():
                        output_path.unlink()
                    output_path.symlink_to(img_path.resolve())
                
                camera_groups[camera_key].append(output_path)
                
            except Exception:
                continue
        
        return camera_groups
    
    def detect_sequences(
        self,
        image_paths: List[Path],
        time_threshold: float = 5.0,  # seconds
        min_sequence_length: int = 3,
    ) -> List[List[Path]]:
        """Detect image sequences based on timestamps.
        
        Args:
            image_paths: List of image paths
            time_threshold: Maximum time gap between images in sequence
            min_sequence_length: Minimum images per sequence
            
        Returns:
            List of sequences (each sequence is a list of paths)
        """
        if not self.extract_metadata:
            return [image_paths]  # Single sequence
        
        extractor = MetadataExtractor()
        
        # Extract timestamps
        timestamped_images = []
        for img_path in image_paths:
            try:
                metadata = extractor.extract(img_path)
                if metadata.capture_time:
                    timestamped_images.append((metadata.capture_time, img_path))
            except Exception:
                continue
        
        if not timestamped_images:
            return [image_paths]
        
        # Sort by timestamp
        timestamped_images.sort(key=lambda x: x[0])
        
        # Detect sequences
        sequences = []
        current_sequence = [timestamped_images[0][1]]
        
        for i in range(1, len(timestamped_images)):
            prev_time = timestamped_images[i-1][0]
            curr_time = timestamped_images[i][0]
            
            time_gap = (curr_time - prev_time).total_seconds()
            
            if time_gap <= time_threshold:
                # Continue current sequence
                current_sequence.append(timestamped_images[i][1])
            else:
                # Start new sequence
                if len(current_sequence) >= min_sequence_length:
                    sequences.append(current_sequence)
                current_sequence = [timestamped_images[i][1]]
        
        # Add final sequence
        if len(current_sequence) >= min_sequence_length:
            sequences.append(current_sequence)
        
        return sequences


__all__ = [
    'ImageImporter',
    'IMAGE_EXTENSIONS',
]
