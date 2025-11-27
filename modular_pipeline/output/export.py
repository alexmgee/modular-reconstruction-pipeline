"""
SplatForge Export: Format Conversion and Viewer Module

Converts outputs to standard formats and provides preview capabilities.

Export formats:
- PLY: Point clouds and Gaussian splats
- OBJ: Meshes with materials
- GLB/GLTF: Web-ready 3D (three.js compatible)
- USDZ: Apple AR Quick Look

Viewer integration:
- Local HTTP server for web viewers
- Gaussian splat viewer
- Standard mesh viewer

Usage:
    # CLI
    python -m modular_pipeline.output.export ./mesh ./output --format glb
    
    # Python API
    from modular_pipeline.output.export import ExportModule, ExportConfig
    
    config = ExportConfig(formats=["glb", "ply"])
    export = ExportModule(config)
    result = export.process(input_path, output_path)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import logging

# Core imports
from modular_pipeline.core import (
    BaseModule, BaseConfig, StageResult, ItemResult,
    QualityLevel, Backend,
    ensure_directory, format_time,
    HAS_TORCH, HAS_NUMPY
)

try:
    import numpy as np
except ImportError:
    np = None


# =============================================================================
# CONFIGURATION
# =============================================================================

class ExportFormat(Enum):
    """Export format options."""
    PLY = "ply"             # Point cloud / Gaussian splat
    OBJ = "obj"             # Mesh with MTL materials
    GLB = "glb"             # Binary GLTF (web-ready)
    GLTF = "gltf"           # Text GLTF (editable)
    USDZ = "usdz"           # Apple AR Quick Look
    FBX = "fbx"             # Autodesk FBX


@dataclass
class ExportConfig(BaseConfig):
    """Configuration for export and format conversion.
    
    Attributes:
        formats: List of export formats to generate
        
        # Compression
        compress: Enable compression for supported formats
        compression_level: Compression quality (0-9)
        
        # LOD generation
        generate_lods: Generate multiple levels of detail
        lod_levels: Number of LOD levels (if enabled)
        lod_ratios: Decimation ratios for each LOD
        
        # Thumbnails
        generate_thumbnail: Create preview image
        thumbnail_size: Thumbnail dimensions (width, height)
        
        # Viewer
        launch_viewer: Launch local viewer after export
        viewer_port: HTTP server port for viewer
    """
    # Export formats
    formats: List[str] = field(default_factory=lambda: ["glb", "ply"])
    
    # Compression
    compress: bool = True
    compression_level: int = 7           # 0-9 (9 = max compression)
    
    # LOD generation
    generate_lods: bool = False
    lod_levels: int = 3
    lod_ratios: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    
    # Thumbnails
    generate_thumbnail: bool = True
    thumbnail_size: Tuple[int, int] = (512, 512)
    
    # Viewer
    launch_viewer: bool = False
    viewer_port: int = 8000
    
    def __post_init__(self):
        super().__post_init__()
        # Convert format strings to enums
        self.formats = [ExportFormat(f) if isinstance(f, str) else f 
                       for f in self.formats]


# =============================================================================
# EXPORT MODULE
# =============================================================================

class ExportModule(BaseModule):
    """Export 3D assets to standard formats with viewer integration.
    
    Handles format conversion, compression, LOD generation, and
    provides local preview capabilities.
    
    Example:
        >>> export = ExportModule(ExportConfig(formats=["glb", "ply"]))
        >>> result = export.process(
        ...     input_path=Path("./mesh/model.obj"),
        ...     output_path=Path("./export")
        ... )
        >>> print(f"Exported: {len(result.metrics['files'])} files")
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        super().__init__(config)
    
    def _default_config(self) -> ExportConfig:
        return ExportConfig()
    
    def _initialize(self) -> None:
        """Initialize export backends."""
        self._converters = {}
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check for available export libraries."""
        # Check for trimesh (mesh conversion)
        try:
            import trimesh
            self._converters['trimesh'] = True
        except ImportError:
            self.log("trimesh not available (pip install trimesh)", "warning")
            self._converters['trimesh'] = False
        
        # Check for pygltflib (GLTF export)
        try:
            import pygltflib
            self._converters['gltf'] = True
        except ImportError:
            self.log("pygltflib not available (pip install pygltflib)", "warning")
            self._converters['gltf'] = False
    
    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------
    
    def process(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> StageResult:
        """Export 3D asset to multiple formats.
        
        Args:
            input_path: Path to input 3D asset
            output_path: Output directory
        
        Returns:
            StageResult with export statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        ensure_directory(output_path)
        
        # Validate input
        if not input_path.exists():
            return self._create_result(
                success=False,
                output_path=output_path,
                errors=[f"Input not found: {input_path}"]
            )
        
        self.log(f"Exporting to {len(self.config.formats)} formats")
        
        start_time = time.perf_counter()
        exported_files = []
        errors = []
        
        # Export to each format
        for fmt in self.config.formats:
            try:
                output_file = self._export_format(input_path, output_path, fmt)
                if output_file:
                    exported_files.append(output_file)
                    self.log(f"  ✓ Exported {fmt.value}: {output_file.name}")
            except Exception as e:
                errors.append(f"Failed to export {fmt.value}: {e}")
                self.log(f"  ✗ Failed {fmt.value}: {e}", "error")
        
        # Generate thumbnail
        if self.config.generate_thumbnail:
            try:
                thumb_path = self._generate_thumbnail(input_path, output_path)
                if thumb_path:
                    exported_files.append(thumb_path)
                    self.log(f"  ✓ Generated thumbnail")
            except Exception as e:
                self.log(f"  ✗ Thumbnail failed: {e}", "warning")
        
        elapsed = time.perf_counter() - start_time
        
        # Quality assessment
        if len(exported_files) == len(self.config.formats) + (1 if self.config.generate_thumbnail else 0):
            quality = QualityLevel.EXCELLENT
            confidence = 0.95
        elif len(exported_files) >= len(self.config.formats):
            quality = QualityLevel.GOOD
            confidence = 0.85
        elif len(exported_files) > 0:
            quality = QualityLevel.REVIEW
            confidence = 0.70
        else:
            quality = QualityLevel.POOR
            confidence = 0.50
        
        self.log(f"\nExport complete in {format_time(elapsed)}")
        self.log(f"  Files exported: {len(exported_files)}")
        
        # Launch viewer if requested
        if self.config.launch_viewer and exported_files:
            self._launch_viewer(output_path)
        
        return self._create_result(
            success=len(exported_files) > 0,
            output_path=output_path,
            processing_time=elapsed,
            quality=quality,
            confidence=confidence,
            metrics={
                'num_files': len(exported_files),
                'files': [str(f) for f in exported_files],
                'formats': [fmt.value for fmt in self.config.formats],
            },
            errors=errors
        )
    
    def _export_format(
        self,
        input_path: Path,
        output_path: Path,
        format: ExportFormat
    ) -> Optional[Path]:
        """Export to a specific format (stub implementation).
        
        Full implementation would use libraries like:
        - trimesh for mesh conversion
        - pygltflib for GLTF/GLB
        - open3d for point clouds
        - Custom exporters for specialized formats
        
        Returns:
            Path to exported file or None if failed
        """
        output_file = output_path / f"model.{format.value}"
        
        self.log(f"Exporting to {format.value} (stub implementation)", "warning")
        
        # Return None to indicate stub
        return None
    
    def _generate_thumbnail(
        self,
        input_path: Path,
        output_path: Path
    ) -> Optional[Path]:
        """Generate preview thumbnail (stub implementation).
        
        Full implementation would:
        1. Load 3D model
        2. Setup camera and lighting
        3. Render to image
        4. Save as PNG/JPEG
        
        Returns:
            Path to thumbnail or None if failed
        """
        thumb_path = output_path / "thumbnail.png"
        
        self.log("Thumbnail generation (stub implementation)", "warning")
        
        return None
    
    def _launch_viewer(self, output_path: Path) -> None:
        """Launch local HTTP viewer (stub implementation).
        
        Full implementation would:
        1. Start HTTP server on configured port
        2. Serve static viewer HTML/JS
        3. Load exported model in viewer
        4. Open browser to viewer URL
        """
        self.log(f"Viewer launch (stub implementation)", "warning")
        self.log(f"Would launch viewer at http://localhost:{self.config.viewer_port}", "info")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for export."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export 3D assets to standard formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input", type=Path, help="Input 3D asset")
    parser.add_argument("output", type=Path, help="Output directory")
    
    # Format selection
    parser.add_argument("--formats", type=str, nargs="+",
                       default=["glb", "ply"],
                       choices=["ply", "obj", "glb", "gltf", "usdz", "fbx"],
                       help="Export formats")
    
    # Compression
    parser.add_argument("--compress", action="store_true", default=True,
                       help="Enable compression")
    parser.add_argument("--compression-level", type=int, default=7,
                       choices=range(10),
                       help="Compression quality (0-9)")
    
    # LOD
    parser.add_argument("--lods", action="store_true",
                       help="Generate LODs")
    parser.add_argument("--lod-levels", type=int, default=3,
                       help="Number of LOD levels")
    
    # Viewer
    parser.add_argument("--viewer", action="store_true",
                       help="Launch viewer after export")
    parser.add_argument("--viewer-port", type=int, default=8000,
                       help="Viewer HTTP port")
    
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    config = ExportConfig(
        formats=args.formats,
        compress=args.compress,
        compression_level=args.compression_level,
        generate_lods=args.lods,
        lod_levels=args.lod_levels,
        launch_viewer=args.viewer,
        viewer_port=args.viewer_port,
        verbose=args.verbose,
    )
    
    export = ExportModule(config)
    result = export.process(args.input, args.output)
    
    if result.success:
        print(f"\n✓ Export complete: {result.output_path}")
        print(f"  Files: {result.metrics['num_files']}")
        print(f"  Formats: {', '.join(result.metrics['formats'])}")
        print(f"  Time: {format_time(result.processing_time_seconds)}")
    else:
        print(f"\n✗ Export failed")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
