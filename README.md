# SplatForge Pipeline

A modular reconstruction pipeline for producing high-quality COLMAP sparse reconstructions from video or images. Designed to feed PostShot, Lichtfeld Studio, and other Gaussian splatting training tools.

**Primary Output:** COLMAP sparse reconstruction (`cameras.bin`, `images.bin`, `points3D.bin`)

---

## Table of Contents

1. [What This Pipeline Does](#what-this-pipeline-does)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Module Reference](#module-reference)
6. [Configuration](#configuration)
7. [Output Formats](#output-formats)
8. [Troubleshooting](#troubleshooting)
9. [Optional: In-Pipeline Training](#optional-in-pipeline-training)

---

## What This Pipeline Does

SplatForge converts raw capture data into optimized COLMAP sparse reconstructions:

```
Input                    Pipeline Stages              Output
─────────────────────   ──────────────────────────   ─────────────────────
Video/Images      →     Extract + Match + SfM   →    COLMAP reconstruction
360° footage      →     Reframe → Extract...    →    COLMAP reconstruction  
Cluttered scenes  →     Mask → Extract...       →    COLMAP reconstruction
```

**Then load into:**
- PostShot
- Lichtfeld Studio
- Nerfstudio
- gsplat
- Any tool that accepts COLMAP format

---

## Installation

### Core Dependencies (Required)

```bash
# Python environment
conda create -n splatforge python=3.10
conda activate splatforge

# Core packages
pip install numpy opencv-python torch torchvision h5py pyyaml tqdm pillow piexif

# Feature extraction (ALIKED + LightGlue)
pip install lightglue

# SfM (GLOMAP + COLMAP)
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization && pip install -e .
# Or alternatively COLMAP: sudo apt install colmap
```

### Optional Dependencies

```bash
# For video extraction
sudo apt install ffmpeg

# For masking (SAM3)
git clone https://github.com/facebookresearch/sam3
cd sam3 && pip install -e .
wget https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt -P checkpoints/

# For 360° reframing
# (Uses existing reframe_v2.py - no additional install)
```

---

## Quick Start

### Scenario 1: Mavic 3 Pro / Air 2S Images

**Example: 6,900 drone images → COLMAP reconstruction**

```bash
# Step 1: Extract features (ALIKED @ 8000 keypoints)
python -m modular_pipeline.extract \
  ./drone_images \
  ./output \
  --extractor aliked \
  --num-keypoints 8000

# Step 2: Match features (LightGlue with NetVLAD retrieval)
python -m modular_pipeline.match \
  ./output/features.h5 \
  ./output \
  --matcher lightglue \
  --retrieval netvlad

# Step 3: Run SfM reconstruction (GLOMAP, 10-50x faster than COLMAP)
python -m modular_pipeline.sfm \
  ./output \
  --backend glomap

# Output: ./output/sparse/0/
#   ├── cameras.bin
#   ├── images.bin
#   └── points3D.bin
```

**Load into PostShot/Lichtfeld:**
- Point to `./output/sparse/0/`
- Images in `./drone_images/`

**Why these settings:**
- ALIKED: Best quality features (ECCV 2024)
- NetVLAD retrieval: Avoids O(n²) matching for 6,900 images
- GLOMAP: 10-50x faster than COLMAP, same quality

### Scenario 2: You Have Video

```bash
# Step 1: Extract frames
python -m modular_pipeline.ingest.extract \
  ./drone_video.mp4 \
  ./output \
  --fps 2

# Then continue with extract → match → sfm as above
```

### Scenario 3: Osmo Action 360 Video

**Equirectangular footage requires reframing to pinhole views before matching.**

```bash
# Step 1: Extract frames (if starting from video)
python -m modular_pipeline.ingest.extract \
  ./osmo_360_video.mp4 \
  ./output \
  --fps 2

# Step 2: Reframe to pinhole virtual rig (CRITICAL for 360°)
python -m modular_pipeline.prepare.reframe \
  ./output/frames \
  ./output \
  --pattern ring12 \
  --fov 90

# Step 3: (Optional) Remove tripod/selfie stick
python -m modular_pipeline.prepare.masking \
  ./output/rig_views \
  ./output \
  --prompts "tripod,selfie stick"

# Step 4-6: Extract, match, SfM on rig_views
python -m modular_pipeline.extract ./output/rig_views ./output
python -m modular_pipeline.match ./output/features.h5 ./output
python -m modular_pipeline.sfm ./output --backend glomap
```

**Why reframing is required:**
- Equirectangular has extreme distortion at poles
- Standard feature matching fails on 360° projection
- Reframing creates 12 pinhole views that overlap 30%
- COLMAP can then reconstruct the virtual rig

---

## Pipeline Architecture

### Data Flow

```
┌─────────────┐
│   INPUT     │  Video, images, 360° footage
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   INGEST    │  Frame extraction, EXIF, camera detection
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   PREPARE   │  Reframe (360°), Masking (optional)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   EXTRACT   │  ALIKED keypoints (8000/image)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    MATCH    │  LightGlue feature matching
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     SfM     │  GLOMAP → COLMAP reconstruction
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   OUTPUT    │  COLMAP sparse/0/ → External training
└─────────────┘
```

### Module Relationship

```
Core Modules (Always run):
  Extract → Match → SfM

Optional Preprocessing:
  Ingest    (if starting from video)
  Reframe   (if 360° footage)
  Masking   (if objects need removal)

Optional Postprocessing:
  Splat     (if training in-pipeline)
  Mesh      (if extracting surfaces)
  Export    (if converting formats)
```

---

## Module Reference

### 1. Ingest Module

**Purpose:** Convert video to frames with metadata extraction

**When to use:**
- ✅ Starting from video files
- ❌ Already have extracted images

**What it does:**
- Extracts frames via ffmpeg or OpenCV
- Reads EXIF data (camera, GPS, focal length)
- Auto-detects camera profiles (DJI, GoPro, Insta360, iPhone)
- Analyzes quality (blur, exposure, motion)
- Generates metadata manifest

**CLI:**
```bash
python -m modular_pipeline.ingest.extract \
  input.mp4 \
  ./output \
  --fps 2 \
  --quality-check
```

**Python API:**
```python
from modular_pipeline.ingest import IngestModule, IngestConfig

config = IngestConfig(fps=2, quality_threshold=0.7)
ingest = IngestModule(config)
result = ingest.process(video_path, output_dir)
```

**Outputs:**
- `frames/`: Extracted images
- `metadata.json`: EXIF, GPS, camera info
- `quality_report.json`: Frame quality scores

**Defaults:**
- FPS: 2 (for video)
- Quality threshold: 0.7 (discard blurry frames)
- Format: JPEG @ 95% quality

---

### 2. Prepare Module

#### 2a. Reframe (360° → Pinhole)

**Purpose:** Convert equirectangular 360° to pinhole rig views

**When to use:**
- ✅ 360° footage (DJI Osmo, Insta360, GoPro MAX)
- ❌ Standard camera footage

**What it does:**
- Generates virtual camera rig (cube, ring8, ring12, geodesic)
- Creates COLMAP rig JSON for bundle adjustment
- Preserves resolution and quality

**CLI:**
```bash
python -m modular_pipeline.prepare.reframe \
  ./frames \
  ./output \
  --pattern ring12 \
  --fov 90
```

**Rig Patterns:**
- `cube`: 6 views (fast, good coverage)
- `ring8`: 8 views around equator
- `ring12`: 12 views (better density)
- `geodesic`: 20+ views (maximum coverage)

**Outputs:**
- `rig_views/`: Pinhole projections
- `rig.json`: COLMAP rig configuration
- Metadata preserved from input

**Defaults:**
- Pattern: `ring12`
- FOV: 90°
- Resolution: Match input

#### 2b. Masking

**Purpose:** Remove unwanted objects (people, vehicles, tripods)

**When to use:**
- ✅ Scenes with moving people
- ✅ Need to remove specific objects
- ❌ Clean outdoor scenes

**What it does:**
- Applies SAM3, FastSAM, or EfficientSAM
- Text prompt-based masking: "person", "car", "tripod"
- Temporal consistency for video
- Geometry-aware (expands masks in equirect poles)

**CLI:**
```bash
python -m modular_pipeline.prepare.masking \
  ./frames \
  ./output \
  --prompts "person,car" \
  --model sam3
```

**Models:**
- `sam3`: Best quality (slow, requires GPU)
- `fastsam`: Fast (YOLOv8-based)
- `efficient_sam`: Balanced

**Outputs:**
- `masked/`: Masked images
- `masks/`: Binary masks (for inspection)

**Defaults:**
- Model: `sam3`
- Temporal consistency: Enabled
- Dilation: 5px

---

### 3. Extract Module

**Purpose:** Extract keypoints and descriptors

**When to use:**
- ✅ Always (required for reconstruction)

**What it does:**
- Extracts ALIKED, SuperPoint, XFeat, or DISK features
- Optional sub-pixel refinement (+7ms/image, ECCV 2024)
- Stores in HDF5 format

**CLI:**
```bash
python -m modular_pipeline.extract \
  ./images \
  ./output \
  --extractor aliked \
  --num-keypoints 8000 \
  --subpixel-refinement
```

**Extractors:**
- `aliked`: Best quality (default)
- `superpoint`: Fast, good baseline
- `xfeat`: Experimental CVPR 24
- `disk`: Good rotation invariance

**Outputs:**
- `features.h5`: Keypoints + descriptors

**Defaults:**
- Extractor: `aliked`
- Keypoints: 8000/image
- Sub-pixel refinement: Disabled (enable for max quality)
- Device: Auto-detect GPU

---

### 4. Match Module

**Purpose:** Match features between image pairs

**When to use:**
- ✅ Always (required for reconstruction)

**What it does:**
- Generates image pairs (retrieval or exhaustive)
- Runs LightGlue matching
- Filters geometric outliers
- Outputs COLMAP matches format

**CLI:**
```bash
python -m modular_pipeline.match \
  ./output/features.h5 \
  ./output \
  --matcher lightglue \
  --retrieval sequential
```

**Matchers:**
- `lightglue`: Best quality (default)
- `superglue`: Good baseline
- `loftr`: Dense matching

**Retrieval Methods:**
- `sequential`: For video (matches temporal neighbors)
- `exhaustive`: For small datasets (<1000 images)
- `netvlad`: For large datasets (requires HLOC)

**Outputs:**
- `matches.h5`: Feature matches
- `pairs.txt`: Image pairs matched

**Defaults:**
- Matcher: `lightglue`
- Retrieval: `sequential` (for >1000 images)
- Min matches: 20

---

### 5. SfM Module

**Purpose:** Structure from Motion reconstruction

**When to use:**
- ✅ Always (produces final COLMAP output)

**What it does:**
- Runs GLOMAP or COLMAP
- Geometric verification
- Bundle adjustment
- Point triangulation
- Outputs COLMAP sparse reconstruction

**CLI:**
```bash
python -m modular_pipeline.sfm \
  ./output \
  --backend glomap \
  --refine-intrinsics
```

**Backends:**
- `glomap`: 10-50x faster (default)
- `colmap`: More robust for difficult scenes
- `instant`: MASt3R/DUSt3R fallback (30x faster, lower quality)

**Outputs:**
- `sparse/0/cameras.bin`: Camera intrinsics
- `sparse/0/images.bin`: Camera poses
- `sparse/0/points3D.bin`: 3D point cloud

**Defaults:**
- Backend: `glomap`
- Refine intrinsics: `True`
- Min triangulation angle: 1.5°
- Max reprojection error: 4.0px

**Quality Metrics:**
- Registration rate: % of images positioned
- Mean reprojection error: <2.0px is good
- Number of 3D points: More is better

---

## Configuration

### Global Defaults

All modules use sensible defaults optimized for high-quality output:

```python
# Extract
ExtractConfig(
    extractor="aliked",
    num_keypoints=8000,
    subpixel_refinement=False,  # Enable for +2% accuracy
    device="auto"
)

# Match
MatchConfig(
    matcher="lightglue",
    retrieval="sequential",
    min_matches=20
)

# SfM
SfMConfig(
    backend="glomap",
    refine_intrinsics=True,
    tri_min_angle=1.5,        # degrees
    tri_max_error=4.0,        # pixels
    min_reg_ratio=0.1         # 10% minimum
)
```

### When to Change Defaults

**Increase keypoints (8000 → 16000):**
- High-resolution images (>4K)
- Complex scenes with fine detail

**Enable sub-pixel refinement:**
- Maximum quality needed
- Acceptable +7ms/image overhead

**Change retrieval method:**
- `sequential`: Video sequences (default for >1000 images)
- `exhaustive`: Small datasets (<1000 images)
- `netvlad`: Large unordered datasets (requires HLOC)

**Use COLMAP instead of GLOMAP:**
- GLOMAP fails (rare)
- Need absolute maximum quality
- Acceptable 10-50x slower runtime

---

## Output Formats

### COLMAP Directory Structure

```
output/
├── images/                    # Input images (or symlink)
├── sparse/
│   └── 0/
│       ├── cameras.bin       # Camera intrinsics
│       ├── images.bin        # Camera poses (position + rotation)
│       └── points3D.bin      # 3D point cloud
├── features.h5               # Keypoints + descriptors
├── matches.h5                # Feature correspondences
└── metadata.json             # EXIF, camera info (if using ingest)
```

### Loading into External Tools

**PostShot:**
1. Open PostShot
2. Import → COLMAP Project
3. Select `output/sparse/0/`
4. Point to images directory
5. Begin training

**Lichtfeld Studio:**
1. File → Import → COLMAP
2. Navigate to `output/sparse/0/`
3. Verify image paths
4. Configure training settings

**Nerfstudio:**
```bash
ns-train splatfacto \
  --data output \
  colmap \
  --colmap-model-path output/sparse/0
```

---

## Troubleshooting

### "GLOMAP failed to register images"

**Symptoms:** Low registration rate (<50%)

**Solutions:**
1. Try COLMAP backend: `--backend colmap`
2. Check image quality: `python -m modular_pipeline.ingest.quality ./images`
3. Increase matches: `--min-matches 15`
4. Enable sub-pixel: `--subpixel-refinement` in extract

### "Too many images for exhaustive matching"

**Symptoms:** >1M pairs warning

**Solution:** Use retrieval
```bash
python -m modular_pipeline.match \
  ./features.h5 \
  ./output \
  --retrieval sequential  # or netvlad
```

### "Point cloud is sparse"

**Symptoms:** <10,000 3D points

**Solutions:**
1. Increase keypoints: `--num-keypoints 16000`
2. Lower match threshold: `--min-matches 15`
3. Adjust triangulation: `--tri-max-error 8.0`

### "High reprojection error"

**Symptoms:** >2.0px mean error

**Solutions:**
1. Enable sub-pixel refinement
2. Use COLMAP instead of GLOMAP
3. Check image quality (blur, motion)

### "360° reconstruction failed"

**Symptoms:** Poor results with equirect

**Solutions:**
1. Use reframe module first: `--pattern ring12`
2. Ensure rig.json is in output
3. Check FOV is correct (usually 90°)

---

## Automation with Presets

For simplified workflows, use the Pipeline orchestrator with camera-specific presets that include optimized settings and camera intrinsics.

### Available Presets

**Specific Presets (Recommended):**
- `mavic_3_pro`: DJI Mavic 3 Pro (Hasselblad L2D-20c, 4/3" sensor, 24mm equiv)
- `mavic_air_2s`: DJI Mavic Air 2S (1" sensor, 22mm equiv)
- `osmo_360`: DJI Osmo Action 360 (dual fisheye, equirectangular output)

**Generic Presets:**
- `drone`: Generic drone footage (when specific model unknown)
- `360`: Generic 360° camera
- `default`: Standard pinhole camera

### Why Use Specific Presets?

Specific presets include camera intrinsics (sensor size, focal length) that:
1. Give SfM a better starting point → faster convergence
2. Reduce calibration errors → higher quality
3. Handle distortion correctly → better matches

**Example: Generic vs Specific**
```bash
# Generic preset (slower, auto-calibrates)
python -m modular_pipeline.pipeline ./project --preset drone

# Specific preset (faster, uses known intrinsics)
python -m modular_pipeline.pipeline ./project --preset mavic_3_pro
```

### Using the Pipeline Orchestrator

**One-command automation:**

```bash
# Mavic 3 Pro: Extract → Match → SfM (fully automated)
python -m modular_pipeline.pipeline ./my_project --preset mavic_3_pro

# Osmo 360: Ingest → Reframe → Mask → Extract → Match → SfM
python -m modular_pipeline.pipeline ./360_project --preset osmo_360
```

**Resume from a specific stage:**
```bash
python -m modular_pipeline.pipeline ./project --resume-from match
```

**Run specific stages only:**
```bash
python -m modular_pipeline.pipeline ./project --stages extract match sfm
```

### Preset Configuration Details

**mavic_3_pro:**
```yaml
extractor: aliked (8000 keypoints)
matcher: lightglue
retrieval: netvlad (50 neighbors)
mapper: glomap
camera_model: OPENCV (handles distortion)
focal_length: 24mm (4/3" sensor)
mesh_backend: gof (outdoor scenes)
```

**mavic_air_2s:**
```yaml
extractor: aliked (8000 keypoints)
matcher: lightglue
retrieval: netvlad (50 neighbors)
mapper: glomap
camera_model: OPENCV
focal_length: 22mm (1" sensor)
mesh_backend: gof (outdoor scenes)
```

**osmo_360:**
```yaml
requires_reframe: true (ring12 pattern, 90° FOV)
requires_masking: true (removes tripod/selfie stick)
extractor: aliked (8000 keypoints)
matcher: lightglue
retrieval: netvlad (50 neighbors)
mapper: glomap
camera_model: PINHOLE (after reframing)
mesh_backend: pgsr (better for 360°)
```

### When to Use Modular vs Automated

**Use Modular Approach When:**
- You need full control over each parameter
- Debugging a specific stage
- Integrating with external tools mid-pipeline
- Testing different configurations

**Use Automated Pipeline When:**
- Standard workflow with known camera
- Quick turnaround needed
- Batch processing multiple captures

---

## Optional: In-Pipeline Training

If you don't have PostShot or Lichtfeld, you can train in-pipeline:

### Splat Module

**Train Gaussian splats from COLMAP:**

```bash
python -m modular_pipeline.output.splat \
  ./output/sparse/0 \
  ./splat_output \
  --iterations 30000 \
  --sh-degree 3
```

**Outputs:** Trained `.ply` splat file

### Mesh Module

**Extract mesh from splat:**

```bash
python -m modular_pipeline.output.mesh \
  ./splat_output \
  ./mesh_output \
  --backend pgsr \
  --resolution 512
```

**Backends:**
- `gof`: Outdoor scenes (~45 min)
- `pgsr`: Best textures (~30 min)
- `2dgs`: Thin surfaces (~30 min)
- `sugar`: Blender/Unity (~20 min)

### Export Module

**Convert to web formats:**

```bash
python -m modular_pipeline.output.export \
  ./mesh_output \
  ./export \
  --formats glb ply
```

**Note:** These modules require additional dependencies:
```bash
pip install gsplat trimesh pygltflib
```

---

## Project Directory Example

```
my_reconstruction/
├── input_video.mp4           # Source
├── output/
│   ├── frames/               # Extracted (if video)
│   ├── rig_views/            # Reframed (if 360°)
│   ├── masked/               # Masked (if needed)
│   ├── features.h5
│   ├── matches.h5
│   ├── sparse/
│   │   └── 0/
│   │       ├── cameras.bin   ← Load this into PostShot
│   │       ├── images.bin
│   │       └── points3D.bin
│   └── metadata.json
└── README.txt                # Your notes
```

---

## Module Selection Guide

**I have video:**
```
Ingest → Extract → Match → SfM
```

**I have images:**
```
Extract → Match → SfM
```

**I have 360° video:**
```
Ingest → Reframe → Extract → Match → SfM
```

**I have images with people:**
```
Masking → Extract → Match → SfM
```

**I have 360° with people:**
```
Ingest → Reframe → Masking → Extract → Match → SfM
```

---

## Performance

**Typical runtimes (6,900 images, RTX 3090 Ti):**

| Stage | Time | Notes |
|-------|------|-------|
| Extract (ALIKED) | ~45 min | 8000 keypoints/image |
| Match (LightGlue) | ~2 hours | Sequential retrieval |
| SfM (GLOMAP) | ~15 min | 10-50x faster than COLMAP |
| **Total** | **~3 hours** | COLMAP ready |

**For comparison:**
- COLMAP (instead of GLOMAP): +4-8 hours
- SuperPoint (instead of ALIKED): -10 min (lower quality)
- Exhaustive (instead of sequential): +8-12 hours for large datasets

---

## Future Enhancements

- [ ] Quality gate system (interactive/review/autonomous)
- [ ] LiDAR integration (iPhone/iPad depth)
- [ ] Rolling shutter compensation
- [ ] Multi-rig support (synchronized cameras)
- [ ] Incremental reconstruction (add images to existing)

---

## References

**Papers Implemented:**
- ALIKED (ECCV 2024) - Feature extraction
- LightGlue (ICCV 2023) - Feature matching
- GLOMAP (2024) - Fast SfM
- SAM3 (2024) - Segmentation

**External Tools:**
- [COLMAP](https://colmap.github.io/)
- [HLOC](https://github.com/cvg/Hierarchical-Localization)
- [PostShot](https://www.postshot.app/)
- [Lichtfeld Studio](https://lichtfeld-studio.com/)

---

**Questions or issues?** Check IMPLEMENTATION_STATUS.md for development status and known issues.
