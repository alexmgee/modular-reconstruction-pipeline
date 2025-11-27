# MRP: Modular Reconstruction Pipeline

A modular 3D reconstruction pipeline that converts raw capture data (video, images, 360° footage) into COLMAP sparse reconstructions for Gaussian splatting training.

**Primary Output:** COLMAP sparse reconstruction (`cameras.bin`, `images.bin`, `points3D.bin`)

---

## Table of Contents

1. [What This Pipeline Does](#1-what-this-pipeline-does)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Module Reference](#3-module-reference)
4. [Installation](#4-installation)
5. [Quick Start](#5-quick-start)
6. [Configuration](#6-configuration)
7. [Output Formats](#7-output-formats)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What This Pipeline Does

### The Problem

Standard Structure-from-Motion (SfM) tools assume:
- Pinhole camera geometry (fails on 360° equirectangular)
- Small-to-medium datasets (<1000 images)
- Clean scenes (no moving objects, tripods, or dynamic content)

**When you have 6,900 drone images, 360° footage, or scenes with people/cars, traditional SfM pipelines fail.**

### The Solution

MRP handles the edge cases that break standard reconstruction:

1. **360° Projection → Pinhole:** Converts equirectangular to virtual camera rigs
2. **Large-Scale Matching:** Uses intelligent retrieval (NetVLAD) to avoid O(n²) complexity
3. **Dynamic Object Removal:** Masks people, vehicles, tripods before feature extraction
4. **Camera Intrinsics:** Leverages known sensor parameters for faster convergence

**Result:** Clean COLMAP output ready for PostShot, Lichtfeld Studio, or gsplat training.

### What You Get

```
Input                Output
─────────────────   ─────────────────────────
6,900 drone pics → COLMAP reconstruction
360° video       → COLMAP reconstruction  
Scenes w/people  → COLMAP reconstruction
```

**Then:** Load into PostShot/Lichtfeld → Train Gaussian splat → Done.

---

## 2. Pipeline Architecture

### Data Flow

```
INPUT (Raw Data)
    ↓
┌─────────────┐
│   INGEST    │  Extract frames + metadata from video
└──────┬──────┘
       ↓
┌─────────────┐
│   PREPARE   │  Fix geometry (360°→pinhole) + Remove objects (masking)
└──────┬──────┘
       ↓
┌─────────────┐
│ RECONSTRUCT │  Extract features → Match → Build 3D structure
│  (3 stages) │  [This is the configurable engine]
└──────┬──────┘
       ↓
OUTPUT (COLMAP sparse/0/)
```

### Module Dependencies

**Core (Always Required):**
- Extract → Match → SfM

**Optional Preprocessing:**
- Ingest (if starting from video)
- Reframe (if 360° footage)
- Masking (if removing objects)

**Optional Postprocessing:**
- Splat/Mesh/Export (if training in-pipeline instead of external tools)

---

## 3. Module Reference

### 3.1 Ingest Module

**Purpose:** Extract frames from video with quality filtering and metadata extraction.

**When to Use:**
- ✅ Starting from video files
- ❌ Already have extracted images

**What It Does:**
- Extracts frames via ffmpeg/OpenCV
- Reads EXIF (camera model, GPS, focal length)
- Auto-detects camera profiles (DJI, GoPro, Insta360)
- Filters low-quality frames (blur, exposure)

**CLI Example:**
```bash
python -m modular_pipeline.ingest.extract \
  input_video.mp4 \
  ./output \
  --fps 2 \
  --quality-check
```

**Outputs:**
- `frames/`: Extracted images (JPEG @ 95%)
- `metadata.json`: EXIF + camera info
- `quality_report.json`: Per-frame quality scores

**Defaults:**
- FPS: 2
- Quality threshold: 0.7 (discard blurry frames)

---

### 3.2 Prepare Module

#### 3.2a. Reframe (360° → Pinhole)

**Purpose:** Convert equirectangular projection to pinhole camera rig.

**When to Use:**
- ✅ 360° footage (DJI Osmo, Insta360, GoPro MAX)
- ❌ Standard camera footage

**Why This Matters:**
Equirectangular has extreme distortion at poles. Standard feature detectors fail. Reframing creates 6-20 virtual pinhole views that COLMAP can reconstruct.

**CLI Example:**
```bash
python -m modular_pipeline.prepare.reframe \
  ./frames \
  ./output \
  --pattern ring12 \
  --fov 90
```

**Rig Patterns:**
- `cube`: 6 views (fast)
- `ring8`: 8 views (balanced)
- `ring12`: 12 views (recommended)
- `geodesic`: 20+ views (maximum coverage)

**Outputs:**
- `rig_views/`: Pinhole projections
- `rig.json`: COLMAP rig configuration

---

#### 3.2b. Masking

**Purpose:** Remove dynamic objects (people, vehicles, tripods) before feature extraction.

**When to Use:**
- ✅ Scenes with moving people/cars
- ✅ Tripod/selfie stick visible in 360° footage
- ❌ Clean outdoor scenes

**CLI Example:**
```bash
python -m modular_pipeline.prepare.masking \
  ./frames \
  ./output \
  --prompts "person,car,tripod" \
  --model sam3
```

**Models:**
- `sam3`: Best quality (slow, GPU required)
- `fastsam`: Fast (YOLOv8-based)  
- `efficient_sam`: Balanced

**Outputs:**
- `masked/`: Masked images
- `masks/`: Binary masks (for inspection)

---

### 3.3 Reconstruct Module (The Engine)

**This is where customization happens.** The Reconstruct stage is three sequential operations:

#### 3.3a. Extract (Feature Detection)

**Purpose:** Detect keypoints and compute descriptors.

**Customization Options:**

| Parameter | Options | When to Use |
|-----------|---------|-------------|
| **Extractor** | `aliked` (default)<br>`superpoint`<br>`xfeat`<br>`disk` | aliked: Best quality (ECCV 2024)<br>superpoint: Fast baseline<br>xfeat: Experimental (CVPR 2024)<br>disk: Rotation-invariant |
| **Keypoints** | 2000-20000 | 8000: Standard<br>16000: High-res images (>4K)<br>4096: Fast processing |
| **Sub-pixel** | `True`/`False` | Enable for +2% accuracy (+7ms/image) |

**CLI Example:**
```bash
python -m modular_pipeline.extract \
  ./images \
  ./output \
  --extractor aliked \
  --num-keypoints 8000 \
  --subpixel-refinement  # Optional: +2% accuracy
```

**Output:**
- `features.h5`: Keypoints + descriptors (HDF5 format)

---

#### 3.3b. Match (Feature Correspondence)

**Purpose:** Find matching keypoints between image pairs.

**Customization Options:**

| Parameter | Options | When to Use |
|-----------|---------|-------------|
| **Matcher** | `lightglue` (default)<br>`superglue`<br>`loftr` | lightglue: Best quality (ICCV 2023)<br>superglue: Good baseline<br>loftr: Dense matching |
| **Retrieval** | `sequential`<br>`exhaustive`<br>`netvlad` | sequential: Video (temporal neighbors)<br>exhaustive: Small sets (<1000 images)<br>netvlad: **Large sets (6,900 images)** |
| **Min Matches** | 15-50 | 20: Standard<br>15: Sparse scenes<br>30: Dense scenes |

**Why Retrieval Matters:**
- Without retrieval: 6,900 images = **23.8 million** pair comparisons (weeks of compute)
- With NetVLAD: 6,900 images × 50 neighbors = **345,000** pairs (hours)

**CLI Example:**
```bash
python -m modular_pipeline.match \
  ./output/features.h5 \
  ./output \
  --matcher lightglue \
  --retrieval netvlad \  # CRITICAL for large datasets
  --min-matches 20
```

**Outputs:**
- `matches.h5`: Feature correspondences
- `pairs.txt`: Image pairs matched

---

#### 3.3c. SfM (Structure from Motion)

**Purpose:** Reconstruct 3D structure and camera poses.

**Customization Options:**

| Parameter | Options | When to Use |
|-----------|---------|-------------|
| **Backend** | `glomap` (default)<br>`colmap`<br>`instant` | glomap: 10-50x faster (global SfM)<br>colmap: More robust (incremental SfM)<br>instant: MASt3R fallback (experimental) |
| **Camera Model** | `SIMPLE_RADIAL`<br>`OPENCV`<br>`PINHOLE` | SIMPLE_RADIAL: Generic cameras<br>OPENCV: DJI Mavic (distortion)<br>PINHOLE: Rectified 360° |
| **Triangulation** | `tri_max_error`<br>`tri_min_angle` | max_error 4.0px: Standard<br>max_error 1.0px: Precision<br>min_angle 1.5°: Conservative |

**Intrinsics (Advanced):**
If you know your camera's sensor size and focal length, specify them for faster convergence:
```python
SfMConfig(
    focal_length_mm=24.0,     # Mavic 3 Pro
    sensor_width_mm=17.3,     # 4/3" sensor
    camera_model='OPENCV'
)
```

**CLI Example:**
```bash
python -m modular_pipeline.sfm \
  ./output \
  --backend glomap \
  --refine-intrinsics \
  --tri-max-error 4.0
```

**Outputs:**
- `sparse/0/cameras.bin`: Camera intrinsics
- `sparse/0/images.bin`: Camera poses
- `sparse/0/points3D.bin`: 3D point cloud

**Quality Metrics:**
- Registration rate: >90% is good
- Mean reprojection error: <2.0px is good
- Point count: More is better (depends on scene)

---

## 4. Installation

### Core Dependencies

```bash
# Python environment
conda create -n mrp python=3.10
conda activate mrp

# Core packages
pip install numpy opencv-python torch torchvision h5py pyyaml tqdm pillow piexif

# Feature extraction/matching
pip install lightglue

# SfM
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization && pip install -e .
```

### Optional Dependencies

```bash
# Video extraction
sudo apt install ffmpeg

# Masking (SAM3)
git clone https://github.com/facebookresearch/sam3
cd sam3 && pip install -e .
wget https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt -P checkpoints/

# 360° reframing
# (Uses existing reframe_v2.py - no additional install)
```

---

## 5. Quick Start

### Scenario 1: Mavic 3 Pro / Air 2S (6,900 Images)

```bash
# Extract features
python -m modular_pipeline.extract \
  ./drone_images \
  ./output \
  --extractor aliked \
  --num-keypoints 8000

# Match (NetVLAD retrieval for large datasets)
python -m modular_pipeline.match \
  ./output/features.h5 \
  ./output \
  --matcher lightglue \
  --retrieval netvlad

# Reconstruct
python -m modular_pipeline.sfm \
  ./output \
  --backend glomap

# Output: ./output/sparse/0/ (load into PostShot)
```

**Why these settings:**
- ALIKED: SOTA feature detector (ECCV 2024)
- NetVLAD: Avoids O(n²) matching (6,900² = 47M pairs → 345K pairs)
- GLOMAP: 10-50x faster than COLMAP

---

### Scenario 2: Osmo Action 360

```bash
# Extract frames
python -m modular_pipeline.ingest.extract \
  ./360_video.mp4 \
  ./output \
  --fps 2

# Reframe to pinhole (REQUIRED for 360°)
python -m modular_pipeline.prepare.reframe \
  ./output/frames \
  ./output \
  --pattern ring12

# (Optional) Remove tripod/selfie stick
python -m modular_pipeline.prepare.masking \
  ./output/rig_views \
  ./output \
  --prompts "tripod,selfie stick"

# Reconstruct
python -m modular_pipeline.extract ./output/rig_views ./output
python -m modular_pipeline.match ./output/features.h5 ./output
python -m modular_pipeline.sfm ./output --backend glomap
```

**Why reframing is required:**
- Equirectangular projection fails standard feature detection
- Reframing creates 12 pinhole views with 30% overlap
- COLMAP reconstructs the virtual rig geometry

---

### Automation (Presets)

For standard workflows, use the preset system:

```bash
# Mavic 3 Pro (includes camera intrinsics)
python -m modular_pipeline.pipeline ./project --preset mavic_3_pro

# Osmo 360 (auto-reframes + masks)
python -m modular_pipeline.pipeline ./project --preset osmo_360
```

**Available Presets:**
- `mavic_3_pro`: DJI Mavic 3 Pro (Hasselblad, 4/3" sensor, 24mm)
- `mavic_air_2s`: DJI Mavic Air 2S (1" sensor, 22mm)
- `osmo_360`: DJI Osmo Action 360 (auto-reframe + mask)
- `drone`: Generic drone (no intrinsics)
- `360`: Generic 360° camera
- `default`: Standard pinhole camera

**Preset Advantage:** Presets include camera intrinsics (sensor size, focal length) that give SfM a better starting point → faster convergence, less calibration error.

---

## 6. Configuration

### Default Settings

```python
# Extract
ExtractConfig(
    extractor="aliked",
    num_keypoints=8000,
    subpixel_refinement=False,
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
    tri_min_angle=1.5,
    tri_max_error=4.0
)
```

### When to Override

**High-resolution images (>4K):**
```bash
--num-keypoints 16000
```

**Maximum quality:**
```bash
--subpixel-refinement  # Extract
--backend colmap       # SfM (10-50x slower)
```

**Large unordered datasets:**
```bash
--retrieval netvlad  # Match
```

**Difficult scenes (low texture):**
```bash
--min-matches 15
--tri-max-error 8.0
```

---

## 7. Output Formats

### COLMAP Directory Structure

```
output/
├── images/              # Input images (or symlink)
├── sparse/
│   └── 0/
│       ├── cameras.bin  # Camera intrinsics
│       ├── images.bin   # Camera poses
│       └── points3D.bin # 3D point cloud
├── features.h5          # Keypoints + descriptors
├── matches.h5           # Feature correspondences
└── metadata.json        # EXIF (if using Ingest)
```

### Loading into External Tools

**PostShot:**
1. Import → COLMAP Project
2. Select `output/sparse/0/`
3. Point to images directory
4. Begin training

**Lichtfeld Studio:**
1. File → Import → COLMAP
2. Navigate to `output/sparse/0/`
3. Configure training

**Nerfstudio:**
```bash
ns-train splatfacto \
  --data output \
  colmap \
  --colmap-model-path output/sparse/0
```

---

## 8. Troubleshooting

### Low Registration Rate (<50%)

**Symptoms:** GLOMAP/COLMAP registers <50% of images

**Solutions:**
1. Try COLMAP: `--backend colmap`
2. Check image quality: `python -m modular_pipeline.ingest.quality ./images`
3. Increase keypoints: `--num-keypoints 16000`
4. Enable sub-pixel: `--subpixel-refinement`

---

### Sparse Point Cloud

**Symptoms:** <10,000 3D points

**Solutions:**
1. Increase keypoints: `--num-keypoints 16000`
2. Lower match threshold: `--min-matches 15`
3. Relax triangulation: `--tri-max-error 8.0`

---

### High Reprojection Error

**Symptoms:** Mean error >2.0px

**Solutions:**
1. Enable sub-pixel refinement (Extract)
2. Use COLMAP instead of GLOMAP
3. Check for motion blur/camera shake

---

### 360° Reconstruction Fails

**Symptoms:** Poor results with equirectangular footage

**Solutions:**
1. **Use reframe module:** `--pattern ring12`
2. Verify `rig.json` exists in output
3. Check FOV is correct (usually 90°)

---

### Too Many Images for Exhaustive Matching

**Symptoms:** >1M pair warning

**Solution:**
```bash
--retrieval netvlad  # Or sequential for video
```

---

## Performance Benchmarks

**Test System:** RTX 3090 Ti, 6,900 drone images

| Stage | Time | Notes |
|-------|------|-------|
| Extract (ALIKED) | ~45 min | 8000 keypoints/image |
| Match (NetVLAD) | ~2 hours | 345K pairs |
| SfM (GLOMAP) | ~15 min | Global reconstruction |
| **Total** | **~3 hours** | Ready for PostShot |

**Comparisons:**
- COLMAP (vs GLOMAP): +4-8 hours
- Exhaustive matching (vs NetVLAD): +8-12 hours
- SuperPoint (vs ALIKED): Faster but lower quality

---

## References

**Papers:**
- ALIKED (ECCV 2024) - Feature extraction
- LightGlue (ICCV 2023) - Feature matching
- GLOMAP (2024) - Global SfM
- SAM3 (2024) - Segmentation

**Tools:**
- [COLMAP](https://colmap.github.io/)
- [HLOC](https://github.com/cvg/Hierarchical-Localization)
- [PostShot](https://www.postshot.app/)
- [Lichtfeld Studio](https://lichtfeld-studio.com/)
