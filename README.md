# MRP: Modular Reconstruction Pipeline

**A production-grade SfM pipeline for experts who need control.**

MRP is a modular photogrammetry framework designed to handle diverse capture scenarios—from drone surveys to 360° video to challenging urban scenes with dynamic objects—while maintaining full transparency and control over every stage of reconstruction.

**Output:** COLMAP sparse reconstructions (`cameras.bin`, `images.bin`, `points3D.bin`) compatible with NeRF/Gaussian splatting training frameworks.

---

## Table of Contents

1. [Why Modular?](#1-why-modular)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Module Reference](#3-module-reference)
4. [Parameter Tuning Guide](#4-parameter-tuning-guide)
5. [Installation](#5-installation)
6. [Usage](#6-usage)
7. [Quality Assessment](#7-quality-assessment)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Why Modular?

Standard SfM pipelines (COLMAP, Metashape) make assumptions that fail in specific scenarios:

| Assumption | Breaks When | MRP Solution |
|-----------|-------------|--------------|
| Pinhole geometry | 360° equirectangular footage | `reframe` module: project to pinhole rig |
| Static scene | Moving people, vehicles, tripods | `masking` module: remove with SAM3 + temporal consistency |
| Small-to-medium datasets | >5000 images | Staged matching: NetVLAD retrieval + fallback matchers |
| Single extractor works | ALIKED/SuperPoint tradeoffs | Swap extractors: ALIKED (quality) vs XFeat (speed) vs DISK (density) |
| Incremental SfM works everywhere | Global scenes | Choose backends: GLOMAP (global) vs COLMAP (incremental) |

**The modular approach:** Each stage is independent, with multiple backends and transparent configurations. Debug or replace any stage without rerunning others.

**When you'd use this instead of COLMAP directly:**
- You have 360° footage (manual rig projection is tedious)
- You have >5000 images (exhaustive matching is infeasible)
- You have masking requirements (preprocessing with traditional CV is outdated)
- You want to experiment with feature extractors (swap ALIKED ↔ XFeat without reruns)
- You want full audit trail (manifests at each stage with quality metrics)

---

## 2. Architecture & Data Flow

### 2.1 Pipeline Stages

```
[INPUT: Video/Images/360° Footage]
    ↓
┌─────────────────────────────────────────────────┐
│ INGEST (Conditional)                            │
│ Input: Video files, image folders, OSV format   │
│ - Source type detection                         │
│ - Frame extraction + EXIF parsing               │
│ - Quality filtering (blur, exposure)            │
│ Output: frames/ + metadata.json                 │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ PREPARE (Conditional: Geometry Fix + Masking)   │
│                                                  │
│ ┌──────────────────────────────────────────┐    │
│ │ REFRAME (if 360° equirectangular)        │    │
│ │ - Project to pinhole rig (cube/ring/geo) │    │
│ │ - Generate COLMAP rig.json               │    │
│ │ Output: reframed/ + rig.json             │    │
│ └──────────────────────────────────────────┘    │
│            ↓                                    │
│ ┌──────────────────────────────────────────┐    │
│ │ MASKING (if needed)                      │    │
│ │ - SAM3 segmentation with text prompts    │    │
│ │ - Temporal consistency (for video)       │    │
│ │ - Geometry-aware pole expansion          │    │
│ │ Output: masked/ + quality report         │    │
│ └──────────────────────────────────────────┘    │
│            ↓                                    │
│ Final: prepared/ (pinhole, clean, ready)        │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ RECONSTRUCT (3 Sequential Stages)               │
│                                                  │
│ ┌──────────────────────────────────────────┐    │
│ │ 1. EXTRACT: Feature Keypoints            │    │
│ │ Input: prepared/ images                  │    │
│ │ Output: features.h5                      │    │
│ │ (keypoints, descriptors, confidence)     │    │
│ └──────────────────────────────────────────┘    │
│            ↓                                    │
│ ┌──────────────────────────────────────────┐    │
│ │ 2. RETRIEVE: Pair Selection (Parallel)   │    │
│ │ Input: prepared/ images                  │    │
│ │ Output: pairs.txt                        │    │
│ │ (image pair candidates for matching)     │    │
│ └──────────────────────────────────────────┘    │
│            ↓                                    │
│ ┌──────────────────────────────────────────┐    │
│ │ 3a. MATCH: Feature Correspondence        │    │
│ │ Input: features.h5 + pairs.txt           │    │
│ │ Stage 1: LightGlue (fast, all pairs)     │    │
│ │ Stage 2: MASt3R (fallback, weak pairs)   │    │
│ │ Output: matches.h5                       │    │
│ └──────────────────────────────────────────┘    │
│            ↓                                    │
│ ┌──────────────────────────────────────────┐    │
│ │ 3b. SFM: Structure from Motion           │    │
│ │ Input: images + features.h5 + matches.h5 │    │
│ │ Backend: GLOMAP (global) or COLMAP       │    │
│ │ Output: sparse/0/                        │    │
│ │ (cameras.bin, images.bin, points3D.bin)  │    │
│ └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ OUTPUT (Conditional: In-Pipeline Training)      │
│ - Gaussian splatting (gsplat/Nerfstudio)        │
│ - Mesh extraction (GOF/PGSR/SuGaR)              │
│ - Format export (PLY/OBJ/GLB/USDZ)              │
└─────────────────────────────────────────────────┘
```

### 2.2 Key Design Decisions

**Staged Retrieval (Not All-Pairs Matching)**
- Sequential: O(k×n) pairs for video (temporal neighbors)
- NetVLAD: O(k×n) pairs for general scenes (nearest neighbors from global descriptors)
- Exhaustive: O(n²) pairs for small datasets (<1000 images)
- Hybrid: Sequential + NetVLAD for video with geometric gaps
- **Why:** 6900 images × exhaustive = 23.8M pairs. With NetVLAD (k=50): 345K pairs = 60x speedup.

**Two-Phase Matching (LightGlue + MASt3R Fallback)**
- LightGlue: ~35-40ms/pair, handles 95% of correspondences
- MASt3R: ~300ms/pair (dense 3D-aware), retries pairs with <15 matches
- **Why:** Dense matching on all pairs is prohibitively slow. Selective fallback captures difficult correspondences without destroying runtime.

**Quality Gates & Manifests**
- JSON manifest at each stage with configuration, metrics, quality flags
- Enables resume from any failed stage
- Provides audit trail for reconstruction quality debugging

**Camera Model Auto-Detection**
- SIMPLE_RADIAL: Generic cameras (default)
- OPENCV: Distorted sensors (DJI Mavic, phone cameras)
- PINHOLE: Reframed 360° (no distortion after projection)
- Custom intrinsics from preset/EXIF

### 2.3 File Structure

```
project_root/
├── config.yaml                    # Pipeline configuration
├── state.json                     # Resume state for interrupted runs
│
├── source/                        # INPUT
│   ├── input_video.mp4           # or image folder
│   └── ...
│
├── frames/                        # INGEST OUTPUT
│   ├── frame_000001.jpg
│   └── manifest.json
│
├── reframed/                      # REFRAME OUTPUT (360° only)
│   ├── rig_views/
│   │   ├── cam_00_000001.jpg     # Rig view 1, frame 1
│   │   ├── cam_01_000001.jpg     # Rig view 2, frame 1
│   │   └── ...
│   ├── rig.json                  # COLMAP rig definition
│   └── manifest.json
│
├── masked/                        # MASKING OUTPUT (if used)
│   ├── masks/                    # Binary masks for inspection
│   ├── masked_images/            # Images with masks applied
│   └── manifest.json
│
├── prepared/                      # RECONSTRUCT INPUT
│   ├── symlink to frames/reframed/masked
│   └── manifest.json
│
├── features/                      # EXTRACT OUTPUT
│   ├── feats-aliked-n8000.h5     # Keypoints + descriptors
│   └── manifest.json
│
├── pairs/                         # RETRIEVE OUTPUT
│   ├── pairs.txt                 # Space-separated image pairs
│   └── manifest.json
│
├── matches/                       # MATCH OUTPUT
│   ├── matches.h5                # Feature correspondences
│   └── manifest.json
│
├── sparse/                        # SFM OUTPUT
│   ├── 0/
│   │   ├── cameras.bin           # Camera parameters
│   │   ├── images.bin            # Poses + visibility
│   │   ├── points3D.bin          # 3D points + descriptors
│   │   └── project.ini
│   └── manifest.json
│
└── output/                        # OPTIONAL: In-pipeline training
    ├── splat/trained.ply
    ├── mesh/model.obj
    └── export/model.glb
```

---

## 3. Module Reference

### 3.1 INGEST Module

**Purpose:** Detect source type and extract frames with quality filtering.

**When to Use:**
- Starting from video (MP4, MOV, MKV)
- Starting from OSV container (DJI drones)
- Need auto-detection of camera profiles

**Not needed if:** Frames already extracted and organized in folders.

#### Configuration

```python
# Source handling
source_type: str = "auto"              # auto, video, images, osv
skip_frames: int = 0                   # Skip first N frames
max_frames: int = None                 # Limit total frames
target_fps: float = None               # Resample video to target FPS

# Quality filtering
blur_threshold: float = 100.0          # Laplacian-based blur detection
exposure_range: Tuple = (0.1, 0.9)     # Acceptable histogram bounds
min_quality_score: float = 0.5         # 0-1, discard if below
filter_quality: bool = True            # Enable quality filtering

# Output
output_format: str = "jpg"
output_quality: int = 95               # JPEG quality
rename_pattern: str = "frame_{:06d}.jpg"
```

#### Backends

| Backend | Use When |
|---------|----------|
| `ffmpeg` (recommended) | Large video files, variable frame rates |
| `OpenCV` | Compatibility, smaller files |
| `OSV Extractor` | DJI Mavic/Mini drones (native `.osv`) |

#### Quality Filtering Explained

**Blur Detection:** Laplacian variance over image. High variance = sharp. Threshold of 100 is conservative.
- `blur_threshold` too high → keeps blurry frames → noisy features in reconstruct
- `blur_threshold` too low → discards good frames → fewer matches

**Exposure Filtering:** Histogram-based. Detects over/under-exposed frames.
- `exposure_range = (0.1, 0.9)` means pixels must occupy bins 10%-90% of histogram
- Adjusts for HDR video or scenes with heavy shadows

**Quality Score:** Combined metric (blur + exposure + contrast).
- `min_quality_score = 0.5` keeps ~70% of frames (typical)
- `= 0.7` keeps ~50% (aggressive, for marginal footage)
- `= 0.3` keeps ~90% (permissive, quality-agnostic)

#### Example: Extract 360° video at specific FPS

```bash
python -m modular_pipeline.ingest.extract \
  input_osmo.mov \
  ./output \
  --source-type osv \
  --target-fps 2 \
  --filter-quality \
  --min-quality-score 0.6
```

---

### 3.2 REFRAME Module (360° → Pinhole)

**Purpose:** Project equirectangular imagery to pinhole camera rig.

**When to Use:**
- Insta360 One X2/X3/RS, DJI Osmo Action 4, GoPro MAX (native equirectangular output)
- Need COLMAP-compatible pinhole geometry

**Critical:** Standard SfM assumes pinhole cameras. Equirectangular distortion (especially at poles) breaks feature matching. Reframing solves this.

#### Why This Matters

Equirectangular projection has extreme distortion:
- Poles: 1° of latitude = tiny pixel area (features compressed)
- Equator: 1° of latitude = normal pixel area
- Feature detectors assume roughly uniform scale across image

Reframing to pinhole rig (6-20 views) creates:
- Uniform per-view distortion (handles with camera model)
- Temporal consistency (each 3D point appears in multiple pinhole views)
- COLMAP rig support (matches across views, solves joint bundle adjustment)

#### Configuration

```python
class RigPattern(Enum):
    CUBE_6 = "cube_6"              # 6 views, minimal coverage
    RING_8 = "ring_8"              # 8 views, balanced
    RING_12 = "ring_12"            # 12 views (RECOMMENDED)
    GEODESIC_20 = "geodesic_20"    # 20 views, full coverage
    ADAPTIVE = "adaptive"           # Content-aware rig

# Configuration
pattern: str = "ring_12"
fov_h: float = 90                   # Horizontal FOV (degrees)
fov_v: float = 60                   # Vertical FOV (degrees)
overlap_target: float = 0.30        # Desired overlap between views
output_resolution: Tuple = (1920, 1080)
```

#### Rig Pattern Tradeoffs

| Pattern | Views | Overlap | Use When |
|---------|-------|---------|----------|
| `cube_6` | 6 | ~20% | Texture-rich scenes, speed priority |
| `ring_8` | 8 | ~25% | Balanced quality/speed (most common) |
| `ring_12` | 12 | ~35% | **Recommended**: feature-rich scenes |
| `geodesic_20` | 20 | ~50% | Difficult lighting, sparse features |

**Tradeoffs:**
- More views → better coverage → more 3D points → slower matching (quadratic in pair count)
- Fewer views → faster matching → potential registration gaps

**Recommendation:** Start with `ring_12`. Increase to `geodesic_20` if registration < 80%.

#### Projection Backends

| Backend | Speed | Accuracy | Use When |
|---------|-------|----------|----------|
| PyTorch GPU | 10-50 ms/image | Best | GPU available |
| NumPy (py360) | 100-500 ms | Good | CPU-only, Python-native |
| equilib | 200-1000 ms | Good | CUDA unavailable |
| OpenCV remap | 500-2000 ms | Good | Compatibility |
| NumPy direct | >2000 ms | Best (transparent) | Debugging |

#### Example: Reframe 360° video with adaptive rig

```bash
python -m modular_pipeline.prepare.reframe \
  ./frames \
  ./output \
  --pattern ring_12 \
  --fov-h 90 \
  --fov-v 60 \
  --output-resolution 1920 1080
```

Output produces `rig.json`:
```json
{
  "model_type": "OPENCV",
  "cameras": [
    {
      "camera_id": 0,
      "model": "OPENCV",
      "width": 1920,
      "height": 1080,
      "params": [f_x, f_y, c_x, c_y, k1, k2, p1, p2, k3, k4, k5, k6]
    },
    ...
  ],
  "rig_cameras": [
    {
      "camera_id": 0,
      "image_prefix": "cam_00_",
      "t_i": [0, 0, 0],
      "q_i": [1, 0, 0, 0]
    },
    ...
  ]
}
```

COLMAP loads this and optimizes jointly across rig members.

---

### 3.3 MASKING Module (Object Removal)

**Purpose:** Remove dynamic objects (people, vehicles, tripods) using SAM3 segmentation.

**When to Use:**
- 360° footage with visible selfie stick/tripod
- Urban scenes with people/vehicles
- Need clean scene for feature matching

**Not needed if:** Scene is static and background-only.

#### Why This Matters

Masking improves reconstruction by:
- **Removing false features** on moving objects (unstable across frames)
- **Stabilizing matcher** (fewer outliers from dynamic content)
- **Cleaner SfM** (correct epipolar geometry, fewer ambiguous 3D points)

Skipping masking when needed → high reprojection error, poor registration.

#### Configuration

```python
class SegmentationModel(Enum):
    SAM3 = "sam3"                   # Text prompts, RECOMMENDED
    FASTSAM = "fastsam"             # YOLO-based, faster
    EFFICIENTSAM = "efficient"      # TensorRT optimized
    MOBILESAM = "mobile"            # Mobile GPU

# Configuration
model: str = "sam3"
remove_prompts: List[str] = [
    "person",
    "tripod",
    "camera operator",
    "selfie stick",
    "equipment"
]
keep_prompts: List[str] = []  # Keep specific regions
confidence_threshold: float = 0.70
review_threshold: float = 0.85       # Flag for manual review
min_mask_area: int = 100
max_mask_area_ratio: float = 0.5     # Don't remove >50% of image
use_temporal_consistency: bool = True
temporal_window: int = 5              # Frames for consistency
geometry_aware: bool = True           # Expand masks for pole regions
```

#### Segmentation Model Tradeoffs

| Model | Speed | Quality | Memory | Use When |
|-------|-------|---------|--------|----------|
| SAM3 | 300-500 ms | Excellent | 3GB+ | Text prompts needed, GPU available |
| FastSAM | 50-100 ms | Good | 1GB | Speed priority, YOLOv8 trained |
| EfficientSAM | 100-200 ms | Good | 500MB | Balanced, TensorRT available |
| MobileSAM | 50-100 ms | Good | 200MB | Mobile/embedded GPU |

#### Advanced: Temporal Consistency

When `use_temporal_consistency=True`, masks are smoothed across video frames:
- Frames t-2 to t+2 vote on mask at frame t
- Prevents flickering masks (features appearing/disappearing)
- **Cost:** +temporal_window × processing time
- **Benefit:** Stable scene boundaries

#### Advanced: Geometry-Aware Expansion

For 360° footage with pole regions:
- Pole masks are expanded by factor (default 1.2)
- Prevents self-stick artifacts from bleeding into edge frames
- Geometry-aware = uses image curvature to avoid over-expansion

#### Example: Remove people from 360° footage with temporal smoothing

```bash
python -m modular_pipeline.prepare.masking \
  ./frames \
  ./output \
  --model sam3 \
  --remove-prompts "person" "tripod" "selfie stick" \
  --confidence-threshold 0.7 \
  --use-temporal-consistency \
  --temporal-window 5 \
  --geometry-aware
```

Output:
```
output/
├── masked_images/              # Inpainted images
├── masks/                       # Binary masks (for inspection)
│   ├── frame_000001_mask.png
│   └── ...
└── manifest.json
```

---

### 3.4 EXTRACT Module (Feature Detection)

**Purpose:** Detect keypoints and compute descriptors for matching.

**When to Use:** Every reconstruction needs features. Primary variable: which extractor?

#### Configuration

```python
class ExtractorBackend(Enum):
    ALIKED = "aliked"               # RECOMMENDED (ECCV 2024)
    XFEAT = "xfeat"                 # Fast real-time
    SUPERPOINT = "superpoint"       # Legacy (still good)
    DISK = "disk"                   # Dense, rotation-invariant

# Configuration
extractor: str = "aliked"
max_keypoints: int = 8000           # Per image
nms_radius: int = 3                 # Non-maximum suppression
keypoint_threshold: float = 0.005   # Detection confidence
resize_max: int = 1600              # Max dimension (downscale for speed)
subpixel_refinement: bool = False   # +2% accuracy, +7ms
```

#### Extractor Comparison & Tradeoffs

| Extractor | Speed | Quality | Feature Type | When |
|-----------|-------|---------|--------------|------|
| **ALIKED** | 150-300ms | Excellent | Affine-invariant | **Default: best overall** |
| **XFeat** | 50-100ms | Good | Fast inference | Need real-time processing |
| **SuperPoint** | 200-400ms | Good | CNN (MagicLeap) | Legacy compatibility |
| **DISK** | 100-200ms | Good | Dense, rotation-invariant | Homographic scenes |

**Feature Type Implications:**

- **Affine-invariant** (ALIKED): Handles scale, rotation, perspective. Best for general SfM.
- **Rotation-invariant** (DISK): Good for tilted cameras, struggles with severe perspective.
- **CNN-based** (SuperPoint): Learned from multi-scale data. Generalizes well.

#### Parameter Tuning: max_keypoints

**Effect across pipeline:**

```
max_keypoints: 4000
├─ Extract: Fast (short descriptors)
├─ Match: O(n²) pairs per image pair (16M operations)
├─ SfM: Fewer 3D points, faster triangulation
└─ Output: Sparser 3D point cloud

max_keypoints: 8000 (DEFAULT)
├─ Extract: 1.6x slower
├─ Match: 4x more operations per pair
├─ SfM: Dense correspondences, better registration
└─ Output: Rich 3D cloud

max_keypoints: 16000 (HIGH-RES)
├─ Extract: 2.8x slower than default
├─ Match: 8x more operations per pair (potential bottleneck)
├─ SfM: Maximum detail, risk of overfitting
└─ Output: Very dense cloud (diminishing returns)
```

**Decision Tree:**
- `max_keypoints = 4000`: Low-texture scenes, speed critical, sparse features acceptable
- `max_keypoints = 8000`: **Recommended default**
- `max_keypoints = 16000`: High-res images (>4K), texture-rich scenes, small dataset (<1000 images)

**Practical:** Increasing from 8K to 16K adds ~1 hour per 6900-image dataset, but only if bottleneck is at MATCH stage (depends on retrieval strategy).

#### Parameter Tuning: nms_radius

Non-maximum suppression. Spreads keypoint locations.

- `nms_radius = 1`: Tight clustering, many features per region
- `nms_radius = 3`: Default, balanced
- `nms_radius = 5`: Sparse, high spatial diversity

**When to change:** If features are clustered in texture-heavy regions (e.g., facade), increase to `nms_radius = 5`.

#### Parameter Tuning: subpixel_refinement

Refine keypoint locations to sub-pixel accuracy (ECCV 2024 enhancement).

```
subpixel_refinement = False (DEFAULT)
├─ Extract: 150ms per image
├─ Reprojection error: ~1.5px
└─ Good enough for most scenes

subpixel_refinement = True
├─ Extract: +7ms per image (~5% overhead)
├─ Reprojection error: ~0.8px (-50%)
└─ Visible improvement only for high-precision scenes (close-range)
```

**When to enable:**
- Close-range captures (architectural detail, small objects)
- High-resolution output required
- Bundle adjustment convergence is poor

---

### 3.5 RETRIEVE Module (Pair Generation)

**Purpose:** Select image pairs for matching without exhaustive comparison.

**When to Use:** Always runs in parallel with EXTRACT. Outcome: `pairs.txt`.

#### Configuration

```python
class RetrievalBackend(Enum):
    NETVLAD = "netvlad"             # RECOMMENDED (global descriptors)
    SEQUENTIAL = "sequential"        # Video frame neighbors
    VOCAB_TREE = "vocab_tree"       # COLMAP vocabulary
    EXHAUSTIVE = "exhaustive"        # All pairs
    HYBRID = "hybrid"                # Sequential + NetVLAD

# Configuration
retrieval: str = "netvlad"
num_neighbors: int = 50              # K for k-NN
```

#### Retrieval Strategy Comparison

| Strategy | Complexity | Use When |
|----------|-----------|----------|
| **Sequential** | O(k·n) | Video frames with temporal ordering |
| **NetVLAD** | O(k·n) | Unordered images, large datasets |
| **Vocabulary Tree** | O(k·n) | Revisit detection, loop closures |
| **Hybrid** | O(2k·n) | Video with large temporal gaps |
| **Exhaustive** | O(n²) | <1000 images only |

**Pair Count Analysis:**

```
6,900 images:

Exhaustive: n*(n-1)/2 = 23,790,450 pairs ← INFEASIBLE
Sequential (k=20): 138,000 pairs (temporal neighbors)
NetVLAD (k=50): 345,000 pairs ← RECOMMENDED
Hybrid (k=20+50): 483,000 pairs (better coverage, still feasible)

Time estimate (LightGlue @ 40ms/pair):
Exhaustive: 264 hours
Sequential: 1.5 hours
NetVLAD: 3.8 hours ← PRACTICAL
Hybrid: 5.3 hours
```

#### NetVLAD Details

Uses ResNet50 global descriptors (trained on landmarks):
- Each image → single 4096D vector
- Compute k-nearest neighbors in descriptor space
- Generate image pairs from k-NN results

**Why it works:** Images with similar global appearance likely have features that match.

**When it fails:** Scenes with large baseline changes (drone ascending), extreme lighting variation. Solution: increase `num_neighbors` to 100+.

#### Hybrid Strategy

For video with gaps (drone moving between locations):
1. Sequential k-NN: temporal neighbors (handles continuous motion)
2. NetVLAD k-NN: global neighbors (handles large temporal gaps, loop closures)
3. Union: all pairs from both strategies

**Cost:** 2× retrieval time, but captures both temporal continuity and spatial revisits.

#### Example: Process 10,000 image dataset with NetVLAD + fallback

```bash
python -m modular_pipeline.match \
  ./output/features.h5 \
  ./output \
  --retrieval netvlad \
  --num-neighbors 75 \
  --matcher lightglue \
  --fallback-matcher mast3r \
  --fallback-threshold 15
```

---

### 3.6 MATCH Module (Feature Correspondence)

**Purpose:** Find matching keypoints between image pairs.

**Key decision:** Two-stage matching (fast + fallback).

#### Configuration

```python
class MatcherBackend(Enum):
    LIGHTGLUE = "lightglue"          # RECOMMENDED (sparse, fast)
    MAST3R = "mast3r"                # Dense 3D-aware matching
    ROMA = "roma"                    # Dense optical flow
    SUPERGLUE = "superglue"          # Legacy

# Configuration
matcher: str = "lightglue"
fallback_matcher: str = "mast3r"
min_matches: int = 15
fallback_threshold: int = 50         # Retry if < this many matches
max_pairs_fallback: int = 500        # Max pairs to retry
depth_confidence: float = -1         # -1 = quality, 0.9 = speed
width_confidence: float = -1         # -1 = quality, 0.95 = speed
```

#### Matcher Comparison

| Matcher | Speed | Density | Mode | When |
|---------|-------|---------|------|------|
| **LightGlue** | 35-40 ms | Sparse | ICCV 2023, transformer | **Default: best balance** |
| **MASt3R** | 300-500 ms | Dense | 3D-aware, dense pred | Fallback for weak matches |
| **RoMa** | 200-300 ms | Very Dense | Optical flow | Dense scenes |
| **SuperGlue** | 100-150 ms | Sparse | CVPR 2020 | Legacy, slower than LightGlue |

#### Two-Stage Matching Strategy

**Stage 1: LightGlue (All Pairs)**
```
For each image pair in pairs.txt:
├─ Load ALIKED descriptors
├─ Run LightGlue correspondence
├─ Record matches + confidence
└─ Elapsed: ~40ms per pair
```

**Stage 2: MASt3R Fallback (Selective Retry)**
```
For pairs with < 15 matches in Stage 1:
├─ Run MASt3R (dense matching)
├─ Dense depth prediction + feature tracking
├─ May find 50-300+ matches (depending on baseline)
└─ Elapsed: ~400ms per pair (but only ~5% of pairs)
```

**Why two stages:**
- LightGlue is 10x faster than dense matching
- LightGlue handles 95% of pairs well
- Dense matching (MASt3R) needed only for pairs with wide baseline or sparse texture
- Overall: 3.8 hours (NetVLAD 345K pairs) vs. 50+ hours (dense only)

#### Parameter Tuning: min_matches

Minimum correspondences to accept a pair.

```
min_matches = 10
├─ Permissive: includes weak pairs
├─ SfM: more pairs to triangulate from
├─ Risk: outliers from weak matches
└─ Reprojection error may increase

min_matches = 15 (DEFAULT)
├─ Balanced: filters obvious false pairs
├─ SfM: robust triangulation
└─ Good for most scenes

min_matches = 30
├─ Aggressive: only strong correspondences
├─ SfM: fewer but higher-quality 3D points
└─ Risk: potential registration gaps if too strict
```

**Decision:** Start with default (15). Lower to 10 if registration < 50%. Increase to 30 only if SfM fails with too many outliers.

#### Parameter Tuning: fallback_threshold & max_pairs_fallback

```
fallback_threshold = 50
├─ LightGlue outputs < 50 matches?
├─ Retry with MASt3R fallback
└─ Cost: ~400ms per pair

max_pairs_fallback = 500
├─ Only retry up to 500 pairs
├─ Prevents excessive fallback runtime
└─ If >500 pairs fail, continues with LightGlue-only
```

**Tuning:**
- `fallback_threshold = 20`: Aggressive fallback, more runtime but richer matching
- `fallback_threshold = 50`: Balanced (default)
- `fallback_threshold = 100`: Permissive LightGlue, rarely triggers fallback

---

### 3.7 SFM Module (Structure from Motion)

**Purpose:** Reconstruct 3D structure and camera poses.

**Primary variable:** Backend choice (speed vs. robustness).

#### Configuration

```python
class MapperBackend(Enum):
    GLOMAP = "glomap"               # RECOMMENDED (global SfM)
    COLMAP = "colmap"               # Incremental SfM

class CameraModel(Enum):
    SIMPLE_RADIAL = "SIMPLE_RADIAL" # DEFAULT (generic)
    OPENCV = "OPENCV"               # Distortion model (DJI)
    PINHOLE = "PINHOLE"             # No distortion (reframed)

# Configuration
mapper: str = "glomap"
camera_model: str = "SIMPLE_RADIAL"
max_num_tracks: int = 6_000_000
tri_min_angle: float = 1.5           # Triangulation angle (degrees)
tri_max_error: float = 4.0           # Reprojection error (pixels)
tri_max_dist: float = 100.0          # Max distance (meters)
refine_intrinsics: bool = True       # Optimize focal length, principal point
```

#### Mapper Backend Comparison

| Backend | Speed | Robustness | Best For |
|---------|-------|-----------|----------|
| **GLOMAP** | 10-50x faster | 8% higher recall | Global scenes, large datasets |
| **COLMAP** | Baseline (1x) | Robust, well-tested | Incremental scenes, small datasets |

**GLOMAP (Global SfM):**
1. Global rotations → camera directions
2. Global translations → camera positions
3. Bundle adjustment (optional)
4. Point triangulation

**Advantages:**
- 10-50x faster (avoids incremental growing)
- Better for unordered images (doesn't need initialization)
- Higher recall (8% better registration)

**Disadvantages:**
- Requires good initial rotation estimates (from pairs)
- Sensitive to outliers in early stages
- Less field-tested than COLMAP

**COLMAP (Incremental SfM):**
1. Find connected component of image graph
2. Pick two-view pair with most matches
3. Incrementally add images, triangulate, bundle adjust
4. Repeat for additional components

**Advantages:**
- More robust to outliers
- Incremental BA catches early mistakes
- Well-established, predictable

**Disadvantages:**
- 10-50x slower
- Sensitive to initialization (two-view pair selection)
- May fail on truly unordered datasets

**Recommendation:** Start with GLOMAP for large datasets. Fall back to COLMAP if GLOMAP registration < 80%.

#### Camera Model Selection

| Model | Distortion Parameters | When |
|-------|----------------------|------|
| SIMPLE_RADIAL | f, cx, cy, k | Generic cameras, phones |
| OPENCV | f, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 | DJI drones, strong distortion |
| PINHOLE | f, cx, cy | Reframed 360° (no distortion) |
| RADIAL | More parameters, slower | Over-parameterized, avoid |

**SIMPLE_RADIAL vs OPENCV:**
- SIMPLE_RADIAL: 4 parameters, fast convergence, generic
- OPENCV: 12 parameters, captures complex distortion (DJI Mavic)

**When OPENCV matters:**
- DJI drones (Mavic 3 Pro has visible barrel distortion)
- Phone cameras (strong pincushion)
- Fisheye lenses

**Decision:** Use SIMPLE_RADIAL unless you know camera has strong distortion. Can always try OPENCV in troubleshooting.

#### Parameter Tuning: tri_max_error (Triangulation Reprojection Error)

Reprojection error threshold for 3D point acceptance.

```
tri_max_error = 1.0 px
├─ Very strict: only high-quality 3D points
├─ Output: Fewer points, very clean
└─ Risk: May reject valid points (under-triangulation)

tri_max_error = 4.0 px (DEFAULT)
├─ Balanced: accepts most valid triangulations
├─ Output: Dense, well-registered points
└─ Good for most scenes

tri_max_error = 8.0 px
├─ Permissive: includes marginal points
├─ Output: Maximum density
└─ Risk: Noisy 3D cloud, potential artifacts in rendering
```

**Practical guidance:**
- Start with 4.0. If point cloud is too sparse → increase to 8.0. If too noisy → decrease to 2.0.
- Reprojection error > 2px typically indicates poor alignment. Check camera model choice.

#### Parameter Tuning: tri_min_angle

Triangulation angle threshold (degrees).

```
tri_min_angle = 0.5°
├─ Permissive: even near-parallel viewing rays accepted
├─ Output: Dense points, but high depth uncertainty
└─ Risk: Unstable triangulation (sensitive to noise)

tri_min_angle = 1.5° (DEFAULT)
├─ Balanced: typical baseline for SfM
├─ Output: Stable 3D points
└─ Good for most scenes

tri_min_angle = 3.0°
├─ Strict: only wide baseline pairs contribute
├─ Output: Sparse but very robust
└─ Use if output is too noisy
```

**Decision:** Default 1.5° is good. Only adjust if:
- Sparse point cloud → lower to 0.5°
- Noisy point cloud → increase to 3.0°

#### Example: Reconstruct with GLOMAP + OPENCV camera model

```bash
python -m modular_pipeline.sfm \
  ./output \
  --mapper glomap \
  --camera-model OPENCV \
  --refine-intrinsics \
  --tri-max-error 4.0 \
  --tri-min-angle 1.5
```

---

## 4. Parameter Tuning Guide

### Decision Tree: What to Change When

```
Registration rate < 50%?
├─ Yes
│  ├─ Try COLMAP instead of GLOMAP
│  ├─ Try XFeat extractor (different feature type)
│  ├─ Increase max_keypoints to 16000
│  ├─ Check camera model (SIMPLE_RADIAL vs OPENCV)
│  └─ Inspect image quality (blur, exposure)
└─ No → OK, continue

Registration 50-80%?
├─ Increase max_keypoints from 8000 to 16000
├─ Increase fallback_threshold from 50 to 75
├─ Increase tri_max_error from 4.0 to 8.0
└─ Try COLMAP for more robust incremental SfM

Registration > 90% but point cloud sparse (<100K points)?
├─ Increase max_keypoints to 16000
├─ Decrease tri_max_error from 4.0 to 2.0
├─ Decrease tri_min_angle from 1.5 to 0.5
├─ Lower min_matches from 15 to 10
└─ Enable subpixel_refinement

Processing slow (>4 hours for 6900 images)?
├─ Reduce max_keypoints to 4000
├─ Use sequential retrieval if video (faster than NetVLAD)
├─ Reduce num_neighbors from 50 to 30
├─ Lower max_pairs_fallback from 500 to 100
└─ Use GLOMAP (already selected by default)

Reprojection error > 2.0 px?
├─ Check camera_model (try OPENCV if using SIMPLE_RADIAL)
├─ Enable subpixel_refinement
├─ Lower tri_max_error to 2.0
├─ Inspect for blurry/low-quality images
└─ Check image alignment (are frames in correct order?)

Point cloud has artifacts/speckles?
├─ Increase tri_min_angle from 1.5 to 3.0
├─ Increase tri_max_error threshold
├─ Lower fallback_threshold (reject weak matches)
└─ Try masking (if moving objects present)
```

### Cascading Effects of Common Changes

**Scenario 1: "Need faster processing"**
```
Change: Reduce max_keypoints from 8000 to 4000
Impact:
├─ Extract: 50% faster ✓
├─ Match: 75% fewer comparisons (O(n²) effect) ✓✓
├─ SfM: Faster, but registration may drop
└─ Output: Sparser 3D cloud

Net: Save 1-2 hours, lose some detail. Good if time-critical.
```

**Scenario 2: "Need richer 3D points"**
```
Change: Increase max_keypoints from 8000 to 16000
Impact:
├─ Extract: 2.8x slower (1.5 hours for 6900 images)
├─ Match: 4x more operations, potential bottleneck (could add 8 hours!)
├─ SfM: More triangulation candidates, denser output
└─ Output: 2-4x more 3D points

Net: 8-10 hours additional compute, 2-4x density gain. Diminishing returns.
```

**Scenario 3: "Getting too many false matches"**
```
Change: Increase min_matches from 15 to 30, disable fallback
Impact:
├─ Match: Faster (skip fallback matcher), fewer weak pairs
├─ SfM: Fewer pairs to triangulate from
└─ Output: Cleaner but potentially sparser

Net: Faster match stage, risk of registration gaps.
```

**Scenario 4: "Have 30,000 images, need to finish in 12 hours"**
```
Changes:
├─ max_keypoints: 4000 (fast extraction)
├─ retrieval: "sequential" if video, else NetVLAD with num_neighbors=30
├─ min_matches: 15 (standard)
├─ tri_max_error: 6.0 (permissive)
└─ mapper: "glomap" (already default)

Estimate: 3-4 hours vs. 12+ hours with defaults.
Trade: Sparse features but feasible processing.
```

---

## 5. Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (GPU recommended, CPU fallback available)
- 32GB+ RAM (for large datasets; 16GB minimum)

### Step 1: Clone and Set Up Environment

```bash
git clone <repo>
cd modular-reconstruction-pipeline

conda create -n mrp python=3.10
conda activate mrp

pip install -e .
```

### Step 2: Install Optional Dependencies

**For video support:**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from ffmpeg.org
```

**For GPU acceleration (PyTorch with CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For additional matchers (RoMa, MASt3R):**
```bash
pip install roma mast3r  # Optional
```

---

## 6. Usage

### Basic Pipeline (All Stages)

```bash
python -m modular_pipeline.pipeline ./my_project --preset mavic_3_pro
```

### Stage-by-Stage (Manual Control)

```bash
# Stage 1: Extract features
python -m modular_pipeline.extract \
  ./my_project/frames \
  ./my_project/output \
  --extractor aliked \
  --max-keypoints 8000

# Stage 2: Retrieve pairs (runs in parallel, but can be explicit)
python -m modular_pipeline.retrieve \
  ./my_project/frames \
  ./my_project/output \
  --retrieval netvlad \
  --num-neighbors 50

# Stage 3: Match features
python -m modular_pipeline.match \
  ./my_project/output/features.h5 \
  ./my_project/output \
  --matcher lightglue \
  --fallback-matcher mast3r

# Stage 4: Reconstruct
python -m modular_pipeline.sfm \
  ./my_project/output \
  --mapper glomap \
  --camera-model OPENCV
```

### Module-Only Use Cases

**Just masking:**
```bash
python -m modular_pipeline.prepare.masking \
  ./frames \
  ./output \
  --model sam3 \
  --remove-prompts "person,tripod"
```

**Just reframing (360° to pinhole):**
```bash
python -m modular_pipeline.prepare.reframe \
  ./frames \
  ./output \
  --pattern ring_12 \
  --fov-h 90
```

**Resume from checkpoint:**
```bash
# If pipeline failed at SfM stage, fix and re-run:
python -m modular_pipeline.pipeline ./my_project --resume
```

---

## 7. Quality Assessment

### Metrics at Each Stage

#### Ingest Quality
```json
{
  "total_frames": 6234,
  "filtered": 312,
  "quality": {
    "excellent": 5890,
    "good": 322,
    "review": 20,
    "poor": 2
  }
}
```
**Action:** Inspect "poor" frames for corruption.

#### Extract Quality
```json
{
  "avg_keypoints_per_image": 8120,
  "min_keypoints": 324,
  "max_keypoints": 9840
}
```
**Action:** If min < 1000, check for low-texture images.

#### Match Quality
```json
{
  "total_pairs": 345000,
  "pairs_with_matches": 312450,
  "success_rate": 0.905,
  "fallback_pairs": 18234
}
```
**Action:** If success_rate < 0.80, increase num_neighbors or max_keypoints.

#### SFM Quality
```json
{
  "total_images": 6234,
  "registered_images": 5989,
  "registration_rate": 0.961,
  "total_3d_points": 3456789,
  "mean_track_length": 2.8,
  "mean_reprojection_error": 1.23
}
```

**Good indicators:**
- Registration rate > 90%
- Mean reprojection error < 2.0 px
- Mean track length > 2.0

**Warning signs:**
- Registration rate < 50% → check extractor, camera model
- Reprojection error > 3.0 px → check image blur, camera calibration
- Track length < 1.5 → too few matches, increase max_keypoints

---

## 8. Troubleshooting

### Registration Rate < 50%

**Root causes:**
1. Poor image quality (blur, motion, dark)
2. Wrong camera model
3. Weak feature extraction
4. Insufficient matches

**Solutions (in order):**
```bash
# 1. Try COLMAP (more robust initialization)
python -m modular_pipeline.sfm ./output --mapper colmap

# 2. Increase feature density
python -m modular_pipeline.extract ./images ./output --max-keypoints 16000

# 3. Try different extractor
python -m modular_pipeline.extract ./images ./output --extractor xfeat

# 4. Inspect image quality
python -m modular_pipeline.ingest.quality ./images

# 5. Try OPENCV camera model
python -m modular_pipeline.sfm ./output --camera-model OPENCV --refine-intrinsics
```

### Sparse Point Cloud (<100K Points)

```bash
# Increase features
--max-keypoints 16000

# Lower triangulation thresholds
--tri-max-error 8.0 --tri-min-angle 0.5

# Reduce match threshold
--min-matches 10
```

### High Reprojection Error (>2.0 px)

```bash
# Enable sub-pixel refinement
--subpixel-refinement

# Try COLMAP backend
--mapper colmap

# Check camera model
--camera-model OPENCV
```

### 360° Reconstruction Fails

```bash
# Verify reframing output
ls output/reframed/rig_views/  # Should have cam_00, cam_01, etc.

# Increase rig views
--pattern geodesic_20

# Check rig.json exists
cat output/reframed/rig.json
```

### Processing Too Slow

```bash
# Reduce keypoints
--max-keypoints 4000

# Sequential if video
--retrieval sequential

# Fewer neighbors
--num-neighbors 30

# Skip fallback
--max-pairs-fallback 0
```

---

## References

**Papers:**
- ALIKED (ECCV 2024): https://github.com/Shiaoming/ALIKED
- LightGlue (ICCV 2023): https://github.com/cvg/LightGlue
- GLOMAP (2024): https://github.com/cvg/GLOMAP
- SAM3 (2024): https://github.com/facebookresearch/sam2
- MASt3R (2024): https://github.com/naver/mast3r

**Toolkits:**
- HLOC (Hierarchical Localization): https://github.com/cvg/Hierarchical-Localization
- COLMAP: https://colmap.github.io/
- OpenCV: https://opencv.org/

**External Tools:**
- PostShot: https://www.postshot.app/
- Nerfstudio: https://docs.nerf.studio/
- Lichtfeld Studio: https://lichtfeld-studio.com/
