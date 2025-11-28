# MRP: Modular Reconstruction Pipeline

**A production-grade SfM pipeline for experts who need control.**

Handles 360° footage, large unordered datasets, and dynamic scenes. COLMAP-compatible output for NeRF/Gaussian splatting training.

---

## Quick Decision Trees

**What to use?**
```
Do you have 360° footage?
├─ Yes → --preset osmo_360 (handles reframe + masking)
└─ No → Do you have >5000 images?
   ├─ Yes → --preset drone + --retrieval netvlad
   └─ No → --preset standard (exhaust matching works)

Need to remove objects?
├─ Yes → Add --masking person,tripod,etc
└─ No → Skip it

Just running one module?
├─ Extract only → python -m modular_pipeline.extract ./images ./out --extractor aliked
├─ Match only → python -m modular_pipeline.match ./out/features.h5 ./out --retrieval netvlad
├─ Reframe only → python -m modular_pipeline.prepare.reframe ./frames ./out --pattern ring_12
└─ Mask only → python -m modular_pipeline.prepare.masking ./frames ./out --model sam3
```

---

## Why Modular? (vs COLMAP)

| Problem | Standard SfM Breaks | MRP Solution |
|---------|-------------------|--------------|
| 360° equirectangular | Pole distortion kills feature matching | Reframe to pinhole rig (6-20 views) |
| 6900 images | Exhaustive matching = 23.8M pairs (infeasible) | NetVLAD retrieval = 345K pairs (60x speedup) |
| People/tripods in video | Outliers corrupt SfM | Mask + temporal consistency |
| Need control | Black box | Swap any module (extractor, matcher, backend) |
| Need audit trail | No visibility | Manifests at each stage with metrics |

---

## Architecture

```
Input → Ingest → Prepare (reframe/mask) → Reconstruct (extract→match→sfm) → Output
```

**Data intermediate outputs:**
```
features.h5      ← keypoints + descriptors
pairs.txt        ← image pairs to match
matches.h5       ← correspondences
sparse/0/        ← COLMAP (cameras.bin, images.bin, points3D.bin)
```

---

## Modules & Parameters

### INGEST (Optional: Video/OSV → Frames)

```python
# Extract frames from video with quality filtering
source_type: "auto" | "video" | "images" | "osv"
target_fps: float = None
blur_threshold: float = 100.0        # Laplacian variance
exposure_range: (float, float) = (0.1, 0.9)
min_quality_score: float = 0.5       # Discard below this
```

**Example:** `--target-fps 2 --min-quality-score 0.6`

---

### REFRAME (Optional: 360° → Pinhole Rig)

```python
pattern: "cube_6" | "ring_8" | "ring_12" | "geodesic_20"
fov_h: float = 90      # Horizontal FOV (degrees)
fov_v: float = 60      # Vertical FOV
output_resolution: (1920, 1080)
```

**Pattern tradeoffs:**

| Pattern | Views | Speed | Coverage | Use |
|---------|-------|-------|----------|-----|
| cube_6 | 6 | Fast | 20% overlap | Speed priority |
| ring_8 | 8 | Good | 25% overlap | Balanced |
| ring_12 | 12 | Slower | 35% overlap | **Recommended** |
| geodesic_20 | 20 | Slow | 50% overlap | Sparse textures |

**Decision:** Start ring_12. Go geodesic_20 if registration < 80%.

**Example:** `--pattern ring_12 --fov-h 90`

---

### MASKING (Optional: Remove Objects)

```python
model: "sam3" | "fastsam" | "efficientsam" | "mobilesam"
remove_prompts: ["person", "tripod", "camera operator", ...]
confidence_threshold: float = 0.70
use_temporal_consistency: bool = True
temporal_window: int = 5
```

**Model tradeoffs:**

| Model | Speed | Quality | When |
|-------|-------|---------|------|
| sam3 | 300-500ms | Excellent | Default, GPU available |
| fastsam | 50-100ms | Good | Speed critical |
| efficientsam | 100-200ms | Good | TensorRT optimized |

**Example:** `--model sam3 --remove-prompts "person,tripod" --use-temporal-consistency`

---

### EXTRACT (Feature Detection)

```python
extractor: "aliked" | "xfeat" | "superpoint" | "disk"
max_keypoints: int = 8000
nms_radius: int = 3
subpixel_refinement: bool = False   # +7ms per image, -50% reprojection error
```

**Extractor comparison:**

| Extractor | Speed | Quality | Feature Type |
|-----------|-------|---------|--------------|
| **aliked** | 150-300ms | Excellent | Affine-invariant |
| xfeat | 50-100ms | Good | Fast inference |
| superpoint | 200-400ms | Good | CNN-based |
| disk | 100-200ms | Good | Rotation-invariant |

**max_keypoints cascading effects:**

```
4000  → Extract 50% faster, Match 75% fewer ops, Sparse cloud
8000  → Default, balanced
16000 → Extract 2.8x slower, Match 4x ops (bottleneck!), Dense cloud
```

**Decision:** 8000 default. 4000 for speed. 16000 for high-res/texture-rich scenes.

**Example:** `--extractor aliked --max-keypoints 8000`

---

### RETRIEVE (Pair Selection - runs parallel with EXTRACT)

```python
retrieval: "netvlad" | "sequential" | "exhaustive" | "vocab_tree" | "hybrid"
num_neighbors: int = 50
```

**Pair count analysis (6900 images):**

| Strategy | Pairs | Time (LightGlue 40ms/pair) | Use |
|----------|-------|---------------------------|-----|
| Exhaustive | 23.8M | 264 hours | <1000 images only |
| Sequential (k=20) | 138K | 1.5 hours | Video frames |
| NetVLAD (k=50) | 345K | 3.8 hours | **Default, unordered** |
| Hybrid (k=20+50) | 483K | 5.3 hours | Video with gaps |

**Decision:** NetVLAD for general scenes. Sequential if video. Increase num_neighbors to 75-100 if sparse textures.

**Example:** `--retrieval netvlad --num-neighbors 50`

---

### MATCH (Feature Correspondence)

```python
matcher: "lightglue" | "mast3r" | "roma" | "superglue"
fallback_matcher: "mast3r"
min_matches: int = 15
fallback_threshold: int = 50         # Retry with fallback if < this
max_pairs_fallback: int = 500
```

**Matcher comparison:**

| Matcher | Speed | Density | When |
|---------|-------|---------|------|
| **lightglue** | 35-40ms | Sparse | Default, best balance |
| mast3r | 300-500ms | Dense | Fallback for weak pairs |
| roma | 200-300ms | Very dense | Dense scenes |
| superglue | 100-150ms | Sparse | Legacy |

**Two-stage strategy:**
1. LightGlue all pairs (~40ms each)
2. MASt3R retry pairs with < min_matches (selective, ~400ms each)

**Why:** 10x speedup vs dense-only, captures difficult correspondences.

**min_matches tuning:**
- 10: Permissive (more pairs, risk outliers)
- 15: Default, balanced
- 30: Strict (fewer pairs, cleaner)

**Example:** `--matcher lightglue --fallback-matcher mast3r --min-matches 15`

---

### SFM (Structure from Motion)

```python
mapper: "glomap" | "colmap"
camera_model: "SIMPLE_RADIAL" | "OPENCV" | "PINHOLE"
tri_max_error: float = 4.0           # Reprojection error (pixels)
tri_min_angle: float = 1.5           # Triangulation angle (degrees)
refine_intrinsics: bool = True
```

**Mapper comparison:**

| Backend | Speed | Robustness | When |
|---------|-------|-----------|------|
| **glomap** | 10-50x faster | 8% higher recall | Default, unordered/global |
| colmap | Baseline | Well-tested | Fallback if GLOMAP < 80% registration |

**Camera model:**
- SIMPLE_RADIAL: Generic (default)
- OPENCV: DJI drones, strong distortion
- PINHOLE: Reframed 360°

**tri_max_error tuning:**
```
1.0 px   → Strict, sparse clean cloud
4.0 px   → Default, balanced
8.0 px   → Permissive, maximum density
```

**tri_min_angle tuning:**
```
0.5°  → Dense but unstable
1.5°  → Default, stable
3.0°  → Sparse but robust
```

**Example:** `--mapper glomap --camera-model OPENCV --tri-max-error 4.0`

---

## Parameter Tuning: Decision Tree

```
Registration < 50%?
├─ Try --mapper colmap (more robust)
├─ Try --max-keypoints 16000
├─ Try --extractor xfeat (different features)
├─ Check --camera-model OPENCV
└─ Inspect image quality (blur, exposure)

Registration 50-80%?
├─ --max-keypoints 16000
├─ --fallback-threshold 75
├─ --tri-max-error 8.0
└─ Try --mapper colmap

Registration >90% but sparse cloud (<100K points)?
├─ --max-keypoints 16000
├─ --tri-max-error 2.0
├─ --tri-min-angle 0.5
├─ --min-matches 10
└─ --subpixel-refinement

Too slow (>4hrs for 6900 images)?
├─ --max-keypoints 4000
├─ --retrieval sequential (if video)
├─ --num-neighbors 30
├─ --max-pairs-fallback 0
└─ (GLOMAP already default)

Reprojection error >2.0 px?
├─ --camera-model OPENCV --refine-intrinsics
├─ --subpixel-refinement
├─ --tri-max-error 2.0
└─ Check image blur

Noisy point cloud (artifacts)?
├─ --tri-min-angle 3.0
├─ --min-matches 20
└─ Add masking if moving objects
```

---

## Cascading Effects

**Increase max_keypoints 8K→16K:**
```
Extract:    +1.8x time
Match:      +4x operations per pair (bottleneck!)
Output:     2-4x more 3D points
Total cost: ~8-10 extra hours for 6900 images
```

**Switch NetVLAD to Sequential (video):**
```
Pair count:  345K → 138K (-60%)
Match time:  3.8 hrs → 1.5 hrs
Tradeoff:    Miss geometric revisits
```

**Decrease tri_max_error 4px→2px:**
```
SfM:     Rejects marginal triangulations
Output:  Cleaner but sparser cloud
Quality: Reprojection error drops ~50%
```

---

## Usage

### Full Pipeline (All Stages)
```bash
python -m modular_pipeline.pipeline ./project --preset mavic_3_pro
```

### Stage-by-Stage
```bash
# 1. Extract
python -m modular_pipeline.extract ./frames ./output --extractor aliked --max-keypoints 8000

# 2. Retrieve (parallel)
python -m modular_pipeline.retrieve ./frames ./output --retrieval netvlad --num-neighbors 50

# 3. Match
python -m modular_pipeline.match ./output/features.h5 ./output --matcher lightglue

# 4. SFM
python -m modular_pipeline.sfm ./output --mapper glomap --camera-model OPENCV
```

### Single Module
```bash
# Mask only
python -m modular_pipeline.prepare.masking ./frames ./output --model sam3 --remove-prompts "person,tripod"

# Reframe only
python -m modular_pipeline.prepare.reframe ./frames ./output --pattern ring_12

# Extract only
python -m modular_pipeline.extract ./frames ./output --extractor aliked --max-keypoints 8000
```

### Resume
```bash
python -m modular_pipeline.pipeline ./project --resume
```

---

## Quality Metrics (At Each Stage)

**Ingest:** Check quality distribution. Flag "poor" frames.

**Extract:** Min keypoints > 1000 per image. Avg ~8000.

**Match:** Success rate > 80%. Fallback pairs < 20% of total.

**SFM:**
- Registration rate > 90% (good), < 50% (broken)
- Reprojection error < 2.0 px (good), > 3.0 px (problem)
- Track length > 2.0 (typical), < 1.5 (too few matches)
- Point count: more is better (depends on scene)

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Registration <50% | Bad features, wrong camera model, image quality | Try COLMAP, OPENCV model, increase keypoints |
| Sparse cloud (<100K) | Too few matches | Increase keypoints, lower tri_max_error |
| High reprojection error | Camera model wrong, image blur | Try OPENCV model, subpixel_refinement |
| 360° fails | Wrong rig pattern, pole issues | Try geodesic_20, check rig.json |
| Too slow | Too many features/pairs | Reduce keypoints, use sequential, reduce neighbors |

---

## Installation

```bash
git clone <repo>
cd modular-reconstruction-pipeline
conda create -n mrp python=3.10
conda activate mrp
pip install -e .
```

**Optional:**
```bash
# Video support
sudo apt install ffmpeg

# GPU (PyTorch CUDA 11.8+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## File Structure

```
project/
├── source/                  # Input (video/images/osv)
├── frames/                  # Ingest output
├── reframed/               # Reframe output (360° only)
├── masked/                 # Masking output (optional)
├── features/               # Extract output (features.h5)
├── pairs/                  # Retrieve output (pairs.txt)
├── matches/                # Match output (matches.h5)
└── sparse/0/               # SFM output (cameras.bin, images.bin, points3D.bin)
```

Each stage produces `manifest.json` with config + metrics for debugging.

---

## References

**Papers:**
- ALIKED (ECCV 2024): https://github.com/Shiaoming/ALIKED
- LightGlue (ICCV 2023): https://github.com/cvg/LightGlue
- GLOMAP (2024): https://github.com/cvg/GLOMAP
- SAM3 (2024): https://github.com/facebookresearch/sam2
- MASt3R (2024): https://github.com/naver/mast3r

**Tools:**
- HLOC: https://github.com/cvg/Hierarchical-Localization
- COLMAP: https://colmap.github.io/
- PostShot: https://www.postshot.app/
- Nerfstudio: https://docs.nerf.studio/
