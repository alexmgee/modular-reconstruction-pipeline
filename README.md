# MRP: Modular Reconstruction Pipeline

**A production-grade SfM pipeline for experts who need control.**

Handles 360° footage, large unordered datasets, and dynamic scenes. COLMAP-compatible output for NeRF/Gaussian splatting training.

---

## Quick Start: Decision Trees

**What to use?**
```
Do you have 360° footage?
├─ Yes → --preset osmo_360 (reframe + mask)
└─ No → Do you have >5000 images?
   ├─ Yes → --preset drone + --retrieval netvlad
   └─ No → --preset standard

Need to remove objects?
├─ Yes → Add --masking person,tripod,etc
└─ No → Skip it

Single module only?
├─ Extract: python -m modular_pipeline.extract ./frames ./out --extractor aliked
├─ Match: python -m modular_pipeline.match ./out/features.h5 ./out --retrieval netvlad
├─ Reframe: python -m modular_pipeline.prepare.reframe ./frames ./out --pattern ring_12
└─ Mask: python -m modular_pipeline.prepare.masking ./frames ./out --model sam3
```

---

## Full Parameter Reference

### INGEST (Video/Images → Frames)

**Essential Parameters:**

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `source_type` | str | auto | auto, video, images, osv | Auto-detect or specify source format |
| `target_fps` | float | None | Positive float | Resample video to target fps (None = keep original) |
| `blur_threshold` | float | 100.0 | Positive float | Laplacian variance threshold (higher = permissive) |
| `exposure_range` | tuple | (0.1, 0.9) | (min, max) [0-1] | Acceptable pixel histogram bounds |
| `min_quality_score` | float | 0.5 | [0-1] | Discard frames below this score |
| `filter_quality` | bool | True | True/False | Enable quality filtering |
| `output_format` | str | jpg | jpg, png | Output image format |
| `output_quality` | int | 95 | [1-100] | JPEG quality (if jpg output) |
| `rename_pattern` | str | frame_{:06d}.jpg | Format string | Frame renaming pattern |

**Advanced Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_frames` | int | 0 | Extract every Nth frame (1=skip none, 2=skip half) |
| `max_frames` | int | None | Maximum total frames to extract |
| `start_frame` | int | 0 | First frame to process |
| `end_frame` | int | None | Last frame to process |
| `extract_metadata` | bool | True | Extract EXIF (camera model, GPS, focal length) |
| `sort_by_timestamp` | bool | True | Sort images by EXIF timestamp |
| `copy_files` | bool | True | Copy frames vs symlink to source |
| `recursive_search` | bool | True | Search subdirectories for images |

**Example:** `--target-fps 2 --filter-quality --min-quality-score 0.6 --skip-frames 2`

---

### REFRAME (360° Equirectangular → Pinhole Rig)

**Essential Parameters:**

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `rig_pattern` | str | ring12 | cube, ring8, ring12, geodesic, custom | Camera rig layout (more views = denser but slower matching) |
| `output_width` | int | 1920 | Positive int | Output frame width per rig camera |
| `output_height` | int | 1080 | Positive int | Output frame height per rig camera |
| `fov_h` | float | 90 | [0-180] degrees | Horizontal field of view per camera |
| `fov_v` | float | 60 | [0-180] degrees | Vertical field of view per camera |
| `projection_backend` | str | auto | auto, torch_gpu, py360, equilib, opencv, numpy | Projection compute backend (auto picks fastest available) |

**Advanced Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `custom_rig_config` | str | None | Path to custom rig JSON (overrides pattern) |
| `interpolation` | str | bilinear | bilinear, nearest, cubic - Reprojection interpolation |
| `cache_projection_maps` | bool | True | Cache computed projection maps (faster, higher memory) |
| `auto_detect_geometry` | bool | True | Auto-detect equirectangular vs pinhole from metadata |
| `passthrough_pinhole` | bool | True | Skip reframing if image already pinhole |
| `generate_colmap_rig` | bool | True | Generate rig.json for COLMAP |
| `save_rig_visualization` | bool | False | Save debug visualization of rig pattern |

**Pattern Tradeoffs:**

| Pattern | Views | Speed | Overlap | Use |
|---------|-------|-------|---------|-----|
| cube | 6 | Fast | 20% | Speed priority |
| ring8 | 8 | Good | 25% | Balanced |
| ring12 | 12 | Slower | 35% | **Recommended** |
| geodesic | 20 | Slow | 50% | Sparse/difficult textures |

**Example:** `--pattern ring12 --fov-h 90 --fov-v 60 --projection-backend torch_gpu`

---

### MASKING (Remove Objects with SAM3)

**Essential Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | auto | auto, sam3, fastsam, efficient, sam2 - Segmentation model |
| `remove_prompts` | list | [tripod, camera operator person, equipment, ...] | Text prompts of objects to remove |
| `confidence_threshold` | float | 0.70 | [0-1] - Minimum mask confidence to accept |
| `use_temporal_consistency` | bool | True | Smooth masks across video frames |
| `temporal_window` | int | 5 | Number of frames for temporal voting |
| `geometry_aware` | bool | True | Handle pole/distortion-aware expansion |

**Advanced Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | str | None | Path to custom model checkpoint |
| `keep_prompts` | list | [] | Inverse masking: keep these objects, remove rest |
| `min_mask_area` | int | 100 | Minimum mask size in pixels (remove if smaller) |
| `max_mask_area_ratio` | float | 0.5 | Don't remove >50% of image per frame |
| `handle_distortion` | bool | True | Handle fisheye/distortion artifacts |
| `pole_mask_expand` | float | 1.2 | Expansion for pole region masking (>1.0) |
| `output_format` | str | png | png, jpg, npy - Mask output format |
| `save_confidence_maps` | bool | False | Save confidence maps as .npy for inspection |
| `save_review_images` | bool | True | Save images flagged for manual review |

**Model Tradeoffs:**

| Model | Speed | Quality | Memory | When |
|-------|-------|---------|--------|------|
| sam3 | 300-500ms | Excellent | 3GB | Default, text prompts, GPU |
| fastsam | 50-100ms | Good | 1GB | Speed critical |
| efficientsam | 100-200ms | Good | 500MB | TensorRT optimized |
| sam2 | 200-400ms | Excellent | 2GB | Latest |

**Example:** `--model sam3 --remove-prompts "person,tripod,selfie stick" --use-temporal-consistency --temporal-window 7`

---

### EXTRACT (Feature Keypoint Detection)

**Essential Parameters:**

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `extractor` | str | aliked | aliked, xfeat, superpoint, disk, auto | Feature detector backend |
| `max_keypoints` | int | 8000 | Positive int | Maximum keypoints per image |
| `nms_radius` | int | 3 | Positive int | Non-maximum suppression radius (px) |
| `keypoint_threshold` | float | 0.005 | [0-1] | Detection confidence threshold |
| `subpixel_refinement` | bool | False | True/False | Sub-pixel refinement (+7ms, -50% reprojection error) |

**Advanced Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resize_max` | int | 1600 | Resize images so max dim <= this (-1 = no resize) |
| `resize_force` | tuple | None | Force specific (width, height) - overrides resize_max |
| `grayscale` | bool | True | Convert to grayscale before extraction |
| `remove_borders` | int | 4 | Ignore edge pixels (prevents boundary artifacts) |
| `output_format` | str | h5 | h5, npz - Output file format |
| `feature_name` | str | feats-{extractor}-n{max_keypoints} | Feature filename pattern |

**Extractor Comparison:**

| Extractor | Speed | Quality | Type | When |
|-----------|-------|---------|------|------|
| **aliked** | 150-300ms | Excellent | Affine-invariant | **Default, best overall** |
| xfeat | 50-100ms | Good | Fast inference | Real-time processing |
| superpoint | 200-400ms | Good | CNN-based | Legacy compatibility |
| disk | 100-200ms | Good | Rotation-invariant | Homographic scenes |

**max_keypoints cascading effects:**

```
4000   → Extract 50% faster, Match 75% fewer ops, Sparse cloud
8000   → Default, balanced
16000  → Extract 2.8x slower, Match 4x ops (bottleneck!), Dense cloud
```

**Example:** `--extractor aliked --max-keypoints 8000 --subpixel-refinement --remove-borders 4`

---

### RETRIEVE (Image Pair Selection)

**Essential Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | netvlad | netvlad, sequential, exhaustive, vocab_tree, cosplace, eigenplaces, hybrid |
| `num_neighbors` | int | 50 | k-NN neighbors per image |

**Advanced Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_pairs` | int | 1_000_000 | Safety limit (stop if exceeds) |
| `sequential_window` | int | 50 | Frame window for sequential pairing |
| `sequential_overlap` | bool | True | Overlapping windows in sequential mode |
| `vlad_cluster_size` | int | 64 | NetVLAD cluster centers (32 or 64) |
| `vlad_dim` | int | 4096 | NetVLAD descriptor dimension |
| `min_score` | float | 0.0 | [0-1] - Minimum similarity to keep pair |
| `filter_duplicates` | bool | True | Remove (A,B) if (B,A) exists |
| `output_format` | str | txt | txt, json - Pair list format |

**Backend Pair Counts (6900 images):**

| Backend | Pairs | Time (40ms/pair) | Use |
|---------|-------|-----------------|-----|
| Exhaustive | 23.8M | 264 hours | <1000 images only |
| Sequential (k=20) | 138K | 1.5 hours | Video frames |
| NetVLAD (k=50) | 345K | 3.8 hours | **Default, unordered** |
| Hybrid | 483K | 5.3 hours | Video with gaps |

**Example:** `--backend netvlad --num-neighbors 50 --min-score 0.0 --filter-duplicates`

---

### MATCH (Feature Correspondence)

**Essential Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `matcher` | str | lightglue | lightglue, mast3r, roma, superglue, auto |
| `fallback_matcher` | str | mast3r | Retry difficult pairs with this |
| `min_matches` | int | 15 | Minimum correspondences per pair |
| `fallback_threshold` | int | 50 | Retry if < this many matches |
| `max_pairs_fallback` | int | 500 | Max pairs to retry with fallback |

**Advanced Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_threshold` | float | 0.1 | [0-1] - Match confidence threshold (lower = more matches) |
| `depth_confidence` | float | -1 | -1 (max quality) or [0-1] (speed threshold for LightGlue) |
| `width_confidence` | float | -1 | -1 (max quality) or [0.95] (pruning for LightGlue) |
| `feature_type` | str | aliked | Feature type (must match EXTRACT extractor) |
| `output_format` | str | h5 | h5, txt - Output file format |
| `force` | bool | False | Force overwrite of existing matches |

**Matcher Comparison:**

| Matcher | Speed | Density | When |
|---------|-------|---------|------|
| **lightglue** | 35-40ms | Sparse | **Default, best balance** |
| mast3r | 300-500ms | Dense | Fallback for weak pairs |
| roma | 200-300ms | Very dense | Dense scenes |
| superglue | 100-150ms | Sparse | Legacy |

**Two-stage matching:**
1. LightGlue all pairs (~40ms each)
2. MASt3R retry pairs with < min_matches (selective, ~400ms each)
3. Why: 10x speedup vs dense-only

**min_matches tuning:**
- 10: Permissive (risk outliers)
- 15: **Default, balanced**
- 30: Strict (cleaner, risk gaps)

**Example:** `--matcher lightglue --fallback-matcher mast3r --min-matches 15 --fallback-threshold 50 --max-pairs-fallback 500`

---

### SFM (Structure from Motion)

**Essential Parameters:**

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `mapper` | str | glomap | glomap, colmap | SfM algorithm |
| `camera_model` | str | SIMPLE_RADIAL | SIMPLE_RADIAL, OPENCV, PINHOLE | Distortion model |
| `tri_max_error` | float | 4.0 | Positive float | Triangulation reprojection error threshold (px) |
| `tri_min_angle` | float | 1.5 | Positive float | Triangulation angle threshold (degrees) |
| `refine_intrinsics` | bool | True | True/False | Optimize camera intrinsics |

**Advanced Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_num_tracks` | int | 6_000_000 | Maximum 3D points to track |
| `tri_max_dist` | float | 100.0 | Maximum 3D point distance (meters) |
| `run_point_triangulator` | bool | True | Triangulate 3D points |
| `run_bundle_adjustment` | bool | False | Final bundle adjustment pass |
| `camera_mode` | str | auto | auto, single, per_folder, per_image - Intrinsic sharing |
| `global_positioning_max_iter` | int | 100 | GLOMAP iteration limit |
| `bundle_adjustment_max_iter` | int | 100 | Bundle adjustment iteration limit |

**Mapper Comparison:**

| Backend | Speed | Robustness | When |
|---------|-------|-----------|------|
| **glomap** | 10-50x faster | 8% higher recall | Default, unordered, large datasets |
| colmap | Baseline | Well-tested | Fallback if GLOMAP < 80% registration |

**Camera Model:**
- SIMPLE_RADIAL: Generic (default), fast convergence
- OPENCV: Strong distortion (DJI Mavic, phones, fisheye)
- PINHOLE: Reframed 360° (no distortion after projection)

**tri_max_error tuning:**
```
1.0 px   → Strict, sparse clean cloud
4.0 px   → **Default, balanced**
8.0 px   → Permissive, maximum density, noisier
```

**tri_min_angle tuning:**
```
0.5°  → Dense but unstable (near-parallel rays)
1.5°  → **Default, stable**
3.0°  → Sparse but robust (wide baseline only)
```

**Example:** `--mapper glomap --camera-model OPENCV --refine-intrinsics --tri-max-error 4.0 --tri-min-angle 1.5`

---

### OUTPUT: SPLAT (Gaussian Splatting Training)

**Essential Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | gsplat | gsplat, nerfstudio, inria_3dgs, auto |
| `num_iterations` | int | 30_000 | Total training iterations |
| `sh_degree` | int | 3 | [0-3] Spherical harmonics (0=diffuse, 3=view-dependent) |

**Training Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position_lr_init` | float | 0.00016 | Initial position learning rate |
| `position_lr_final` | float | 0.0000016 | Final position learning rate |
| `position_lr_delay_mult` | float | 0.01 | Position LR delay multiplier |
| `position_lr_max_steps` | int | 30_000 | Position LR schedule max steps |
| `feature_lr` | float | 0.0025 | Feature learning rate |
| `opacity_lr` | float | 0.05 | Opacity learning rate |
| `scaling_lr` | float | 0.005 | Scaling learning rate |
| `rotation_lr` | float | 0.001 | Rotation learning rate |

**Densification Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `densification_interval` | int | 100 | Add splats every N iterations |
| `densification_start` | int | 500 | Start densification at iteration N |
| `densification_end` | int | 15_000 | Stop densification at iteration N |
| `densify_grad_threshold` | float | 0.0002 | Gradient threshold for splat generation |
| `opacity_cull_threshold` | float | 0.005 | [0-1] - Opacity threshold for pruning |

**Advanced Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_checkpoint_interval` | int | 5_000 | Save checkpoint every N iterations |
| `export_ply` | bool | True | Export final splat as PLY |
| `viewer_enabled` | bool | False | Enable real-time viewer during training |
| `viewer_port` | int | 7007 | Viewer port (1-65535) |

**Example:** `--num-iterations 40000 --sh-degree 3 --densification-end 20000 --opacity-cull-threshold 0.01`

---

### OUTPUT: MESH (Mesh Extraction)

**Essential Parameters:**

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `backend` | str | pgsr | gof, pgsr, 2dgs, sugar, auto | Mesh extraction method |
| `resolution` | int | 512 | Positive int | Marching cubes grid resolution |
| `texture_resolution` | int | 2048 | Positive int | Output texture size |

**Advanced Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `poisson_depth` | int | 10 | Poisson reconstruction depth (higher = detail, slower) |
| `simplify_ratio` | float | 0.0 | [0-1] Mesh decimation (0=none, 0.5=half faces) |
| `min_face_count` | int | 10_000 | Minimum output face count |
| `max_face_count` | int | 1_000_000 | Maximum output face count (triggers decimation) |
| `bake_textures` | bool | True | Generate texture maps |
| `bake_normals` | bool | False | Generate normal maps |
| `bake_albedo` | bool | True | Generate albedo maps |
| `export_obj` | bool | True | Export as OBJ with materials |
| `export_ply` | bool | False | Export as PLY |
| `export_glb` | bool | True | Export as GLB (web-ready) |

**Backend Comparison:**

| Backend | Use |
|---------|-----|
| **pgsr** | General purpose, good quality/speed balance |
| gof | Unbounded outdoor scenes |
| 2dgs | Thin surface extraction |
| sugar | Blender/Unity editable geometry |

**Example:** `--backend pgsr --resolution 1024 --texture-resolution 4096 --simplify-ratio 0.3 --export-glb`

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

Reprojection error >2.0 px?
├─ --camera-model OPENCV --refine-intrinsics
├─ --subpixel-refinement
├─ --tri-max-error 2.0

Noisy point cloud (artifacts)?
├─ --tri-min-angle 3.0
├─ --min-matches 20
└─ --keypoint-threshold 0.01 (strict detection)
```

---

## Cascading Effects

**Increase max_keypoints 8K→16K:**
```
Extract:    +1.8x time
Match:      +4x operations per pair (bottleneck!)
Output:     2-4x more 3D points
Total:      ~8-10 extra hours for 6900 images
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

**Decrease keypoint_threshold 0.005→0.01:**
```
Extract: Fewer keypoints per image (-40%)
Match:   Fewer pairs match, sparser cloud
Speed:   +25% faster extraction
Quality: Fewer false positives
```

---

## Usage

### Full Pipeline
```bash
python -m modular_pipeline.pipeline ./project --preset mavic_3_pro
```

### Stage-by-Stage
```bash
# Extract features
python -m modular_pipeline.extract ./frames ./output --extractor aliked --max-keypoints 8000 --subpixel-refinement

# Retrieve pairs (parallel with extract)
python -m modular_pipeline.retrieve ./frames ./output --backend netvlad --num-neighbors 50

# Match features
python -m modular_pipeline.match ./output/features.h5 ./output --matcher lightglue --fallback-matcher mast3r --min-matches 15

# SfM
python -m modular_pipeline.sfm ./output --mapper glomap --camera-model OPENCV --tri-max-error 4.0
```

### Single Module
```bash
# Mask
python -m modular_pipeline.prepare.masking ./frames ./output --model sam3 --remove-prompts "person,tripod"

# Reframe
python -m modular_pipeline.prepare.reframe ./frames ./output --pattern ring12 --fov-h 90

# Extract
python -m modular_pipeline.extract ./frames ./output --extractor aliked --max-keypoints 8000

# Ingest
python -m modular_pipeline.ingest.extract ./video.mp4 ./output --target-fps 2 --filter-quality
```

### Resume
```bash
python -m modular_pipeline.pipeline ./project --resume
```

### In-Pipeline Splatting & Mesh
```bash
# Full pipeline with splatting
python -m modular_pipeline.pipeline ./project --preset mavic_3_pro --splat-training --splat-iterations 40000

# Extract mesh from splat
python -m modular_pipeline.output.mesh ./project/output/splat/trained.ply ./project/output/mesh --backend pgsr --poisson-depth 11
```

---

## Quality Metrics (At Each Stage)

**Ingest:** Check quality distribution. Flag "poor" frames.

**Extract:** Min keypoints > 1000 per image. Avg ~8000. Max should be near limit.

**Match:** Success rate > 80%. Fallback pairs < 20% of total.

**SFM:**
- Registration rate > 90% (good), < 50% (broken)
- Reprojection error < 2.0 px (good), > 3.0 px (problem)
- Track length > 2.0 (typical), < 1.5 (too few matches)
- Point count: 100K-10M+ (depends on scene resolution)

---

## Troubleshooting

| Issue | Cause | Primary Fix | Secondary Fixes |
|-------|-------|------------|-----------------|
| Registration <50% | Bad extraction, wrong camera model, image quality | Try COLMAP | OPENCV model, +keypoints, check blur |
| Sparse cloud (<100K) | Too few matches, strict triangulation | Increase keypoints, lower tri_max_error | Lower min_matches, lower tri_min_angle |
| High reprojection error | Camera model wrong, blur, misalignment | Try OPENCV model | Subpixel_refinement, lower tri_max_error |
| 360° fails | Wrong rig pattern, pole artifacts | Try geodesic_20 | Check rig.json, increase geometry_aware |
| Too slow | Too many features/pairs | Reduce keypoints, use sequential | Reduce neighbors, disable fallback |
| Noisy 3D points | Weak matches, low triangulation thresholds | Increase tri_min_angle | Increase min_matches, increase tri_max_error |

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

# GPU (CUDA 11.8+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional backends (RoMa, MASt3R, SuGaR)
pip install roma mast3r sugar3d
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
├── sparse/0/               # SFM output (cameras.bin, images.bin, points3D.bin)
├── output/
│   ├── splat/             # Splatting output (trained.ply)
│   ├── mesh/              # Mesh output (model.obj, model.glb)
│   └── export/            # Export formats (various)
└── (manifest.json at each stage)
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
