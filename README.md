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

## Parameter Tuning: Expert Decision Tree

### 1. REGISTRATION RATE

```
Registration Rate?
├─ < 30% (BROKEN)
│  ├─ Step 1: Try COLMAP (more robust than GLOMAP)
│  │  └─ Still broken? → Check image quality
│  ├─ Step 2: Increase feature count
│  │  ├─ --max-keypoints 16000
│  │  ├─ --subpixel-refinement
│  │  └─ --keypoint-threshold 0.001 (aggressive detection)
│  ├─ Step 3: Try different extractor
│  │  ├─ aliked → xfeat (different feature type, worth trying)
│  │  └─ aliked → superpoint (more conservative)
│  ├─ Step 4: Check camera model
│  │  ├─ If DJI/phone → --camera-model OPENCV
│  │  ├─ If 360° reframed → --camera-model PINHOLE
│  │  └─ Always → --refine-intrinsics
│  ├─ Step 5: Relax match constraints
│  │  ├─ --min-matches 10 (down from 15)
│  │  ├─ --fallback-threshold 30 (retry sooner)
│  │  └─ --max-pairs-fallback 1000 (retry more pairs)
│  ├─ Step 6: Image quality check
│  │  ├─ python -m modular_pipeline.ingest.quality ./images
│  │  ├─ High blur? → --blur-threshold 50 (permissive)
│  │  ├─ Dark images? → Check lighting, rescan
│  │  └─ Motion blur? → Too many frames extracted, increase --skip-frames
│  └─ Still broken? → Data problem (motion, texture-less, incorrect geometry)
│
├─ 30-50% (WEAK)
│  ├─ PRIMARY: Increase features
│  │  └─ --max-keypoints 16000 --subpixel-refinement
│  ├─ SECONDARY: Improve matching
│  │  ├─ --fallback-threshold 75 (more fallback attempts)
│  │  ├─ --num-neighbors 75 (NetVLAD, more candidates)
│  │  └─ --min-matches 10 (permissive)
│  ├─ TERTIARY: Relax SfM
│  │  ├─ --mapper colmap (10x slower but more robust)
│  │  ├─ --tri-max-error 8.0 (accept marginal points)
│  │  └─ --tri-min-angle 0.5 (accept near-parallel rays)
│  └─ LAST: Try different extraction
│      └─ --extractor xfeat or superpoint
│
├─ 50-80% (MARGINAL)
│  ├─ Step 1: Optimize features
│  │  ├─ --max-keypoints 12000 (balance: 8k too few, 16k too slow)
│  │  ├─ --nms-radius 4 (spread features across image)
│  │  └─ --subpixel-refinement (if CPU time allows)
│  ├─ Step 2: Improve matching pairs
│  │  ├─ --retrieval hybrid (if video → sequential + NetVLAD)
│  │  ├─ --num-neighbors 75
│  │  └─ --filter-threshold 0.05 (stricter matches)
│  ├─ Step 3: Better SfM initialization
│  │  ├─ Try --mapper colmap (slower but better at weak cases)
│  │  ├─ --camera-mode per_folder (if images from multiple cameras)
│  │  └─ --run-bundle-adjustment True (final refinement)
│  └─ Step 4: Relax triangulation
│      ├─ --tri-max-error 6.0
│      ├─ --tri-min-angle 1.0
│      └─ --tri-max-dist 200.0 (if outdoor/large scene)
│
├─ 80-90% (GOOD)
│  ├─ No action needed, acceptable
│  ├─ Optional: --mapper colmap for final polish
│  └─ Or: Keep GLOMAP, it's fast enough
│
└─ > 90% (EXCELLENT)
   └─ Stop tuning registration. Focus on cloud density/quality.
```

### 2. POINT CLOUD DENSITY

```
3D Point Count?
├─ < 50K (SPARSE)
│  ├─ PRIMARY: More features per image
│  │  ├─ --max-keypoints 16000 (double from default)
│  │  ├─ --nms-radius 1 (cluster features tighter)
│  │  ├─ --remove-borders 0 (use edge pixels)
│  │  └─ --keypoint-threshold 0.001 (aggressive detection)
│  ├─ SECONDARY: More matches per pair
│  │  ├─ --min-matches 10 (accept weaker pairs)
│  │  ├─ --filter-threshold 0.05 (permissive)
│  │  └─ --fallback-threshold 30 (retry sooner with MASt3R)
│  ├─ TERTIARY: Relax triangulation
│  │  ├─ --tri-max-error 8.0 (accept marginal points)
│  │  ├─ --tri-min-angle 0.5 (accept near-parallel)
│  │  └─ --run-point-triangulator True (if False, try True)
│  └─ LAST: Try dense matcher
│      └─ --fallback-matcher mast3r (dense 3D-aware)
│
├─ 50K-500K (NORMAL)
│  └─ Good range for most scenes. Stop.
│
├─ 500K-5M (RICH)
│  ├─ Acceptable for high-res captures
│  ├─ If too slow in downstream (Splat, Mesh):
│  │  ├─ --tri-max-error 2.0 (prune outliers)
│  │  ├─ --tri-min-angle 2.0 (only confident points)
│  │  └─ Optionally subsample point cloud
│  └─ Else: Keep it, denser is generally better
│
└─ > 5M (EXTREMELY DENSE)
   ├─ Check if you really need this many points
   ├─ Memory/processing bottleneck downstream?
   │  ├─ --tri-max-error 1.0 (strict pruning)
   │  ├─ --tri-min-angle 3.0 (only high-confidence)
   │  └─ --min-matches 25 (only strong pairs)
   └─ If intentional (high-res, close-range): Keep it
```

### 3. REPROJECTION ERROR & REGISTRATION QUALITY

```
Mean Reprojection Error?
├─ < 1.0 px (EXCELLENT)
│  └─ Perfect. Stop.
│
├─ 1.0-2.0 px (GOOD)
│  └─ Normal range. Acceptable.
│
├─ 2.0-3.0 px (MARGINAL)
│  ├─ Camera model issue?
│  │  ├─ If known distortion → --camera-model OPENCV
│  │  ├─ Else → --refine-intrinsics True (optimize focal length, principal point)
│  │  └─ Try: --subpixel-refinement
│  ├─ Image quality issue?
│  │  ├─ Blur: Check blur_threshold, rescan video
│  │  ├─ Motion: If video, reduce --target-fps or increase --skip-frames
│  │  └─ Focus: Check video for out-of-focus frames
│  └─ Feature issue?
│      ├─ --subpixel-refinement (refine to sub-pixel accuracy)
│      └─ --remove-borders 2 (use more image area)
│
└─ > 3.0 px (POOR)
   ├─ Step 1: Camera model
   │  └─ --camera-model OPENCV --refine-intrinsics
   ├─ Step 2: Sub-pixel refinement
   │  └─ --subpixel-refinement
   ├─ Step 3: Strict triangulation
   │  ├─ --tri-max-error 1.0 (reject bad points)
   │  └─ --tri-min-angle 2.5
   ├─ Step 4: Tighter matching
   │  ├─ --min-matches 25
   │  └─ --filter-threshold 0.05
   └─ Step 5: Check images for corruption
      └─ python -m modular_pipeline.ingest.quality ./images
```

### 4. POINT CLOUD QUALITY (Artifacts, Noise, Speckles)

```
Cloud has artifacts/noise?
├─ Floating points far from scene?
│  ├─ PRIMARY: Increase triangulation angle
│  │  └─ --tri-min-angle 3.0 (only confident triangulations)
│  ├─ SECONDARY: Increase reprojection error threshold
│  │  └─ --tri-max-error 2.0 (reject outlier triangulations)
│  ├─ TERTIARY: Stricter matching
│  │  ├─ --min-matches 25
│  │  └─ --filter-threshold 0.05 (low confidence threshold)
│  └─ LAST: Disable fallback matcher
│      └─ --max-pairs-fallback 0 (only LightGlue)
│
├─ Speckles in low-texture areas?
│  ├─ These are false matches from weak features
│  ├─ PRIMARY: Stricter detection
│  │  ├─ --keypoint-threshold 0.01 (high confidence features)
│  │  └─ --min-matches 25
│  ├─ SECONDARY: Increase min angle
│  │  └─ --tri-min-angle 2.0
│  └─ LAST: Increase match confidence
│      └─ --filter-threshold 0.05
│
├─ Ghosting (duplicate surfaces at slight offset)?
│  ├─ Usually indicates lens distortion or mis-calibration
│  ├─ PRIMARY: Strict camera model
│  │  └─ --camera-model OPENCV --refine-intrinsics
│  ├─ SECONDARY: Sub-pixel refinement
│  │  └─ --subpixel-refinement
│  └─ TERTIARY: Tighter triangulation
│      ├─ --tri-max-error 1.0
│      └─ --tri-min-angle 2.5
│
└─ If using masking: Mask boundaries bleeding artifacts?
   ├─ PRIMARY: Increase mask confidence
   │  └─ --confidence-threshold 0.85
   ├─ SECONDARY: Enable temporal consistency
   │  └─ --use-temporal-consistency True --temporal-window 7
   └─ TERTIARY: Expand safe regions
       └─ --max-mask-area-ratio 0.3 (don't remove >30% of image)
```

### 5. PROCESSING TIME & PERFORMANCE

```
Too slow? (>4 hours for 6900 images)
├─ Where is the bottleneck?
│  ├─ Extract stage slow?
│  │  ├─ PRIMARY: Reduce features
│  │  │  └─ --max-keypoints 4000 (50% speedup)
│  │  ├─ SECONDARY: Use faster extractor
│  │  │  └─ --extractor xfeat (3x faster than ALIKED)
│  │  └─ TERTIARY: Skip sub-pixel
│  │      └─ --subpixel-refinement False
│  │
│  ├─ Match stage slow?
│  │  ├─ PRIMARY: Reduce pair candidates
│  │  │  ├─ --num-neighbors 30 (down from 50)
│  │  │  ├─ --max-pairs-fallback 0 (skip MASt3R)
│  │  │  └─ --retrieval sequential (if video, much faster)
│  │  ├─ SECONDARY: Fewer keypoints
│  │  │  └─ --max-keypoints 4000
│  │  └─ TERTIARY: Use faster matcher
│  │      └─ --matcher superglue (faster than lightglue, lower quality)
│  │
│  ├─ SfM stage slow?
│  │  ├─ PRIMARY: GLOMAP is default (already fast)
│  │  ├─ SECONDARY: Reduce keypoints before SfM
│  │  │  └─ --max-keypoints 4000
│  │  └─ TERTIARY: Reduce refinement
│  │      └─ --refine-intrinsics False (skip intrinsic optimization)
│  │
│  ├─ Retrieve stage slow? (rare, usually <1 minute)
│  │  ├─ PRIMARY: Reduce neighbors
│  │  │  └─ --num-neighbors 30
│  │  ├─ SECONDARY: Skip duplicates filtering
│  │  │  └─ --filter-duplicates False
│  │  └─ TERTIARY: Lower VLAD dimensionality
│  │      └─ --vlad-dim 2048 (from 4096)
│  │
│  └─ Overall too slow?
│      ├─ Reduce --target-fps (extract fewer frames)
│      ├─ Reduce --max-keypoints to 4000
│      ├─ Use --retrieval sequential (if video)
│      ├─ Set --max-pairs-fallback 0
│      └─ Use --extractor xfeat (3x faster)
│
├─ Memory issues?
│  ├─ OOM during Extract?
│  │  └─ --resize-max 1200 (downscale images)
│  ├─ OOM during Match?
│  │  ├─ --max-keypoints 4000 (smaller feature sets)
│  │  └─ Reduce dataset size (--max-frames, --skip-frames)
│  └─ OOM during SfM?
│      ├─ --max-num-tracks 3_000_000 (limit 3D points)
│      └─ Process in chunks (run separately on image subsets)
│
└─ GPU not being used?
   ├─ Check: CUDA available? nvidia-smi
   ├─ Extract: Should auto-use GPU (ALIKED, XFeat)
   ├─ Match: Uses GPU for LightGlue
   ├─ Reframe: --projection-backend torch_gpu (explicitly use GPU)
   └─ If still slow: CPU bottleneck elsewhere (I/O, retrieval)
```

### 6. DATA-SPECIFIC SCENARIOS

```
What type of data do you have?

├─ VIDEO (Continuous Frames)
│  ├─ Use: --retrieval sequential (temporal neighbors, fast)
│  ├─ Set: --num-neighbors 20-30 (frame window)
│  ├─ Skip: --skip-frames 1-2 (extract every Nth frame)
│  └─ Watch: Motion blur (blur_threshold might filter too much)
│
├─ DRONE SURVEY (Systematic Coverage)
│  ├─ Expect: Good registration (drone stable, wide baselines)
│  ├─ Use: --retrieval netvlad (unordered, but correlated)
│  ├─ Set: --num-neighbors 50
│  ├─ Camera: --camera-model OPENCV (if DJI)
│  └─ Pattern: Regular grid = good feature distribution
│
├─ HANDHELD / WALK-AROUND
│  ├─ Expect: Variable baselines, frame blur, motion
│  ├─ Use: --retrieval hybrid (sequential + NetVLAD)
│  ├─ Set: --target-fps 2 (subsample to reduce blur)
│  ├─ Use: --masking (remove hand/body shadows)
│  └─ Features: May need --max-keypoints 16000 (fewer reliable features)
│
├─ 360° FOOTAGE
│  ├─ MUST: --rig-pattern ring12 (or geodesic_20)
│  ├─ MUST: --projection-backend torch_gpu (fast reframing)
│  ├─ Use: --masking (remove tripod/selfie stick)
│  ├─ Enable: --geometry-aware True (pole handling)
│  └─ Set: --fov-h 90 --fov-v 60
│
├─ CLOSE-RANGE / ARCHITECTURAL
│  ├─ Enable: --subpixel-refinement (precision matters)
│  ├─ Use: --max-keypoints 16000 (high detail)
│  ├─ Triangulation: --tri-max-error 1.0 (strict)
│  ├─ Camera: --camera-model OPENCV (phones/small sensors have distortion)
│  └─ Watch: Out-of-focus regions (use masking)
│
├─ OUTDOOR / LANDSCAPE
│  ├─ Large scale: --tri-max-dist 500.0 (no distance limit)
│  ├─ Expect: Lighting variation (may affect matching)
│  ├─ Use: --max-keypoints 8000 (sufficient for texture)
│  ├─ If sparse texture: --max-keypoints 12000
│  └─ Watch: Sky/water (no features, don't match)
│
├─ INDOOR / CONFINED SPACE
│  ├─ Small baseline: --tri-min-angle 0.5 (accept small angles)
│  ├─ Watch: Repetitive texture (false matches)
│  │  └─ Use: --min-matches 25 (strict matching)
│  ├─ Lighting: Variable (may need --min-quality-score 0.3)
│  ├─ Masking: Use --masking to remove dynamic objects
│  └─ Dense: --max-keypoints 12000-16000
│
├─ NIGHT / LOW-LIGHT
│  ├─ Challenging! Expect: Poor registration
│  ├─ Increase: --keypoint-threshold 0.001 (aggressive detection)
│  ├─ Increase: --max-keypoints 16000
│  ├─ Relax: --min-matches 10
│  ├─ Ingest: --min-quality-score 0.2 (keep marginal frames)
│  └─ Consider: Adding more light source or higher ISO video
│
└─ TEXTURE-LESS (Walls, Plain Surfaces)
   ├─ HARD! Expect: Poor matching
   ├─ Increase: --keypoint-threshold 0.001
   ├─ Increase: --max-keypoints 16000
   ├─ Relax: --min-matches 10
   ├─ Retrieval: --num-neighbors 100 (more candidates)
   └─ Consider: Adding contrast (lighting) or change capture angle
```

### 7. SPECIFIC SYMPTOMS

```
MATCHING FAILURES
├─ Few pairs match (< 20% success)?
│  ├─ Increase candidates: --num-neighbors 100
│  ├─ Relax threshold: --filter-threshold 0.1
│  ├─ Lower bar: --min-matches 10
│  └─ Fallback sooner: --fallback-threshold 20
│
├─ All pairs match but few points?
│  ├─ Too strict on triangulation
│  │  ├─ --tri-max-error 8.0
│  │  └─ --tri-min-angle 0.5
│  └─ Or too few matches per pair
│      └─ --min-matches 5
│
└─ Random/inconsistent matches?
   ├─ Weak features: --keypoint-threshold 0.01
   ├─ Weak matches: --filter-threshold 0.05
   └─ Dataset issue: Check for duplicates, very similar frames

FEATURE EXTRACTION ISSUES
├─ Uneven feature distribution (clustered in one region)?
│  ├─ Increase NMS: --nms-radius 5
│  ├─ Use more borders: --remove-borders 0
│  └─ Different extractor: --extractor disk (position-diverse)
│
├─ Too few features (< 1000 per image)?
│  ├─ Lower threshold: --keypoint-threshold 0.001
│  ├─ Extract more: --max-keypoints 16000
│  └─ Different extractor: --extractor xfeat
│
└─ Too many duplicates (similar features)?
   ├─ Increase NMS: --nms-radius 5
   └─ Stricter threshold: --keypoint-threshold 0.01

RECONSTRUCTION ISSUES
├─ Drifting/distorted model?
│  ├─ Camera model: --camera-model OPENCV --refine-intrinsics
│  ├─ SfM backend: Try --mapper colmap
│  └─ Bundle adjustment: --run-bundle-adjustment True
│
├─ Model offset/translated?
│  ├─ Usually OK (arbitrary coordinate frame)
│  ├─ Unless you need georeferencing (use camera poses)
│  └─ Check registration rate > 90%
│
└─ Holes/gaps in model?
   ├─ Matching: Increase --num-neighbors, --fallback-threshold
   ├─ Features: Increase --max-keypoints
   └─ Baselines: Is capture coverage continuous?
```

### 8. ITERATIVE TUNING ORDER

**When starting with defaults and nothing works:**

```
Step 1: Diagnose
├─ Check registration rate
├─ Check reprojection error
├─ Check point count
└─ Check image quality (python -m modular_pipeline.ingest.quality)

Step 2: If registration < 50%
├─ Increase features: --max-keypoints 16000
├─ Try different SfM: --mapper colmap
└─ Check camera model: --camera-model OPENCV

Step 3: If registration 50-90%
├─ Optimize features: --max-keypoints 12000
├─ Improve matching: --fallback-threshold 75 --num-neighbors 75
└─ Relax triangulation: --tri-max-error 6.0 --tri-min-angle 1.0

Step 4: If registration > 90% but sparse
├─ Increase features: --max-keypoints 16000
├─ Lower match bar: --min-matches 10
└─ Relax triangulation: --tri-max-error 8.0 --tri-min-angle 0.5

Step 5: If registration > 90% but noisy
├─ Increase match confidence: --min-matches 25 --filter-threshold 0.05
├─ Strict triangulation: --tri-min-angle 2.5 --tri-max-error 2.0
└─ Strict detection: --keypoint-threshold 0.01

Step 6: If too slow
├─ Reduce features: --max-keypoints 4000
├─ Reduce pairs: --num-neighbors 30 --max-pairs-fallback 0
├─ Faster extractor: --extractor xfeat
└─ Faster retrieval: --retrieval sequential (if video)
```

---

## Equipment & Workflow Reference (Your Setup)

### Equipment Profiles & Presets

#### DJI Mavic 3 Pro (Drone)
**Best for:** Aerial surveys, wide coverage, systematic grid patterns

**Preset:** `--preset mavic_3_pro`

**Settings:**
```bash
--camera-model OPENCV              # Strong barrel distortion on this sensor
--refine-intrinsics                # Optimize focal length + principal point
--max-keypoints 8000               # Good balance for aerial imagery
--subpixel-refinement              # Optional: adds precision
--num-neighbors 50                 # NetVLAD: typical drone survey spacing
--retrieval netvlad                # Unordered but spatially correlated
```

**Typical workflow:**
```bash
# From OSV (native DJI format)
python -m modular_pipeline.pipeline ./statue_drone --preset mavic_3_pro

# From extracted frames
python -m modular_pipeline.pipeline ./statue_drone --preset mavic_3_pro
```

**Expect:** 90%+ registration rate, dense cloud, clean reconstruction.

---

#### DJI Mavic Air 2S (Drone)
**Best for:** Similar to Mavic 3 Pro but slightly older sensor characteristics

**Preset:** `--preset mavic_air_2s`

**Settings:**
```bash
--camera-model OPENCV              # Similar distortion to Mavic 3 Pro
--refine-intrinsics                # Always enable for DJI
--max-keypoints 8000               # Standard
--num-neighbors 50
--retrieval netvlad
```

**Workflow:** Same as Mavic 3 Pro (preset handles it).

---

#### DJI Osmo Action 4 (360° Camera)
**Best for:** Comprehensive 360° capture, environment scanning

**Preset:** `--preset osmo_360`

**Critical settings (must use):**
```bash
--rig-pattern ring12               # MUST: reframe to pinhole rig
--projection-backend torch_gpu     # MUST: GPU acceleration for speed
--masking person,tripod,selfie stick  # Remove visible stand/operator
--geometry-aware True              # Pole region artifact handling
--fov-h 90 --fov-v 60             # Standard 90° FOV per camera
```

**Full workflow:**
```bash
# Step 1: Extract frames from video
python -m modular_pipeline.ingest.extract \
  ./osmo360_video.mov \
  ./statue_360 \
  --target-fps 2 \
  --filter-quality

# Step 2: Full pipeline with reframe + mask + reconstruct
python -m modular_pipeline.pipeline ./statue_360 \
  --preset osmo_360 \
  --masking person,tripod \
  --rig-pattern ring12 \
  --geometry-aware
```

**Output:** `statue_360/sparse/0/` with rig.json for COLMAP multi-camera handling.

**Expect:** Good registration after reframing, temporal smoothing helps consistency.

---

#### Fuji XPro2 (DSLR with Interchangeable Lenses)
**Best for:** High-quality detail capture, multiple perspectives, variable focal lengths

**Preset:** None (use `standard`), but customize by lens

**Settings (per lens):**
```bash
# Wide lens (18-55mm equivalent)
--camera-model SIMPLE_RADIAL       # Fuji lenses: minimal distortion
--max-keypoints 8000-12000         # Wider FOV = more texture
--subpixel-refinement              # DSLR detail matters
--refine-intrinsics                # Optimize for this specific lens

# Standard lens (35mm equivalent)
--camera-model SIMPLE_RADIAL
--max-keypoints 8000
--subpixel-refinement

# Telephoto (50mm+)
--camera-model SIMPLE_RADIAL
--max-keypoints 6000               # Narrow FOV = fewer features per image
--min-matches 10                   # Accept weaker matches (tighter baseline)
```

**Workflow (mixed lenses):**
```bash
# If all photos from same lens
python -m modular_pipeline.pipeline ./statue_fuji \
  --camera-model SIMPLE_RADIAL \
  --max-keypoints 10000 \
  --subpixel-refinement

# If mixed lenses (wide + standard + tele)
python -m modular_pipeline.pipeline ./statue_fuji \
  --camera-mode per_folder \
  --camera-model SIMPLE_RADIAL \
  --max-keypoints 10000 \
  --subpixel-refinement \
  --run-bundle-adjustment True      # Critical for multi-focal-length
```

**Expect:** High quality, but variable registration depending on lens diversity.

**Tips:**
- Organize by lens folder for `--camera-mode per_folder`
- EXIF includes focal length (pipeline uses it)
- Telephoto photos may have fewer matches (wider baselines)

---

#### iPhone 16 Pro Max (Phone Camera)
**Best for:** Quick supplemental capture, high resolution, computational photography

**Settings:**
```bash
--camera-model OPENCV              # Phone sensors have barrel distortion
--refine-intrinsics                # Optimize distortion model
--max-keypoints 8000-12000         # High-res 12MP sensor
--subpixel-refinement              # Precision helps
--resize-max 2048                  # Don't downscale too much (keep detail)
```

**Workflow:**
```bash
# From Photos app export (assumes RAW or high-quality JPEG)
python -m modular_pipeline.pipeline ./statue_iphone \
  --camera-model OPENCV \
  --max-keypoints 10000 \
  --subpixel-refinement \
  --refine-intrinsics
```

**Watch out:**
- Auto-focus can vary between photos (focus breathing)
- Computational photography may introduce artifacts
- ISO variation between photos is normal

---

#### iPad Pro 2024 (Tablet Camera)
**Best for:** Wide FOV supplemental capture, panoramic

**Settings:**
```bash
--camera-model OPENCV              # Similar distortion to iPhone
--refine-intrinsics
--max-keypoints 8000
--resize-max 1920                  # Smaller sensor than iPhone
```

**Workflow:** Similar to iPhone.

**Note:** Wider FOV than iPhone, fewer reliable features per image.

---

### Common Single-Source Workflows

#### Scenario A: All Drone (Mavic 3 Pro)
**Your data:** OSV file from drone + extracted frames

```bash
# Option 1: Direct from OSV video
python -m modular_pipeline.pipeline ./statue_drone --preset mavic_3_pro

# Option 2: From already-extracted frames (if pre-processed)
python -m modular_pipeline.pipeline ./statue_drone --preset mavic_3_pro
```

**Expected outcome:**
- Registration: 95%+
- Point count: 500K-2M
- Time: ~2-3 hours (6900 images)
- Quality: Excellent clean reconstruction

---

#### Scenario B: All 360° (Osmo Action 4)
**Your data:** MOV file from 360° camera

```bash
# Full workflow (ingest + reframe + mask + reconstruct)
python -m modular_pipeline.ingest.extract \
  ./video.mov \
  ./statue_360 \
  --target-fps 2

python -m modular_pipeline.pipeline ./statue_360 --preset osmo_360
```

**Expected outcome:**
- Registration: 85-95% (reframing helps significantly)
- Point count: 300K-1M
- Time: ~2-3 hours
- Quality: Good (pole/tripod removed)

---

#### Scenario C: All DSLR (Fuji XPro2)
**Your data:** 300 JPEG files from multiple lenses

```bash
python -m modular_pipeline.pipeline ./statue_fuji \
  --camera-model SIMPLE_RADIAL \
  --max-keypoints 10000 \
  --subpixel-refinement \
  --run-bundle-adjustment True
```

**Expected outcome:**
- Registration: 80-95% (depends on coverage)
- Point count: 200K-800K (fewer images = fewer points)
- Time: ~30-60 minutes
- Quality: Very high (DSLR detail)

---

### Your Workflow: Roadside Statues (Mixed Data)

**Setup:** Drone (5 min video) + 360° (5 min video) + DSLR Fuji (300 photos)

**Decision:** Process separately, then compare or combine.

#### Option 1: Separate Reconstructions (Recommended for testing)

```bash
# ============ DRONE PROCESSING ============
# Extract drone video
python -m modular_pipeline.ingest.extract \
  ./drone.osv \
  ./statue_drone_raw \
  --target-fps 2 \
  --filter-quality

# Full drone pipeline
python -m modular_pipeline.pipeline ./statue_drone_raw \
  --preset mavic_3_pro \
  --output-splat                           # Optional: train splat
  --splat-iterations 30000

# ============ 360° PROCESSING ============
# Extract 360° video
python -m modular_pipeline.ingest.extract \
  ./osmo360.mov \
  ./statue_360_raw \
  --target-fps 2 \
  --filter-quality

# Full 360° pipeline
python -m modular_pipeline.pipeline ./statue_360_raw \
  --preset osmo_360 \
  --masking person,tripod,selfie_stick

# ============ DSLR PROCESSING ============
python -m modular_pipeline.pipeline ./statue_fuji_raw \
  --camera-model SIMPLE_RADIAL \
  --max-keypoints 10000 \
  --subpixel-refinement \
  --run-bundle-adjustment True

# ============ INSPECT RESULTS ============
# Check each:
# statue_drone_raw/sparse/0/ → registration rate, point count
# statue_360_raw/sparse/0/   → registration rate, point count
# statue_fuji_raw/sparse/0/  → registration rate, point count
```

**Then decide:** Which is best? Combine them? Use drone + DSLR for detail?

---

#### Option 2: Combined Reconstruction (Advanced)

**Goal:** Single 3D model using all data sources

**Warning:** This is complex; only if individual reconstructions are all good (>90% registration).

```bash
# Combine image sources into single folder
mkdir statue_combined
cp statue_drone_raw/frames/* statue_combined/
cp statue_360_raw/reframed/rig_views/* statue_combined/  # Post-reframed
cp statue_fuji_raw/frames/* statue_combined/

# Run single extraction + matching + SfM
python -m modular_pipeline.extract ./statue_combined ./output \
  --max-keypoints 8000

python -m modular_pipeline.match ./output/features.h5 ./output \
  --retrieval netvlad \
  --num-neighbors 100              # More candidates (mixed sensors)

python -m modular_pipeline.sfm ./output \
  --camera-model SIMPLE_RADIAL \
  --camera-mode per_folder \
  --run-bundle-adjustment True
```

**Tradeoff:**
- ✅ Single unified reconstruction
- ❌ Complex multi-camera bundle adjustment
- ❌ Mixed image qualities may cause issues
- ⚠️ Only attempt if individual sources all work

---

#### Option 3: Drone + DSLR (Practical Hybrid)

**Best approach for your statues:** High-res context (drone) + detail (DSLR)

```bash
# Organize data
mkdir statue_hybrid
mkdir statue_hybrid/drone_frames    # From extracted drone video
mkdir statue_hybrid/dslr_frames     # Your 300 Fuji photos

cp statue_drone_raw/frames/* statue_hybrid/drone_frames/
cp statue_fuji_raw/frames/* statue_hybrid/dslr_frames/

# Run combined pipeline with per-folder camera mode
python -m modular_pipeline.extract ./statue_hybrid ./output \
  --max-keypoints 8000

python -m modular_pipeline.retrieve ./statue_hybrid ./output \
  --backend hybrid                  # Sequential (drone) + NetVLAD (DSLR)
  --num-neighbors 50

python -m modular_pipeline.match ./output/features.h5 ./output \
  --retrieval hybrid \
  --num-neighbors 50

python -m modular_pipeline.sfm ./output \
  --camera-model OPENCV            # Drone (OPENCV) dominates
  --camera-mode per_folder \
  --run-bundle-adjustment True

# Train splat
python -m modular_pipeline.output.splat ./output \
  --backend gsplat \
  --num-iterations 40000 \
  --sh-degree 3
```

**Result:**
- Drone provides coverage, DSLR provides detail
- Temporal consistency (drone video) + spatial detail (DSLR stills)
- ~1M-3M 3D points
- Good splat training data

---

### Quick Decision: Which Approach for Your Project?

```
Do you want to test each source individually?
├─ Yes → Option 1 (Separate, compare results)
└─ No → Do drone + DSLR coverage + detail matter equally?
   ├─ Yes → Option 3 (Hybrid: best practical approach)
   └─ No → Just use best single source (probably drone or DSLR)
```

**My recommendation for roadside statues:** **Option 3 (Hybrid Drone + DSLR)**
- Drone gives you the environment (context, lighting, scale)
- DSLR gives you fine detail (statue texture, weathering, inscriptions)
- Combined → rich, detailed reconstruction perfect for Gaussian splatting

---

### Running Your First Project

**Copy-paste ready (roadside statues example):**

```bash
# Set up directories
mkdir -p roadside_statues/{drone,dslr}

# Place your files
# Put extracted drone frames in: roadside_statues/drone/
# Put your DSLR photos in: roadside_statues/dslr/

# Run the pipeline
python -m modular_pipeline.pipeline ./roadside_statues \
  --max-keypoints 8000 \
  --camera-mode per_folder \
  --retrieval hybrid \
  --num-neighbors 50 \
  --mapper glomap \
  --camera-model OPENCV \
  --refine-intrinsics \
  --run-bundle-adjustment True

# Check results
ls -la roadside_statues/sparse/0/

# Train splat
python -m modular_pipeline.output.splat ./roadside_statues \
  --backend gsplat \
  --num-iterations 40000 \
  --sh-degree 3

# Final model in: roadside_statues/output/splat/trained.ply
```

**Time estimate:**
- Extract: 10 min
- Match: 20 min
- SfM: 5 min
- Splat training: 20 min
- **Total: ~1 hour**

---



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
