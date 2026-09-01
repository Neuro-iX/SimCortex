# SimCortex Evaluation

This directory contains the public evaluation pipeline for SimCortex cortical-surface reconstruction.

The evaluator is intentionally **SimCortex-only**. It validates and evaluates SimCortex predictions; it is not a multi-method benchmarking or ranking framework.

The evaluation pipeline operates on the four reconstructed cortical surfaces:

- `lh_white`
- `lh_pial`
- `rh_white`
- `rh_pial`

and reports geometric reconstruction accuracy, self-intersections, cortical-thickness error, and inter-surface collision measurements.

---

## Evaluation workflow

The complete evaluation is orchestrated by:

```bash
evaluation/run_evaluation.sh
```

It executes seven stages in order:

1. `build_case_manifest.py`
2. `build_gt_manifest.py`
3. `build_pred_manifest.py`
4. `audit_predictions.py`
5. `evaluate_metrics.py`
6. `evaluate_collisions.py`
7. `summarize_results.py`

Inference is not performed by this evaluation directory. Generate SimCortex predictions first, for example through the main pipeline runner under `scripts/`.

---

## Required inputs

The top-level runner requires three roots:

```text
--sample-root
--pred-root
--eval-root
```

### `--sample-root`

Root containing the evaluation cases and their FreeSurfer reference data.

The case and ground-truth manifest builders use these files to identify the evaluation cohort and construct the four reference cortical surfaces.

### `--pred-root`

Root containing SimCortex deformation predictions.

The canonical session-based prediction layout is:

```text
<pred-root>/
└── <dataset>/
    └── <subject>/
        └── <session>/
            └── surfaces/
                ├── <subject>_<session>_space-native_desc-deform_hemi-L_white.surf.ply
                ├── <subject>_<session>_space-native_desc-deform_hemi-L_pial.surf.ply
                ├── <subject>_<session>_space-native_desc-deform_hemi-R_white.surf.ply
                └── <subject>_<session>_space-native_desc-deform_hemi-R_pial.surf.ply
```

The prediction manifest records the actual resolved file for each case and surface.

SimCortex native-space deformation outputs are expected to be compatible with scanner RAS coordinates. The public evaluator therefore does not perform an additional prediction-space coordinate conversion.

### `--eval-root`

Destination for manifests, reports, metrics, collision results, and final summaries.

A new evaluation typically produces:

```text
<eval-root>/
├── manifests/
├── reports/
├── metrics/
├── collisions/
└── summary/
```

---

## Running the complete evaluation

Example:

```bash
evaluation/run_evaluation.sh \
  --eval-root /path/to/evaluation_output \
  --sample-root /path/to/evaluation_sample \
  --pred-root /path/to/simcortex_predictions \
  --device cuda:0 \
  --strict
```

For CPU metric evaluation:

```bash
evaluation/run_evaluation.sh \
  --eval-root /path/to/evaluation_output \
  --sample-root /path/to/evaluation_sample \
  --pred-root /path/to/simcortex_predictions \
  --device cpu \
  --strict
```

Inspect the complete command sequence without executing it:

```bash
evaluation/run_evaluation.sh \
  --eval-root /path/to/evaluation_output \
  --sample-root /path/to/evaluation_sample \
  --pred-root /path/to/simcortex_predictions \
  --dry-run
```

Use:

```bash
evaluation/run_evaluation.sh --help
```

for the current runner options.

---

## Historical sample40 defaults

The default full-evaluation contract corresponds to the historical SimCortex sample40 evaluation:

```text
14 datasets
40 cases per dataset
560 cases total
4 surfaces per case
```

These are validation defaults, not hard requirements of the individual scientific metric implementations.

The top-level runner exposes:

```text
--expected-cases
--expected-cases-per-dataset
--expected-n-datasets
```

so the evaluation can be used with a different cohort.

For dataset subsets or more specialized runs, the Python stages can also be executed individually.

---

## Stage 1: case manifest

```text
build_case_manifest.py
```

Discovers the evaluation cases from `--sample-root` and writes the canonical case manifest.

The manifest defines the subjects and sessions that all later stages must evaluate.

This stage performs cohort discovery and integrity validation only. It does not compute surface metrics.

---

## Stage 2: ground-truth manifest

```text
build_gt_manifest.py
```

Exports the four FreeSurfer reference surfaces into the coordinate system used by the evaluation.

FreeSurfer surfaces are represented in `tkRAS`. The evaluator converts reference vertices to scanner RAS using:

```text
scannerRAS = vox2ras @ inverse(vox2ras_tkr) @ tkRAS
```

This conversion is applied to the ground truth only.

The generated ground-truth manifest records one reference mesh for every expected case and cortical surface.

---

## Stage 3: prediction manifest

```text
build_pred_manifest.py
```

Locates the four SimCortex deformation outputs for every case in the canonical case manifest.

For each prediction it records, among other metadata:

```text
method = SimCortex
surface
pred_path
status
selected_variant = space-native_desc-deform
raw_format = ply
raw_space_assumption = native_RAS_scannerRAS_compatible
conversion_required = none
```

Only valid `OK` prediction rows are consumed by the metric and collision stages.

---

## Stage 4: prediction audit

```text
audit_predictions.py
```

Checks the prediction manifest and the resolved meshes before expensive scientific evaluation.

The audit includes checks for:

- missing surfaces;
- unreadable meshes;
- non-finite vertices;
- invalid or empty mesh geometry;
- vertex and face counts;
- manifest consistency;
- gross prediction-versus-ground-truth centroid, bounding-box, and extent disagreement.

This stage does not transform prediction coordinates and does not modify the meshes.

Alignment sanity warnings are diagnostic unless the corresponding strict behavior is explicitly requested by the stage.

---

## Stage 5: geometric metrics

```text
evaluate_metrics.py
```

The scientific metric implementation is contained in:

```text
metrics_core.py
```

The core implementation is kept separate from the orchestration code so the metric definitions remain stable.

### ASSD

`ASSD_mm` is the symmetric average sampled point-to-triangle surface distance.

Points sampled from the prediction are measured against the ground-truth mesh, and points sampled from the ground truth are measured against the prediction mesh. The two directions are combined symmetrically.

Units are millimeters.

### HD90

`HD90_mm` is the symmetric 90th-percentile sampled point-to-triangle surface distance.

It provides a tail-distance measure that is less sensitive to a very small number of extreme samples than the maximum Hausdorff distance.

Units are millimeters.

### ChamferPCL1

`ChamferPCL1_mm` is the symmetric mean nearest-neighbor distance between sampled point clouds from the prediction and ground-truth surfaces.

It is a sampled point-cloud metric and is distinct from the point-to-triangle ASSD calculation.

Units are millimeters.

### SIF

`SIF_pct` is the percentage of predicted mesh faces identified as self-intersecting by the PyMeshLab self-intersection procedure used by the frozen metric implementation.

It is reported as a percentage of predicted faces.

### Thickness

Thickness evaluation is computed separately for the left and right hemispheres.

For each hemisphere, the evaluator estimates the mean white-to-pial separation using sampled nearest-surface distances.

The public output includes the prediction thickness estimate, the corresponding ground-truth estimate, and their absolute error.

This measurement is a nearest-surface thickness estimate. It is **not** a vertex-correspondence cortical-thickness measurement.

The final case summary reports:

```text
ThicknessAbsErr_mm
```

as the mean of the left- and right-hemisphere thickness absolute errors.

### Metric sampling defaults

The historical scientific defaults are:

```text
surface samples:    150000
thickness samples:   50000
seed:                 12345
```

The deterministic sampling identity includes the method name `SimCortex`, dataset, subject, session, and surface or thickness stream.

---

## Stage 6: FCL collision evaluation

```text
evaluate_collisions.py
```

Collision evaluation uses `python-fcl`.

For each case, six anatomically relevant surface pairs are evaluated:

```text
white_pial_left
    lh_white vs lh_pial

white_pial_right
    rh_white vs rh_pial

pial_lr
    lh_pial vs rh_pial

white_lr
    lh_white vs rh_white

cross_lhwhite_rhpial
    lh_white vs rh_pial

cross_rhwhite_lhpial
    rh_white vs lh_pial
```

The evaluator first performs a boolean FCL collision query.

For colliding pairs it then performs a contact query and records the unique intersecting face IDs for both participating meshes.

### Contact retry ladder

The historical default maximum-contact ladder is:

```text
50000
200000
500000
```

If a contact query reaches its configured maximum, the evaluator retries the pair with the next cap.

The collision runner also preserves the historical defaults:

```text
case timeout:       600 seconds
worker memory cap:   48 GB
```

### Surface-union collision percentage

Pairwise percentages alone can count the same surface face more than once when that face intersects multiple other cortical surfaces.

The primary case-level collision measurement therefore first computes, for each of the four surfaces, the union of its unique colliding face IDs across the other three surfaces:

```text
surface_collision_pct_union =
    100 * unique_colliding_faces / total_surface_faces
```

The final case-level metric is:

```text
collision_pct_union_mean4 =
    mean(
        lh_white_collision_pct_union,
        lh_pial_collision_pct_union,
        rh_white_collision_pct_union,
        rh_pial_collision_pct_union
    )
```

The collision output also records:

```text
collision_pct_union_max4
collision_faces_union_sum4
```

The final summary exposes the mean-four metric as:

```text
CollisionPctUnion_mean4
```

---

## Stage 7: result aggregation

```text
summarize_results.py
```

This stage performs aggregation only.

It does **not**:

- load meshes;
- resample surfaces;
- recompute geometric distances;
- run PyMeshLab;
- run FCL;
- alter scientific metric values.

For each case:

```text
ChamferPCL1_mm
ASSD_mm
HD90_mm
SIF_pct
```

are averaged across:

```text
lh_white
lh_pial
rh_white
rh_pial
```

`ThicknessAbsErr_mm` is the mean of the left- and right-hemisphere thickness absolute errors.

`CollisionPctUnion_mean4` is copied from the collision evaluator's case-level `collision_pct_union_mean4`.

Dataset-level and overall summaries report:

```text
count
mean
std
median
min
max
```

for each final metric.

---

## Output files

### Manifests and reports

The first four stages populate:

```text
<eval-root>/manifests/
<eval-root>/reports/
```

These contain the canonical case list, scanner-RAS ground-truth manifest, SimCortex prediction manifest, and prediction-audit outputs.

### Geometric metrics

```text
<eval-root>/metrics/
```

Important files include:

```text
surface_metrics_long.csv
pair_metrics_thickness_collisions.csv
missing_or_failed.csv
run_summary.json
```

`pair_metrics_thickness_collisions.csv` retains its historical filename for output compatibility. Collision percentages used by the final public evaluation are produced by the dedicated FCL collision evaluator.

### Collision results

```text
<eval-root>/collisions/
```

Important outputs include the pair-level FCL collision results and:

```text
collision_surface_union_case_level.csv
collision_missing_or_failed.csv
collision_run_summary.json
```

### Final summaries

```text
<eval-root>/summary/
```

contains:

```text
case_metrics.csv
by_dataset.csv
overall.csv
run_summary.json
summarize_results.log
```

`case_metrics.csv` is the most convenient case-level table.

`by_dataset.csv` contains one aggregate row per dataset.

`overall.csv` contains the aggregate SimCortex evaluation across all selected cases.

---

## Strict mode

Use:

```text
--strict
```

to make supported stages fail when validation detects scientific or structural problems rather than only recording them.

Depending on the stage, strict checks include conditions such as:

- missing or failed cases;
- unexpected case/surface counts;
- non-finite geometric metrics;
- invalid SIF results;
- FCL worker errors or timeouts;
- incomplete or non-exact collision counts;
- invalid collision percentages;
- inconsistent final case sets.

For release evaluation, strict mode is recommended.

---

## Overwrite behavior

Use:

```text
--overwrite
```

when rerunning stages that protect existing outputs.

For a final release evaluation, using a new empty `--eval-root` is preferable because it gives a clear provenance boundary between evaluation runs.

---

## Device selection

`evaluate_metrics.py` uses PyTorch/PyTorch3D for surface sampling and distance calculations.

The top-level runner accepts:

```text
--device auto
--device cpu
--device cuda:0
```

`auto` lets the metric runner choose the available execution device.

The FCL collision evaluator is CPU/subprocess based and is independent of the PyTorch device option.

---

## Reproducibility

For reproducible comparison with an existing SimCortex evaluation:

- use the same evaluation cases;
- use the same prediction meshes;
- use the same metric sample counts;
- use the same seed;
- preserve the method identity `SimCortex`;
- use the same metric implementation and dependency versions where exact numerical reproduction is required.

The metric seed streams depend on the `SimCortex` method identity as well as case and surface identifiers.

---

## Scope

This directory evaluates **SimCortex**.

Code for converting, ranking, or comparing external reconstruction methods is intentionally outside the public evaluation pipeline. External methods with different file or coordinate conventions should first be adapted independently to a compatible evaluation representation rather than adding method-specific branches to the scientific evaluator.
