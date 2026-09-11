# SimCortex v2.0.0 Docker Guide

This document explains how to build and run the **SimCortex v2.0.0** Docker image.
It covers containerized execution of the `simcortex` CLI for all four stages:
**Stage 1 (Preprocessing)**, **Stage 2 (Segmentation)**, **Stage 3 (InitSurf)**,
and **Stage 4 (Deform)**.

For detailed stage logic, expected inputs and outputs, naming conventions, and
workflow order, see the repository root `README.md`.

The Docker image provides a reproducible SimCortex runtime, including the
Python / CUDA / PyTorch / PyTorch3D / ANTsPy stack used by the project.

---

## Table of Contents

- [Overview](#overview)
- [Published Resources](#published-resources)
- [Image Tags](#image-tags)
- [Build the Image](#build-the-image)
- [Quick Validation](#quick-validation)
- [Recommended Runtime Pattern](#recommended-runtime-pattern)
- [Mounting Datasets and Outputs](#mounting-datasets-and-outputs)
- [Hydra Configuration from Docker](#hydra-configuration-from-docker)
- [Inspect Packaged Config Files](#inspect-packaged-config-files)
- [Use Your Own YAML Config](#use-your-own-yaml-config)
- [Examples by Stage](#examples-by-stage)
- [GPU Support](#gpu-support)
- [Shared Server and HPC Notes](#shared-server-and-hpc-notes)
- [Docker Hub Publication](#docker-hub-publication)
- [Apptainer / Singularity Notes](#apptainer--singularity-notes)

---

## Overview

The Docker image allows users to run SimCortex without recreating the full local
environment manually. This is useful for:

- reproducibility across workstations and servers;
- simpler setup for collaborators;
- preserving a validated CUDA / PyTorch / PyTorch3D environment;
- CLI-based workflows where datasets and outputs are mounted from the host;
- running the same four public SimCortex stages documented in the root README.

The main Docker image supports the complete four-stage workflow:

1. **Stage 1 — Preprocessing**
2. **Stage 2 — Segmentation**
3. **Stage 3 — InitSurf**
4. **Stage 4 — Deform**

Stage 1 reads existing FreeSurfer subject outputs and performs MNI152
registration with **ANTsPy**. It does not require FreeSurfer executables
inside the container. The MNI152 reference image is an external
scientific input and must be mounted into the container and supplied explicitly
with `--mni-template`.

The Docker guide complements the repository root `README.md`; it does not define
a separate workflow or naming convention.

---

## Published Resources

- **Source code:** [Neuro-iX/SimCortex](https://github.com/Neuro-iX/SimCortex)
- **Docker Hub:** [kavehmoradkhani/simcortex](https://hub.docker.com/r/kavehmoradkhani/simcortex)
- **Pre-trained models and reproducibility files:** [Zenodo](https://doi.org/10.5281/zenodo.22710781)
- **Reproducibility dataset:** [FRDR](https://doi.org/10.20383/103.01769)

---

## Image Tags

The recommended versioned release is:

```text
kavehmoradkhani/simcortex:2.0.0
```

The convenience alias:

```text
kavehmoradkhani/simcortex:latest
```

currently points to the same SimCortex v2.0.0 image.

For exact reproducibility, use the immutable registry digest:

```text
kavehmoradkhani/simcortex@sha256:b4dfdfa36a10eeedadd1eb4d62adfe751025e45336e6fdfeab717a2f6245ec25
```

The published image was built from Git revision:

```text
59f167e02ebe9680f9d9f2293827f7983d3325a3
```

Pull the versioned image with:

```bash
docker pull kavehmoradkhani/simcortex:2.0.0
```

or pin the exact published artifact:

```bash
docker pull kavehmoradkhani/simcortex@sha256:b4dfdfa36a10eeedadd1eb4d62adfe751025e45336e6fdfeab717a2f6245ec25
```

For local development builds from source, the local tag used in this guide is:

```text
simcortex:2.0.0
```

For published workflows and reproducible analyses, prefer the versioned Docker
Hub tag or the immutable digest.

---

## Build the Image

The Dockerfile intentionally separates the validated scientific runtime from the
SimCortex source tree. It first unpacks a `conda-pack` archive containing the
validated Python/CUDA dependencies, then installs the SimCortex package from the
current repository checkout with `--no-deps`. This avoids resolving a different
PyTorch/PyTorch3D stack during the Docker build.

### Prepare the packed runtime archive

The required build input is:

```text
docker/simcortex-env.tar.gz
```

This archive is intentionally ignored by Git and must not be committed. A clean
source checkout therefore needs this build prerequisite to be created before
running `docker build`.

Create it from a **dedicated Conda environment that already contains the
validated runtime stack described in the root README**. Before packing, inspect
editable installs:

```bash
python -m pip list --editable
```

Do not pack unrelated editable projects. SimCortex itself does not need to be
embedded in the archive because the Dockerfile installs the current checkout
after unpacking the runtime.

For the v2.0.0 environment, pack the runtime from the activated environment with:

```bash
ENV_PREFIX="$(python -c 'import sys; print(sys.prefix)')"

conda-pack \
  --prefix "$ENV_PREFIX" \
  --output docker/simcortex-env.tar.gz \
  --format tar.gz \
  --compress-level 4 \
  --n-threads -1 \
  --ignore-editable-packages \
  --exclude 'bin/simcortex' \
  --exclude 'lib/python3.10/site-packages/__editable__.simcortex-2.0.0.pth' \
  --exclude 'lib/python3.10/site-packages/simcortex-2.0.0.dist-info/*' \
  --force
```

The exclusion entries keep the archive dependency-only; the image receives the
release's SimCortex code from the checked-out source tree.

Optionally verify that no editable SimCortex installation was embedded:

```bash
if tar -tzf docker/simcortex-env.tar.gz | \
  grep -Eq '(^|/)bin/simcortex$|__editable__\.simcortex|simcortex-2\.0\.0\.dist-info/'; then
  echo "ERROR: editable SimCortex artifacts found in packed runtime" >&2
  exit 1
fi
```

Then build from the repository root:

```bash
docker build -f docker/Dockerfile -t simcortex:2.0.0 .
```

The resulting image bundles the validated runtime stack together with the
packaged `simcortex` CLI, ANTsPy, PyTorch, PyTorch3D, and Hydra configuration
files.

---

## Quick Validation

Show the main CLI:

```bash
docker run --rm kavehmoradkhani/simcortex:2.0.0 simcortex --help
```

Show help for all four public stages:

```bash
docker run --rm kavehmoradkhani/simcortex:2.0.0 simcortex fs-to-mni --help
docker run --rm kavehmoradkhani/simcortex:2.0.0 simcortex seg --help
docker run --rm kavehmoradkhani/simcortex:2.0.0 simcortex initsurf --help
docker run --rm kavehmoradkhani/simcortex:2.0.0 simcortex deform --help
```

Verify important Python packages:

```bash
docker run --rm kavehmoradkhani/simcortex:2.0.0 \
  python -c "import ants, torch, pytorch3d, pymeshlab, simcortex; print(torch.__version__)"
```

Verify GPU visibility:

```bash
docker run --rm --gpus all kavehmoradkhani/simcortex:2.0.0 \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

You can also run the repository smoke test:

```bash
bash scripts/docker_smoke_test.sh kavehmoradkhani/simcortex:2.0.0
```

---

## Recommended Runtime Pattern

For shared systems, prefer:

```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  [mounts] \
  kavehmoradkhani/simcortex:2.0.0 \
  <command>
```

For GPU stages, add:

```bash
--gpus all
```

For example:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex --help
```

Using the host UID/GID prevents container-created output files from being owned
by root.

---

## Mounting Datasets and Outputs

A practical host layout is:

```text
/home/<user>/datasets
/home/<user>/runs
/home/<user>/checkpoints
/home/<user>/templates
```

Mount them into predictable container paths:

```bash
-v /home/<user>/datasets:/data
-v /home/<user>/runs:/runs
-v /home/<user>/checkpoints:/checkpoints:ro
-v /home/<user>/templates:/templates:ro
```

For a multi-stage workflow, derivative roots typically appear under the mounted
dataset tree as:

```text
/data/<dataset>/derivatives/sc-preproc
/data/<dataset>/derivatives/sc-seg
/data/<dataset>/derivatives/sc-initsurf
/data/<dataset>/derivatives/sc-deform
```

Run directories can remain separate:

```text
/runs/seg
/runs/initsurf
/runs/deform
```

Keeping datasets and experiment outputs separate makes commands easier to read
and reduces accidental overwrites.

---

## Hydra Configuration from Docker

SimCortex uses Hydra configurations for Segmentation, InitSurf, and Deform.
Overrides can be passed directly through the CLI inside `docker run`.

General pattern:

```bash
docker run --rm [docker-options] kavehmoradkhani/simcortex:2.0.0 \
  simcortex <stage> <command> key=value key=value
```

Example:

```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex initsurf generate \
  dataset.split_file=/data/splits/dataset_split.csv \
  dataset.split_name=all \
  dataset.roots.HCP_YA=/data/hcpya-u100/derivatives/sc-preproc \
  dataset.seg_roots.HCP_YA=/data/hcpya-u100/derivatives/sc-seg \
  outputs.out_roots.HCP_YA=/data/hcpya-u100/derivatives/sc-initsurf \
  outputs.log_dir=/runs/initsurf/exp01/logs
```

---

## Inspect Packaged Config Files

Print the installed package location:

```bash
docker run --rm kavehmoradkhani/simcortex:2.0.0 \
  python -c "import simcortex, pathlib; print(pathlib.Path(simcortex.__file__).resolve().parent)"
```

Print an InitSurf config:

```bash
docker run --rm kavehmoradkhani/simcortex:2.0.0 \
  python -c "import simcortex, pathlib; p=pathlib.Path(simcortex.__file__).resolve().parent/'configs'/'initsurf'/'generate.yaml'; print(p.read_text())"
```

Print a Deform training config:

```bash
docker run --rm kavehmoradkhani/simcortex:2.0.0 \
  python -c "import simcortex, pathlib; p=pathlib.Path(simcortex.__file__).resolve().parent/'configs'/'deform'/'train.yaml'; print(p.read_text())"
```

Save a packaged config to the host:

```bash
mkdir -p /tmp/simcortex_cfg

docker run --rm \
  -v /tmp/simcortex_cfg:/out \
  kavehmoradkhani/simcortex:2.0.0 \
  python -c "import simcortex, pathlib; p=pathlib.Path(simcortex.__file__).resolve().parent/'configs'/'initsurf'/'generate.yaml'; open('/out/generate.yaml','w').write(p.read_text())"
```

Then edit:

```text
/tmp/simcortex_cfg/generate.yaml
```

on the host.

---

## Use Your Own YAML Config

If a stage supports a `user_config` pattern, mount the custom YAML file and pass
it explicitly.

Example:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  -v /home/<user>/myconfigs:/cfg:ro \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex deform train user_config=/cfg/train.yaml
```

If the stage does not use a `user_config` field, override individual values
directly on the CLI.

---

## Examples by Stage

The examples below use the same four-stage workflow documented in the repository
root `README.md`.

### Stage 1 — Preprocessing

Stage 1 reads an existing FreeSurfer derivatives tree and an external MNI152
reference image. The MNI template is not bundled with SimCortex, so mount it
explicitly.

```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /path/to/MNI152_T1_1mm.nii.gz:/templates/MNI152_T1_1mm.nii.gz:ro \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex fs-to-mni \
  --freesurfer-root /data/<dataset>/derivatives/freesurfer-7.4.1 \
  --out-deriv-root /data/<dataset>/derivatives/sc-preproc \
  --mni-template /templates/MNI152_T1_1mm.nii.gz \
  --transform-type affine \
  --n4 \
  --with-aparc-aseg \
  --with-filled
```

The MNI reference used for exact reproduction of previously generated SimCortex
derivatives must match the reference documented in the root README.

### Stage 2 — Segmentation train

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex seg train \
  dataset.path=/data/<dataset>/derivatives/sc-preproc \
  dataset.split_file=/data/splits/<dataset>_split.csv \
  outputs.root=/runs/seg/exp01
```

### Stage 2 — Segmentation inference

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/checkpoints:/checkpoints:ro \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex seg infer \
  dataset.path=/data/<dataset>/derivatives/sc-preproc \
  dataset.split_file=/data/splits/<dataset>_split.csv \
  dataset.split_name=test \
  model.ckpt_path=/checkpoints/seg_best_dice.pt \
  outputs.out_root=/data/<dataset>/derivatives/sc-seg
```

### Stage 3 — InitSurf

```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex initsurf generate \
  dataset.split_file=/data/splits/dataset_split.csv \
  dataset.split_name=all \
  dataset.roots.HCP_YA=/data/hcpya-u100/derivatives/sc-preproc \
  dataset.seg_roots.HCP_YA=/data/hcpya-u100/derivatives/sc-seg \
  outputs.out_roots.HCP_YA=/data/hcpya-u100/derivatives/sc-initsurf \
  outputs.log_dir=/runs/initsurf/exp01/logs
```

### Stage 4 — Deformation train

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex deform train \
  outputs.root=/runs/deform/exp01
```

### Stage 4 — Deformation inference

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/checkpoints:/checkpoints:ro \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex deform infer \
  model.ckpt_path=/checkpoints/deform_best_rmse.pth
```

### Stage 4 — Deformation evaluation

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex deform eval
```

For the full set of stage-specific Hydra overrides, use the repository root
README and the packaged configuration files.

---

## GPU Support

If Docker and the NVIDIA Container Toolkit are configured correctly on the host,
enable GPU access with:

```bash
--gpus all
```

Check visibility with:

```bash
docker run --rm --gpus all kavehmoradkhani/simcortex:2.0.0 \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Segmentation and Deform normally use GPU acceleration. Stage 1 preprocessing and
InitSurf do not require a GPU.

If the CUDA test fails, first verify the host Docker / NVIDIA runtime
configuration independently of SimCortex.

---

## Shared Server and HPC Notes

On shared systems:

- run containers with the host UID/GID when writing to shared project storage;
- set a writable `HOME`, for example `-e HOME=/tmp`;
- mount datasets and run directories explicitly;
- use read-only mounts for immutable checkpoints and templates where practical;
- avoid writing large intermediate files into the container filesystem;
- request GPUs through the site's scheduler before using `--gpus all`;
- use versioned image tags for reproducible runs.

Example permission-safe invocation:

```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /project/data:/data \
  -v /project/runs:/runs \
  kavehmoradkhani/simcortex:2.0.0 \
  simcortex --help
```

---

## Docker Hub Publication

SimCortex v2.0.0 is published on Docker Hub. Both the `2.0.0` and `latest`
tags currently resolve to:

```text
sha256:b4dfdfa36a10eeedadd1eb4d62adfe751025e45336e6fdfeab717a2f6245ec25
```

The published image was built from Git revision:

```text
59f167e02ebe9680f9d9f2293827f7983d3325a3
```

The following procedure documents how the versioned release image was built
and published.

From the repository root, attach the source revision and release metadata as
OCI labels:

```bash
REV="$(git rev-parse HEAD)"
CREATED="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

docker build \
  -f docker/Dockerfile \
  -t simcortex:2.0.0 \
  --label "org.opencontainers.image.title=SimCortex" \
  --label "org.opencontainers.image.description=SimCortex v2 cortical surface reconstruction pipeline" \
  --label "org.opencontainers.image.version=2.0.0" \
  --label "org.opencontainers.image.revision=$REV" \
  --label "org.opencontainers.image.created=$CREATED" \
  --label "org.opencontainers.image.source=https://github.com/Neuro-iX/SimCortex" \
  --label "org.opencontainers.image.url=https://github.com/Neuro-iX/SimCortex" \
  --label "org.opencontainers.image.licenses=Apache-2.0" \
  .
```

Tag the validated image for Docker Hub:

```bash
docker tag simcortex:2.0.0 kavehmoradkhani/simcortex:2.0.0
```

Authenticate and push the versioned release:

```bash
docker login
docker push kavehmoradkhani/simcortex:2.0.0
```

After validating the published versioned image, update `latest` to the same
tested image:

```bash
docker tag simcortex:2.0.0 kavehmoradkhani/simcortex:latest
docker push kavehmoradkhani/simcortex:latest
```

The versioned tag remains the primary human-readable reproducibility reference;
the registry digest is the immutable reference.

---

## Apptainer / Singularity Notes

On HPC systems where Docker is unavailable, the published Docker image can be
converted or pulled with Apptainer/Singularity according to local cluster
policy.

Example:

```bash
apptainer pull simcortex_2.0.0.sif \
  docker://kavehmoradkhani/simcortex:2.0.0
```

Run CLI help:

```bash
apptainer exec simcortex_2.0.0.sif simcortex --help
```

For GPU stages:

```bash
apptainer exec --nv simcortex_2.0.0.sif simcortex seg --help
```

Bind datasets, outputs, checkpoints, and the MNI template using the mount syntax
required by the local HPC environment.

