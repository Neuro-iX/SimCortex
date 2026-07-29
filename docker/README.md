# SimCortex v2.0 Docker Guide

This document explains how to build and run the **SimCortex v2.0** Docker image.
It documents containerized execution of the `simcortex` CLI for **Stage 1 (Preprocessing)**, **Stage 2 (Segmentation)**, **Stage 3 (InitSurf)**, and **Stage 4 (Deform)**. For stage logic, expected inputs and outputs, naming conventions, and workflow order, see the repository root `README.md`.

The Docker image is intended to provide a reproducible runtime for SimCortex, including the Python / CUDA / PyTorch / PyTorch3D stack used by the project.

---

## Table of Contents

- [Overview](#overview)
- [Project Resources](#project-resources)
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

The Docker image provides a validated runtime for all four SimCortex stages:

- **Stage 1 — Preprocessing**
- **Stage 2 — Segmentation**
- **Stage 3 — InitSurf**
- **Stage 4 — Deform**

Stage 1 does not execute FreeSurfer. It consumes existing FreeSurfer subject directories mounted from the host and performs N4 correction, linear registration, image resampling, and surface conversion with ANTsPy, nibabel, and the SimCortex Python implementation.

The image does not require the external NiftyReg command-line tools used by older preprocessing implementations.

The container is useful for:

- reproducibility across workstations and servers
- simpler setup for collaborators
- preserving the validated CUDA, PyTorch, and PyTorch3D stack
- CLI workflows with explicitly mounted datasets and outputs

The Docker guide complements rather than replaces the project `README.md`.

---

## Project Resources

- **Docker Hub target repository:** [kavehmoradkhani/simcortex](https://hub.docker.com/r/kavehmoradkhani/simcortex)
- **Zenodo pre-trained weights and splits:** [SimCortex v2.0: Pre-trained Models and Dataset Splits](https://zenodo.org/records/18974730)

---

## Image Tags

Examples below use the local image tag:

```text
simcortex:2.0.0
```

After publishing to Docker Hub, the tag can be:

```text
kavehmoradkhani/simcortex:2.0.0
```

Target Docker Hub repository:

```text
https://hub.docker.com/r/kavehmoradkhani/simcortex
```

Keep versioned tags even if you later publish `latest`, so users can pin an exact image for reproducibility.

---

## Build the Image

### Required environment archive

A local build requires the separately supplied `docker/simcortex-env.tar.gz` archive.
The archive is approximately 4.1 GB and is intentionally excluded from Git.

Place it at the following path before starting the build:

```text
docker/simcortex-env.tar.gz
```

Validate the archive:

```bash
test -f docker/simcortex-env.tar.gz

sha256sum docker/simcortex-env.tar.gz

tar -tzf docker/simcortex-env.tar.gz >/dev/null
echo "Docker environment archive: PASS"
```

Expected SHA256:

```text
88dc6be16aed3d756314fca7acd5af7732da2774643d11e2327f1917ee165b63  docker/simcortex-env.tar.gz
```

The validated archive contains the project runtime, including Python 3.10, PyTorch 2.1.0, CUDA 12.1, PyTorch3D 0.7.8, MONAI 1.3.2, ANTsPy 0.6.1, python-fcl 0.7.0.10, and PyMeshLab 2025.7.post1.

### Build command

From the repository root:

```bash
docker build -f docker/Dockerfile -t simcortex:2.0.0 .
```

During the build, the Dockerfile copies the environment archive, extracts it under `/opt/conda-env`, runs `conda-unpack`, copies the SimCortex repository, and installs the package without resolving dependencies again.

The resulting image exposes the packaged `simcortex` CLI and the installed Hydra configuration files.

---

## Quick Validation

After building `simcortex:2.0.0`, validate the main CLI and all four stage command groups.

### Main CLI

```bash
docker run --rm simcortex:2.0.0 simcortex --help
```

### Stage command groups

```bash
docker run --rm simcortex:2.0.0 simcortex fs-to-mni --help

docker run --rm simcortex:2.0.0 simcortex seg --help

docker run --rm simcortex:2.0.0 simcortex initsurf --help

docker run --rm simcortex:2.0.0 simcortex deform --help
```

### Python dependency imports

Verify that the principal preprocessing, segmentation, surface-processing, collision, and deep-learning dependencies import successfully:

```bash
docker run --rm simcortex:2.0.0 \
  python -c "import ants, fcl, monai, nibabel, pymeshlab, pytorch3d, simcortex, torch, torchvision; print('Dependency imports: PASS')"
```

### GPU access

On a host with the NVIDIA Container Toolkit, verify that the container can access the requested GPUs:

```bash
docker run --rm --gpus all simcortex:2.0.0 \
  python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); raise SystemExit(0 if torch.cuda.is_available() else 1)"
```

The CLI-help and dependency-import checks do not require a GPU. Segmentation and deformation workflows normally require GPU access.

---

## Recommended Runtime Pattern

On shared Linux systems, Docker writes files as `root` by default unless told otherwise. For ordinary SimCortex runs, the safest pattern is to run as the host user and mount data and outputs explicitly.

Recommended runtime block:

```bash
--user $(id -u):$(id -g) \
-e HOME=/tmp \
-e UMASK=002
```

Recommended day-to-day command pattern:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
  simcortex --help
```

This gives you:

- host-owned output files instead of root-owned files
- explicit mounted paths for datasets and run outputs
- GPU access when Docker and the NVIDIA runtime are configured correctly

If you do not need GPU access, omit `--gpus all`.

---

## Mounting Datasets and Outputs

Docker containers do not automatically see host files. Mount them explicitly with `-v`:

```bash
-v /host/path:/container/path
```

A simple and maintainable layout is:

```bash
-v /home/<user>/datasets:/data \
-v /home/<user>/runs:/runs
```

Inside the container:

- `/data` points to datasets and derivatives
- `/runs` points to experiment outputs and logs

Example host layout:

```text
/home/<user>/datasets/
  hcpya-u100/
  oasis-1/
  splits/
/home/<user>/runs/
  seg/
  initsurf/
  deform/
```

Example mounted layout inside the container:

```text
/data/hcpya-u100
/data/oasis-1
/data/splits
/runs/seg
/runs/initsurf
/runs/deform
```

Keeping datasets and run outputs mounted separately makes commands easier to read and reduces mistakes.

---

## Hydra Configuration from Docker

SimCortex uses Hydra configs. You can pass overrides directly through the CLI inside `docker run`.

General pattern:

```bash
docker run --rm [docker-options] simcortex:2.0.0 \
  simcortex <stage> <command> key=value key=value
```

Example:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
  simcortex initsurf generate \
  dataset.split_file=/data/splits/dataset_split.csv \
  dataset.split_name=all \
  dataset.roots.HCP_YA=/data/hcpya-u100/derivatives/sc-preproc \
  dataset.seg_roots.HCP_YA=/data/hcpya-u100/derivatives/sc-seg \
  outputs.out_roots.HCP_YA=/data/hcpya-u100/derivatives/sc-initsurf \
  outputs.log_dir=/runs/initsurf/exp01/logs
```

This is usually the simplest approach for quick tests and one-off runs.

---

## Inspect Packaged Config Files

If you want to inspect the packaged Hydra configs inside the image, you can locate the installed package and print a config file directly.

Print the package location:

```bash
docker run --rm simcortex:2.0.0 \
  python -c "import simcortex, pathlib; print(pathlib.Path(simcortex.__file__).resolve().parent)"
```

Print an InitSurf config to stdout:

```bash
docker run --rm simcortex:2.0.0 \
  python -c "import simcortex, pathlib; p=pathlib.Path(simcortex.__file__).resolve().parent/'configs'/'initsurf'/'generate.yaml'; print(p.read_text())"
```

Print a Deform train config to stdout:

```bash
docker run --rm simcortex:2.0.0 \
  python -c "import simcortex, pathlib; p=pathlib.Path(simcortex.__file__).resolve().parent/'configs'/'deform'/'train.yaml'; print(p.read_text())"
```

Save a packaged config to the host:

```bash
mkdir -p /tmp/simcortex_cfg

docker run --rm \
  -v /tmp/simcortex_cfg:/out \
  simcortex:2.0.0 \
  python -c "import simcortex, pathlib; p=pathlib.Path(simcortex.__file__).resolve().parent/'configs'/'initsurf'/'generate.yaml'; open('/out/generate.yaml','w').write(p.read_text())"
```

After that, edit `/tmp/simcortex_cfg/generate.yaml` on the host.

---

## Use Your Own YAML Config

If a stage supports a `user_config` pattern, mount your custom YAML file and pass it explicitly.

Example:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  -v /home/<user>/myconfigs:/cfg \
  simcortex:2.0.0 \
  simcortex deform train user_config=/cfg/train.yaml
```

If your stage does not use a `user_config` field, override individual values directly on the CLI instead.

---

## Examples by Stage

> **Stage 1 is supported by the Docker image.** It consumes existing FreeSurfer subject directories mounted from the host and writes preprocessing outputs under `sc-preproc`. All stages use the canonical derivative directories `sc-preproc`, `sc-seg`, `sc-initsurf`, and `sc-deform`.

### Stage 1 — FreeSurfer to MNI152 preprocessing

Stage 1 consumes existing FreeSurfer outputs mounted from the host. It does not run FreeSurfer itself and does not require GPU access.

The example below mounts the dataset root as writable and mounts the MNI152 template separately as read-only.

```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/templates:/templates:ro \
  simcortex:2.0.0 \
  simcortex fs-to-mni \
  --freesurfer-root /data/<dataset>/derivatives/freesurfer-7.4.1 \
  --out-deriv-root /data/<dataset>/derivatives/sc-preproc \
  --mni-template /templates/MNI152_T1_1mm.nii.gz \
  --transform-type affine \
  --n4 \
  --with-aparc-aseg \
  --with-filled \
  -v
```

The FreeSurfer root must already contain subject directories with the required `mri/` and `surf/` outputs.

### Stage 2 — Segmentation train

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
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
  simcortex:2.0.0 \
  simcortex seg infer \
  dataset.path=/data/<dataset>/derivatives/sc-preproc \
  dataset.split_file=/data/splits/<dataset>_split.csv \
  dataset.split_name=test \
  model.ckpt_path=/data/checkpoints/seg_best_dice.pt \
  outputs.out_root=/data/<dataset>/derivatives/sc-seg
```

### Stage 3 — InitSurf

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
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
  simcortex:2.0.0 \
  simcortex deform train \
  outputs.root=/runs/deform/exp01
```

### Stage 4 — Deformation eval

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e UMASK=002 \
  -v /home/<user>/datasets:/data \
  -v /home/<user>/runs:/runs \
  simcortex:2.0.0 \
  simcortex deform eval
```

---

## GPU Support

If Docker and the NVIDIA runtime are configured correctly on the host, enable GPU access with:

```bash
--gpus all
```

First validate the host Docker GPU setup:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 \
  nvidia-smi
```

Then validate GPU access inside the SimCortex image:

```bash
docker run --rm --gpus all simcortex:2.0.0 \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If the CUDA test container fails, the problem is with the host Docker / NVIDIA runtime setup rather than SimCortex.

---

## Shared Server and HPC Notes

On some managed Linux systems, the Docker daemon cannot bind-mount arbitrary paths such as:

```text
/project/...
```

Even if the path exists for the user, Docker may return an error like:

```text
error while creating mount source path ... permission denied
```

This is typically a host-side Docker policy issue, not an SimCortex issue.

Practical workarounds:

- mount from `$HOME/...`
- mount from `/tmp/...`
- use another host path explicitly allowed by your Docker configuration
- create a symlink, copy, or temporary view under an allowed path before running Docker

---

## Docker Hub Publication

Tag the local image:

```bash
docker tag simcortex:2.0.0 kavehmoradkhani/simcortex:2.0.0
```

Push it:

```bash
docker push kavehmoradkhani/simcortex:2.0.0
```

Then users can pull it with:

```bash
docker pull kavehmoradkhani/simcortex:2.0.0
```

If you later publish a `latest` tag, keep the versioned tag as well.

---

## Apptainer / Singularity Notes

Many neuroimaging and HPC systems prefer **Apptainer / Singularity** rather than Docker. A common pattern is to convert the Docker image into a `.sif` file.

Build from the local Docker image:

```bash
apptainer build simcortex_2.0.0.sif docker-daemon://simcortex:2.0.0
```

Build from Docker Hub after publication:

```bash
apptainer build simcortex_2.0.0.sif docker://kavehmoradkhani/simcortex:2.0.0
```

After conversion, re-check:

- CLI behavior
- writable output directories
- environment variables
- GPU access on the target system

Apptainer runtime behavior is often close to Docker, but it is not always identical.

---

