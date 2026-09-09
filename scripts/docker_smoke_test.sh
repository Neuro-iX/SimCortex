#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-simcortex:2.0.0}"

echo "==> Smoke test for image: ${IMAGE}"

echo
echo "==> [1/7] CLI version"
docker run --rm "${IMAGE}" simcortex --version

echo
echo "==> [2/7] CLI root help"
docker run --rm "${IMAGE}" simcortex --help >/dev/null

echo
echo "==> [3/7] Preprocessing CLI help"
docker run --rm "${IMAGE}" simcortex fs-to-mni --help >/dev/null

echo
echo "==> [4/7] Segmentation CLI help"
docker run --rm "${IMAGE}" simcortex seg --help >/dev/null

echo
echo "==> [5/7] InitSurf CLI help"
docker run --rm "${IMAGE}" simcortex initsurf --help >/dev/null

echo
echo "==> [6/7] Deform CLI help"
docker run --rm "${IMAGE}" simcortex deform --help >/dev/null

echo
echo "==> [7/7] Python imports + CUDA"
docker run --rm --gpus all -i "${IMAGE}" python - <<'PY'
from importlib import metadata

import ants
import pymeshlab
import pytorch3d
import simcortex
import torch

print("simcortex:", simcortex.__version__)
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("pytorch3d:", getattr(pytorch3d, "__version__", "unknown"))
print("antspyx:", metadata.version("antspyx"))
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside container")
PY

echo
echo "==> All smoke tests passed."
