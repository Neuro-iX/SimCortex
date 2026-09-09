#!/usr/bin/env python3
"""Run SimCortex end-to-end inference across multiple datasets.

This script is an orchestration wrapper around ``scripts/run_pipeline.py``.
It does not implement any scientific pipeline stages itself.

Datasets can be supplied either with repeated::

    --dataset NAME=/path/to/bids/root

arguments, or through a CSV manifest containing:

    dataset,bids_root

Each dataset is executed independently. A failure in one dataset does not
prevent later datasets from running. The process exits nonzero if any dataset
fails.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


DATASET_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    bids_root: Path


def normalize_dataset_name(value: str) -> str:
    name = str(value).strip()
    if not DATASET_NAME_RE.fullmatch(name):
        raise ValueError(
            "Invalid dataset name "
            f"{name!r}. Use letters, numbers, '.', '_', '+', or '-' and "
            "start with a letter or number."
        )
    return name


def resolve_path(value: str | Path, base: Optional[Path] = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def parse_dataset_argument(value: str) -> DatasetSpec:
    name, sep, root = value.partition("=")
    if not sep or not name.strip() or not root.strip():
        raise ValueError(
            f"Invalid --dataset value {value!r}. Expected NAME=/path/to/bids/root."
        )

    return DatasetSpec(
        name=normalize_dataset_name(name),
        bids_root=resolve_path(root),
    )


def load_dataset_manifest(path: Path) -> list[DatasetSpec]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    specs: list[DatasetSpec] = []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Dataset manifest has no header: {path}")

        required = {"dataset", "bids_root"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Dataset manifest is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )

        for row_number, row in enumerate(reader, start=2):
            name_raw = str(row.get("dataset", "")).strip()
            root_raw = str(row.get("bids_root", "")).strip()

            if not name_raw and not root_raw:
                continue

            if not name_raw or not root_raw:
                raise ValueError(
                    f"Incomplete dataset manifest row {row_number}: "
                    "both dataset and bids_root are required."
                )

            specs.append(
                DatasetSpec(
                    name=normalize_dataset_name(name_raw),
                    bids_root=resolve_path(root_raw, base=path.parent),
                )
            )

    if not specs:
        raise ValueError(f"Dataset manifest contains no datasets: {path}")

    return specs


def collect_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []

    if args.dataset_manifest is not None:
        specs.extend(load_dataset_manifest(args.dataset_manifest))

    for value in args.dataset:
        specs.append(parse_dataset_argument(value))

    if not specs:
        raise ValueError(
            "No datasets provided. Use --dataset NAME=PATH and/or "
            "--dataset-manifest MANIFEST.csv."
        )

    seen: dict[str, Path] = {}
    for spec in specs:
        if spec.name in seen:
            raise ValueError(
                f"Duplicate dataset name {spec.name!r}: "
                f"{seen[spec.name]} and {spec.bids_root}"
            )
        seen[spec.name] = spec.bids_root

    return specs


def check_required_paths(
    datasets: list[DatasetSpec],
    project_root: Path,
    mni: Path,
    seg_ckpt: Path,
    deform_ckpt: Path,
) -> list[str]:
    errors: list[str] = []

    if not project_root.is_dir():
        errors.append(f"Project root not found: {project_root}")

    pipeline_script = project_root / "scripts" / "run_pipeline.py"
    if not pipeline_script.is_file():
        errors.append(f"Pipeline script not found: {pipeline_script}")

    source_package = project_root / "src" / "simcortex"
    if not source_package.is_dir():
        errors.append(f"SimCortex source package not found: {source_package}")

    if not mni.is_file():
        errors.append(f"MNI template not found: {mni}")

    if not seg_ckpt.is_file():
        errors.append(f"Segmentation checkpoint not found: {seg_ckpt}")

    if not deform_ckpt.is_file():
        errors.append(f"Deformation checkpoint not found: {deform_ckpt}")

    for spec in datasets:
        if not spec.bids_root.is_dir():
            errors.append(
                f"Dataset root not found [{spec.name}]: {spec.bids_root}"
            )

    return errors


def build_pipeline_command(
    spec: DatasetSpec,
    args: argparse.Namespace,
    project_root: Path,
    mni: Path,
    seg_ckpt: Path,
    deform_ckpt: Path,
    out_root: Path,
    work_root: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_pipeline.py"),
        "bids",
        "--bids-root",
        str(spec.bids_root),
        "--out-root",
        str(out_root),
        "--work-root",
        str(work_root),
        "--project-root",
        str(project_root),
        "--mni",
        str(mni),
        "--seg-ckpt",
        str(seg_ckpt),
        "--deform-ckpt",
        str(deform_ckpt),
        "--device",
        str(args.device),
        "--space",
        str(args.space),
        "--transform-type",
        str(args.transform_type),
        "--initsurf-workers",
        str(max(1, int(args.initsurf_workers))),
    ]

    if args.overwrite:
        cmd.append("--overwrite")

    if args.keep_work:
        cmd.append("--keep-work")

    if args.export_native:
        cmd.append("--export-native")

    if args.participant_label:
        cmd.append("--participant-label")
        cmd.extend(args.participant_label)

    if args.session:
        cmd.append("--session")
        cmd.extend(args.session)

    return cmd


def prepare_environment(
    project_root: Path,
    cuda_visible_devices: Optional[str],
) -> dict[str, str]:
    env = os.environ.copy()

    src = str(project_root / "src")
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not old_pythonpath else f"{src}:{old_pythonpath}"

    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    return env


def initialize_summary(path: Path) -> list[str]:
    fieldnames = [
        "dataset",
        "bids_root",
        "status",
        "returncode",
        "elapsed_sec",
        "out_root",
        "work_root",
        "log_file",
        "error",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

    return fieldnames


def append_summary(
    path: Path,
    fieldnames: list[str],
    row: dict[str, object],
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            delimiter="\t",
        )
        writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser(
        description=(
            "Run SimCortex end-to-end inference across multiple BIDS-like datasets "
            "using scripts/run_pipeline.py."
        )
    )

    dataset_group = p.add_argument_group("datasets")
    dataset_group.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Dataset name and BIDS-like root. Repeat for multiple datasets. "
            "May be combined with --dataset-manifest."
        ),
    )
    dataset_group.add_argument(
        "--dataset-manifest",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "CSV with columns 'dataset,bids_root'. Relative bids_root values "
            "are resolved relative to the manifest file."
        ),
    )

    required = p.add_argument_group("required model/input paths")
    required.add_argument("--mni", required=True, type=Path)
    required.add_argument("--seg-ckpt", required=True, type=Path)
    required.add_argument("--deform-ckpt", required=True, type=Path)

    outputs = p.add_argument_group("output paths")
    outputs.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Base prediction directory. One child directory is created per dataset.",
    )
    outputs.add_argument(
        "--work-root",
        required=True,
        type=Path,
        help="Base working directory. One child directory is created per dataset.",
    )
    outputs.add_argument(
        "--log-root",
        required=True,
        type=Path,
        help="Base orchestration log directory. One log is written per dataset.",
    )

    runtime = p.add_argument_group("runtime")
    runtime.add_argument(
        "--project-root",
        type=Path,
        default=repo_root,
        help=f"SimCortex source checkout. Default: {repo_root}",
    )
    runtime.add_argument("--device", default="cuda:0")
    runtime.add_argument(
        "--cuda-visible-devices",
        default=None,
        metavar="DEVICES",
        help=(
            "Optional CUDA_VISIBLE_DEVICES value for child processes, e.g. '0' "
            "or '1'. The logical --device is passed separately."
        ),
    )
    runtime.add_argument("--space", default="MNI152")
    runtime.add_argument(
        "--transform-type",
        choices=["Rigid", "Affine"],
        default="Affine",
    )
    runtime.add_argument("--initsurf-workers", type=int, default=1)
    runtime.add_argument("--overwrite", action="store_true")
    runtime.add_argument("--keep-work", action="store_true")
    runtime.add_argument("--export-native", action="store_true")
    runtime.add_argument(
        "--participant-label",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "Optional participant filter applied to every dataset. "
            "Repeat for multiple labels."
        ),
    )
    runtime.add_argument(
        "--session",
        action="append",
        default=[],
        metavar="SESSION",
        help=(
            "Optional session filter applied to every dataset. "
            "Repeat for multiple sessions."
        ),
    )
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate inputs, print/write the commands, and create the global "
            "summary without running inference."
        ),
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        datasets = collect_datasets(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    project_root = args.project_root.expanduser().resolve()
    mni = args.mni.expanduser().resolve()
    seg_ckpt = args.seg_ckpt.expanduser().resolve()
    deform_ckpt = args.deform_ckpt.expanduser().resolve()

    out_base = args.out_root.expanduser().resolve()
    work_base = args.work_root.expanduser().resolve()
    log_base = args.log_root.expanduser().resolve()

    errors = check_required_paths(
        datasets=datasets,
        project_root=project_root,
        mni=mni,
        seg_ckpt=seg_ckpt,
        deform_ckpt=deform_ckpt,
    )
    if errors:
        print("Preflight validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    out_base.mkdir(parents=True, exist_ok=True)
    work_base.mkdir(parents=True, exist_ok=True)
    log_base.mkdir(parents=True, exist_ok=True)

    summary_path = out_base / "multi_dataset_summary.tsv"
    summary_fields = initialize_summary(summary_path)

    env = prepare_environment(
        project_root=project_root,
        cuda_visible_devices=args.cuda_visible_devices,
    )

    print("=" * 72)
    print("SimCortex multi-dataset inference")
    print(f"Datasets       : {len(datasets)}")
    print(f"Project root   : {project_root}")
    print(f"MNI template   : {mni}")
    print(f"Seg checkpoint : {seg_ckpt}")
    print(f"Deform ckpt    : {deform_ckpt}")
    print(f"Output root    : {out_base}")
    print(f"Work root      : {work_base}")
    print(f"Log root       : {log_base}")
    print(f"Device         : {args.device}")
    print(
        "CUDA visible   : "
        f"{env.get('CUDA_VISIBLE_DEVICES', '<inherited/unset>')}"
    )
    print(f"Dry run        : {args.dry_run}")
    print("=" * 72)

    failed: list[str] = []

    for index, spec in enumerate(datasets, start=1):
        dataset_out = out_base / spec.name
        dataset_work = work_base / spec.name
        dataset_log = log_base / f"{spec.name}.log"

        cmd = build_pipeline_command(
            spec=spec,
            args=args,
            project_root=project_root,
            mni=mni,
            seg_ckpt=seg_ckpt,
            deform_ckpt=deform_ckpt,
            out_root=dataset_out,
            work_root=dataset_work,
        )

        print()
        print("=" * 72)
        print(f"[{index}/{len(datasets)}] {spec.name}")
        print(f"BIDS root : {spec.bids_root}")
        print(f"Output    : {dataset_out}")
        print(f"Work      : {dataset_work}")
        print(f"Log       : {dataset_log}")
        print(f"Command   : {shlex.join(cmd)}")
        print("=" * 72)

        dataset_log.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        returncode = 0
        error = ""

        with dataset_log.open("w", encoding="utf-8") as log_file:
            log_file.write("COMMAND:\n")
            log_file.write(shlex.join(cmd))
            log_file.write("\n\n")

            if args.cuda_visible_devices is not None:
                log_file.write(
                    "CUDA_VISIBLE_DEVICES="
                    f"{args.cuda_visible_devices}\n\n"
                )

            log_file.flush()

            if args.dry_run:
                status = "DRY_RUN"
            else:
                proc = subprocess.run(
                    cmd,
                    cwd=str(project_root),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                returncode = int(proc.returncode)

                if returncode == 0:
                    status = "OK"
                else:
                    status = "FAILED"
                    error = f"run_pipeline.py exited with code {returncode}"
                    failed.append(spec.name)

        elapsed = time.time() - start

        append_summary(
            summary_path,
            summary_fields,
            {
                "dataset": spec.name,
                "bids_root": str(spec.bids_root),
                "status": status,
                "returncode": returncode,
                "elapsed_sec": round(elapsed, 3),
                "out_root": str(dataset_out),
                "work_root": str(dataset_work),
                "log_file": str(dataset_log),
                "error": error,
            },
        )

        if status == "FAILED":
            print(f"FAILED: {spec.name} — see {dataset_log}")
        elif status == "DRY_RUN":
            print(f"DRY RUN: {spec.name}")
        else:
            print(f"DONE: {spec.name}")

    print()
    print("=" * 72)
    print("Multi-dataset inference finished.")
    print(f"Summary : {summary_path}")
    print(f"Failed  : {', '.join(failed) if failed else 'none'}")
    print("=" * 72)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
