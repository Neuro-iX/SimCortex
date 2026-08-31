#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Evaluate SimCortex cortical-surface reconstruction metrics.

The scientific metric implementation is provided by ``metrics_core.py``.
This runner pairs SimCortex native-space predictions with scannerRAS
ground-truth surfaces, executes the frozen metric implementation, and writes
per-surface and per-case results.

Inputs
------
- <eval-root>/manifests/pred_manifest.tsv
- <eval-root>/manifests/gt_scannerRAS_manifest.tsv

Outputs
-------
- <eval-root>/metrics/surface_metrics_long.csv
- <eval-root>/metrics/pair_metrics_thickness_collisions.csv
- <eval-root>/metrics/pair_metrics_thickness.csv
- <eval-root>/metrics/missing_or_failed.csv
- <eval-root>/metrics/run_summary.json

Pairwise FCL collision metrics are computed separately by the collision
evaluation stage. Metric definitions are provided by
``metrics_core.METRIC_DEFINITIONS``.
"""

import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from metrics_core import (
    HAS_PYMESHLAB,
    HAS_SIF,
    METRIC_DEFINITIONS,
    METRICS_SCHEMA_VERSION,
    SURFACE_KEYS,
    choose_device,
    evaluate_subject_meshes,
    load_mesh,
)

METHOD_NAME = "SimCortex"

SURFACE_DIST_COLS = ["ASSD_mm", "HD90_mm", "ChamferPCL1_mm"]

OFFICIAL_PAIR_CSV = "pair_metrics_thickness_collisions.csv"
LEGACY_PAIR_CSV = "pair_metrics_thickness.csv"

SURFACE_OUTPUT_COLUMNS = [
    "schema_version",
    "method",
    "eval_set",
    "dataset",
    "subject",
    "session",
    "case_id",
    "surface",
    "ASSD_mm",
    "HD90_mm",
    "ChamferPCL1_mm",
    "SIF_pct",
    "SIF_status",
    "SIF_error",
    "SIF_filter",
    "pred_gt_centroid_dist_mm",
    "pred_n_vertices",
    "pred_n_faces",
    "gt_n_vertices",
    "gt_n_faces",
    "pred_path",
    "gt_path",
]

PAIR_OUTPUT_COLUMNS = [
    "schema_version",
    "method",
    "eval_set",
    "dataset",
    "subject",
    "session",
    "case_id",
    "lh_thickness_pred_mean_mm",
    "lh_thickness_gt_mean_mm",
    "lh_thickness_abs_error_mm",
    "lh_thickness_bias_mm",
    "rh_thickness_pred_mean_mm",
    "rh_thickness_gt_mean_mm",
    "rh_thickness_abs_error_mm",
    "rh_thickness_bias_mm",
]

MISSING_OUTPUT_COLUMNS = [
    "method",
    "dataset",
    "subject",
    "session",
    "case_id",
    "status",
    "error",
    "traceback",
]


# --------------------------------------------------------------------------- #
# CLI / setup
# --------------------------------------------------------------------------- #
def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval_metrics.log"

    # Avoid duplicated log lines if this script is invoked from an interactive
    # Python session or test harness that already configured root handlers.
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compute SimCortex ASSD/HD90/ChamferPCL1/SIF/"
            "thickness metrics."
        )
    )
    ap.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Root directory for evaluation inputs and outputs.",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional subset of dataset keys to evaluate.",
    )
    ap.add_argument(
        "--max-cases-per-dataset",
        type=int,
        default=None,
    )
    ap.add_argument(
        "--device",
        default="auto",
        help=(
            "Device for PyTorch3D sampling/distances. "
            "Use 'auto', 'cpu', or e.g. 'cuda:0'."
        ),
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=150000,
    )
    ap.add_argument(
        "--n-thickness-samples",
        type=int,
        default=50000,
    )
    ap.add_argument(
        "--eval-set",
        default="sample40",
        help="Metadata label written to metric rows.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=12345,
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=25,
        help=(
            "Write partial outputs every N attempted cases. "
            "Use 0 to disable partial writes."
        ),
    )
    ap.add_argument(
        "--slow-case-warn-sec",
        type=float,
        default=300.0,
        help=(
            "Log a warning when a case exceeds this duration; "
            "use 0 to disable."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail at the end if any case failed, any distance metric "
            "is non-finite/negative, any SIF row failed, or output "
            "row counts are internally inconsistent."
        ),
    )
    ap.add_argument(
        "--pred-manifest",
        type=Path,
        default=None,
        help="Default: <eval-root>/manifests/pred_manifest.tsv.",
    )
    ap.add_argument(
        "--gt-manifest",
        type=Path,
        default=None,
        help="Default: <eval-root>/manifests/gt_scannerRAS_manifest.tsv.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <eval-root>/metrics.",
    )

    args = ap.parse_args()

    eval_root = args.eval_root.expanduser().resolve()

    if args.pred_manifest is None:
        args.pred_manifest = (
            eval_root
            / "manifests"
            / "pred_manifest.tsv"
        )
    else:
        args.pred_manifest = (
            args.pred_manifest.expanduser().resolve()
        )

    if args.gt_manifest is None:
        args.gt_manifest = (
            eval_root
            / "manifests"
            / "gt_scannerRAS_manifest.tsv"
        )
    else:
        args.gt_manifest = (
            args.gt_manifest.expanduser().resolve()
        )

    if args.output_dir is None:
        args.output_dir = eval_root / "metrics"
    else:
        args.output_dir = (
            args.output_dir.expanduser().resolve()
        )

    args.eval_root = eval_root
    return args





# --------------------------------------------------------------------------- #
# Manifest selection / maps
# --------------------------------------------------------------------------- #

def select_cases(
    pred: pd.DataFrame,
    datasets: list[str] | None,
    max_cases_per_dataset: int | None,
) -> pd.DataFrame:
    required = {
        "method",
        "dataset",
        "case_id",
        "subject",
        "session",
        "surface",
        "status",
    }
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(
            "Prediction manifest missing columns: "
            f"{sorted(missing)}"
        )

    df = pred[
        (pred["method"].astype(str) == METHOD_NAME)
        & (pred["status"].astype(str) == "OK")
    ].copy()

    if datasets is not None:
        df = df[
            df["dataset"].isin(datasets)
        ].copy()

    expected_surface_set = {
        str(x)
        for x in SURFACE_KEYS
    }

    per_case = (
        df.groupby(
            ["method", "dataset", "case_id"],
            dropna=False,
        )
        .agg(
            n_rows=("surface", "size"),
            n_unique_surfaces=(
                "surface",
                "nunique",
            ),
            surfaces=(
                "surface",
                lambda x: sorted(
                    str(v)
                    for v in set(x)
                ),
            ),
        )
        .reset_index()
    )

    bad_surface_sets = per_case[
        (
            per_case["n_rows"]
            != len(SURFACE_KEYS)
        )
        | (
            per_case["n_unique_surfaces"]
            != len(SURFACE_KEYS)
        )
        | (
            ~per_case["surfaces"].map(
                lambda values: (
                    set(values)
                    == expected_surface_set
                )
            )
        )
    ]

    if len(bad_surface_sets):
        preview = (
            bad_surface_sets
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Selected prediction manifest contains incomplete "
            "or invalid surface sets; "
            f"first rows={preview}"
        )

    cases = (
        df[
            [
                "method",
                "dataset",
                "case_id",
                "subject",
                "session",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "method",
                "dataset",
                "case_id",
            ]
        )
        .reset_index(drop=True)
    )

    if max_cases_per_dataset is not None:
        if max_cases_per_dataset <= 0:
            raise ValueError(
                "--max-cases-per-dataset must be "
                "positive when provided."
            )

        cases = (
            cases.groupby(
                ["method", "dataset"],
                group_keys=False,
            )
            .head(
                int(
                    max_cases_per_dataset
                )
            )
            .reset_index(drop=True)
        )

    return cases




def build_maps(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
) -> tuple[dict, dict]:
    pred_required = {
        "method",
        "dataset",
        "case_id",
        "surface",
        "pred_path",
        "status",
    }
    gt_required = {
        "dataset",
        "case_id",
        "surface",
        "output_path",
        "status",
    }

    missing_pred = (
        pred_required
        - set(pred.columns)
    )
    missing_gt = (
        gt_required
        - set(gt.columns)
    )

    if missing_pred:
        raise ValueError(
            "Prediction manifest missing columns: "
            f"{sorted(missing_pred)}"
        )
    if missing_gt:
        raise ValueError(
            "GT manifest missing columns: "
            f"{sorted(missing_gt)}"
        )

    ok_pred = pred[
        pred["status"].astype(str) == "OK"
    ].copy()
    ok_pred = ok_pred[
        ok_pred["method"].astype(str)
        == METHOD_NAME
    ].copy()

    ok_gt = gt[
        gt["status"].astype(str) == "OK"
    ].copy()

    pred_key_cols = [
        "method",
        "dataset",
        "case_id",
        "surface",
    ]
    gt_key_cols = [
        "dataset",
        "case_id",
        "surface",
    ]

    pred_dups = ok_pred[
        ok_pred.duplicated(
            pred_key_cols,
            keep=False,
        )
    ]
    if len(pred_dups):
        preview = pred_dups[
            pred_key_cols
            + ["pred_path"]
        ].head(10).to_dict(
            orient="records"
        )
        raise ValueError(
            "Duplicate OK prediction rows for "
            f"{pred_key_cols}: "
            f"first rows={preview}"
        )

    gt_dups = ok_gt[
        ok_gt.duplicated(
            gt_key_cols,
            keep=False,
        )
    ]
    if len(gt_dups):
        preview = gt_dups[
            gt_key_cols
            + ["output_path"]
        ].head(10).to_dict(
            orient="records"
        )
        raise ValueError(
            "Duplicate OK GT rows for "
            f"{gt_key_cols}: "
            f"first rows={preview}"
        )

    pred_map: dict[
        tuple[str, str, str, str],
        Path,
    ] = {}

    for row in ok_pred.itertuples(
        index=False
    ):
        pred_map[
            (
                str(row.method),
                str(row.dataset),
                str(row.case_id),
                str(row.surface),
            )
        ] = Path(
            str(row.pred_path)
        )

    gt_map: dict[
        tuple[str, str, str],
        Path,
    ] = {}

    for row in ok_gt.itertuples(
        index=False
    ):
        gt_map[
            (
                str(row.dataset),
                str(row.case_id),
                str(row.surface),
            )
        ] = Path(
            str(row.output_path)
        )

    return pred_map, gt_map



# --------------------------------------------------------------------------- #
# Per-case evaluation
# --------------------------------------------------------------------------- #
def evaluate_one_case(
    *,
    method: str,
    dataset: str,
    case_id: str,
    subject: str,
    session: str,
    pred_map: dict,
    gt_map: dict,
    device: torch.device,
    eval_set: str,
    n_samples: int,
    n_thickness_samples: int,
    seed: int,
) -> tuple[list[dict], dict]:
    pred_paths: dict[str, Path] = {}
    gt_paths: dict[str, Path] = {}

    for skey in SURFACE_KEYS:
        pk = (method, dataset, case_id, skey)
        gk = (dataset, case_id, skey)

        if pk not in pred_map:
            raise FileNotFoundError(f"Missing pred map row: {pk}")
        if gk not in gt_map:
            raise FileNotFoundError(f"Missing GT map row: {gk}")

        gt_paths[skey] = gt_map[gk]
        pred_paths[skey] = pred_map[pk]

        if not pred_paths[skey].exists():
            raise FileNotFoundError(f"Missing pred file: {pred_paths[skey]}")
        if not gt_paths[skey].exists():
            raise FileNotFoundError(f"Missing GT file: {gt_paths[skey]}")

    # Load prediction and GT reference meshes separately. GT is only the
    # evaluation reference, not a method/baseline in this DL-only benchmark.
    pred_meshes = {k: load_mesh(pred_paths[k], device) for k in SURFACE_KEYS}
    gt_meshes = {k: load_mesh(gt_paths[k], device) for k in SURFACE_KEYS}

    surface_rows, pair_row = evaluate_subject_meshes(
        method=method,
        eval_set=eval_set,
        dataset=dataset,
        subject=subject,
        session=session,
        pred_meshes=pred_meshes,
        gt_meshes=gt_meshes,
        pred_paths=pred_paths,
        gt_paths=gt_paths,
        n_samples=n_samples,
        n_thickness_samples=n_thickness_samples,
        seed=seed,
    )

    for row in surface_rows:
        row["case_id"] = case_id
    pair_row["case_id"] = case_id

    return surface_rows, pair_row


# --------------------------------------------------------------------------- #
# Output helpers / quality checks
# --------------------------------------------------------------------------- #
def dataframe_with_columns(rows: list[dict], preferred_columns: list[str]) -> pd.DataFrame:
    """Create a DataFrame with stable headers even when rows is empty.

    If rows contain additional keys not in preferred_columns, they are appended
    at the end so no information is lost.
    """
    if rows:
        df = pd.DataFrame(rows)
        extra_cols = [c for c in df.columns if c not in preferred_columns]
        return df.reindex(columns=preferred_columns + extra_cols)
    return pd.DataFrame(columns=preferred_columns)


def json_safe(obj: Any) -> Any:
    """JSON serializer for numpy/pandas/path objects used in run_summary."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return [{str(k): json_safe(v) for k, v in row.items()} for row in obj.to_dict(orient="records")]
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    try:
        scalar_na = pd.isna(obj)
        if isinstance(scalar_na, (bool, np.bool_)) and bool(scalar_na):
            return None
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_safe)


def write_outputs(out_dir: Path, surface_rows: list[dict], pair_rows: list[dict], missing_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    surface_df = dataframe_with_columns(surface_rows, SURFACE_OUTPUT_COLUMNS)
    pair_df = dataframe_with_columns(pair_rows, PAIR_OUTPUT_COLUMNS)
    missing_df = dataframe_with_columns(missing_rows, MISSING_OUTPUT_COLUMNS)

    surface_df.to_csv(out_dir / "surface_metrics_long.csv", index=False, encoding="utf-8")

    # Official final pipeline name.
    pair_df.to_csv(out_dir / OFFICIAL_PAIR_CSV, index=False, encoding="utf-8")
    # Backward-compatible alias for older helper scripts. This can be removed in
    # a future major cleanup after all downstream scripts use OFFICIAL_PAIR_CSV.
    pair_df.to_csv(out_dir / LEGACY_PAIR_CSV, index=False, encoding="utf-8")

    missing_df.to_csv(out_dir / "missing_or_failed.csv", index=False, encoding="utf-8")


def compute_quality_counts(surface_rows: list[dict]) -> dict[str, Any]:
    counts: dict[str, Any] = {
        # Backward-compatible names retained.
        "surface_rows_with_nan_distance": 0,
        "surface_metric_nan_values": 0,
        # Stricter final checks.
        "surface_rows_with_nonfinite_distance": 0,
        "surface_metric_nonfinite_values": 0,
        "surface_rows_with_negative_distance": 0,
        "surface_metric_negative_values": 0,
        "sif_failures": 0,
        "sif_nonfinite_values": 0,
        "max_pred_gt_centroid_dist_mm": 0.0,
    }

    if not surface_rows:
        return counts

    sdf = pd.DataFrame(surface_rows)

    nan_row_mask = pd.Series(False, index=sdf.index)
    nonfinite_row_mask = pd.Series(False, index=sdf.index)
    negative_row_mask = pd.Series(False, index=sdf.index)
    nan_value_count = 0
    nonfinite_value_count = 0
    negative_value_count = 0

    for col in SURFACE_DIST_COLS:
        if col not in sdf.columns:
            # Missing distance columns are represented as non-finite for all rows.
            nan_row_mask = pd.Series(True, index=sdf.index)
            nonfinite_row_mask = pd.Series(True, index=sdf.index)
            nan_value_count += int(len(sdf))
            nonfinite_value_count += int(len(sdf))
            continue

        numeric = pd.to_numeric(sdf[col], errors="coerce")
        arr = numeric.to_numpy(dtype=float)
        col_nan = numeric.isna()
        col_nonfinite = ~np.isfinite(arr)
        col_negative = np.isfinite(arr) & (arr < 0)

        nan_row_mask = nan_row_mask | col_nan
        nonfinite_row_mask = nonfinite_row_mask | pd.Series(col_nonfinite, index=sdf.index)
        negative_row_mask = negative_row_mask | pd.Series(col_negative, index=sdf.index)
        nan_value_count += int(col_nan.sum())
        nonfinite_value_count += int(col_nonfinite.sum())
        negative_value_count += int(col_negative.sum())

    counts["surface_rows_with_nan_distance"] = int(nan_row_mask.sum())
    counts["surface_metric_nan_values"] = int(nan_value_count)
    counts["surface_rows_with_nonfinite_distance"] = int(nonfinite_row_mask.sum())
    counts["surface_metric_nonfinite_values"] = int(nonfinite_value_count)
    counts["surface_rows_with_negative_distance"] = int(negative_row_mask.sum())
    counts["surface_metric_negative_values"] = int(negative_value_count)

    if {"SIF_status", "SIF_pct"}.issubset(sdf.columns):
        sif_pct = pd.to_numeric(sdf["SIF_pct"], errors="coerce")
        sif_arr = sif_pct.to_numpy(dtype=float)
        sif_nonfinite = ~np.isfinite(sif_arr)
        counts["sif_nonfinite_values"] = int(sif_nonfinite.sum())
        counts["sif_failures"] = int(((sdf["SIF_status"] != "OK") | sif_nonfinite).sum())
    elif "SIF_status" in sdf.columns:
        counts["sif_failures"] = int((sdf["SIF_status"] != "OK").sum())
    else:
        counts["sif_failures"] = int(len(sdf))

    if "pred_gt_centroid_dist_mm" in sdf.columns:
        values = pd.to_numeric(sdf["pred_gt_centroid_dist_mm"], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            counts["max_pred_gt_centroid_dist_mm"] = float(np.max(finite_values))
        else:
            counts["max_pred_gt_centroid_dist_mm"] = float("nan")

    return counts


def strict_failures(
    *,
    quality: dict[str, Any],
    n_cases_attempted: int,
    n_cases_succeeded: int,
    missing_rows: list[dict],
    surface_rows: list[dict],
    pair_rows: list[dict],
) -> list[str]:
    problems: list[str] = []

    if missing_rows:
        problems.append(f"missing_or_failed rows = {len(missing_rows)}")

    if n_cases_attempted > 0 and n_cases_succeeded == 0:
        problems.append("no cases succeeded")

    expected_surface_rows = n_cases_succeeded * len(SURFACE_KEYS)
    if len(surface_rows) != expected_surface_rows:
        problems.append(f"surface_rows = {len(surface_rows)} but expected {expected_surface_rows}")

    if len(pair_rows) != n_cases_succeeded:
        problems.append(f"pair_rows = {len(pair_rows)} but expected {n_cases_succeeded}")

    if int(quality.get("surface_rows_with_nonfinite_distance", 0)) != 0:
        problems.append(
            f"surface_rows_with_nonfinite_distance = {quality.get('surface_rows_with_nonfinite_distance')}"
        )

    # Keep the older NaN-only field in the report, but strict mode uses the stronger non-finite check above.
    if int(quality.get("surface_rows_with_negative_distance", 0)) != 0:
        problems.append(
            f"surface_rows_with_negative_distance = {quality.get('surface_rows_with_negative_distance')}"
        )

    if int(quality.get("sif_failures", 0)) != 0:
        problems.append(f"sif_failures = {quality.get('sif_failures')}")

    return problems


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    setup_logging(out_dir)

    required_existing_outputs = [
        out_dir
        / "surface_metrics_long.csv",
        out_dir
        / OFFICIAL_PAIR_CSV,
        out_dir
        / LEGACY_PAIR_CSV,
        out_dir
        / "missing_or_failed.csv",
        out_dir
        / "run_summary.json",
    ]

    existing_outputs = [
        path
        for path in required_existing_outputs
        if path.exists()
    ]

    if existing_outputs and not args.overwrite:
        existing_list = ", ".join(
            str(path)
            for path in existing_outputs
        )
        raise SystemExit(
            "Existing metric output(s) found in "
            f"{out_dir}: {existing_list}. "
            "Use --overwrite to rerun and replace "
            "partial/previous outputs."
        )

    logging.info(
        "schema_version=%s",
        METRICS_SCHEMA_VERSION,
    )
    logging.info(
        "eval_root=%s",
        args.eval_root,
    )
    logging.info(
        "pred_manifest=%s",
        args.pred_manifest,
    )
    logging.info(
        "gt_manifest=%s",
        args.gt_manifest,
    )
    logging.info(
        "method=%s",
        METHOD_NAME,
    )
    logging.info(
        "device=%s n_samples=%s "
        "n_thickness_samples=%s",
        args.device,
        args.n_samples,
        args.n_thickness_samples,
    )
    logging.info(
        "eval_set=%s seed=%s",
        args.eval_set,
        args.seed,
    )
    logging.info(
        "HAS_PYMESHLAB=%s HAS_SIF=%s",
        HAS_PYMESHLAB,
        HAS_SIF,
    )
    logging.info(
        "Writing official pair output as %s",
        OFFICIAL_PAIR_CSV,
    )
    logging.info(
        "Also writing compatibility alias as %s",
        LEGACY_PAIR_CSV,
    )

    device = choose_device(
        args.device
    )

    pred_manifest = Path(
        args.pred_manifest
    )
    gt_manifest = Path(
        args.gt_manifest
    )

    if not pred_manifest.exists():
        raise FileNotFoundError(
            "Missing pred manifest: "
            f"{pred_manifest}"
        )
    if not gt_manifest.exists():
        raise FileNotFoundError(
            "Missing GT manifest: "
            f"{gt_manifest}"
        )

    pred = pd.read_csv(
        pred_manifest,
        sep="\t",
        low_memory=False,
    )
    gt = pd.read_csv(
        gt_manifest,
        sep="\t",
        low_memory=False,
    )

    try:
        cases = select_cases(
            pred,
            args.datasets,
            args.max_cases_per_dataset,
        )
        pred_map, gt_map = build_maps(
            pred,
            gt,
        )
    except Exception as exc:
        logging.error(
            "Manifest preparation failed: %s",
            exc,
        )
        raise SystemExit(1) from exc

    logging.info(
        "cases=%d",
        len(cases),
    )

    if len(cases) == 0:
        raise SystemExit(
            "No SimCortex cases selected. "
            "Check --datasets and prediction "
            "manifest status."
        )

    logging.info(
        "cases by dataset:\n%s",
        cases.groupby(
            "dataset"
        ).size().to_string(),
    )

    surface_rows: list[dict] = []
    pair_rows: list[dict] = []
    missing_rows: list[dict] = []

    t0 = time.time()
    attempted = 0
    succeeded = 0

    for row in tqdm(
        cases.itertuples(index=False),
        total=len(cases),
        desc="Metric eval",
    ):
        attempted += 1
        t_case = time.time()

        try:
            surface_result, pair_result = (
                evaluate_one_case(
                    method=row.method,
                    dataset=row.dataset,
                    case_id=row.case_id,
                    subject=row.subject,
                    session=row.session,
                    pred_map=pred_map,
                    gt_map=gt_map,
                    device=device,
                    eval_set=args.eval_set,
                    n_samples=args.n_samples,
                    n_thickness_samples=(
                        args.n_thickness_samples
                    ),
                    seed=args.seed,
                )
            )

            surface_rows.extend(
                surface_result
            )
            pair_rows.append(
                pair_result
            )
            succeeded += 1

        except Exception as exc:
            logging.error(
                "FAILED %s %s %s: %s",
                row.method,
                row.dataset,
                row.case_id,
                exc,
            )
            missing_rows.append(
                {
                    "method": row.method,
                    "dataset": row.dataset,
                    "subject": row.subject,
                    "session": row.session,
                    "case_id": row.case_id,
                    "status": "FAILED",
                    "error": repr(exc),
                    "traceback": (
                        traceback.format_exc()
                    ),
                }
            )

        finally:
            case_elapsed = (
                time.time()
                - t_case
            )

            if (
                args.slow_case_warn_sec
                and case_elapsed
                > float(
                    args.slow_case_warn_sec
                )
            ):
                logging.warning(
                    "Slow case: %s %s %s "
                    "took %.1f sec",
                    row.method,
                    row.dataset,
                    row.case_id,
                    case_elapsed,
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()

        if (
            args.save_every > 0
            and attempted
            % args.save_every
            == 0
        ):
            write_outputs(
                out_dir,
                surface_rows,
                pair_rows,
                missing_rows,
            )

    write_outputs(
        out_dir,
        surface_rows,
        pair_rows,
        missing_rows,
    )

    elapsed = time.time() - t0

    quality = compute_quality_counts(
        surface_rows
    )

    strict_problems = strict_failures(
        quality=quality,
        n_cases_attempted=attempted,
        n_cases_succeeded=succeeded,
        missing_rows=missing_rows,
        surface_rows=surface_rows,
        pair_rows=pair_rows,
    )

    summary = {
        "stage": "evaluate_metrics",
        "schema_version": (
            METRICS_SCHEMA_VERSION
        ),
        "pipeline_schema_version": (
            "simcortex_evaluation_metrics_v1.0"
        ),
        "eval_root": str(
            args.eval_root
        ),
        "output_dir": str(
            out_dir
        ),
        "method": METHOD_NAME,
        "datasets": args.datasets,
        "n_cases_requested": int(
            len(cases)
        ),
        "n_cases_attempted": int(
            attempted
        ),
        "n_cases_succeeded": int(
            succeeded
        ),
        "n_cases_failed": int(
            len(missing_rows)
        ),
        "n_cases_processed": int(
            succeeded
        ),
        "surface_rows": int(
            len(surface_rows)
        ),
        "pair_rows": int(
            len(pair_rows)
        ),
        "missing_or_failed": int(
            len(missing_rows)
        ),
        "device": str(device),
        "n_samples": int(
            args.n_samples
        ),
        "n_thickness_samples": int(
            args.n_thickness_samples
        ),
        "eval_set": args.eval_set,
        "seed": int(args.seed),
        "seed_strategy": (
            "metrics_core.set_seed(base_seed, method, "
            "dataset, subject, session, surface/hemi, "
            "metric_part)"
        ),
        "slow_case_warn_sec": float(
            args.slow_case_warn_sec
        ),
        "HAS_PYMESHLAB": bool(
            HAS_PYMESHLAB
        ),
        "HAS_SIF": bool(
            HAS_SIF
        ),
        "quality_counts": quality,
        "strict_problems": (
            strict_problems
        ),
        "metric_definitions": (
            METRIC_DEFINITIONS
        ),
        "manifests": {
            "pred_manifest": str(
                pred_manifest
            ),
            "gt_manifest": str(
                gt_manifest
            ),
        },
        "outputs": {
            "surface_metrics_long": str(
                out_dir
                / "surface_metrics_long.csv"
            ),
            "pair_metrics_thickness_collisions": str(
                out_dir
                / OFFICIAL_PAIR_CSV
            ),
            "pair_metrics_thickness_legacy": str(
                out_dir
                / LEGACY_PAIR_CSV
            ),
            "missing_or_failed": str(
                out_dir
                / "missing_or_failed.csv"
            ),
            "run_summary": str(
                out_dir
                / "run_summary.json"
            ),
        },
        "elapsed_sec": float(
            elapsed
        ),
        "elapsed_min": float(
            elapsed / 60.0
        ),
    }

    write_json(
        out_dir / "run_summary.json",
        summary,
    )

    logging.info(
        "Done. attempted=%d succeeded=%d "
        "failed=%d surface_rows=%d "
        "pair_rows=%d elapsed_min=%.2f",
        attempted,
        succeeded,
        len(missing_rows),
        len(surface_rows),
        len(pair_rows),
        elapsed / 60.0,
    )

    logging.info(
        "quality_counts=%s",
        json.dumps(
            quality,
            default=json_safe,
        ),
    )

    if strict_problems:
        logging.error(
            "strict_problems=%s",
            json.dumps(
                strict_problems,
                default=json_safe,
            ),
        )

    if (
        args.strict
        and strict_problems
    ):
        raise SystemExit(1)



if __name__ == "__main__":
    main()
