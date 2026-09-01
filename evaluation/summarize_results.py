#!/usr/bin/env python3
"""Summarize SimCortex evaluation metrics and collision results.

This stage performs aggregation only. It does not load meshes, sample points,
compute geometric distances, or run FCL.

Inputs
------
  <eval-root>/metrics/surface_metrics_long.csv
  <eval-root>/metrics/pair_metrics_thickness_collisions.csv
  <eval-root>/metrics/missing_or_failed.csv
  <eval-root>/collisions/collision_surface_union_case_level.csv
  <eval-root>/collisions/collision_missing_or_failed.csv

Outputs
-------
  <eval-root>/summary/case_metrics.csv
  <eval-root>/summary/by_dataset.csv
  <eval-root>/summary/overall.csv
  <eval-root>/summary/run_summary.json
  <eval-root>/summary/summarize_results.log
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_NAME = "SimCortex"

SURFACES = [
    "lh_white",
    "lh_pial",
    "rh_white",
    "rh_pial",
]

CASE_KEY_COLS = [
    "dataset",
    "subject",
    "session",
    "case_id",
]

SURFACE_METRICS = [
    "ChamferPCL1_mm",
    "ASSD_mm",
    "HD90_mm",
    "SIF_pct",
]

FINAL_METRICS = [
    "ChamferPCL1_mm",
    "ASSD_mm",
    "HD90_mm",
    "SIF_pct",
    "ThicknessAbsErr_mm",
    "CollisionPctUnion_mean4",
]

LH_THICKNESS_COL = "lh_thickness_abs_error_mm"
RH_THICKNESS_COL = "rh_thickness_abs_error_mm"

COLLISION_SOURCE_COL = "collision_pct_union_mean4"
COLLISION_OUTPUT_COL = "CollisionPctUnion_mean4"

SCHEMA_VERSION = "simcortex_evaluation_summary_v1.0"

LOG = logging.getLogger("simcortex.evaluation.summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SimCortex evaluation results."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Root directory containing evaluation outputs.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Default: <eval-root>/metrics.",
    )
    parser.add_argument(
        "--collisions-dir",
        type=Path,
        default=None,
        help="Default: <eval-root>/collisions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <eval-root>/summary.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional subset of dataset keys.",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=560,
        help="Expected total cases when --datasets is not supplied.",
    )
    parser.add_argument(
        "--expected-cases-per-dataset",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
    )

    args = parser.parse_args()

    eval_root = args.eval_root.expanduser().resolve()

    if args.metrics_dir is None:
        args.metrics_dir = eval_root / "metrics"
    else:
        args.metrics_dir = (
            args.metrics_dir.expanduser().resolve()
        )

    if args.collisions_dir is None:
        args.collisions_dir = eval_root / "collisions"
    else:
        args.collisions_dir = (
            args.collisions_dir.expanduser().resolve()
        )

    if args.output_dir is None:
        args.output_dir = eval_root / "summary"
    else:
        args.output_dir = (
            args.output_dir.expanduser().resolve()
        )

    args.eval_root = eval_root
    return args


def setup_logging(
    out_dir: Path,
) -> Path:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    log_path = out_dir / "summarize_results.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    return log_path


def safe_json_value(
    obj: Any,
) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(
            orient="records"
        )

    if isinstance(obj, pd.Series):
        return obj.to_dict()

    if isinstance(obj, dict):
        return {
            str(key): safe_json_value(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [
            safe_json_value(value)
            for value in obj
        ]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def normalize_case_keys(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    for column in CASE_KEY_COLS:
        if column not in out.columns:
            raise ValueError(
                f"Missing case key column: {column}"
            )

        out[column] = (
            out[column]
            .fillna("")
            .astype(str)
        )

    return out


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{label} missing required columns: "
            f"{missing}"
        )


def read_csv_allow_empty(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    try:
        return pd.read_csv(
            path,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def validate_method(
    df: pd.DataFrame,
    label: str,
) -> None:
    if "method" not in df.columns:
        raise ValueError(
            f"{label} missing method column"
        )

    methods = sorted(
        df["method"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if methods != [METHOD_NAME]:
        raise ValueError(
            f"{label} must contain only "
            f"{METHOD_NAME}; found {methods}"
        )


def filter_datasets(
    df: pd.DataFrame,
    datasets: list[str] | None,
) -> pd.DataFrame:
    if datasets is None:
        return df.copy()

    wanted = set(
        map(str, datasets)
    )

    return df[
        df["dataset"]
        .astype(str)
        .isin(wanted)
    ].copy()


def build_case_surface_metrics(
    surface_df: pd.DataFrame,
    datasets: list[str] | None,
) -> pd.DataFrame:
    required = {
        "method",
        *CASE_KEY_COLS,
        "surface",
        *SURFACE_METRICS,
    }

    require_columns(
        surface_df,
        required,
        "surface_metrics_long.csv",
    )

    validate_method(
        surface_df,
        "surface_metrics_long.csv",
    )

    df = normalize_case_keys(
        surface_df
    )

    df = filter_datasets(
        df,
        datasets,
    )

    for metric in SURFACE_METRICS:
        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce",
        )

    duplicate = df[
        df.duplicated(
            CASE_KEY_COLS + ["surface"],
            keep=False,
        )
    ]

    if len(duplicate):
        raise ValueError(
            "surface_metrics_long.csv contains "
            f"{len(duplicate)} duplicate "
            "case/surface rows"
        )

    expected_surface_string = ",".join(
        sorted(SURFACES)
    )

    per_case_surface = (
        df.groupby(
            CASE_KEY_COLS,
            dropna=False,
        )
        .agg(
            n_surface_rows=(
                "surface",
                "size",
            ),
            n_surfaces=(
                "surface",
                "nunique",
            ),
            surfaces=(
                "surface",
                lambda values: ",".join(
                    sorted(
                        set(
                            map(
                                str,
                                values,
                            )
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )

    bad = per_case_surface[
        (
            per_case_surface[
                "n_surface_rows"
            ]
            != len(SURFACES)
        )
        | (
            per_case_surface[
                "n_surfaces"
            ]
            != len(SURFACES)
        )
        | (
            per_case_surface[
                "surfaces"
            ]
            != expected_surface_string
        )
    ]

    if len(bad):
        raise ValueError(
            f"{len(bad)} cases do not have "
            "exactly the expected four surfaces"
        )

    # average the four cortical-surface metrics per case.
    out = (
        df.groupby(
            CASE_KEY_COLS,
            dropna=False,
        )[SURFACE_METRICS]
        .mean()
        .reset_index()
    )

    out.insert(
        0,
        "method",
        METHOD_NAME,
    )

    return out


def build_case_thickness_metrics(
    pair_df: pd.DataFrame,
    datasets: list[str] | None,
) -> pd.DataFrame:
    required = {
        "method",
        *CASE_KEY_COLS,
        LH_THICKNESS_COL,
        RH_THICKNESS_COL,
    }

    require_columns(
        pair_df,
        required,
        "pair_metrics_thickness_collisions.csv",
    )

    validate_method(
        pair_df,
        "pair_metrics_thickness_collisions.csv",
    )

    df = normalize_case_keys(
        pair_df
    )

    df = filter_datasets(
        df,
        datasets,
    )

    duplicate = df[
        df.duplicated(
            CASE_KEY_COLS,
            keep=False,
        )
    ]

    if len(duplicate):
        raise ValueError(
            "pair_metrics_thickness_collisions.csv "
            f"contains {len(duplicate)} duplicate "
            "case rows"
        )

    df[LH_THICKNESS_COL] = pd.to_numeric(
        df[LH_THICKNESS_COL],
        errors="coerce",
    )
    df[RH_THICKNESS_COL] = pd.to_numeric(
        df[RH_THICKNESS_COL],
        errors="coerce",
    )

    out = df[
        CASE_KEY_COLS
        + [
            LH_THICKNESS_COL,
            RH_THICKNESS_COL,
        ]
    ].copy()

    # mean of left/right thickness absolute errors.
    out["ThicknessAbsErr_mm"] = out[
        [
            LH_THICKNESS_COL,
            RH_THICKNESS_COL,
        ]
    ].mean(
        axis=1
    )

    out = out[
        CASE_KEY_COLS
        + ["ThicknessAbsErr_mm"]
    ]

    out.insert(
        0,
        "method",
        METHOD_NAME,
    )

    return out


def build_case_collision_metrics(
    union_df: pd.DataFrame,
    datasets: list[str] | None,
) -> pd.DataFrame:
    required = {
        "method",
        *CASE_KEY_COLS,
        COLLISION_SOURCE_COL,
    }

    require_columns(
        union_df,
        required,
        "collision_surface_union_case_level.csv",
    )

    validate_method(
        union_df,
        "collision_surface_union_case_level.csv",
    )

    df = normalize_case_keys(
        union_df
    )

    df = filter_datasets(
        df,
        datasets,
    )

    duplicate = df[
        df.duplicated(
            CASE_KEY_COLS,
            keep=False,
        )
    ]

    if len(duplicate):
        raise ValueError(
            "collision_surface_union_case_level.csv "
            f"contains {len(duplicate)} duplicate "
            "case rows"
        )

    if "union_status" in df.columns:
        non_ok = df[
            df["union_status"]
            .astype(str)
            != "OK"
        ]

        if len(non_ok):
            raise ValueError(
                f"{len(non_ok)} collision union rows "
                "have union_status != OK"
            )

    out = df[
        CASE_KEY_COLS
        + [COLLISION_SOURCE_COL]
    ].copy()

    out[COLLISION_OUTPUT_COL] = pd.to_numeric(
        out[COLLISION_SOURCE_COL],
        errors="coerce",
    )

    out = out[
        CASE_KEY_COLS
        + [COLLISION_OUTPUT_COL]
    ]

    out.insert(
        0,
        "method",
        METHOD_NAME,
    )

    return out


def compare_key_sets(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    label: str,
) -> list[str]:
    key = [
        "method",
        *CASE_KEY_COLS,
    ]

    reference_keys = (
        reference[key]
        .drop_duplicates()
    )
    other_keys = (
        other[key]
        .drop_duplicates()
    )

    missing = (
        reference_keys.merge(
            other_keys,
            on=key,
            how="left",
            indicator=True,
        )
        .query("_merge == 'left_only'")
    )

    extra = (
        other_keys.merge(
            reference_keys,
            on=key,
            how="left",
            indicator=True,
        )
        .query("_merge == 'left_only'")
    )

    problems: list[str] = []

    if len(missing):
        problems.append(
            f"{label}: missing {len(missing)} "
            "cases relative to surface metrics"
        )

    if len(extra):
        problems.append(
            f"{label}: has {len(extra)} "
            "extra cases relative to surface metrics"
        )

    return problems


def build_case_metrics(
    surface_case: pd.DataFrame,
    thickness_case: pd.DataFrame,
    collision_case: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    key = [
        "method",
        *CASE_KEY_COLS,
    ]

    problems: list[str] = []

    problems.extend(
        compare_key_sets(
            surface_case,
            thickness_case,
            "thickness metrics",
        )
    )
    problems.extend(
        compare_key_sets(
            surface_case,
            collision_case,
            "collision metrics",
        )
    )

    merged = surface_case.merge(
        thickness_case,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    merged = merged.merge(
        collision_case,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    merged = merged.sort_values(
        [
            "dataset",
            "case_id",
        ]
    ).reset_index(
        drop=True
    )

    return merged, problems


def summarize_series(
    values: pd.Series,
) -> dict[str, float | int]:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    return {
        "count": int(
            numeric.notna().sum()
        ),
        "mean": float(
            numeric.mean()
        ),
        "std": float(
            numeric.std()
        ),
        "median": float(
            numeric.median()
        ),
        "min": float(
            numeric.min()
        ),
        "max": float(
            numeric.max()
        ),
    }


def summarize_overall(
    case_df: pd.DataFrame,
) -> pd.DataFrame:
    row: dict[str, Any] = {
        "method": METHOD_NAME,
        "n_cases": int(
            len(case_df)
        ),
    }

    for metric in FINAL_METRICS:
        stats = summarize_series(
            case_df[metric]
        )

        for stat_name, value in stats.items():
            row[
                f"{metric}_{stat_name}"
            ] = value

    return pd.DataFrame(
        [row]
    )


def summarize_by_dataset(
    case_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for dataset, subset in case_df.groupby(
        "dataset",
        dropna=False,
        sort=True,
    ):
        row: dict[str, Any] = {
            "method": METHOD_NAME,
            "dataset": dataset,
            "n_cases": int(
                len(subset)
            ),
        }

        for metric in FINAL_METRICS:
            stats = summarize_series(
                subset[metric]
            )

            for stat_name, value in stats.items():
                row[
                    f"{metric}_{stat_name}"
                ] = value

        rows.append(row)

    return pd.DataFrame(
        rows
    ).sort_values(
        "dataset"
    ).reset_index(
        drop=True
    )


def numeric_problems(
    case_df: pd.DataFrame,
) -> list[str]:
    problems: list[str] = []

    for metric in FINAL_METRICS:
        values = pd.to_numeric(
            case_df[metric],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            problems.append(
                f"{metric}: non-finite values found"
            )
            continue

        if (
            values < 0.0
        ).any():
            problems.append(
                f"{metric}: negative values found"
            )

    collision_values = pd.to_numeric(
        case_df[
            COLLISION_OUTPUT_COL
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if np.isfinite(
        collision_values
    ).all() and (
        collision_values
        > 100.0 + 1e-9
    ).any():
        problems.append(
            "CollisionPctUnion_mean4: "
            "values above 100% found"
        )

    return problems


def count_problems(
    case_df: pd.DataFrame,
    *,
    datasets: list[str] | None,
    expected_cases: int,
    expected_cases_per_dataset: int,
) -> list[str]:
    problems: list[str] = []

    if datasets is None:
        if len(case_df) != expected_cases:
            problems.append(
                f"Expected {expected_cases} cases, "
                f"got {len(case_df)}"
            )

    expected_dataset_set = (
        None
        if datasets is None
        else set(
            map(str, datasets)
        )
    )

    observed_dataset_set = set(
        case_df["dataset"]
        .astype(str)
        .unique()
        .tolist()
    )

    if (
        expected_dataset_set is not None
        and observed_dataset_set
        != expected_dataset_set
    ):
        problems.append(
            "Dataset set mismatch: expected "
            f"{sorted(expected_dataset_set)}, "
            f"got {sorted(observed_dataset_set)}"
        )

    counts = (
        case_df.groupby(
            "dataset"
        )
        .size()
    )

    bad_counts = {
        str(dataset): int(count)
        for dataset, count in counts.items()
        if int(count)
        != expected_cases_per_dataset
    }

    if bad_counts:
        problems.append(
            "Dataset case-count problems: "
            f"{bad_counts}"
        )

    return problems


def write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    out_dir = Path(
        args.output_dir
    )

    output_paths = [
        out_dir / "case_metrics.csv",
        out_dir / "by_dataset.csv",
        out_dir / "overall.csv",
        out_dir / "run_summary.json",
        out_dir / "summarize_results.log",
    ]

    existing = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing and not args.overwrite:
        raise SystemExit(
            "Summary output files already exist: "
            + ", ".join(
                str(path)
                for path in existing
            )
            + ". Use --overwrite."
        )

    log_path = setup_logging(
        out_dir
    )

    metrics_dir = Path(
        args.metrics_dir
    )
    collisions_dir = Path(
        args.collisions_dir
    )

    surface_path = (
        metrics_dir
        / "surface_metrics_long.csv"
    )
    thickness_path = (
        metrics_dir
        / "pair_metrics_thickness_collisions.csv"
    )
    metric_fail_path = (
        metrics_dir
        / "missing_or_failed.csv"
    )

    collision_path = (
        collisions_dir
        / "collision_surface_union_case_level.csv"
    )
    collision_fail_path = (
        collisions_dir
        / "collision_missing_or_failed.csv"
    )

    LOG.info(
        "=== Summarize SimCortex evaluation ==="
    )
    LOG.info(
        "surface_metrics=%s",
        surface_path,
    )
    LOG.info(
        "thickness_metrics=%s",
        thickness_path,
    )
    LOG.info(
        "collision_union=%s",
        collision_path,
    )

    surface_df = read_csv_allow_empty(
        surface_path
    )
    thickness_df = read_csv_allow_empty(
        thickness_path
    )
    collision_df = read_csv_allow_empty(
        collision_path
    )
    metric_fail_df = read_csv_allow_empty(
        metric_fail_path
    )
    collision_fail_df = read_csv_allow_empty(
        collision_fail_path
    )

    surface_case = build_case_surface_metrics(
        surface_df,
        args.datasets,
    )
    thickness_case = build_case_thickness_metrics(
        thickness_df,
        args.datasets,
    )
    collision_case = build_case_collision_metrics(
        collision_df,
        args.datasets,
    )

    case_df, merge_problems = build_case_metrics(
        surface_case,
        thickness_case,
        collision_case,
    )

    problems: list[str] = []

    if len(metric_fail_df):
        problems.append(
            "Metric evaluation contains "
            f"{len(metric_fail_df)} "
            "missing/failed rows"
        )

    if len(collision_fail_df):
        problems.append(
            "Collision evaluation contains "
            f"{len(collision_fail_df)} "
            "missing/failed rows"
        )

    problems.extend(
        merge_problems
    )

    problems.extend(
        numeric_problems(
            case_df
        )
    )

    problems.extend(
        count_problems(
            case_df,
            datasets=args.datasets,
            expected_cases=int(
                args.expected_cases
            ),
            expected_cases_per_dataset=int(
                args.expected_cases_per_dataset
            ),
        )
    )

    overall_df = summarize_overall(
        case_df
    )
    by_dataset_df = summarize_by_dataset(
        case_df
    )

    case_path = (
        out_dir
        / "case_metrics.csv"
    )
    by_dataset_path = (
        out_dir
        / "by_dataset.csv"
    )
    overall_path = (
        out_dir
        / "overall.csv"
    )
    summary_path = (
        out_dir
        / "run_summary.json"
    )

    write_csv(
        case_df,
        case_path,
    )
    write_csv(
        by_dataset_df,
        by_dataset_path,
    )
    write_csv(
        overall_df,
        overall_path,
    )

    summary = {
        "stage": "summarize_results",
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "eval_root": args.eval_root,
        "metrics_dir": metrics_dir,
        "collisions_dir": collisions_dir,
        "output_dir": out_dir,
        "datasets": args.datasets,
        "expected_cases": int(
            args.expected_cases
        ),
        "expected_cases_per_dataset": int(
            args.expected_cases_per_dataset
        ),
        "n_cases": int(
            len(case_df)
        ),
        "n_datasets": int(
            case_df["dataset"].nunique()
        ),
        "metric_missing_or_failed_rows": int(
            len(metric_fail_df)
        ),
        "collision_missing_or_failed_rows": int(
            len(collision_fail_df)
        ),
        "strict": bool(
            args.strict
        ),
        "problems": problems,
        "definitions": {
            "surface_case_aggregation": (
                "Mean across lh_white, lh_pial, "
                "rh_white, rh_pial."
            ),
            "ThicknessAbsErr_mm": (
                "Mean of left and right hemisphere "
                "thickness absolute errors."
            ),
            "CollisionPctUnion_mean4": (
                "collision_pct_union_mean4 copied "
                "directly from evaluate_collisions.py."
            ),
            "summary_statistics": [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ],
        },
        "final_metrics": FINAL_METRICS,
        "outputs": {
            "case_metrics": case_path,
            "by_dataset": by_dataset_path,
            "overall": overall_path,
            "run_summary": summary_path,
            "log": log_path,
        },
    }

    summary_path.write_text(
        json.dumps(
            safe_json_value(
                summary
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    LOG.info(
        "Cases: %d",
        len(case_df),
    )
    LOG.info(
        "Datasets: %d",
        case_df[
            "dataset"
        ].nunique(),
    )
    LOG.info(
        "Problems: %d",
        len(problems),
    )

    if problems:
        for problem in problems:
            LOG.error(
                " - %s",
                problem,
            )

        if args.strict:
            raise SystemExit(1)

    LOG.info(
        "Result summary complete."
    )


if __name__ == "__main__":
    main()
