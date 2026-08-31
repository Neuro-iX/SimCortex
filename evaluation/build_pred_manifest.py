#!/usr/bin/env python3
"""Build the SimCortex prediction manifest for evaluation.

The input is a SimCortex prediction root produced by the public inference
pipeline. Predictions are expected to contain the four native-space cortical
surfaces for every evaluation case.

Outputs
-------
  <eval-root>/manifests/pred_manifest.tsv
  <eval-root>/reports/pred_manifest_qc_report.json
  <eval-root>/reports/pred_manifest.log

SimCortex native-space cortical surfaces are treated as scannerRAS-compatible,
matching the validated evaluation convention used by the original benchmark.
No coordinate conversion is performed by this stage.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm


METHOD_NAME = "SimCortex"

SURFACES = [
    "lh_white",
    "lh_pial",
    "rh_white",
    "rh_pial",
]

EXPECTED_SURFACE_SET = set(SURFACES)

CASE_REQUIRED_COLUMNS = {
    "dataset",
    "sample_dataset_dir",
    "subject",
    "session",
    "case_id",
    "status",
}

OUTPUT_COLUMNS = [
    "method",
    "dataset",
    "sample_dataset_dir",
    "subject",
    "session",
    "case_id",
    "surface",
    "pred_path",
    "path_exists",
    "candidate_paths",
    "candidate_index",
    "selected_variant",
    "raw_format",
    "raw_space_assumption",
    "conversion_required",
    "status",
    "load_error",
    "n_vertices",
    "n_faces",
    "finite_vertices",
    "nan_count",
    "inf_count",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_center_z",
    "extent_x",
    "extent_y",
    "extent_z",
    "bbox_diag",
]

LOG = logging.getLogger("simcortex.evaluation.pred_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SimCortex prediction manifest."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Root directory for evaluation inputs and outputs.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        required=True,
        help=(
            "Root containing SimCortex prediction dataset directories, "
            "for example <pred-root>/<dataset>/<subject>/<session>/surfaces/."
        ),
    )
    parser.add_argument(
        "--case-manifest",
        type=Path,
        default=None,
        help="Default: <eval-root>/manifests/case_manifest.tsv.",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="Default: <eval-root>/manifests/pred_manifest.tsv.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Default: <eval-root>/reports.",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=560,
    )
    parser.add_argument(
        "--expected-cases-per-dataset",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--expected-surfaces-per-case",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest and report files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if manifest validation problems are found.",
    )

    args = parser.parse_args()

    eval_root = args.eval_root.expanduser().resolve()
    pred_root = args.pred_root.expanduser().resolve()

    if args.case_manifest is None:
        args.case_manifest = eval_root / "manifests" / "case_manifest.tsv"
    else:
        args.case_manifest = args.case_manifest.expanduser().resolve()

    if args.out_manifest is None:
        args.out_manifest = eval_root / "manifests" / "pred_manifest.tsv"
    else:
        args.out_manifest = args.out_manifest.expanduser().resolve()

    if args.report_dir is None:
        args.report_dir = eval_root / "reports"
    else:
        args.report_dir = args.report_dir.expanduser().resolve()

    args.eval_root = eval_root
    args.pred_root = pred_root
    return args


def setup_logging(report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "pred_manifest.log"

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


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()

    try:
        scalar_na = pd.isna(obj)
        if isinstance(
            scalar_na,
            (bool, np.bool_),
        ) and bool(scalar_na):
            return None
    except Exception:
        pass

    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_safe(obj),
            handle,
            indent=2,
        )


def write_tsv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(
        path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )


def dataframe_with_columns(
    rows: list[dict[str, Any]],
    columns: Sequence[str],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA

    extra = [
        column
        for column in df.columns
        if column not in columns
    ]

    return df[list(columns) + extra]


def hemi_label(
    surface: str,
) -> tuple[str, str, str]:
    hemi, surf = surface.split(
        "_",
        1,
    )
    hemi_lr = "L" if hemi == "lh" else "R"
    return hemi, hemi_lr, surf


def first_existing(
    candidates: list[Path],
) -> tuple[Path, list[str], int]:
    if not candidates:
        raise ValueError(
            "candidate path list is empty"
        )

    for index, path in enumerate(candidates):
        if path.exists():
            return (
                path,
                [str(p) for p in candidates],
                index,
            )

    return (
        candidates[0],
        [str(p) for p in candidates],
        -1,
    )


def prediction_path_candidates(
    *,
    pred_root: Path,
    dataset: str,
    sample_dataset_dir: str,
    subject: str,
    session: str,
    case_id: str,
    surface: str,
) -> list[Path]:
    """Return deterministic SimCortex prediction path candidates."""
    _, hemi_lr, surf = hemi_label(surface)

    filename = (
        f"{subject}_{session}_space-native_"
        f"desc-deform_hemi-{hemi_lr}_{surf}.surf.ply"
    )

    case_filename = (
        f"{case_id}_space-native_"
        f"desc-deform_hemi-{hemi_lr}_{surf}.surf.ply"
    )

    candidates = [
        pred_root
        / dataset
        / subject
        / session
        / "surfaces"
        / filename,
        pred_root
        / dataset
        / subject
        / "surfaces"
        / filename,
        pred_root
        / dataset
        / case_id
        / "surfaces"
        / case_filename,
    ]

    # Some evaluation input directories use a dataset-directory name that
    # differs from the canonical dataset key.
    if sample_dataset_dir != dataset:
        candidates.append(
            pred_root
            / sample_dataset_dir
            / subject
            / session
            / "surfaces"
            / filename
        )

    return candidates


def validate_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path,
) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Invalid vertices shape for {path}: "
            f"{vertices.shape}"
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Invalid faces shape for {path}: "
            f"{faces.shape}"
        )

    if len(vertices) == 0:
        raise ValueError(
            f"Zero vertices in {path}"
        )

    if len(faces) == 0:
        raise ValueError(
            f"Zero faces in {path}"
        )

    if (
        faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError(
            f"Face index out of range for {path}: "
            f"min={int(faces.min())}, "
            f"max={int(faces.max())}, "
            f"n_vertices={len(vertices)}"
        )


def bbox_basic(
    vertices: np.ndarray,
) -> dict[str, float]:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (
        minimum + maximum
    )
    extent = maximum - minimum

    return {
        "bbox_center_x": float(center[0]),
        "bbox_center_y": float(center[1]),
        "bbox_center_z": float(center[2]),
        "extent_x": float(extent[0]),
        "extent_y": float(extent[1]),
        "extent_z": float(extent[2]),
        "bbox_diag": float(
            np.linalg.norm(extent)
        ),
    }


def load_mesh_basic(
    path: Path,
) -> dict[str, Any]:
    mesh = trimesh.load(
        str(path),
        process=False,
    )

    if isinstance(mesh, trimesh.Scene):
        geometries = list(
            mesh.geometry.values()
        )
        if not geometries:
            raise ValueError(
                f"Empty scene: {path}"
            )
        mesh = trimesh.util.concatenate(
            geometries
        )

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )
    faces = np.asarray(
        mesh.faces,
        dtype=np.int64,
    )

    validate_mesh_arrays(
        vertices,
        faces,
        path,
    )

    finite = bool(
        np.isfinite(vertices).all()
    )

    result: dict[str, Any] = {
        "n_vertices": int(vertices.shape[0]),
        "n_faces": int(faces.shape[0]),
        "finite_vertices": finite,
        "nan_count": int(
            np.isnan(vertices).sum()
        ),
        "inf_count": int(
            np.isinf(vertices).sum()
        ),
    }

    if finite:
        result.update(
            bbox_basic(vertices)
        )
    else:
        result.update(
            {
                "bbox_center_x": np.nan,
                "bbox_center_y": np.nan,
                "bbox_center_z": np.nan,
                "extent_x": np.nan,
                "extent_y": np.nan,
                "extent_z": np.nan,
                "bbox_diag": np.nan,
            }
        )

    return result


def validate_case_manifest(
    cases: pd.DataFrame,
    expected_cases: int,
    expected_cases_per_dataset: int,
) -> list[str]:
    problems: list[str] = []

    missing = (
        CASE_REQUIRED_COLUMNS
        - set(cases.columns)
    )

    if missing:
        return [
            "case_manifest is missing required "
            f"columns: {sorted(missing)}"
        ]

    if len(cases) != expected_cases:
        problems.append(
            f"Expected {expected_cases} cases, "
            f"got {len(cases)}"
        )

    non_ok = cases[
        cases["status"].astype(str) != "OK"
    ]
    if len(non_ok):
        problems.append(
            "case_manifest has "
            f"{len(non_ok)} non-OK rows"
        )

    duplicate = cases[
        cases.duplicated(
            ["dataset", "case_id"],
            keep=False,
        )
    ]
    if len(duplicate):
        problems.append(
            "case_manifest has "
            f"{len(duplicate)} duplicate "
            "dataset/case_id rows"
        )

    cases_by_dataset = cases.groupby(
        "dataset"
    )["case_id"].nunique()

    bad_counts = cases_by_dataset[
        cases_by_dataset
        != expected_cases_per_dataset
    ]

    if len(bad_counts):
        problems.append(
            "Datasets with case count != "
            f"{expected_cases_per_dataset}: "
            + json.dumps(
                {
                    str(k): int(v)
                    for k, v
                    in bad_counts.items()
                }
            )
        )

    return problems


def build_manifest(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cases = pd.read_csv(
        args.case_manifest,
        sep="\t",
        low_memory=False,
    )

    case_problems = validate_case_manifest(
        cases=cases,
        expected_cases=int(
            args.expected_cases
        ),
        expected_cases_per_dataset=int(
            args.expected_cases_per_dataset
        ),
    )

    if case_problems:
        raise SystemExit(
            "Invalid case manifest:\n"
            + "\n".join(
                f" - {problem}"
                for problem in case_problems
            )
        )

    rows: list[dict[str, Any]] = []

    iterator = [
        (case, surface)
        for _, case in cases.iterrows()
        for surface in SURFACES
    ]

    for case, surface in tqdm(
        iterator,
        desc="Build prediction manifest",
    ):
        dataset = str(
            case["dataset"]
        )
        sample_dataset_dir = str(
            case["sample_dataset_dir"]
        )
        subject = str(
            case["subject"]
        )
        session = str(
            case["session"]
        )
        case_id = str(
            case["case_id"]
        )

        candidates = prediction_path_candidates(
            pred_root=args.pred_root,
            dataset=dataset,
            sample_dataset_dir=sample_dataset_dir,
            subject=subject,
            session=session,
            case_id=case_id,
            surface=surface,
        )

        pred_path, candidate_paths, candidate_index = (
            first_existing(candidates)
        )

        row: dict[str, Any] = {
            "method": METHOD_NAME,
            "dataset": dataset,
            "sample_dataset_dir": sample_dataset_dir,
            "subject": subject,
            "session": session,
            "case_id": case_id,
            "surface": surface,
            "pred_path": str(pred_path),
            "path_exists": bool(
                pred_path.exists()
            ),
            "candidate_paths": " | ".join(
                candidate_paths
            ),
            "candidate_index": int(
                candidate_index
            ),
            "selected_variant": (
                "space-native_desc-deform"
            ),
            "raw_format": "ply",
            "raw_space_assumption": (
                "native_RAS_scannerRAS_compatible"
            ),
            "conversion_required": "none",
            "status": "",
            "load_error": "",
        }

        if not pred_path.exists():
            row.update(
                {
                    "status": "MISSING_PRED",
                    "n_vertices": "",
                    "n_faces": "",
                    "finite_vertices": "",
                    "nan_count": "",
                    "inf_count": "",
                }
            )
            rows.append(row)
            continue

        try:
            basic = load_mesh_basic(
                pred_path
            )
            row.update(basic)
            row["status"] = (
                "OK"
                if basic["finite_vertices"]
                else "NONFINITE_PRED"
            )
        except Exception as exc:
            row.update(
                {
                    "status": (
                        "PRED_LOAD_ERROR:"
                        f"{repr(exc)}"
                    ),
                    "load_error": repr(exc),
                    "n_vertices": "",
                    "n_faces": "",
                    "finite_vertices": "",
                    "nan_count": "",
                    "inf_count": "",
                }
            )

        rows.append(row)

    df = dataframe_with_columns(
        rows,
        OUTPUT_COLUMNS,
    )

    summary = summarize_manifest(
        df=df,
        expected_cases=int(
            args.expected_cases
        ),
        expected_cases_per_dataset=int(
            args.expected_cases_per_dataset
        ),
        expected_surfaces_per_case=int(
            args.expected_surfaces_per_case
        ),
    )

    return df, summary


def summarize_manifest(
    *,
    df: pd.DataFrame,
    expected_cases: int,
    expected_cases_per_dataset: int,
    expected_surfaces_per_case: int,
) -> dict[str, Any]:
    problems: list[str] = []

    expected_rows = (
        expected_cases
        * len(SURFACES)
    )

    expected_rows_per_dataset = (
        expected_cases_per_dataset
        * expected_surfaces_per_case
    )

    if len(df) != expected_rows:
        problems.append(
            f"Expected {expected_rows} rows, "
            f"got {len(df)}"
        )

    status_counts = df[
        "status"
    ].value_counts(
        dropna=False
    )

    ok = df[
        df["status"] == "OK"
    ].copy()

    if len(ok) != expected_rows:
        problems.append(
            f"Expected {expected_rows} OK rows, "
            f"got {len(ok)}"
        )

    bad = df[
        df["status"] != "OK"
    ].copy()

    if len(bad):
        problems.append(
            f"{len(bad)} prediction rows "
            "are not OK"
        )

    duplicate = df[
        df.duplicated(
            [
                "dataset",
                "case_id",
                "surface",
            ],
            keep=False,
        )
    ].copy()

    if len(duplicate):
        problems.append(
            f"{len(duplicate)} duplicate "
            "dataset/case_id/surface rows found"
        )

    expected_surfaces = ",".join(
        sorted(EXPECTED_SURFACE_SET)
    )

    if not ok.empty:
        per_case = (
            ok.groupby(
                ["dataset", "case_id"]
            )["surface"]
            .agg(
                lambda values: ",".join(
                    sorted(
                        set(
                            map(
                                str,
                                values,
                            )
                        )
                    )
                )
            )
            .reset_index(
                name="surfaces"
            )
        )

        per_case["n_surfaces"] = (
            per_case["surfaces"].apply(
                lambda value: (
                    len(value.split(","))
                    if value
                    else 0
                )
            )
        )

        per_case[
            "expected_surfaces"
        ] = expected_surfaces

        per_case[
            "has_expected_surface_set"
        ] = per_case[
            "surfaces"
        ].eq(
            expected_surfaces
        )

        bad_case = per_case[
            (
                per_case["n_surfaces"]
                != len(SURFACES)
            )
            | (
                ~per_case[
                    "has_expected_surface_set"
                ]
            )
        ].copy()

        if len(bad_case):
            problems.append(
                f"{len(bad_case)} dataset/case "
                "groups do not have exactly "
                "the expected four surfaces"
            )

        dataset_counts = (
            ok.groupby("dataset")
            .agg(
                n_rows=(
                    "surface",
                    "size",
                ),
                n_cases=(
                    "case_id",
                    "nunique",
                ),
                n_surfaces=(
                    "surface",
                    "nunique",
                ),
            )
            .reset_index()
            .sort_values("dataset")
        )

        bad_dataset = dataset_counts[
            (
                dataset_counts["n_cases"]
                != expected_cases_per_dataset
            )
            | (
                dataset_counts["n_rows"]
                != expected_rows_per_dataset
            )
            | (
                dataset_counts["n_surfaces"]
                != expected_surfaces_per_case
            )
        ].copy()

        if len(bad_dataset):
            problems.append(
                f"{len(bad_dataset)} datasets "
                "do not have the expected "
                "case/row/surface counts"
            )
    else:
        bad_case = pd.DataFrame()
        dataset_counts = pd.DataFrame()
        problems.append(
            "No OK prediction rows found"
        )

    return {
        "expected_rows": int(expected_rows),
        "n_rows": int(len(df)),
        "n_ok_rows": int(len(ok)),
        "n_non_ok_rows": int(len(bad)),
        "status_counts": {
            str(k): int(v)
            for k, v
            in status_counts.to_dict().items()
        },
        "bad_rows": bad,
        "duplicate_rows": duplicate,
        "bad_case_rows": bad_case,
        "dataset_counts": dataset_counts,
        "problems": problems,
    }


def fail_if_exists(
    paths: Iterable[Path],
    overwrite: bool,
) -> None:
    if overwrite:
        return

    existing = [
        str(path)
        for path in paths
        if Path(path).exists()
    ]

    if existing:
        raise FileExistsError(
            "Prediction manifest output files "
            "already exist. Use --overwrite:\n"
            + "\n".join(existing)
        )


def main() -> None:
    args = parse_args()

    output_paths = [
        args.out_manifest,
        args.report_dir
        / "pred_manifest.log",
        args.report_dir
        / "pred_manifest_dataset_counts.tsv",
        args.report_dir
        / "pred_manifest_non_ok_rows.tsv",
        args.report_dir
        / "pred_manifest_duplicate_rows.tsv",
        args.report_dir
        / "pred_manifest_bad_surface_count.tsv",
        args.report_dir
        / "pred_manifest_qc_report.json",
    ]

    fail_if_exists(
        output_paths,
        overwrite=bool(args.overwrite),
    )

    log_path = setup_logging(
        args.report_dir
    )

    LOG.info(
        "=== Build SimCortex prediction manifest ==="
    )
    LOG.info(
        "eval_root=%s",
        args.eval_root,
    )
    LOG.info(
        "case_manifest=%s",
        args.case_manifest,
    )
    LOG.info(
        "pred_root=%s",
        args.pred_root,
    )
    LOG.info(
        "out_manifest=%s",
        args.out_manifest,
    )

    df, summary = build_manifest(args)

    write_tsv(
        df,
        args.out_manifest,
    )

    write_tsv(
        summary["dataset_counts"],
        args.report_dir
        / "pred_manifest_dataset_counts.tsv",
    )

    write_tsv(
        summary["bad_rows"],
        args.report_dir
        / "pred_manifest_non_ok_rows.tsv",
    )

    write_tsv(
        summary["duplicate_rows"],
        args.report_dir
        / "pred_manifest_duplicate_rows.tsv",
    )

    write_tsv(
        summary["bad_case_rows"],
        args.report_dir
        / "pred_manifest_bad_surface_count.tsv",
    )

    report = {
        "stage": "build_pred_manifest",
        "schema_version": (
            "simcortex_evaluation_pred_manifest_v1.0"
        ),
        "method": METHOD_NAME,
        "eval_root": args.eval_root,
        "case_manifest": args.case_manifest,
        "pred_root": args.pred_root,
        "pred_manifest": args.out_manifest,
        "expected_cases": int(
            args.expected_cases
        ),
        "expected_cases_per_dataset": int(
            args.expected_cases_per_dataset
        ),
        "expected_surfaces_per_case": int(
            args.expected_surfaces_per_case
        ),
        "expected_surfaces": SURFACES,
        "n_rows": summary["n_rows"],
        "expected_rows": summary[
            "expected_rows"
        ],
        "n_ok_rows": summary[
            "n_ok_rows"
        ],
        "n_non_ok_rows": summary[
            "n_non_ok_rows"
        ],
        "status_counts": summary[
            "status_counts"
        ],
        "problems": summary["problems"],
        "strict": bool(args.strict),
        "reports": {
            "dataset_counts": (
                args.report_dir
                / "pred_manifest_dataset_counts.tsv"
            ),
            "non_ok_rows": (
                args.report_dir
                / "pred_manifest_non_ok_rows.tsv"
            ),
            "duplicate_rows": (
                args.report_dir
                / "pred_manifest_duplicate_rows.tsv"
            ),
            "bad_case_rows": (
                args.report_dir
                / "pred_manifest_bad_surface_count.tsv"
            ),
            "log": log_path,
        },
    }

    report_path = (
        args.report_dir
        / "pred_manifest_qc_report.json"
    )

    write_json(
        report_path,
        report,
    )

    LOG.info(
        "Rows: %d",
        summary["n_rows"],
    )
    LOG.info(
        "Expected rows: %d",
        summary["expected_rows"],
    )
    LOG.info(
        "OK rows: %d",
        summary["n_ok_rows"],
    )
    LOG.info(
        "Status counts: %s",
        summary["status_counts"],
    )

    if summary["problems"]:
        LOG.error(
            "Prediction manifest QC found "
            "%d problem(s):",
            len(summary["problems"]),
        )
        for problem in summary[
            "problems"
        ]:
            LOG.error(
                " - %s",
                problem,
            )

        if args.strict:
            raise SystemExit(
                "Prediction manifest failed "
                "strict validation."
            )

    LOG.info(
        "Prediction manifest QC passed."
    )


if __name__ == "__main__":
    main()
