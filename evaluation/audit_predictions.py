#!/usr/bin/env python3
"""Audit SimCortex predictions before metric and collision evaluation.

This stage validates the prediction manifest produced by
``build_pred_manifest.py`` and checks each SimCortex native-space prediction
against the corresponding scannerRAS ground-truth surface.

SimCortex ``space-native_desc-deform`` outputs are used as-is. No coordinate
transform or mesh rewrite is performed.

Outputs
-------
  <eval-root>/reports/prediction_audit.tsv
  <eval-root>/reports/prediction_audit_failures.tsv
  <eval-root>/reports/prediction_alignment_warnings.tsv
  <eval-root>/reports/prediction_audit_dataset_counts.tsv
  <eval-root>/reports/prediction_audit_report.json
  <eval-root>/reports/prediction_audit.log
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

EXPECTED_VARIANT = "space-native_desc-deform"
EXPECTED_RAW_FORMAT = "ply"
EXPECTED_SPACE = "native_RAS_scannerRAS_compatible"
EXPECTED_CONVERSION_REQUIRED = "none"
CONVERSION_APPLIED = "none_space_native_as_is"

REQUIRED_PRED_COLUMNS = {
    "method",
    "dataset",
    "subject",
    "session",
    "case_id",
    "surface",
    "pred_path",
    "selected_variant",
    "raw_format",
    "raw_space_assumption",
    "conversion_required",
    "status",
    "n_vertices",
    "n_faces",
    "finite_vertices",
}

REQUIRED_GT_COLUMNS = {
    "dataset",
    "case_id",
    "surface",
    "output_path",
    "status",
}

AUDIT_COLUMNS = [
    "method",
    "dataset",
    "subject",
    "session",
    "case_id",
    "surface",
    "pred_path",
    "gt_path",
    "selected_variant",
    "raw_format",
    "raw_space_assumption",
    "conversion_required",
    "conversion_applied",
    "pred_n_vertices",
    "pred_n_faces",
    "gt_n_vertices",
    "gt_n_faces",
    "finite_pred_vertices",
    "pred_gt_centroid_dist_mm",
    "pred_gt_bbox_center_dist_mm",
    "pred_gt_extent_max_abs_diff_mm",
    "pred_gt_diag_diff_mm",
    "alignment_warning",
    "audit_status",
    "audit_error",
]

LOG = logging.getLogger("simcortex.evaluation.prediction_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SimCortex predictions before evaluation."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Root directory for evaluation inputs and outputs.",
    )
    parser.add_argument(
        "--pred-manifest",
        type=Path,
        default=None,
        help="Default: <eval-root>/manifests/pred_manifest.tsv.",
    )
    parser.add_argument(
        "--gt-manifest",
        type=Path,
        default=None,
        help="Default: <eval-root>/manifests/gt_scannerRAS_manifest.tsv.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Default: <eval-root>/reports.",
    )
    parser.add_argument(
        "--expected-n-datasets",
        type=int,
        default=14,
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
        "--center-warn-mm",
        type=float,
        default=15.0,
        help="Centroid/bbox-center distance threshold for alignment warnings.",
    )
    parser.add_argument(
        "--extent-warn-mm",
        type=float,
        default=20.0,
        help="Maximum extent-difference threshold for alignment warnings.",
    )
    parser.add_argument(
        "--fail-on-alignment-warnings",
        action="store_true",
        help="Treat alignment warnings as audit failures.",
    )

    args = parser.parse_args()

    eval_root = args.eval_root.expanduser().resolve()

    if args.pred_manifest is None:
        args.pred_manifest = eval_root / "manifests" / "pred_manifest.tsv"
    else:
        args.pred_manifest = args.pred_manifest.expanduser().resolve()

    if args.gt_manifest is None:
        args.gt_manifest = (
            eval_root
            / "manifests"
            / "gt_scannerRAS_manifest.tsv"
        )
    else:
        args.gt_manifest = args.gt_manifest.expanduser().resolve()

    if args.report_dir is None:
        args.report_dir = eval_root / "reports"
    else:
        args.report_dir = args.report_dir.expanduser().resolve()

    args.eval_root = eval_root
    return args


def setup_logging(report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "prediction_audit.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

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
    if isinstance(obj, (pd.Series, pd.Index)):
        return json_safe(obj.tolist())

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


def write_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_safe(payload),
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
    rows: List[Dict[str, Any]],
    columns: List[str],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    df = pd.DataFrame(rows)

    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA

    extra = [
        column
        for column in df.columns
        if column not in columns
    ]

    return df[columns + extra]


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(
        required - set(df.columns)
    )
    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{missing}"
        )


def bool_like(value: Any) -> bool | None:
    try:
        scalar_na = pd.isna(value)
        if isinstance(
            scalar_na,
            (bool, np.bool_),
        ) and bool(scalar_na):
            return None
    except Exception:
        pass

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
        return None

    if isinstance(value, (float, np.floating)):
        if float(value) in (0.0, 1.0):
            return bool(int(value))
        return None

    text = str(value).strip().lower()

    if text in {
        "true",
        "t",
        "yes",
        "y",
        "1",
    }:
        return True

    if text in {
        "false",
        "f",
        "no",
        "n",
        "0",
        "",
        "none",
        "nan",
        "na",
        "null",
    }:
        return False

    return None


def validate_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path,
) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Invalid vertices in {path}: "
            f"shape={vertices.shape}"
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Invalid faces in {path}: "
            f"shape={faces.shape}"
        )

    if len(vertices) == 0:
        raise ValueError(
            f"Zero vertices in {path}"
        )

    if len(faces) == 0:
        raise ValueError(
            f"Zero faces in {path}"
        )

    if not np.isfinite(vertices).all():
        raise ValueError(
            f"Non-finite vertices in {path}"
        )

    if (
        faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError(
            f"Face index out of bounds in {path}: "
            f"min={int(faces.min())}, "
            f"max={int(faces.max())}, "
            f"n_vertices={len(vertices)}"
        )


def load_ply(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(
        str(path),
        process=False,
    )

    if isinstance(mesh, trimesh.Scene):
        geoms = list(
            mesh.geometry.values()
        )
        if not geoms:
            raise RuntimeError(
                f"Empty scene: {path}"
            )
        mesh = trimesh.util.concatenate(
            geoms
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

    return vertices, faces


# Keep these calculations identical to the historical scannerRAS audit.
def bbox_stats(v: np.ndarray) -> Dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    center = 0.5 * (mn + mx)
    extent = mx - mn
    centroid = v.mean(axis=0)
    return {
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "bbox_center_x": float(center[0]),
        "bbox_center_y": float(center[1]),
        "bbox_center_z": float(center[2]),
        "extent_x": float(extent[0]),
        "extent_y": float(extent[1]),
        "extent_z": float(extent[2]),
        "bbox_diag": float(np.linalg.norm(extent)),
    }


def compare_vertices(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    ps = bbox_stats(pred)
    gs = bbox_stats(gt)

    pred_centroid = np.array(
        [
            ps["centroid_x"],
            ps["centroid_y"],
            ps["centroid_z"],
        ]
    )
    gt_centroid = np.array(
        [
            gs["centroid_x"],
            gs["centroid_y"],
            gs["centroid_z"],
        ]
    )

    pred_bbox = np.array(
        [
            ps["bbox_center_x"],
            ps["bbox_center_y"],
            ps["bbox_center_z"],
        ]
    )
    gt_bbox = np.array(
        [
            gs["bbox_center_x"],
            gs["bbox_center_y"],
            gs["bbox_center_z"],
        ]
    )

    pred_ext = np.array(
        [
            ps["extent_x"],
            ps["extent_y"],
            ps["extent_z"],
        ]
    )
    gt_ext = np.array(
        [
            gs["extent_x"],
            gs["extent_y"],
            gs["extent_z"],
        ]
    )

    return {
        "pred_gt_centroid_dist_mm": float(
            np.linalg.norm(
                pred_centroid - gt_centroid
            )
        ),
        "pred_gt_bbox_center_dist_mm": float(
            np.linalg.norm(
                pred_bbox - gt_bbox
            )
        ),
        "pred_gt_extent_max_abs_diff_mm": float(
            np.max(
                np.abs(
                    pred_ext - gt_ext
                )
            )
        ),
        "pred_gt_diag_diff_mm": float(
            abs(
                ps["bbox_diag"]
                - gs["bbox_diag"]
            )
        ),
    }


def build_gt_map(
    gt_df: pd.DataFrame,
) -> Dict[Tuple[str, str, str], Path]:
    require_columns(
        gt_df,
        REQUIRED_GT_COLUMNS,
        "GT manifest",
    )

    ok_gt = gt_df[
        gt_df["status"].astype(str) == "OK"
    ].copy()

    duplicate = ok_gt[
        ok_gt.duplicated(
            [
                "dataset",
                "case_id",
                "surface",
            ],
            keep=False,
        )
    ]

    if len(duplicate):
        raise ValueError(
            f"GT manifest contains "
            f"{len(duplicate)} duplicate OK "
            "dataset/case_id/surface rows"
        )

    return {
        (
            str(row.dataset),
            str(row.case_id),
            str(row.surface),
        ): Path(
            str(row.output_path)
        )
        for row in ok_gt.itertuples(
            index=False
        )
    }


def audit_one(
    row: Dict[str, Any],
    *,
    gt_map: Dict[
        Tuple[str, str, str],
        Path,
    ],
    center_warn_mm: float,
    extent_warn_mm: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "method": str(
            row.get("method", "")
        ),
        "dataset": str(
            row.get("dataset", "")
        ),
        "subject": str(
            row.get("subject", "")
        ),
        "session": str(
            row.get("session", "")
        ),
        "case_id": str(
            row.get("case_id", "")
        ),
        "surface": str(
            row.get("surface", "")
        ),
        "pred_path": str(
            row.get("pred_path", "")
        ),
        "gt_path": "",
        "selected_variant": str(
            row.get("selected_variant", "")
        ),
        "raw_format": str(
            row.get("raw_format", "")
        ),
        "raw_space_assumption": str(
            row.get(
                "raw_space_assumption",
                "",
            )
        ),
        "conversion_required": str(
            row.get(
                "conversion_required",
                "",
            )
        ),
        "conversion_applied": (
            CONVERSION_APPLIED
        ),
        "audit_status": "FAILED",
        "audit_error": "",
    }

    try:
        method = result["method"]
        dataset = result["dataset"]
        case_id = result["case_id"]
        surface = result["surface"]

        if method != METHOD_NAME:
            raise ValueError(
                f"Unexpected method: {method!r}"
            )

        if surface not in EXPECTED_SURFACE_SET:
            raise ValueError(
                f"Unexpected surface: {surface!r}"
            )

        if str(row.get("status", "")) != "OK":
            raise ValueError(
                "Prediction manifest row is not OK: "
                f"{row.get('status')!r}"
            )

        if (
            result["selected_variant"]
            != EXPECTED_VARIANT
        ):
            raise ValueError(
                "Unexpected selected_variant: "
                f"{result['selected_variant']!r}"
            )

        if (
            result["raw_format"]
            != EXPECTED_RAW_FORMAT
        ):
            raise ValueError(
                "Unexpected raw_format: "
                f"{result['raw_format']!r}"
            )

        if (
            result["raw_space_assumption"]
            != EXPECTED_SPACE
        ):
            raise ValueError(
                "Unexpected raw_space_assumption: "
                f"{result['raw_space_assumption']!r}"
            )

        if (
            result["conversion_required"]
            != EXPECTED_CONVERSION_REQUIRED
        ):
            raise ValueError(
                "Unexpected conversion_required: "
                f"{result['conversion_required']!r}"
            )

        manifest_finite = bool_like(
            row.get("finite_vertices")
        )
        if manifest_finite is not True:
            raise ValueError(
                "Prediction manifest does not "
                "report finite_vertices=True"
            )

        pred_path = Path(
            result["pred_path"]
        )
        if not pred_path.exists():
            raise FileNotFoundError(
                f"Missing prediction: {pred_path}"
            )

        gt_key = (
            dataset,
            case_id,
            surface,
        )
        if gt_key not in gt_map:
            raise KeyError(
                "Missing GT scannerRAS row for "
                f"{gt_key}"
            )

        gt_path = gt_map[gt_key]
        result["gt_path"] = str(gt_path)

        if not gt_path.exists():
            raise FileNotFoundError(
                f"Missing GT surface: {gt_path}"
            )

        pred_v, pred_f = load_ply(
            pred_path
        )
        gt_v, gt_f = load_ply(
            gt_path
        )

        result[
            "pred_n_vertices"
        ] = int(len(pred_v))
        result[
            "pred_n_faces"
        ] = int(len(pred_f))
        result[
            "gt_n_vertices"
        ] = int(len(gt_v))
        result[
            "gt_n_faces"
        ] = int(len(gt_f))
        result[
            "finite_pred_vertices"
        ] = bool(
            np.isfinite(pred_v).all()
        )

        manifest_n_vertices = pd.to_numeric(
            pd.Series(
                [row.get("n_vertices")]
            ),
            errors="coerce",
        ).iloc[0]

        manifest_n_faces = pd.to_numeric(
            pd.Series(
                [row.get("n_faces")]
            ),
            errors="coerce",
        ).iloc[0]

        if (
            not np.isfinite(
                manifest_n_vertices
            )
            or int(manifest_n_vertices)
            != len(pred_v)
        ):
            raise ValueError(
                "Prediction vertex count differs "
                "from pred_manifest.tsv"
            )

        if (
            not np.isfinite(
                manifest_n_faces
            )
            or int(manifest_n_faces)
            != len(pred_f)
        ):
            raise ValueError(
                "Prediction face count differs "
                "from pred_manifest.tsv"
            )

        comparison = compare_vertices(
            pred_v,
            gt_v,
        )
        result.update(comparison)

        suspicious = (
            float(
                result[
                    "pred_gt_centroid_dist_mm"
                ]
            )
            > center_warn_mm
            or float(
                result[
                    "pred_gt_bbox_center_dist_mm"
                ]
            )
            > center_warn_mm
            or float(
                result[
                    "pred_gt_extent_max_abs_diff_mm"
                ]
            )
            > extent_warn_mm
        )

        result[
            "alignment_warning"
        ] = bool(suspicious)

        result["audit_status"] = "OK"
        result["audit_error"] = ""

    except Exception as exc:
        result["audit_status"] = "FAILED"
        result["audit_error"] = repr(exc)

    return result


def summarize_audit(
    pred_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    *,
    expected_n_datasets: int,
    expected_cases_per_dataset: int,
    expected_surfaces_per_case: int,
    fail_on_alignment_warnings: bool,
) -> Tuple[
    List[str],
    List[str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    problems: List[str] = []
    warnings: List[str] = []

    expected_rows_per_dataset = (
        expected_cases_per_dataset
        * expected_surfaces_per_case
    )
    expected_total_rows = (
        expected_n_datasets
        * expected_rows_per_dataset
    )

    if len(pred_df) != expected_total_rows:
        problems.append(
            f"Expected {expected_total_rows} "
            "prediction rows, "
            f"got {len(pred_df)}"
        )

    methods = sorted(
        pred_df["method"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if methods != [METHOD_NAME]:
        problems.append(
            "Prediction manifest must contain "
            f"only {METHOD_NAME}; found {methods}"
        )

    duplicate = pred_df[
        pred_df.duplicated(
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
            "dataset/case_id/surface rows"
        )

    failed = audit_df[
        audit_df["audit_status"] != "OK"
    ].copy()

    if len(failed):
        problems.append(
            f"{len(failed)} prediction rows "
            "failed the audit"
        )

    ok = audit_df[
        audit_df["audit_status"] == "OK"
    ].copy()

    dataset_values = sorted(
        pred_df["dataset"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if (
        len(dataset_values)
        != expected_n_datasets
    ):
        problems.append(
            f"Expected {expected_n_datasets} "
            "datasets, got "
            f"{len(dataset_values)}: "
            f"{dataset_values}"
        )

    if not ok.empty:
        per_case = (
            ok.groupby(
                ["dataset", "case_id"]
            )
            .agg(
                n_rows=(
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

        expected_surface_string = (
            ",".join(
                sorted(
                    EXPECTED_SURFACE_SET
                )
            )
        )

        bad_case = per_case[
            (
                per_case["n_rows"]
                != expected_surfaces_per_case
            )
            | (
                per_case["n_surfaces"]
                != expected_surfaces_per_case
            )
            | (
                per_case["surfaces"]
                != expected_surface_string
            )
        ].copy()

        if len(bad_case):
            problems.append(
                f"{len(bad_case)} cases do "
                "not contain the expected "
                "four cortical surfaces"
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
                dataset_counts["n_rows"]
                != expected_rows_per_dataset
            )
            | (
                dataset_counts["n_cases"]
                != expected_cases_per_dataset
            )
            | (
                dataset_counts["n_surfaces"]
                != expected_surfaces_per_case
            )
        ]

        if len(bad_dataset):
            problems.append(
                f"{len(bad_dataset)} datasets "
                "have incorrect case/row/"
                "surface counts"
            )
    else:
        dataset_counts = pd.DataFrame(
            columns=[
                "dataset",
                "n_rows",
                "n_cases",
                "n_surfaces",
            ]
        )
        problems.append(
            "No prediction rows passed "
            "the audit"
        )

    numeric_columns = [
        "pred_gt_centroid_dist_mm",
        "pred_gt_bbox_center_dist_mm",
        "pred_gt_extent_max_abs_diff_mm",
        "pred_gt_diag_diff_mm",
    ]

    for column in numeric_columns:
        if column not in ok.columns:
            problems.append(
                f"Missing audit column: {column}"
            )
            continue

        values = pd.to_numeric(
            ok[column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            problems.append(
                "Non-finite values found in "
                f"{column}"
            )

    alignment_mask = (
        ok["alignment_warning"]
        .map(bool_like)
        .fillna(False)
        .astype(bool)
    )

    alignment_warnings = ok[
        alignment_mask
    ].copy()

    if len(alignment_warnings):
        message = (
            f"{len(alignment_warnings)} "
            "prediction rows have "
            "alignment_warning=True"
        )

        if fail_on_alignment_warnings:
            problems.append(message)
        else:
            warnings.append(message)

    return (
        problems,
        warnings,
        failed,
        alignment_warnings,
        dataset_counts,
    )


def main() -> None:
    args = parse_args()

    log_path = setup_logging(
        args.report_dir
    )

    LOG.info(
        "=== Audit SimCortex predictions ==="
    )
    LOG.info(
        "pred_manifest=%s",
        args.pred_manifest,
    )
    LOG.info(
        "gt_manifest=%s",
        args.gt_manifest,
    )

    if not args.pred_manifest.exists():
        raise SystemExit(
            "Missing prediction manifest: "
            f"{args.pred_manifest}"
        )

    if not args.gt_manifest.exists():
        raise SystemExit(
            "Missing GT manifest: "
            f"{args.gt_manifest}"
        )

    pred_df = pd.read_csv(
        args.pred_manifest,
        sep="\t",
        low_memory=False,
    )
    gt_df = pd.read_csv(
        args.gt_manifest,
        sep="\t",
        low_memory=False,
    )

    try:
        require_columns(
            pred_df,
            REQUIRED_PRED_COLUMNS,
            "Prediction manifest",
        )
        gt_map = build_gt_map(
            gt_df
        )
    except Exception as exc:
        raise SystemExit(
            f"Manifest preparation failed: {exc}"
        ) from exc

    rows: List[Dict[str, Any]] = []

    for row in tqdm(
        pred_df.to_dict(
            orient="records"
        ),
        desc="Audit predictions",
    ):
        rows.append(
            audit_one(
                row,
                gt_map=gt_map,
                center_warn_mm=float(
                    args.center_warn_mm
                ),
                extent_warn_mm=float(
                    args.extent_warn_mm
                ),
            )
        )

    audit_df = dataframe_with_columns(
        rows,
        AUDIT_COLUMNS,
    )

    (
        problems,
        warnings,
        failed,
        alignment_warnings,
        dataset_counts,
    ) = summarize_audit(
        pred_df,
        audit_df,
        expected_n_datasets=int(
            args.expected_n_datasets
        ),
        expected_cases_per_dataset=int(
            args.expected_cases_per_dataset
        ),
        expected_surfaces_per_case=int(
            args.expected_surfaces_per_case
        ),
        fail_on_alignment_warnings=bool(
            args.fail_on_alignment_warnings
        ),
    )

    audit_path = (
        args.report_dir
        / "prediction_audit.tsv"
    )
    failures_path = (
        args.report_dir
        / "prediction_audit_failures.tsv"
    )
    warnings_path = (
        args.report_dir
        / "prediction_alignment_warnings.tsv"
    )
    counts_path = (
        args.report_dir
        / "prediction_audit_dataset_counts.tsv"
    )
    report_path = (
        args.report_dir
        / "prediction_audit_report.json"
    )

    write_tsv(
        audit_df,
        audit_path,
    )
    write_tsv(
        failed,
        failures_path,
    )
    write_tsv(
        alignment_warnings,
        warnings_path,
    )
    write_tsv(
        dataset_counts,
        counts_path,
    )

    expected_total_rows = (
        int(args.expected_n_datasets)
        * int(
            args.expected_cases_per_dataset
        )
        * int(
            args.expected_surfaces_per_case
        )
    )

    report = {
        "stage": "audit_predictions",
        "schema_version": (
            "simcortex_evaluation_prediction_audit_v1.0"
        ),
        "method": METHOD_NAME,
        "eval_root": args.eval_root,
        "pred_manifest": args.pred_manifest,
        "gt_manifest": args.gt_manifest,
        "report_dir": args.report_dir,
        "expected_n_datasets": int(
            args.expected_n_datasets
        ),
        "expected_cases_per_dataset": int(
            args.expected_cases_per_dataset
        ),
        "expected_surfaces_per_case": int(
            args.expected_surfaces_per_case
        ),
        "expected_total_rows": int(
            expected_total_rows
        ),
        "n_rows": int(
            len(audit_df)
        ),
        "n_ok_rows": int(
            (
                audit_df["audit_status"]
                == "OK"
            ).sum()
        ),
        "n_failed_rows": int(
            len(failed)
        ),
        "alignment_warning_rows": int(
            len(alignment_warnings)
        ),
        "alignment_warnings_fatal": bool(
            args.fail_on_alignment_warnings
        ),
        "center_warn_mm": float(
            args.center_warn_mm
        ),
        "extent_warn_mm": float(
            args.extent_warn_mm
        ),
        "coordinate_contract": {
            "selected_variant": (
                EXPECTED_VARIANT
            ),
            "raw_format": (
                EXPECTED_RAW_FORMAT
            ),
            "raw_space_assumption": (
                EXPECTED_SPACE
            ),
            "conversion_required": (
                EXPECTED_CONVERSION_REQUIRED
            ),
            "conversion_applied": (
                CONVERSION_APPLIED
            ),
        },
        "warnings": warnings,
        "problems": problems,
        "audit_passed": (
            len(problems) == 0
        ),
        "outputs": {
            "audit_rows": audit_path,
            "failures": failures_path,
            "alignment_warnings": (
                warnings_path
            ),
            "dataset_counts": counts_path,
            "log": log_path,
        },
    }

    write_json(
        report_path,
        report,
    )

    LOG.info(
        "Rows: %d",
        len(audit_df),
    )
    LOG.info(
        "OK rows: %d",
        (
            audit_df["audit_status"]
            == "OK"
        ).sum(),
    )
    LOG.info(
        "Alignment warnings: %d",
        len(alignment_warnings),
    )

    if warnings:
        for warning in warnings:
            LOG.warning(
                "%s",
                warning,
            )

    if problems:
        LOG.error(
            "Prediction audit failed with "
            "%d problem(s):",
            len(problems),
        )
        for problem in problems:
            LOG.error(
                " - %s",
                problem,
            )
        raise SystemExit(1)

    LOG.info(
        "Prediction audit passed."
    )


if __name__ == "__main__":
    main()
