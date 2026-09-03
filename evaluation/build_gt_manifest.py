#!/usr/bin/env python3
"""
Build the SimCortex ground-truth scannerRAS manifest.

FreeSurfer cortical surfaces are read in tkRAS coordinates and exported to
scannerRAS for comparison with SimCortex native-space predictions.

Inputs
------
- <eval-root>/manifests/case_manifest.tsv
- FreeSurfer orig.mgz and cortical surfaces referenced by that manifest

Outputs
-------
- <eval-root>/gt_scannerRAS/<dataset>/<case_id>/*.surf.ply
- <eval-root>/manifests/gt_scannerRAS_manifest.tsv
- <eval-root>/reports/gt_scannerRAS_manifest_qc_report.json
- <eval-root>/reports/gt_scannerRAS_manifest.log

Coordinate convention
---------------------
The conversion is:

    scannerRAS = vox2ras @ inv(vox2ras_tkr) @ tkRAS

where both matrices are read from each subject's orig.mgz header.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm


SURFACES = ["lh_white", "lh_pial", "rh_white", "rh_pial"]
SURFACE_TO_CASE_COL = {
    "lh_white": "lh_white_gt",
    "lh_pial": "lh_pial_gt",
    "rh_white": "rh_white_gt",
    "rh_pial": "rh_pial_gt",
}

REQUIRED_CASE_COLUMNS = [
    "dataset",
    "subject",
    "session",
    "case_id",
    "orig_mgz",
    "status",
    "lh_white_gt",
    "lh_pial_gt",
    "rh_white_gt",
    "rh_pial_gt",
]

OUTPUT_COLUMNS = [
    "dataset",
    "subject",
    "session",
    "case_id",
    "surface",
    "gt_source_path",
    "orig_mgz",
    "output_path",
    "source_space",
    "target_space",
    "transform_formula",
    "n_vertices",
    "n_faces",
    "finite_vertices",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_center_z",
    "extent_x",
    "extent_y",
    "extent_z",
    "bbox_diag",
    "vox2ras_flat",
    "vox2ras_tkr_flat",
    "tkr_to_scanner_flat",
    "status",
    "error",
    "nan_count",
    "inf_count",
]

LOG = logging.getLogger("simcortex.evaluation.gt_scannerRAS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build scannerRAS ground-truth surfaces and the SimCortex "
            "evaluation GT manifest."
        )
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Root directory for evaluation inputs and outputs.",
    )
    parser.add_argument(
        "--case-manifest",
        type=Path,
        default=None,
        help="Default: <eval-root>/manifests/case_manifest.tsv.",
    )
    parser.add_argument(
        "--gt-manifest",
        type=Path,
        default=None,
        help="Default: <eval-root>/manifests/gt_scannerRAS_manifest.tsv.",
    )
    parser.add_argument(
        "--gt-out-root",
        type=Path,
        default=None,
        help="Default: <eval-root>/gt_scannerRAS.",
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
        default=True,
        help=(
            "Overwrite output PLY files and manifest. "
            "Enabled by default for reproducibility."
        ),
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if an output PLY already exists.",
    )

    args = parser.parse_args()

    eval_root = args.eval_root.expanduser().resolve()

    if args.case_manifest is None:
        args.case_manifest = eval_root / "manifests" / "case_manifest.tsv"
    else:
        args.case_manifest = args.case_manifest.expanduser().resolve()

    if args.gt_manifest is None:
        args.gt_manifest = eval_root / "manifests" / "gt_scannerRAS_manifest.tsv"
    else:
        args.gt_manifest = args.gt_manifest.expanduser().resolve()

    if args.gt_out_root is None:
        args.gt_out_root = eval_root / "gt_scannerRAS"
    else:
        args.gt_out_root = args.gt_out_root.expanduser().resolve()

    if args.report_dir is None:
        args.report_dir = eval_root / "reports"
    else:
        args.report_dir = args.report_dir.expanduser().resolve()

    args.eval_root = eval_root
    return args

def setup_logging(report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "gt_scannerRAS_manifest.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_path


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(json_safe(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        scalar_na = pd.isna(obj)
        if isinstance(scalar_na, (bool, np.bool_)) and scalar_na:
            return None
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dataframe_with_columns(rows: List[Dict[str, Any]], columns: Iterable[str]) -> pd.DataFrame:
    columns = list(columns)
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    extra_cols = [c for c in df.columns if c not in columns]
    return df[columns + extra_cols]


def apply_affine(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    affine = np.asarray(affine, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3); got {vertices.shape}")
    if affine.shape != (4, 4):
        raise ValueError(f"affine must have shape (4, 4); got {affine.shape}")
    return (np.c_[vertices, np.ones(len(vertices), dtype=np.float64)] @ affine.T)[:, :3]


def bbox_stats(vertices: np.ndarray) -> Tuple[List[float], List[float], float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"cannot compute bbox for vertices with shape {vertices.shape}")
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    center = 0.5 * (mn + mx)
    extent = mx - mn
    diag = float(np.linalg.norm(extent))
    return center.astype(float).tolist(), extent.astype(float).tolist(), diag


def validate_geometry(vertices: np.ndarray, faces: np.ndarray, label: str) -> None:
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{label}: vertices must have shape (N, 3); got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{label}: faces must have shape (M, 3); got {faces.shape}")
    if len(vertices) == 0:
        raise ValueError(f"{label}: zero vertices")
    if len(faces) == 0:
        raise ValueError(f"{label}: zero faces")
    if not np.isfinite(vertices).all():
        raise ValueError(
            f"{label}: non-finite vertices; "
            f"nan={int(np.isnan(vertices).sum())}, inf={int(np.isinf(vertices).sum())}"
        )
    if not np.issubdtype(faces.dtype, np.integer):
        faces = faces.astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(
            f"{label}: face index out of range; min={int(faces.min())}, "
            f"max={int(faces.max())}, n_vertices={len(vertices)}"
        )


def save_ply(vertices: np.ndarray, faces: np.ndarray, out_path: Path, overwrite: bool) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists and --no-overwrite was used: {out_path}")
    validate_geometry(vertices, faces, str(out_path))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.export(str(out_path))


def output_path_for(row: pd.Series, surface: str, gt_out_root: Path) -> Path:
    dataset = str(row["dataset"])
    case_id = str(row["case_id"])
    hemi, surf = surface.split("_")
    hemi_label = "L" if hemi == "lh" else "R"
    fname = f"{case_id}_space-scannerRAS_desc-gt_hemi-{hemi_label}_{surf}.surf.ply"
    return Path(gt_out_root) / dataset / case_id / fname


def ensure_case_manifest_valid(
    cases: pd.DataFrame,
    expected_cases: int,
    expected_cases_per_dataset: int,
) -> List[str]:
    problems: List[str] = []
    missing_cols = sorted(set(REQUIRED_CASE_COLUMNS) - set(cases.columns))
    if missing_cols:
        return [f"Case manifest is missing required columns: {missing_cols}"]

    if len(cases) != expected_cases:
        problems.append(f"Expected {expected_cases} cases, got {len(cases)}")

    bad_status = cases[cases["status"].astype(str) != "OK"].copy()
    if len(bad_status):
        problems.append(f"{len(bad_status)} case_manifest rows have status != OK")

    dup = cases[cases.duplicated(["dataset", "case_id"], keep=False)].copy()
    if len(dup):
        problems.append(f"{len(dup)} duplicate dataset/case_id rows in case manifest")

    n_by_ds = cases.groupby("dataset")["case_id"].nunique().sort_index()
    if not n_by_ds.empty and not n_by_ds.eq(expected_cases_per_dataset).all():
        bad = {
            str(k): int(v)
            for k, v in n_by_ds[
                ~n_by_ds.eq(expected_cases_per_dataset)
            ].to_dict().items()
        }
        problems.append(
            "Datasets with case count != "
            f"{expected_cases_per_dataset}: {bad}"
        )

    return problems


def base_row(case: pd.Series, surface: str, gt_out_root: Path) -> Dict[str, Any]:
    return {
        "dataset": str(case.get("dataset", "")),
        "subject": str(case.get("subject", "")),
        "session": str(case.get("session", "")),
        "case_id": str(case.get("case_id", "")),
        "surface": surface,
        "gt_source_path": str(case.get(SURFACE_TO_CASE_COL.get(surface, ""), "")),
        "orig_mgz": str(case.get("orig_mgz", "")),
        "output_path": str(output_path_for(case, surface, gt_out_root)),
        "source_space": "FreeSurfer_tkRAS",
        "target_space": "scannerRAS",
        "transform_formula": "scannerRAS = vox2ras @ inv(vox2ras_tkr) @ tkRAS",
        "n_vertices": np.nan,
        "n_faces": np.nan,
        "finite_vertices": pd.NA,
        "bbox_center_x": np.nan,
        "bbox_center_y": np.nan,
        "bbox_center_z": np.nan,
        "extent_x": np.nan,
        "extent_y": np.nan,
        "extent_z": np.nan,
        "bbox_diag": np.nan,
        "vox2ras_flat": "",
        "vox2ras_tkr_flat": "",
        "tkr_to_scanner_flat": "",
        "status": "UNKNOWN",
        "error": "",
        "nan_count": np.nan,
        "inf_count": np.nan,
    }


def error_rows_for_case(case: pd.Series, status: str, error: str, gt_out_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for surface in SURFACES:
        row = base_row(case, surface, gt_out_root)
        row["status"] = status
        row["error"] = error
        rows.append(row)
    return rows


def export_one_surface(
    case: pd.Series,
    surface: str,
    gt_out_root: Path,
    tkr_to_scanner: np.ndarray,
    vox2ras: np.ndarray,
    vox2ras_tkr: np.ndarray,
    overwrite: bool,
) -> Dict[str, Any]:
    row = base_row(case, surface, gt_out_root)
    gt_source = Path(str(case[SURFACE_TO_CASE_COL[surface]]))
    out_path = Path(row["output_path"])

    if not gt_source.exists():
        row.update(status="MISSING_GT_SOURCE", error=f"Missing GT source: {gt_source}")
        return row

    try:
        vertices_tkr, faces = fsio.read_geometry(str(gt_source))
        vertices_tkr = np.asarray(vertices_tkr, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        validate_geometry(vertices_tkr, faces, str(gt_source))

        vertices_scan = apply_affine(vertices_tkr, tkr_to_scanner)
        if not np.isfinite(vertices_scan).all():
            row.update(
                status="NONFINITE_GT_SCANNERRAS",
                error="Non-finite vertices after tkRAS-to-scannerRAS transform",
                nan_count=int(np.isnan(vertices_scan).sum()),
                inf_count=int(np.isinf(vertices_scan).sum()),
            )
            return row

        validate_geometry(vertices_scan, faces, str(out_path))
        save_ply(vertices_scan, faces, out_path, overwrite=overwrite)

        center, extent, diag = bbox_stats(vertices_scan)
        row.update(
            n_vertices=int(vertices_scan.shape[0]),
            n_faces=int(faces.shape[0]),
            finite_vertices=True,
            bbox_center_x=float(center[0]),
            bbox_center_y=float(center[1]),
            bbox_center_z=float(center[2]),
            extent_x=float(extent[0]),
            extent_y=float(extent[1]),
            extent_z=float(extent[2]),
            bbox_diag=float(diag),
            vox2ras_flat=" ".join(f"{x:.10g}" for x in vox2ras.reshape(-1)),
            vox2ras_tkr_flat=" ".join(f"{x:.10g}" for x in vox2ras_tkr.reshape(-1)),
            tkr_to_scanner_flat=" ".join(f"{x:.10g}" for x in tkr_to_scanner.reshape(-1)),
            status="OK",
            error="",
            nan_count=0,
            inf_count=0,
        )
        return row
    except Exception as exc:
        row.update(status="GT_EXPORT_ERROR", error=repr(exc))
        return row


def export_gt_manifest(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    case_manifest = Path(args.case_manifest)
    gt_manifest = Path(args.gt_manifest)
    gt_out_root = Path(args.gt_out_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    gt_out_root.mkdir(parents=True, exist_ok=True)
    gt_manifest.parent.mkdir(parents=True, exist_ok=True)

    if not case_manifest.exists():
        raise FileNotFoundError(f"Missing case manifest: {case_manifest}")

    cases = pd.read_csv(case_manifest, sep="\t", low_memory=False)
    manifest_problems = ensure_case_manifest_valid(
        cases,
        expected_cases=int(args.expected_cases),
        expected_cases_per_dataset=int(args.expected_cases_per_dataset),
    )
    if manifest_problems:
        for problem in manifest_problems:
            LOG.error(problem)
        raise SystemExit(1)

    rows: List[Dict[str, Any]] = []

    for _, case in tqdm(cases.iterrows(), total=len(cases), desc="Export GT scannerRAS"):
        orig_mgz = Path(str(case["orig_mgz"]))
        if not orig_mgz.exists():
            rows.extend(error_rows_for_case(case, "MISSING_ORIG", f"Missing orig.mgz: {orig_mgz}", gt_out_root))
            continue

        try:
            img = nib.load(str(orig_mgz))
            vox2ras = np.asarray(img.affine, dtype=np.float64)
            vox2ras_tkr = np.asarray(img.header.get_vox2ras_tkr(), dtype=np.float64)
            if vox2ras.shape != (4, 4) or vox2ras_tkr.shape != (4, 4):
                raise ValueError(f"Invalid affine shapes: vox2ras={vox2ras.shape}, vox2ras_tkr={vox2ras_tkr.shape}")
            tkr_to_scanner = vox2ras @ np.linalg.inv(vox2ras_tkr)
        except Exception as exc:
            rows.extend(error_rows_for_case(case, "ORIG_LOAD_OR_XFM_ERROR", repr(exc), gt_out_root))
            continue

        for surface in SURFACES:
            rows.append(
                export_one_surface(
                    case=case,
                    surface=surface,
                    gt_out_root=gt_out_root,
                    tkr_to_scanner=tkr_to_scanner,
                    vox2ras=vox2ras,
                    vox2ras_tkr=vox2ras_tkr,
                    overwrite=bool(args.overwrite),
                )
            )

    df = dataframe_with_columns(rows, OUTPUT_COLUMNS)
    df.to_csv(gt_manifest, sep="\t", index=False)

    report = build_qc_report(df, cases, args)
    return df, report


def build_qc_report(df: pd.DataFrame, cases: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    report_dir = Path(args.report_dir)
    gt_manifest = Path(args.gt_manifest)
    gt_out_root = Path(args.gt_out_root)
    expected_rows = int(args.expected_cases) * int(args.expected_surfaces_per_case)
    problems: List[str] = []

    ok = df[df["status"] == "OK"].copy() if "status" in df.columns else pd.DataFrame()

    status_counts = df["status"].value_counts(dropna=False).to_dict() if "status" in df.columns else {}
    rows_by_surface = ok.groupby("surface").size().sort_index() if not ok.empty else pd.Series(dtype=int)
    cases_by_dataset = ok.groupby("dataset")["case_id"].nunique().sort_index() if not ok.empty else pd.Series(dtype=int)

    if len(df) != expected_rows:
        problems.append(f"Expected {expected_rows} GT surface rows, got {len(df)}")
    if len(ok) != expected_rows:
        problems.append(f"Expected {expected_rows} OK rows, got {len(ok)}")

    duplicate_keys = pd.DataFrame()
    if not ok.empty:
        duplicate_keys = ok[ok.duplicated(["dataset", "case_id", "surface"], keep=False)].copy()
        if len(duplicate_keys):
            problems.append(f"{len(duplicate_keys)} duplicate dataset/case_id/surface OK rows")
            duplicate_keys.to_csv(report_dir / "gt_scannerRAS_duplicate_surface_rows.tsv", sep="\t", index=False)

        per_case = (
            ok.groupby(["dataset", "case_id"])
            .agg(n_surfaces=("surface", "size"), surfaces=("surface", lambda x: ",".join(sorted(map(str, x)))))
            .reset_index()
        )
        expected_surface_str = ",".join(sorted(SURFACES))
        per_case["expected_surfaces"] = expected_surface_str
        per_case["has_expected_surface_set"] = per_case["surfaces"].eq(expected_surface_str)
        bad_case = per_case[
            (per_case["n_surfaces"] != int(args.expected_surfaces_per_case))
            | (~per_case["has_expected_surface_set"])
        ].copy()
        if len(bad_case):
            problems.append(f"{len(bad_case)} cases do not have exactly the expected GT surface set")
            bad_case.to_csv(report_dir / "gt_scannerRAS_bad_surface_count_per_case.tsv", sep="\t", index=False)

        missing_outputs = ok[~ok["output_path"].map(lambda p: Path(str(p)).exists())].copy()
        if len(missing_outputs):
            problems.append(f"{len(missing_outputs)} OK rows have missing output_path files")
            missing_outputs.to_csv(report_dir / "gt_scannerRAS_missing_output_files.tsv", sep="\t", index=False)

        for col in ["n_vertices", "n_faces", "bbox_diag", "extent_x", "extent_y", "extent_z"]:
            vals = pd.to_numeric(ok[col], errors="coerce")
            if vals.isna().any() or not np.isfinite(vals.to_numpy(dtype=float)).all():
                problems.append(f"Non-finite or non-numeric values in OK rows for {col}")
            if col in ["n_vertices", "n_faces", "bbox_diag"] and (vals <= 0).any():
                problems.append(f"Non-positive values in OK rows for {col}")

        expected_cases_per_dataset = int(args.expected_cases_per_dataset)
        if (
            not cases_by_dataset.empty
            and not cases_by_dataset.eq(expected_cases_per_dataset).all()
        ):
            bad = {
                str(k): int(v)
                for k, v in cases_by_dataset[
                    ~cases_by_dataset.eq(expected_cases_per_dataset)
                ].to_dict().items()
            }
            problems.append(
                "GT OK case counts by dataset are not all "
                f"{expected_cases_per_dataset}: {bad}"
            )

    non_ok = df[df["status"] != "OK"].copy() if "status" in df.columns else df.copy()
    if len(non_ok):
        non_ok.to_csv(report_dir / "gt_scannerRAS_non_ok_rows.tsv", sep="\t", index=False)

    rows_by_surface.to_csv(report_dir / "gt_scannerRAS_rows_by_surface.tsv", sep="\t", header=["n"])
    cases_by_dataset.to_csv(report_dir / "gt_scannerRAS_cases_by_dataset.tsv", sep="\t", header=["n_cases"])

    report = {
        "case_manifest": str(args.case_manifest),
        "gt_manifest": str(gt_manifest),
        "gt_output_root": str(gt_out_root),
        "n_case_manifest_rows": int(len(cases)),
        "n_rows": int(len(df)),
        "n_ok_rows": int(len(ok)),
        "n_non_ok_rows": int(len(non_ok)),
        "expected_rows": int(expected_rows),
        "expected_cases": int(args.expected_cases),
        "expected_cases_per_dataset": int(args.expected_cases_per_dataset),
        "expected_surfaces_per_case": int(args.expected_surfaces_per_case),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "rows_by_surface": {str(k): int(v) for k, v in rows_by_surface.to_dict().items()},
        "cases_by_dataset": {str(k): int(v) for k, v in cases_by_dataset.to_dict().items()},
        "problems": problems,
    }
    return report


def main() -> None:
    args = parse_args()
    log_path = setup_logging(Path(args.report_dir))

    LOG.info("=== Build GT scannerRAS manifest ===")
    LOG.info("case_manifest=%s", args.case_manifest)
    LOG.info("gt_manifest=%s", args.gt_manifest)
    LOG.info("gt_out_root=%s", args.gt_out_root)
    LOG.info("report_dir=%s", args.report_dir)
    LOG.info("log_path=%s", log_path)

    df, report = export_gt_manifest(args)

    report_path = Path(args.report_dir) / "gt_scannerRAS_manifest_qc_report.json"
    with open(report_path, "w") as f:
        json.dump(json_safe(report), f, indent=2)

    LOG.info("Wrote: %s", args.gt_manifest)
    LOG.info("Wrote report: %s", report_path)
    LOG.info("Rows: %d", int(report["n_rows"]))
    LOG.info("OK rows: %d", int(report["n_ok_rows"]))
    LOG.info("Status counts: %s", report["status_counts"])
    LOG.info("Rows per surface: %s", report["rows_by_surface"])
    LOG.info("Cases by dataset: %s", report["cases_by_dataset"])

    problems = report.get("problems", [])
    if problems:
        LOG.error("GT scannerRAS MANIFEST QC FAILED with %d problem(s):", len(problems))
        for problem in problems:
            LOG.error(" - %s", problem)
        raise SystemExit(1)

    LOG.info("GT scannerRAS manifest QC passed.")


if __name__ == "__main__":
    main()
