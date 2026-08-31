#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Build the SimCortex evaluation case manifest.

This script discovers FreeSurfer subject/session folders from an evaluation
input tree and writes the canonical case manifest used by downstream
evaluation stages.

Outputs
-------
  <eval-root>/manifests/case_manifest.tsv
  <eval-root>/reports/case_manifest_excluded_sessions.tsv
  <eval-root>/reports/case_manifest_run_summary.json

The default benchmark contract contains 14 datasets with 40 cases per dataset
(560 cases total). These expectations can be overridden from the command line.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_DATASETS = [
    "cnp",
    "ds000115",
    "ds000144",
    "ds001486",
    "ds001748",
    "ds002424",
    "ds002862",
    "ds002886",
    "ds003499",
    "ds003568",
    "ds003763",
    "ds005234",
    "ds006067",
    "hcp_oasis",
]

SAMPLE_DIR_TO_DATASET = {
    "hcpya_oasis_sample40": "hcp_oasis",
}

REQUIRED_OUTPUT_COLUMNS = [
    "dataset",
    "sample_dataset_dir",
    "subject",
    "session",
    "case_id",
    "fsroot",
    "orig_mgz",
    "lh_white_gt",
    "lh_pial_gt",
    "rh_white_gt",
    "rh_pial_gt",
    "lh_pial_kind",
    "rh_pial_kind",
    "has_orig_mgz",
    "has_lh_white",
    "has_lh_pial",
    "has_rh_white",
    "has_rh_pial",
    "status",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the SimCortex evaluation case manifest."
    )
    p.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Root directory for evaluation outputs.",
    )
    p.add_argument(
        "--sample-root",
        type=Path,
        required=True,
        help="Root containing the evaluation FreeSurfer dataset folders.",
    )
    p.add_argument(
        "--expected-datasets",
        nargs="+",
        default=EXPECTED_DATASETS,
        help="Expected evaluation dataset keys.",
    )
    p.add_argument(
        "--expected-cases-per-dataset",
        type=int,
        default=40,
    )
    p.add_argument(
        "--expected-total-cases",
        type=int,
        default=560,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if validation problems are found.",
    )
    return p.parse_args()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def dataset_key_from_dir(ds_dir: Path) -> str:
    if ds_dir.name in SAMPLE_DIR_TO_DATASET:
        return SAMPLE_DIR_TO_DATASET[ds_dir.name]
    if ds_dir.name.endswith("_sample40"):
        return ds_dir.name.replace("_sample40", "")
    return ds_dir.name


def find_pial(surf_dir: Path, hemi: str) -> tuple[Path | None, str | None]:
    candidates = [
        (surf_dir / f"{hemi}.pial.T1", "pial.T1"),
        (surf_dir / f"{hemi}.pial", "pial"),
    ]
    for path, kind in candidates:
        if path.exists():
            return path, kind
    return None, None


def is_t2_session_name(name: str) -> bool:
    return "T2" in str(name).upper()


def discover_cases(sample_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    if not sample_root.exists():
        raise FileNotFoundError(f"Missing sample root: {sample_root}")

    ds_dirs = sorted([p for p in sample_root.glob("*_sample40") if p.is_dir()])

    for ds_dir in ds_dirs:
        dataset = dataset_key_from_dir(ds_dir)

        for sub_dir in sorted(ds_dir.glob("sub-*")):
            if not sub_dir.is_dir():
                continue

            sessions = sorted([p for p in sub_dir.glob("ses-*") if p.is_dir()])

            if sessions:
                session_dirs = []
                for ses_dir in sessions:
                    if is_t2_session_name(ses_dir.name):
                        excluded_rows.append({
                            "dataset": dataset,
                            "sample_dataset_dir": ds_dir.name,
                            "subject": sub_dir.name,
                            "session": ses_dir.name,
                            "fsroot": str(ses_dir),
                            "reason": "excluded_T2_session",
                        })
                        continue
                    session_dirs.append(ses_dir)

                if not session_dirs:
                    excluded_rows.append({
                        "dataset": dataset,
                        "sample_dataset_dir": ds_dir.name,
                        "subject": sub_dir.name,
                        "session": "",
                        "fsroot": str(sub_dir),
                        "reason": "subject_has_no_usable_non_T2_session",
                    })
                    continue
            else:
                # CNP / HCP-OASIS style: FreeSurfer tree directly under subject.
                session_dirs = [sub_dir]

            for fsroot in session_dirs:
                subject = sub_dir.name
                session = fsroot.name if fsroot.name.startswith("ses-") else "ses-01"
                case_id = f"{subject}_{session}"

                mri_dir = fsroot / "mri"
                surf_dir = fsroot / "surf"

                orig_mgz = mri_dir / "orig.mgz"

                lh_white = surf_dir / "lh.white"
                rh_white = surf_dir / "rh.white"

                lh_pial, lh_pial_kind = find_pial(surf_dir, "lh")
                rh_pial, rh_pial_kind = find_pial(surf_dir, "rh")

                row = {
                    "dataset": dataset,
                    "sample_dataset_dir": ds_dir.name,
                    "subject": subject,
                    "session": session,
                    "case_id": case_id,
                    "fsroot": str(fsroot),
                    "orig_mgz": str(orig_mgz),
                    "lh_white_gt": str(lh_white),
                    "lh_pial_gt": str(lh_pial) if lh_pial else "",
                    "rh_white_gt": str(rh_white),
                    "rh_pial_gt": str(rh_pial) if rh_pial else "",
                    "lh_pial_kind": lh_pial_kind or "",
                    "rh_pial_kind": rh_pial_kind or "",
                    "has_orig_mgz": orig_mgz.exists(),
                    "has_lh_white": lh_white.exists(),
                    "has_lh_pial": lh_pial is not None and lh_pial.exists(),
                    "has_rh_white": rh_white.exists(),
                    "has_rh_pial": rh_pial is not None and rh_pial.exists(),
                }

                row["status"] = (
                    "OK"
                    if row["has_orig_mgz"]
                    and row["has_lh_white"]
                    and row["has_lh_pial"]
                    and row["has_rh_white"]
                    and row["has_rh_pial"]
                    else "MISSING_REQUIRED_FILE"
                )

                rows.append(row)

    df = pd.DataFrame(rows)
    excluded_df = pd.DataFrame(excluded_rows)

    for col in REQUIRED_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[REQUIRED_OUTPUT_COLUMNS].copy(), excluded_df


def validate_manifest(
    df: pd.DataFrame,
    expected_datasets: list[str],
    expected_cases_per_dataset: int,
    expected_total_cases: int,
) -> list[str]:
    problems: list[str] = []

    if df.empty:
        return ["No cases discovered."]

    found_datasets = sorted(df["dataset"].dropna().astype(str).unique().tolist())
    missing_datasets = sorted(set(expected_datasets) - set(found_datasets))
    extra_datasets = sorted(set(found_datasets) - set(expected_datasets))

    if missing_datasets:
        problems.append(f"Missing datasets: {missing_datasets}")
    if extra_datasets:
        problems.append(f"Unexpected datasets: {extra_datasets}")

    for ds in expected_datasets:
        n = int((df["dataset"].astype(str) == ds).sum())
        if n != expected_cases_per_dataset:
            problems.append(
                f"Dataset {ds} has {n} cases; expected {expected_cases_per_dataset}."
            )

    if len(df) != expected_total_cases:
        problems.append(f"Total cases = {len(df)}; expected {expected_total_cases}.")

    bad_status = df[df["status"].astype(str) != "OK"]
    if len(bad_status):
        problems.append(f"{len(bad_status)} cases have status != OK.")

    dup = df[df.duplicated(["dataset", "case_id"], keep=False)]
    if len(dup):
        problems.append(f"{len(dup)} duplicate dataset/case_id rows found.")

    # Dataset integrity check for the released benchmark cohort.
    if "ds003499" in found_datasets:
        ds3499 = df[df["dataset"].astype(str) == "ds003499"]
        has_sub111 = bool((ds3499["subject"].astype(str) == "sub-111").any())
        has_sub151 = bool((ds3499["subject"].astype(str) == "sub-151").any())

        if has_sub111:
            problems.append("ds003499 still contains sub-111; this should be removed.")
        if not has_sub151:
            problems.append("ds003499 does not contain sub-151; this replacement should exist.")

    return problems


def fail_if_exists(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [p for p in paths if p.exists()]
    if existing:
        raise FileExistsError(
            "Output files already exist:\n"
            + "\n".join(str(p) for p in existing)
            + "\nUse --overwrite to replace them."
        )


def main() -> None:
    args = parse_args()

    eval_root = args.eval_root.resolve()
    sample_root = args.sample_root.resolve()

    manifest_dir = eval_root / "manifests"
    report_dir = eval_root / "reports"

    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    out_manifest = manifest_dir / "case_manifest.tsv"
    excluded_path = report_dir / "case_manifest_excluded_sessions.tsv"
    summary_path = report_dir / "case_manifest_run_summary.json"

    fail_if_exists([out_manifest, excluded_path, summary_path], overwrite=args.overwrite)

    df, excluded_df = discover_cases(sample_root)

    df.to_csv(out_manifest, sep="\t", index=False)
    excluded_df.to_csv(excluded_path, sep="\t", index=False)

    problems = validate_manifest(
        df=df,
        expected_datasets=list(args.expected_datasets),
        expected_cases_per_dataset=int(args.expected_cases_per_dataset),
        expected_total_cases=int(args.expected_total_cases),
    )

    cases_by_dataset = (
        df.groupby("dataset")["case_id"]
        .nunique()
        .sort_index()
        .astype(int)
        .to_dict()
        if not df.empty
        else {}
    )

    rows_by_sample_dataset_dir = (
        df.groupby("sample_dataset_dir")["case_id"]
        .nunique()
        .sort_index()
        .astype(int)
        .to_dict()
        if not df.empty
        else {}
    )

    status_counts = (
        df["status"].value_counts(dropna=False).astype(int).to_dict()
        if "status" in df.columns
        else {}
    )

    summary = {
        "stage": "build_case_manifest",
        "schema_version": "simcortex_evaluation_case_manifest_v1.0",
        "eval_root": str(eval_root),
        "sample_root": str(sample_root),
        "outputs": {
            "case_manifest": str(out_manifest),
            "excluded_sessions": str(excluded_path),
            "run_summary": str(summary_path),
        },
        "expected_datasets": list(args.expected_datasets),
        "expected_cases_per_dataset": int(args.expected_cases_per_dataset),
        "expected_total_cases": int(args.expected_total_cases),
        "n_rows": int(len(df)),
        "n_excluded_session_rows": int(len(excluded_df)),
        "status_counts": status_counts,
        "cases_by_dataset": {str(k): int(v) for k, v in cases_by_dataset.items()},
        "rows_by_sample_dataset_dir": {str(k): int(v) for k, v in rows_by_sample_dataset_dir.items()},
        "problems": problems,
        "strict": bool(args.strict),
    }

    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    print("Wrote:")
    print(out_manifest)
    print(excluded_path)
    print(summary_path)

    print("\nStatus:")
    print(pd.Series(status_counts).to_string() if status_counts else "EMPTY")

    print("\nCases by dataset:")
    print(pd.Series(cases_by_dataset).to_string() if cases_by_dataset else "EMPTY")

    print("\nProblems:")
    if problems:
        for p in problems:
            print(f"  - {p}")
    else:
        print("  none")

    if args.strict and problems:
        raise SystemExit("Case manifest failed strict validation.")

    print("\nCASE MANIFEST PASSED ✅")


if __name__ == "__main__":
    main()
