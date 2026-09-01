from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "evaluation"

PUBLIC_EVALUATION_FILES = [
    "build_case_manifest.py",
    "build_gt_manifest.py",
    "build_pred_manifest.py",
    "audit_predictions.py",
    "metrics_core.py",
    "evaluate_metrics.py",
    "evaluate_collisions.py",
    "summarize_results.py",
    "run_evaluation.sh",
    "README.md",
]

SURFACE_KEYS = [
    "lh_white",
    "lh_pial",
    "rh_white",
    "rh_pial",
]

COLLISION_PAIRS = {
    "white_pial_left": (
        "lh_white",
        "lh_pial",
    ),
    "white_pial_right": (
        "rh_white",
        "rh_pial",
    ),
    "pial_lr": (
        "lh_pial",
        "rh_pial",
    ),
    "white_lr": (
        "lh_white",
        "rh_white",
    ),
    "cross_lhwhite_rhpial": (
        "lh_white",
        "rh_pial",
    ),
    "cross_rhwhite_lhpial": (
        "rh_white",
        "lh_pial",
    ),
}

FINAL_SUMMARY_METRICS = [
    "ChamferPCL1_mm",
    "ASSD_mm",
    "HD90_mm",
    "SIF_pct",
    "ThicknessAbsErr_mm",
    "CollisionPctUnion_mean4",
]

RUNNER_STAGES = [
    "build_case_manifest.py",
    "build_gt_manifest.py",
    "build_pred_manifest.py",
    "audit_predictions.py",
    "evaluate_metrics.py",
    "evaluate_collisions.py",
    "summarize_results.py",
]

# metrics_core.py was migrated byte-for-byte from the scientifically
# validated historical implementation. Changing this digest therefore
# requires an explicit scientific-regression review.
METRICS_CORE_SHA256 = (
    "c18865e45d462c1e4eb7f3dc656ccd7a"
    "e1e2e0fa0f8335affa1ff496a1ba7718"
)


def _read(name: str) -> str:
    return (
        EVAL_DIR / name
    ).read_text(
        encoding="utf-8"
    )


def _tree(name: str) -> ast.Module:
    return ast.parse(
        _read(name)
    )


def _assignment_value(
    filename: str,
    variable: str,
) -> Any:
    tree = _tree(filename)

    for node in tree.body:
        value_node = None

        if isinstance(
            node,
            ast.Assign,
        ):
            targets = node.targets
            value_node = node.value

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            targets = [node.target]
            value_node = node.value

        else:
            continue

        for target in targets:
            if (
                isinstance(
                    target,
                    ast.Name,
                )
                and target.id == variable
            ):
                assert value_node is not None

                return ast.literal_eval(
                    value_node
                )

    raise AssertionError(
        f"{variable!r} not found in "
        f"{filename}"
    )


def _argparse_defaults(
    filename: str,
) -> dict[str, Any]:
    tree = _tree(filename)

    defaults: dict[str, Any] = {}

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if not (
            isinstance(
                node.func,
                ast.Attribute,
            )
            and node.func.attr
            == "add_argument"
        ):
            continue

        flags = [
            arg.value
            for arg in node.args
            if (
                isinstance(
                    arg,
                    ast.Constant,
                )
                and isinstance(
                    arg.value,
                    str,
                )
                and arg.value.startswith(
                    "--"
                )
            )
        ]

        if not flags:
            continue

        default_found = False
        default_value = None

        for keyword in node.keywords:
            if keyword.arg != "default":
                continue

            default_found = True

            try:
                default_value = (
                    ast.literal_eval(
                        keyword.value
                    )
                )
            except Exception:
                default_value = (
                    ast.unparse(
                        keyword.value
                    )
                )

        for flag in flags:
            if default_found:
                defaults[
                    flag
                ] = default_value

    return defaults


def test_public_evaluation_files_exist() -> None:
    missing = [
        name
        for name
        in PUBLIC_EVALUATION_FILES
        if not (
            EVAL_DIR / name
        ).is_file()
    ]

    assert not missing, (
        "Missing public evaluation files: "
        f"{missing}"
    )


def test_metrics_core_is_frozen() -> None:
    path = (
        EVAL_DIR
        / "metrics_core.py"
    )

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    assert digest == (
        METRICS_CORE_SHA256
    )


def test_method_identity_is_simcortex() -> None:
    for filename in [
        "evaluate_metrics.py",
        "evaluate_collisions.py",
        "summarize_results.py",
    ]:
        assert (
            _assignment_value(
                filename,
                "METHOD_NAME",
            )
            == "SimCortex"
        )


def test_metric_surface_contract() -> None:
    actual = _assignment_value(
        "metrics_core.py",
        "SURFACE_KEYS",
    )

    assert list(actual) == (
        SURFACE_KEYS
    )


def test_collision_surface_contract() -> None:
    actual = _assignment_value(
        "evaluate_collisions.py",
        "SURFACE_KEYS",
    )

    assert list(actual) == (
        SURFACE_KEYS
    )


def test_collision_pair_contract() -> None:
    actual = _assignment_value(
        "evaluate_collisions.py",
        "COLLISION_PAIRS",
    )

    assert actual == (
        COLLISION_PAIRS
    )

    # Dict insertion order is part of the
    # historical six-pair evaluation order.
    assert list(
        actual.keys()
    ) == list(
        COLLISION_PAIRS.keys()
    )


def test_metric_runner_scientific_defaults() -> None:
    defaults = _argparse_defaults(
        "evaluate_metrics.py"
    )

    assert defaults[
        "--n-samples"
    ] == 150000

    assert defaults[
        "--n-thickness-samples"
    ] == 50000

    assert defaults[
        "--seed"
    ] == 12345

    assert defaults[
        "--save-every"
    ] == 25

    assert defaults[
        "--slow-case-warn-sec"
    ] == 300.0


def test_collision_runner_operational_defaults() -> None:
    defaults = _argparse_defaults(
        "evaluate_collisions.py"
    )

    assert defaults[
        "--max-contact-ladder"
    ] == "50000,200000,500000"

    assert defaults[
        "--timeout-sec"
    ] == 600

    assert defaults[
        "--mem-gb"
    ] == 48

    assert defaults[
        "--save-every"
    ] == 5

    assert defaults[
        "--slow-case-warn-sec"
    ] == 1800.0


def test_summary_metric_contract() -> None:
    metrics = _assignment_value(
        "summarize_results.py",
        "FINAL_METRICS",
    )

    surfaces = _assignment_value(
        "summarize_results.py",
        "SURFACES",
    )

    assert metrics == (
        FINAL_SUMMARY_METRICS
    )

    assert surfaces == (
        SURFACE_KEYS
    )


def test_summary_historical_cohort_defaults() -> None:
    defaults = _argparse_defaults(
        "summarize_results.py"
    )

    assert defaults[
        "--expected-cases"
    ] == 560

    assert defaults[
        "--expected-cases-per-dataset"
    ] == 40


def test_evaluation_runner_stage_order() -> None:
    text = _read(
        "run_evaluation.sh"
    )

    positions = []

    for stage in RUNNER_STAGES:
        assert text.count(
            stage
        ) == 1, (
            f"{stage} must appear exactly once "
            "in run_evaluation.sh"
        )

        positions.append(
            text.index(stage)
        )

    assert positions == sorted(
        positions
    ), (
        "Evaluation stages are not in the "
        "expected execution order"
    )


def test_public_evaluation_has_no_private_or_competitor_paths() -> None:
    banned = [
        "CFPP",
        "CortexODE",
        "V2C-Flow",
        "final6",
        "results_per_method",
        "/project/",
    ]

    files = [
        path
        for path in EVAL_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix
            in {
                ".py",
                ".sh",
                ".md",
            }
        )
    ]

    problems = []

    for path in files:
        text = path.read_text(
            encoding="utf-8"
        )

        for token in banned:
            if token in text:
                problems.append(
                    (
                        path.name,
                        token,
                    )
                )

    assert not problems, (
        "Private/competitor-specific content "
        "found in public evaluation files: "
        f"{problems}"
    )
