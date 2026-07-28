import re
from pathlib import Path

from simcortex.preproc.fs_to_mni import PIPELINE_NAME
from simcortex.preproc.mri_to_mni_inference import (
    PREPROC_DERIVATIVE_NAME,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_CODE_FILES = (
    Path("src/simcortex/preproc/fs_to_mni.py"),
    Path(
        "src/simcortex/preproc/"
        "mri_to_mni_inference.py"
    ),
    Path("src/simcortex/initsurf/generate.py"),
    Path("src/simcortex/deform/inference.py"),
)

DERIVATIVE_RUN_PATTERN = re.compile(
    r"^sc-"
    r"(preproc|seg|initsurf|deform)-"
    r"[0-9]+\.[0-9]+$"
)

PIPELINE_ROOT_PATTERN = re.compile(
    r"\bsc-"
    r"(preproc|seg|initsurf|deform)-"
    r"([0-9]+\.[0-9]+)\b"
)


def test_preprocessing_names_use_sc_namespace() -> None:
    assert DERIVATIVE_RUN_PATTERN.fullmatch(
        PIPELINE_NAME
    )

    assert DERIVATIVE_RUN_PATTERN.fullmatch(
        PREPROC_DERIVATIVE_NAME
    )


def test_full_pipeline_defines_all_stage_roots() -> None:
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "run_pipeline.py"
    ).read_text(
        encoding="utf-8",
    )

    matches = PIPELINE_ROOT_PATTERN.findall(
        source
    )

    stages = {
        stage
        for stage, _run_label in matches
    }

    assert stages == {
        "preproc",
        "seg",
        "initsurf",
        "deform",
    }


def test_active_code_contains_no_legacy_scpp_identifier() -> None:
    for relative_path in ACTIVE_CODE_FILES:
        source = (
            REPOSITORY_ROOT
            / relative_path
        ).read_text(
            encoding="utf-8",
        )

        assert "scpp" not in source, (
            "Legacy scpp identifier remains in "
            f"{relative_path}"
        )
