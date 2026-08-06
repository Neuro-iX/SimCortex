import re
from pathlib import Path

from simcortex.preproc.fs_to_mni import PIPELINE_NAME
from simcortex.preproc.mri_to_mni_inference import (
    PREPROC_DERIVATIVE_NAME,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_CODE_FILES = (
    Path("scripts/run_pipeline.py"),
    Path("src/simcortex/preproc/fs_to_mni.py"),
    Path(
        "src/simcortex/preproc/"
        "mri_to_mni_inference.py"
    ),
    Path("src/simcortex/initsurf/generate.py"),
    Path("src/simcortex/deform/inference.py"),
    Path(
        "src/simcortex/deform/data/"
        "dataloader.py"
    ),
)

CANONICAL_DERIVATIVE_NAMES = {
    "sc-preproc",
    "sc-seg",
    "sc-initsurf",
    "sc-deform",
}

PIPELINE_ROOT_PATTERN = re.compile(
    r'cfg\.work_root / "'
    r'(sc-(?:preproc|seg|initsurf|deform))'
    r'"'
)

VERSIONED_DERIVATIVE_PATTERN = re.compile(
    r'\b(?:sc|scpp|simcortex)-'
    r'(?:preproc|seg|initsurf|deform)'
    r'(?:-[A-Za-z0-9_.+]+)*-'
    r'[0-9]+\.[0-9]+\b'
)

LEGACY_DERIVATIVE_NAMESPACE_PATTERN = re.compile(
    r'\b(?:scpp|simcortex)-'
    r'(?:preproc|seg|initsurf|deform)\b'
)

def test_preprocessing_names_are_canonical() -> None:
    assert PIPELINE_NAME == "sc-preproc"
    assert (
        PREPROC_DERIVATIVE_NAME
        == "sc-preproc"
    )


def test_full_pipeline_uses_canonical_stage_roots() -> None:
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "run_pipeline.py"
    ).read_text(
        encoding="utf-8",
    )

    roots = set(
        PIPELINE_ROOT_PATTERN.findall(source)
    )

    assert roots == CANONICAL_DERIVATIVE_NAMES
    assert (
        VERSIONED_DERIVATIVE_PATTERN.search(source)
        is None
    )


def test_active_code_has_no_legacy_or_versioned_roots() -> None:
    for relative_path in ACTIVE_CODE_FILES:
        source = (
            REPOSITORY_ROOT
            / relative_path
        ).read_text(
            encoding="utf-8",
        )

        assert (
            LEGACY_DERIVATIVE_NAMESPACE_PATTERN.search(
                source
            )
            is None
        ), (
            "Legacy derivative namespace remains in "
            f"{relative_path}"
        )

        assert (
            VERSIONED_DERIVATIVE_PATTERN.search(source)
            is None
        ), (
            "Versioned derivative root remains in "
            f"{relative_path}"
        )
