import subprocess
import sys
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path


def test_wheel_declares_preprocessing_extra(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "dist"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output_dir),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        completed.stdout + "\n" + completed.stderr
    )

    wheels = sorted(output_dir.glob("*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        metadata_entries = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]

        assert len(metadata_entries) == 1

        metadata = BytesParser(
            policy=default,
        ).parsebytes(
            archive.read(metadata_entries[0])
        )

    provided_extras = set(
        metadata.get_all("Provides-Extra", [])
    )
    requirements = metadata.get_all(
        "Requires-Dist",
        [],
    )

    normalized_requirements = [
        requirement.lower().replace("'", '"')
        for requirement in requirements
    ]

    assert "preproc" in provided_extras

    assert any(
        requirement.startswith("antspyx>=0.6.1")
        and 'extra == "preproc"' in requirement
        for requirement in normalized_requirements
    )
