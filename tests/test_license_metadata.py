import subprocess
import sys
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path


def test_wheel_uses_modern_apache_license_metadata(tmp_path):
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

    build_output = (
        completed.stdout
        + "\n"
        + completed.stderr
    )

    assert completed.returncode == 0, build_output

    deprecated_warnings = (
        "`project.license` as a TOML table is deprecated",
        "'tool.setuptools.license-files' is deprecated",
        "License classifiers are deprecated",
    )

    for warning in deprecated_warnings:
        assert warning not in build_output

    wheels = sorted(output_dir.glob("*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_entries = set(archive.namelist())

        metadata_entries = [
            name
            for name in wheel_entries
            if name.endswith(".dist-info/METADATA")
        ]

        license_entries = [
            name
            for name in wheel_entries
            if name.endswith(".dist-info/licenses/LICENSE")
        ]

        assert len(metadata_entries) == 1
        assert len(license_entries) == 1

        metadata = BytesParser(
            policy=default,
        ).parsebytes(
            archive.read(metadata_entries[0])
        )

    assert (
        metadata.get("License-Expression")
        == "Apache-2.0"
    )

    license_files = metadata.get_all(
        "License-File",
        [],
    )
    classifiers = set(
        metadata.get_all("Classifier", [])
    )

    assert "LICENSE" in license_files
    assert (
        "License :: OSI Approved :: "
        "Apache Software License"
        not in classifiers
    )
