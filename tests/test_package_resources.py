import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


SOURCE_LUT_PATH = Path(
    "src/simcortex/utils/critical186LUT.raw.gz"
)
WHEEL_LUT_PATH = (
    "simcortex/utils/critical186LUT.raw.gz"
)
SDIST_LUT_SUFFIX = (
    "/src/simcortex/utils/critical186LUT.raw.gz"
)


def test_source_lut_exists():
    repository_root = Path(__file__).resolve().parents[1]
    lut_path = repository_root / SOURCE_LUT_PATH

    assert lut_path.is_file()
    assert lut_path.stat().st_size > 0


def test_built_distributions_contain_lut(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "dist"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
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
    source_distributions = sorted(
        output_dir.glob("*.tar.gz")
    )

    assert len(wheels) == 1
    assert len(source_distributions) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_entries = set(archive.namelist())

    assert WHEEL_LUT_PATH in wheel_entries

    with tarfile.open(
        source_distributions[0],
        mode="r:gz",
    ) as archive:
        source_entries = {
            member.name
            for member in archive.getmembers()
        }

    assert any(
        entry.endswith(SDIST_LUT_SUFFIX)
        for entry in source_entries
    )
