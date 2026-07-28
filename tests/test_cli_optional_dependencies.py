import subprocess
import sys
import textwrap


BLOCK_ANTS_IMPORT = """
import importlib.abc
import sys


class BlockAntsImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ants" or fullname.startswith("ants."):
            raise ModuleNotFoundError(
                "ANTsPy intentionally blocked for CLI test"
            )
        return None


sys.meta_path.insert(0, BlockAntsImport())
"""


def run_without_antspy(code: str) -> subprocess.CompletedProcess:
    script = (
        textwrap.dedent(BLOCK_ANTS_IMPORT)
        + "\n"
        + textwrap.dedent(code)
    )

    return subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
    )


def assert_subprocess_passed(
    completed: subprocess.CompletedProcess,
) -> None:
    assert completed.returncode == 0, (
        "STDOUT:\n"
        + completed.stdout
        + "\nSTDERR:\n"
        + completed.stderr
    )


def test_base_package_imports_without_antspy():
    completed = run_without_antspy(
        """
        import simcortex

        print("Base package import: PASS")
        """
    )

    assert_subprocess_passed(completed)


def test_root_cli_help_works_without_antspy():
    completed = run_without_antspy(
        """
        from typer.testing import CliRunner

        from simcortex.cli.main import app

        result = CliRunner().invoke(
            app,
            ["--help"],
        )

        print(result.stdout)

        assert result.exit_code == 0
        assert "fs-to-mni" in result.stdout
        assert "seg" in result.stdout
        assert "initsurf" in result.stdout
        assert "deform" in result.stdout
        """
    )

    assert_subprocess_passed(completed)


def test_fs_to_mni_help_works_without_antspy():
    completed = run_without_antspy(
        """
        from typer.testing import CliRunner

        from simcortex.cli.main import app

        result = CliRunner().invoke(
            app,
            [
                "fs-to-mni",
                "--help",
            ],
        )

        print(result.stdout)

        assert result.exit_code == 0
        assert "--freesurfer-root" in result.stdout
        assert "--out-deriv-root" in result.stdout
        assert "--mni-template" in result.stdout
        assert "--transform-type" in result.stdout
        """
    )

    assert_subprocess_passed(completed)


def test_preprocessing_operation_reports_missing_antspy():
    completed = run_without_antspy(
        """
        from pathlib import Path

        from simcortex.preproc.fs_to_mni import (
            ants_affine_to_homogeneous_lps,
        )

        try:
            ants_affine_to_homogeneous_lps(
                Path("unused-transform.mat")
            )
        except RuntimeError as exc:
            message = str(exc)

            assert (
                "ANTsPy is required for Stage 1 preprocessing"
                in message
            )
            assert (
                'python -m pip install "simcortex[preproc]"'
                in message
            )
            assert "pip install antspyx" not in message
        else:
            raise AssertionError(
                "Expected missing ANTsPy RuntimeError"
            )
        """
    )

    assert_subprocess_passed(completed)


def test_mri_only_preprocessing_module_imports_without_antspy():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        r"""
        import importlib.abc
        import sys


        class BlockAntsFinder(importlib.abc.MetaPathFinder):
            def find_spec(
                self,
                fullname,
                path=None,
                target=None,
            ):
                if (
                    fullname == "ants"
                    or fullname.startswith("ants.")
                ):
                    raise ModuleNotFoundError(
                        "ANTsPy intentionally blocked "
                        "for MRI preprocessing import test"
                    )

                return None


        sys.meta_path.insert(
            0,
            BlockAntsFinder(),
        )

        import simcortex.preproc.mri_to_mni_inference

        print(
            "MRI-only preprocessing import "
            "without ANTsPy: PASS"
        )
        """
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        completed.stdout
        + "\n"
        + completed.stderr
    )


def test_mri_only_preprocessing_reports_missing_antspy():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        r"""
        import importlib.abc
        import sys
        from pathlib import Path


        class BlockAntsFinder(importlib.abc.MetaPathFinder):
            def find_spec(
                self,
                fullname,
                path=None,
                target=None,
            ):
                if (
                    fullname == "ants"
                    or fullname.startswith("ants.")
                ):
                    raise ModuleNotFoundError(
                        "ANTsPy intentionally blocked "
                        "for MRI preprocessing runtime test"
                    )

                return None


        sys.meta_path.insert(
            0,
            BlockAntsFinder(),
        )

        from simcortex.preproc.mri_to_mni_inference import (
            ants_affine_to_homogeneous_lps,
        )

        try:
            ants_affine_to_homogeneous_lps(
                Path("missing-transform.mat")
            )
        except RuntimeError as exc:
            message = str(exc)

            assert (
                "ANTsPy is required for "
                "Stage 1 preprocessing"
                in message
            )
            assert (
                'python -m pip install '
                '"simcortex[preproc]"'
                in message
            )
            assert (
                "pip install antspyx"
                not in message
            )
        else:
            raise AssertionError(
                "Expected missing ANTsPy RuntimeError"
            )

        print(
            "MRI-only preprocessing missing "
            "ANTsPy message: PASS"
        )
        """
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        completed.stdout
        + "\n"
        + completed.stderr
    )
