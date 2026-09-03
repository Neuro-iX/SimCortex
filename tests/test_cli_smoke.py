"""Smoke/regression tests for the public SimCortex CLI."""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

import pytest
import typer
from typer.main import get_command
from typer.testing import CliRunner

import simcortex.cli.main as cli


runner = CliRunner()


def test_top_level_public_command_contract():
    """The public CLI exposes exactly the four finalized stage commands/groups."""
    command = get_command(cli.app)

    assert list(command.commands) == [
        "fs-to-mni",
        "seg",
        "initsurf",
        "deform",
    ]


def test_top_level_help_smoke():
    """The installed application must render top-level help successfully."""
    result = runner.invoke(
        cli.app,
        ["--help"],
    )

    assert result.exit_code == 0, result.output

    assert "SimCortex" in result.output
    assert "fs-to-mni" in result.output
    assert "seg" in result.output
    assert "initsurf" in result.output
    assert "deform" in result.output


@pytest.mark.parametrize(
    "flag",
    [
        "--version",
        "-V",
    ],
)
def test_top_level_version_option(flag):
    """Both public version flags must report the package version and exit."""
    result = runner.invoke(
        cli.app,
        [flag],
    )

    assert result.exit_code == 0, result.output
    assert result.output == f"simcortex {cli.__version__}\n"


def test_top_level_no_arguments_preserves_missing_command_behavior():
    """Running the root CLI without a command must remain an error."""
    result = runner.invoke(
        cli.app,
        [],
    )

    assert result.exit_code == 2
    assert "Missing command." in result.output


def test_preproc_is_not_a_public_top_level_command():
    """Stage 1 is intentionally named fs-to-mni, not preproc."""
    command = get_command(cli.app)

    assert "preproc" not in command.commands
    assert "fs-to-mni" in command.commands


@pytest.mark.parametrize(
    ("args", "expected_commands"),
    [
        (
            ["seg", "--help"],
            ["train", "infer", "eval"],
        ),
        (
            ["initsurf", "--help"],
            ["generate"],
        ),
        (
            ["deform", "--help"],
            ["train", "infer", "eval"],
        ),
    ],
)
def test_stage_group_help_smoke(
    args,
    expected_commands,
):
    """Stage sub-apps must render help and expose their intended subcommands."""
    result = runner.invoke(
        cli.app,
        args,
    )

    assert result.exit_code == 0, result.output

    for name in expected_commands:
        assert name in result.output


def test_fs_to_mni_help_smoke():
    """The direct Stage-1 Typer application must remain reachable."""
    result = runner.invoke(
        cli.app,
        [
            "fs-to-mni",
            "--help",
        ],
    )

    assert result.exit_code == 0, result.output


def test_run_module_constructs_standard_python_module_command(
    monkeypatch,
):
    """Non-DDP commands must use the active interpreter and python -m."""
    calls = []

    def fake_run(cmd, *, check):
        calls.append(
            {
                "cmd": list(cmd),
                "check": check,
            }
        )
        return SimpleNamespace(
            returncode=7,
        )

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        fake_run,
    )

    rc = cli.run_module(
        "simcortex.seg.inference",
        [
            "dataset.path=/tmp/data",
            "trainer.device=cpu",
        ],
    )

    assert rc == 7

    assert calls == [
        {
            "cmd": [
                sys.executable,
                "-m",
                "simcortex.seg.inference",
                "dataset.path=/tmp/data",
                "trainer.device=cpu",
            ],
            "check": False,
        }
    ]


def test_run_module_constructs_torchrun_command(
    monkeypatch,
):
    """DDP training must use torch.distributed.run from the active environment."""
    calls = []

    def fake_run(cmd, *, check):
        calls.append(
            {
                "cmd": list(cmd),
                "check": check,
            }
        )
        return SimpleNamespace(
            returncode=0,
        )

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        fake_run,
    )

    rc = cli.run_module(
        "simcortex.deform.train",
        [
            "trainer.num_epochs=10",
        ],
        torchrun=True,
        nproc_per_node=2,
    )

    assert rc == 0

    assert calls == [
        {
            "cmd": [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=2",
                "-m",
                "simcortex.deform.train",
                "trainer.num_epochs=10",
            ],
            "check": False,
        }
    ]


def test_run_module_rejects_invalid_process_count():
    """Invalid torchrun process counts must fail before launching anything."""
    with pytest.raises(
        typer.BadParameter,
        match="nproc-per-node",
    ):
        cli.run_module(
            "simcortex.deform.train",
            torchrun=True,
            nproc_per_node=0,
        )


@pytest.mark.parametrize(
    ("cli_args", "expected_module"),
    [
        (
            [
                "seg",
                "infer",
                "dataset.path=/tmp/preproc",
            ],
            "simcortex.seg.inference",
        ),
        (
            [
                "seg",
                "eval",
                "dataset.split_name=test",
            ],
            "simcortex.seg.eval",
        ),
        (
            [
                "initsurf",
                "generate",
                "dataset.split_name=test",
            ],
            "simcortex.initsurf.generate",
        ),
        (
            [
                "deform",
                "infer",
                "dataset.split_name=test",
            ],
            "simcortex.deform.inference",
        ),
        (
            [
                "deform",
                "eval",
                "dataset.split_name=test",
            ],
            "simcortex.deform.eval",
        ),
    ],
)
def test_cli_routes_hydra_overrides_to_correct_module(
    monkeypatch,
    cli_args,
    expected_module,
):
    """Hydra-style arguments must be forwarded unchanged to stage modules."""
    calls = []

    def fake_run_module(
        module,
        overrides=None,
        *,
        torchrun=False,
        nproc_per_node=1,
    ):
        calls.append(
            {
                "module": module,
                "overrides": list(overrides or []),
                "torchrun": torchrun,
                "nproc_per_node": nproc_per_node,
            }
        )
        return 0

    monkeypatch.setattr(
        cli,
        "run_module",
        fake_run_module,
    )

    result = runner.invoke(
        cli.app,
        cli_args,
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1

    assert calls[0]["module"] == expected_module

    # The final CLI item in each case is the Hydra override.
    assert calls[0]["overrides"] == [
        cli_args[-1]
    ]

    assert calls[0]["torchrun"] is False
    assert calls[0]["nproc_per_node"] == 1


@pytest.mark.parametrize(
    ("stage", "expected_module"),
    [
        (
            "seg",
            "simcortex.seg.train",
        ),
        (
            "deform",
            "simcortex.deform.train",
        ),
    ],
)
def test_training_cli_routes_torchrun_options(
    monkeypatch,
    stage,
    expected_module,
):
    """Training commands must preserve explicit multi-GPU launch controls."""
    calls = []

    def fake_run_module(
        module,
        overrides=None,
        *,
        torchrun=False,
        nproc_per_node=1,
    ):
        calls.append(
            {
                "module": module,
                "overrides": list(overrides or []),
                "torchrun": torchrun,
                "nproc_per_node": nproc_per_node,
            }
        )
        return 0

    monkeypatch.setattr(
        cli,
        "run_module",
        fake_run_module,
    )

    result = runner.invoke(
        cli.app,
        [
            stage,
            "train",
            "--torchrun",
            "--nproc-per-node",
            "2",
            "trainer.num_epochs=25",
        ],
    )

    assert result.exit_code == 0, result.output

    assert calls == [
        {
            "module": expected_module,
            "overrides": [
                "trainer.num_epochs=25",
            ],
            "torchrun": True,
            "nproc_per_node": 2,
        }
    ]


def test_cli_source_contains_no_stale_scpp_name():
    """The finalized public CLI must not expose the former scpp naming."""
    source = (
        cli.__file__
        and open(
            cli.__file__,
            encoding="utf-8",
        ).read()
    )

    assert source is not None
    assert "scpp" not in source.lower()


def test_pyproject_exposes_simcortex_console_script_only():
    """Packaging must point the public console command at this Typer app."""
    text = open(
        "pyproject.toml",
        encoding="utf-8",
    ).read()

    assert re.search(
        r'(?m)^\s*simcortex\s*=\s*["\']'
        r'simcortex\.cli\.main:app["\']\s*$',
        text,
    )

    assert not re.search(
        r'(?m)^\s*scpp\s*=',
        text,
    )
