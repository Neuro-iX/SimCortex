"""Regression tests for SimCortex checkpoint compatibility contracts."""

from pathlib import Path

import pytest
import torch

import simcortex.seg.inference as seg_infer
import simcortex.seg.train as seg_train
import simcortex.deform.train as deform_train


def _state():
    """Return a small deterministic state_dict-like mapping."""
    return {
        "layer.weight": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
        ),
        "layer.bias": torch.tensor(
            [5.0, 6.0],
            dtype=torch.float32,
        ),
    }


def _module_state():
    return {
        f"module.{key}": value.clone()
        for key, value in _state().items()
    }


def _assert_state_equal(actual, expected):
    assert list(actual.keys()) == list(expected.keys())

    for key in expected:
        assert actual[key].dtype == expected[key].dtype
        assert actual[key].shape == expected[key].shape
        assert torch.equal(actual[key], expected[key]), key


@pytest.mark.parametrize(
    "wrapper_key",
    [
        "model",
        "state_dict",
        "model_state_dict",
    ],
)
def test_segmentation_extracts_supported_wrapped_formats(wrapper_key):
    """Segmentation inference must accept all supported wrapper keys."""
    expected = _state()

    checkpoint = {
        wrapper_key: {
            key: value.clone()
            for key, value in expected.items()
        }
    }

    actual = seg_infer._extract_state_dict(checkpoint)

    _assert_state_equal(actual, expected)


def test_segmentation_extracts_raw_state_dict():
    """Raw segmentation state_dict checkpoints remain supported."""
    expected = _state()

    actual = seg_infer._extract_state_dict(
        {
            key: value.clone()
            for key, value in expected.items()
        }
    )

    _assert_state_equal(actual, expected)


@pytest.mark.parametrize(
    "wrapper_key",
    [
        "model",
        "state_dict",
        "model_state_dict",
    ],
)
def test_segmentation_strips_ddp_prefix_in_wrapped_formats(wrapper_key):
    """Segmentation checkpoints saved through DDP must load without module."""
    checkpoint = {
        wrapper_key: _module_state(),
    }

    actual = seg_infer._extract_state_dict(checkpoint)

    _assert_state_equal(
        actual,
        _state(),
    )


def test_segmentation_strips_ddp_prefix_in_raw_state_dict():
    """A raw DDP segmentation state_dict must also be normalized."""
    actual = seg_infer._extract_state_dict(
        _module_state()
    )

    _assert_state_equal(
        actual,
        _state(),
    )


def test_segmentation_extract_rejects_non_dict_checkpoint():
    """Invalid segmentation checkpoint types must fail explicitly."""
    with pytest.raises(TypeError):
        seg_infer._extract_state_dict(
            ["not", "a", "checkpoint"]
        )


@pytest.mark.parametrize(
    "wrapper_key",
    [
        "model",
        "state_dict",
        "model_state_dict",
    ],
)
def test_deformation_extracts_supported_wrapped_formats(wrapper_key):
    """Deformation training must accept all supported checkpoint wrappers."""
    expected = _state()

    checkpoint = {
        wrapper_key: {
            key: value.clone()
            for key, value in expected.items()
        }
    }

    actual = deform_train.extract_model_state_dict(
        checkpoint
    )

    _assert_state_equal(
        actual,
        expected,
    )


def test_deformation_extracts_raw_state_dict():
    """Raw deformation model state_dict checkpoints remain supported."""
    expected = _state()

    actual = deform_train.extract_model_state_dict(
        {
            key: value.clone()
            for key, value in expected.items()
        }
    )

    _assert_state_equal(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "wrapper_key",
    [
        "model",
        "state_dict",
        "model_state_dict",
    ],
)
def test_deformation_strips_ddp_prefix(wrapper_key):
    """Fully DDP-prefixed deformation checkpoints remain loadable."""
    actual = deform_train.extract_model_state_dict(
        {
            wrapper_key: _module_state(),
        }
    )

    _assert_state_equal(
        actual,
        _state(),
    )


def test_deformation_raw_ddp_state_dict_is_normalized():
    """Raw DDP deformation state_dicts must lose the module prefix."""
    actual = deform_train.extract_model_state_dict(
        _module_state()
    )

    _assert_state_equal(
        actual,
        _state(),
    )


def test_deformation_extract_rejects_non_dict_checkpoint():
    """Invalid deformation checkpoint types must fail explicitly."""
    with pytest.raises(TypeError):
        deform_train.extract_model_state_dict(
            ["not", "a", "checkpoint"]
        )


@pytest.mark.parametrize(
    "loader",
    [
        seg_infer._load_trusted_checkpoint,
        seg_train._load_trusted_checkpoint,
        deform_train._load_trusted_checkpoint,
    ],
)
def test_trusted_checkpoint_loaders_preserve_full_checkpoint(
    tmp_path,
    loader,
):
    """Trusted loaders must preserve tensors and non-model checkpoint state."""
    path = tmp_path / "checkpoint.pth"

    payload = {
        "epoch": 17,
        "model": _state(),
        "optimizer": {
            "state": {},
            "param_groups": [
                {
                    "lr": 1.0e-4,
                    "params": [0],
                }
            ],
        },
        "scheduler": {
            "best": 1.234,
        },
        "rng_state": {
            "torch": torch.tensor(
                [1, 2, 3],
                dtype=torch.uint8,
            ),
        },
    }

    torch.save(
        payload,
        path,
    )

    loaded = loader(
        path,
        map_location="cpu",
    )

    assert loaded["epoch"] == 17

    _assert_state_equal(
        loaded["model"],
        payload["model"],
    )

    assert (
        loaded["optimizer"]["param_groups"][0]["lr"]
        == 1.0e-4
    )
    assert loaded["scheduler"]["best"] == 1.234

    assert torch.equal(
        loaded["rng_state"]["torch"],
        payload["rng_state"]["torch"],
    )


@pytest.mark.parametrize(
    "module",
    [
        seg_infer,
        seg_train,
        deform_train,
    ],
)
def test_trusted_loader_requests_weights_only_false(
    monkeypatch,
    module,
):
    """Modern PyTorch loads must explicitly preserve full trusted checkpoints."""
    calls = []

    sentinel = {
        "model": _state(),
        "optimizer": {
            "state": {},
        },
    }

    def fake_load(path, *, map_location, **kwargs):
        calls.append(
            {
                "path": path,
                "map_location": map_location,
                **kwargs,
            }
        )
        return sentinel

    monkeypatch.setattr(
        module.torch,
        "load",
        fake_load,
    )

    result = module._load_trusted_checkpoint(
        Path("/trusted/test.pth"),
        map_location="cpu",
    )

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0]["map_location"] == "cpu"
    assert calls[0]["weights_only"] is False


@pytest.mark.parametrize(
    "module",
    [
        seg_infer,
        seg_train,
        deform_train,
    ],
)
def test_trusted_loader_falls_back_for_older_pytorch(
    monkeypatch,
    module,
):
    """PyTorch versions without weights_only must use the legacy call."""
    calls = []

    sentinel = {
        "model": _state(),
    }

    def fake_load(path, *, map_location, **kwargs):
        calls.append(
            {
                "path": path,
                "map_location": map_location,
                **kwargs,
            }
        )

        if "weights_only" in kwargs:
            raise TypeError(
                "unexpected keyword argument 'weights_only'"
            )

        return sentinel

    monkeypatch.setattr(
        module.torch,
        "load",
        fake_load,
    )

    result = module._load_trusted_checkpoint(
        Path("/trusted/legacy.pth"),
        map_location="cpu",
    )

    assert result is sentinel

    assert len(calls) == 2

    assert calls[0]["weights_only"] is False

    assert "weights_only" not in calls[1]
    assert calls[1]["map_location"] == "cpu"


def test_checkpoint_tensor_values_survive_save_load_extract_roundtrip(
    tmp_path,
):
    """Checkpoint compatibility must never alter learned tensor values."""
    expected = _state()

    path = tmp_path / "full_checkpoint.pth"

    torch.save(
        {
            "epoch": 42,
            "model": {
                f"module.{key}": value.clone()
                for key, value in expected.items()
            },
            "optimizer": {
                "state": {},
                "param_groups": [],
            },
        },
        path,
    )

    loaded = deform_train._load_trusted_checkpoint(
        str(path),
        map_location="cpu",
    )

    actual = deform_train.extract_model_state_dict(
        loaded
    )

    _assert_state_equal(
        actual,
        expected,
    )

    assert loaded["epoch"] == 42
