"""Regression tests for the SimCortex deformation model contract."""

import inspect

import pytest
import torch

import simcortex.deform.models.surfdeform as surfmod
from simcortex.deform.models.surfdeform import (
    DualMUNetV2,
    SurfDeform,
)


SMALL_KWARGS = dict(
    C_in=2,
    C_hid=(4, 4, 8, 8, 8, 8),
    inshape=(16, 16, 16),
    sigma=1.0,
    geom_ratio=0.5,
    gn_groups=4,
    gate_init=-3.0,
    dropout=0.0,
)


def _assert_nested_equal(a, b):
    """Require bitwise equality for tensors and nested tensor outputs."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert torch.equal(a, b)
        return

    if isinstance(a, (tuple, list)):
        assert type(a) is type(b)
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_nested_equal(x, y)
        return

    raise TypeError(f"Unexpected model output type: {type(a)!r}")


def test_production_constructor_defaults():
    """Public architecture defaults must match the finalized production model."""
    dual = inspect.signature(DualMUNetV2)
    surf = inspect.signature(SurfDeform)

    assert dual.parameters["C_in"].default == 2
    assert dual.parameters["C_hid"].default == (
        8,
        16,
        32,
        64,
        128,
        128,
    )
    assert dual.parameters["geom_ratio"].default == 0.5
    assert dual.parameters["geom_depth"].default == 6
    assert dual.parameters["gn_groups"].default == 8
    assert dual.parameters["gate_init"].default == -3.0
    assert dual.parameters["dropout"].default == 0.0

    assert surf.parameters["C_in"].default == 2
    assert surf.parameters["C_hid"].default == (
        8,
        16,
        32,
        64,
        128,
        128,
    )
    assert surf.parameters["geom_ratio"].default == 0.5
    assert surf.parameters["geom_depth"].default == 6
    assert surf.parameters["gn_groups"].default == 8
    assert surf.parameters["gate_init"].default == -3.0
    assert surf.parameters["dropout"].default == 0.0


def test_default_depth_six_equals_explicit_depth_six_state():
    """Omitting geom_depth must be identical to explicitly requesting six."""
    torch.manual_seed(12345)
    default_model = SurfDeform(
        **SMALL_KWARGS,
    )

    torch.manual_seed(12345)
    explicit_model = SurfDeform(
        **SMALL_KWARGS,
        geom_depth=6,
    )

    default_state = default_model.state_dict()
    explicit_state = explicit_model.state_dict()

    assert list(default_state) == list(explicit_state)

    for key in default_state:
        assert default_state[key].shape == explicit_state[key].shape
        assert torch.equal(
            default_state[key],
            explicit_state[key],
        ), key

    assert default_model.munet.geom_depth == 6
    assert explicit_model.munet.geom_depth == 6


def test_explicit_historical_depth_four_remains_supported():
    """The historical shallower geometry encoder must remain constructible."""
    model = SurfDeform(
        **SMALL_KWARGS,
        geom_depth=4,
    )

    assert model.munet.geom_depth == 4

    assert model.munet.g1 is not None
    assert model.munet.g2 is not None
    assert model.munet.g3 is not None
    assert model.munet.g4 is not None

    assert model.munet.g5 is None
    assert model.munet.g6 is None


def test_default_and_explicit_depth_six_forward_are_bitwise_identical():
    """The production default must match explicit depth-six execution."""
    torch.manual_seed(2026)
    default_model = SurfDeform(
        **SMALL_KWARGS,
    ).eval()

    torch.manual_seed(2026)
    explicit_model = SurfDeform(
        **SMALL_KWARGS,
        geom_depth=6,
    ).eval()

    # Make the equivalence contract explicit.
    explicit_model.load_state_dict(
        default_model.state_dict(),
        strict=True,
    )

    torch.manual_seed(99)
    vol = torch.randn(
        1,
        2,
        16,
        16,
        16,
    )

    vert = torch.tensor(
        [
            [
                [4.0, 4.0, 4.0],
                [8.0, 8.0, 8.0],
                [12.0, 10.0, 6.0],
            ]
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        out_default = default_model(
            vert.clone(),
            vol.clone(),
            n_steps=1,
        )
        out_explicit = explicit_model(
            vert.clone(),
            vol.clone(),
            n_steps=1,
        )

    _assert_nested_equal(
        out_default,
        out_explicit,
    )


def test_interpolation_contract(monkeypatch):
    """SurfDeform interpolation must keep the historical grid_sample policy."""
    model = SurfDeform(
        **SMALL_KWARGS,
        geom_depth=6,
    )

    captured = {}

    def fake_grid_sample(input_tensor, grid, **kwargs):
        captured["input_shape"] = tuple(input_tensor.shape)
        captured["grid_shape"] = tuple(grid.shape)
        captured.update(kwargs)

        # Shape is irrelevant for this contract test.
        return torch.empty(
            0,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

    monkeypatch.setattr(
        surfmod.F,
        "grid_sample",
        fake_grid_sample,
    )

    src = torch.zeros(
        1,
        3,
        16,
        16,
        16,
    )

    coord = torch.tensor(
        [
            [
                [4.0, 4.0, 4.0],
                [8.0, 8.0, 8.0],
            ]
        ],
        dtype=torch.float32,
    )

    model.interpolate(
        coord,
        src,
    )

    assert captured["mode"] == "bilinear"
    assert captured["padding_mode"] == "border"
    assert captured["align_corners"] is True


def test_decoder_upsampling_contract():
    """Decoder upsampling must remain trilinear with aligned corners."""
    model = DualMUNetV2(
        C_in=2,
        C_hid=(4, 4, 8, 8, 8, 8),
        geom_ratio=0.5,
        geom_depth=6,
        gn_groups=4,
        gate_init=-3.0,
        dropout=0.0,
    )

    assert model.up.mode == "trilinear"
    assert model.up.align_corners is True


def test_four_flow_heads_preserve_three_vector_channels():
    """Each multi-scale SVF head must continue to predict 3-D vectors."""
    model = DualMUNetV2(
        C_in=2,
        C_hid=(4, 4, 8, 8, 8, 8),
        geom_ratio=0.5,
        geom_depth=6,
        gn_groups=4,
        gate_init=-3.0,
        dropout=0.0,
    )

    assert model.flow1.out_channels == 3
    assert model.flow2.out_channels == 3
    assert model.flow3.out_channels == 3
    assert model.flow4.out_channels == 3


@pytest.mark.parametrize(
    "geom_depth",
    [0, 7],
)
def test_invalid_geom_depth_is_rejected(geom_depth):
    """Only geometry depths one through six are valid."""
    with pytest.raises(ValueError):
        DualMUNetV2(
            C_in=2,
            geom_depth=geom_depth,
        )


def test_invalid_single_channel_input_contract_is_rejected():
    """The model requires MRI plus at least one geometry/probability channel."""
    with pytest.raises(ValueError):
        DualMUNetV2(
            C_in=1,
        )


@pytest.mark.parametrize(
    "inshape",
    [
        (15, 16, 16),
        (16, 0, 16),
        (16, 16),
    ],
)
def test_invalid_surfdeform_inshape_is_rejected(inshape):
    """Spatial dimensions must be a positive 3-tuple divisible by eight."""
    with pytest.raises(ValueError):
        SurfDeform(
            C_in=2,
            C_hid=(4, 4, 8, 8, 8, 8),
            inshape=inshape,
            geom_depth=6,
            gn_groups=4,
        )


def test_forward_rejects_wrong_volume_channel_count():
    """Production deformation input remains two channels."""
    model = SurfDeform(
        **SMALL_KWARGS,
        geom_depth=6,
    )

    vert = torch.zeros(
        1,
        3,
        3,
    )
    vol = torch.zeros(
        1,
        1,
        16,
        16,
        16,
    )

    with pytest.raises(
        ValueError,
        match="channels",
    ):
        model(
            vert,
            vol,
            n_steps=1,
        )


def test_forward_rejects_negative_integration_steps():
    """Negative scaling-and-squaring step counts must fail explicitly."""
    model = SurfDeform(
        **SMALL_KWARGS,
        geom_depth=6,
    )

    vert = torch.zeros(
        1,
        3,
        3,
    )
    vol = torch.zeros(
        1,
        2,
        16,
        16,
        16,
    )

    with pytest.raises(
        ValueError,
        match="n_steps",
    ):
        model(
            vert,
            vol,
            n_steps=-1,
        )
