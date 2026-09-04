"""Regression tests for finalized SimCortex configuration contracts."""

import re
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "src" / "simcortex" / "configs"

SEG_INFER = CONFIG_ROOT / "seg" / "inference.yaml"
SEG_EVAL = CONFIG_ROOT / "seg" / "eval.yaml"
SEG_TRAIN = CONFIG_ROOT / "seg" / "train.yaml"

INITSURF = CONFIG_ROOT / "initsurf" / "generate.yaml"

DEFORM_INFER = CONFIG_ROOT / "deform" / "inference.yaml"
DEFORM_EVAL = CONFIG_ROOT / "deform" / "eval.yaml"
DEFORM_TRAIN = CONFIG_ROOT / "deform" / "train.yaml"

PUBLIC_CONFIGS = [
    SEG_INFER,
    SEG_EVAL,
    SEG_TRAIN,
    INITSURF,
    DEFORM_INFER,
    DEFORM_EVAL,
    DEFORM_TRAIN,
]


def load(path):
    return OmegaConf.load(path)


def test_public_configs_contain_no_private_or_legacy_paths():
    """Release configs must not expose private cluster or stale derivative names."""
    forbidden = [
        "/project/",
        "sc-preproc-0.1",
        "sc-preproc-0.2",
        "sc-seg-0.1",
        "sc-seg-0.2",
        "sc-initsurf-0.1",
        "sc-initsurf-0.2",
        "sc-deform-0.1",
        "sc-deform-0.2",
        "exp13",
        "exp33",
        "Ablation",
        "3Losses",
        "oneInput",
    ]

    unapproved_derivative = re.compile(
        r"\\b(?!sc-)[a-z][a-z0-9]*-"
        r"(?:preproc|seg|initsurf|deform)\\b",
        flags=re.IGNORECASE,
    )

    for path in PUBLIC_CONFIGS:
        text = path.read_text()

        for value in forbidden:
            assert value not in text, (
                f"{value!r} found in {path}"
            )

        match = unapproved_derivative.search(text)
        assert match is None, (
            f"unapproved derivative namespace "
            f"{match.group(0)!r} found in {path}"
        )


def test_public_configs_use_placeholder_paths():
    """Release configs must use portable /path/to examples."""
    for path in PUBLIC_CONFIGS:
        text = path.read_text()

        # Every public config has at least one filesystem example.
        assert "/path/to/" in text, path


def test_segmentation_training_scientific_contract():
    """Lock the finalized segmentation training recipe."""
    cfg = load(SEG_TRAIN)

    assert cfg.dataset.train_split == "train"
    assert cfg.dataset.val_split == "val"
    assert cfg.dataset.session_label == "01"
    assert cfg.dataset.space == "MNI152"
    assert cfg.dataset.pad_mult == 16
    assert cfg.dataset.augment is True

    assert cfg.model.in_channels == 1
    assert cfg.model.out_channels == 9
    assert list(cfg.model.features) == [
        16,
        32,
        64,
        128,
        128,
    ]
    assert cfg.model.norm == "instance"
    assert cfg.model.dropout == 0.10

    assert cfg.trainer.batch_size == 2
    assert cfg.trainer.num_workers == 4
    assert cfg.trainer.use_ddp is True
    assert cfg.trainer.data_parallel is False

    assert cfg.trainer.seed == 12345

    # Important: preserve the historical non-deterministic training policy.
    assert cfg.trainer.deterministic is False

    assert cfg.trainer.num_epochs == 1000
    assert cfg.trainer.validation_interval == 5

    assert cfg.trainer.optimizer == "adamw"
    assert cfg.trainer.learning_rate == 1.0e-4
    assert cfg.trainer.weight_decay == 1.0e-5

    assert cfg.trainer.dice_weight == 1.0
    assert cfg.trainer.dice_exclude_bg is True
    assert (
        cfg.trainer.dice_ignore_absent_target_classes
        is True
    )
    assert cfg.trainer.ce_class_weights is None

    assert cfg.trainer.amp is True
    assert cfg.trainer.grad_clip_norm == 1.0

    assert cfg.trainer.scheduler.name == "cosine"
    assert cfg.trainer.scheduler.eta_min == 1.0e-6

    assert cfg.trainer.save_interval == 25
    assert cfg.trainer.keep_last_n_checkpoints == 3
    assert cfg.trainer.early_stop_patience == 30
    assert cfg.trainer.resume_from is None

    assert (
        cfg.outputs.root
        == "/path/to/simcortex-runs/seg/exp01_hcpya+oasis1"
    )


def test_segmentation_inference_contract():
    """Lock segmentation inference architecture and execution defaults."""
    cfg = load(SEG_INFER)

    assert cfg.dataset.split_name == "all"
    assert cfg.dataset.session_label == "01"
    assert cfg.dataset.space == "MNI152"
    assert cfg.dataset.pad_mult == 16

    assert cfg.model.in_channels == 1
    assert cfg.model.out_channels == 9
    assert list(cfg.model.features) == [
        16,
        32,
        64,
        128,
        128,
    ]
    assert cfg.model.norm == "instance"
    assert cfg.model.dropout == 0.10

    assert cfg.trainer.batch_size == 1
    assert cfg.trainer.num_workers == 2
    assert cfg.trainer.amp is True

    assert cfg.outputs.overwrite is False
    assert cfg.outputs.overwrite_dataset_description is False
    assert cfg.outputs.tensorboard is True


def test_segmentation_evaluation_contract():
    """Lock segmentation evaluation metrics."""
    cfg = load(SEG_EVAL)

    assert cfg.dataset.split_name == "test"
    assert cfg.dataset.space == "MNI152"

    assert cfg.evaluation.num_classes == 9
    assert cfg.evaluation.exclude_background is True
    assert cfg.evaluation.eps == 1.0e-6
    assert cfg.evaluation.compute_nsd is True
    assert cfg.evaluation.nsd_tolerance_vox == 1.0
    assert cfg.evaluation.nsd_include_background is False
    assert list(cfg.evaluation.spacing) == [
        1.0,
        1.0,
        1.0,
    ]


def test_initsurf_geometry_contract():
    """Lock the historical InitSurf geometry-generation parameters."""
    cfg = load(INITSURF)

    assert cfg.n_workers == 4

    assert cfg.dataset.split_name == "all"
    assert cfg.dataset.session_label == "01"
    assert cfg.dataset.space == "MNI152"

    assert cfg.params.overwrite is False
    assert cfg.params.validate_affine is True
    assert cfg.params.affine_atol == 1.0e-4

    assert cfg.params.gap_size == 1
    assert cfg.params.sdf_sigma == 0.5
    assert cfg.params.topo_threshold == 16.0
    assert cfg.params.n_smooth == 1

    assert cfg.params.wm_start_level == -0.2
    assert cfg.params.wm_step == -0.08
    assert cfg.params.wm_min_level == -3.0
    assert cfg.params.wm_inset == 1.0

    assert cfg.params.pial_min_level == 1.8
    assert cfg.params.pial_max_level == 2.7
    assert cfg.params.pial_grid_step == 0.1
    assert cfg.params.pial_absolute_floor == 0.1


def test_deformation_training_model_contract():
    """Lock the finalized deformation architecture."""
    cfg = load(DEFORM_TRAIN)

    assert cfg.model.sigma == 1
    assert cfg.model.n_steps == 8
    assert cfg.model.c_in == 2

    assert list(cfg.model.inshape) == [
        184,
        224,
        184,
    ]
    assert list(cfg.model.c_hid) == [
        8,
        16,
        32,
        64,
        128,
        128,
    ]

    assert cfg.model.geom_ratio == 0.5
    assert cfg.model.geom_depth == 6
    assert cfg.model.gn_groups == 8
    assert cfg.model.gate_init == -3.0
    assert cfg.model.dropout == 0.1

    assert cfg.model.init_ckpt is None
    assert cfg.model.init_strict is True


def test_deformation_training_optimizer_contract():
    """Lock deformation optimizer, sampling, and scheduler values."""
    cfg = load(DEFORM_TRAIN)

    assert cfg.trainer.seed == 2025
    assert cfg.trainer.img_batch_size == 1
    assert cfg.trainer.grad_accum_steps == 4

    assert cfg.trainer.num_epochs == 600
    assert cfg.trainer.validation_interval == 5
    assert cfg.trainer.collision_interval == 5

    assert cfg.trainer.learning_rate == 1.0e-4
    assert cfg.trainer.weight_decay == 1.0e-4

    assert cfg.trainer.points_per_image == 180000
    assert cfg.trainer.val_points_per_image == 150000

    assert cfg.trainer.num_workers == 4
    assert cfg.trainer.mesh_chunk == 4
    assert cfg.trainer.grad_clip_norm == 1.0

    assert cfg.trainer.scheduler_patience == 4
    assert cfg.trainer.scheduler_factor == 0.5
    assert cfg.trainer.scheduler_min_lr == 1.0e-6
    assert cfg.trainer.scheduler_threshold_mm == 0.0004
    assert cfg.trainer.scheduler_threshold_mode == "abs"
    assert cfg.trainer.scheduler_cooldown == 1

    assert cfg.trainer.early_stop_patience == 15
    assert cfg.trainer.early_stop_min_delta_mm == 0.0003
    assert cfg.trainer.resume_from is None


def test_deformation_training_objective_contract():
    """Lock the finalized full deformation objective."""
    cfg = load(DEFORM_TRAIN)

    objective = cfg.objective

    assert objective.chamfer_weight == 1.0
    assert objective.chamfer_scale == 1.0

    assert objective.edge_loss_weight == 2.5
    assert objective.normal_weight == 0.25
    assert objective.reg_warmup_epochs == 25

    # White/pial partial-Hausdorff guardrail.
    assert objective.hd_weight == 5.5
    assert objective.hd_p == 0.05
    assert objective.hd_lambda_mm == 0.75
    assert objective.hd_points == 50000

    # Signed nesting term.
    assert objective.signed_nested_weight == 8.0
    assert objective.signed_margin_mm == 0.55
    assert objective.signed_points == 50000

    # Left/right pial separation term.
    assert objective.pial_lr_hd_weight == 1.5
    assert objective.pial_lr_hd_p == 0.05
    assert objective.pial_lr_hd_lambda_mm == 1.0
    assert objective.pial_lr_hd_points == 30000


def test_deformation_checkpoint_selection_contract():
    """Lock the finalized collision-aware checkpoint-selection policy."""
    cfg = load(DEFORM_TRAIN)

    assert cfg.checkpoint.alpha_wp == 0.03
    assert cfg.checkpoint.alpha_lr == 0.015
    assert cfg.checkpoint.rmse_guardrail_rel == 1.03
    assert cfg.checkpoint.min_delta_score == 0.0001
    assert cfg.checkpoint.require_collision_for_best is True


def test_deformation_inference_contract():
    """Training and inference must use the same deformation architecture."""
    train = load(DEFORM_TRAIN)
    infer = load(DEFORM_INFER)

    assert infer.model.c_in == train.model.c_in == 2
    assert infer.model.sigma == train.model.sigma == 1
    assert infer.model.n_steps == train.model.n_steps == 8

    assert list(infer.model.inshape) == list(train.model.inshape)
    assert list(infer.model.c_hid) == list(train.model.c_hid)

    assert infer.model.geom_ratio == train.model.geom_ratio == 0.5
    assert infer.model.geom_depth == train.model.geom_depth == 6
    assert infer.model.gn_groups == train.model.gn_groups == 8
    assert infer.model.gate_init == train.model.gate_init == -3.0

    assert infer.model.strict_load is True

    # Historical inference behavior intentionally preserved.
    assert infer.inference.overwrite is True

    assert list(infer.dataset.surface_name) == [
        "lh_pial",
        "lh_white",
        "rh_pial",
        "rh_white",
    ]

    assert infer.dataset.prob_clip_min == 0.0
    assert infer.dataset.prob_clip_max == 1.0
    assert infer.dataset.prob_gamma == 1.0
    assert infer.dataset.add_prob_grad is False


def test_deformation_evaluation_contract():
    """Lock deformation evaluation sampling and metric settings."""
    cfg = load(DEFORM_EVAL)

    assert cfg.dataset.split_name == "test"

    assert cfg.eval.seed == 1234
    assert cfg.eval.pred_desc == "deform"
    assert cfg.eval.n_chamfer == 150000
    assert cfg.eval.n_assd_hd == 150000
    assert cfg.eval.log_level == "INFO"


def test_canonical_derivative_roots_are_consistent():
    """All stages must agree on canonical version-independent derivative names."""
    seg_train = load(SEG_TRAIN)
    seg_infer = load(SEG_INFER)
    seg_eval = load(SEG_EVAL)
    init = load(INITSURF)
    deform_train = load(DEFORM_TRAIN)
    deform_infer = load(DEFORM_INFER)
    deform_eval = load(DEFORM_EVAL)

    datasets = ["HCP_YA", "OASIS1"]

    for dataset in datasets:
        # Preprocessing inputs.
        expected_preproc_suffix = "/derivatives/sc-preproc"

        assert seg_train.dataset.roots[dataset].endswith(
            expected_preproc_suffix
        )
        assert seg_infer.dataset.roots[dataset].endswith(
            expected_preproc_suffix
        )
        assert seg_eval.dataset.roots[dataset].endswith(
            expected_preproc_suffix
        )
        assert init.dataset.roots[dataset].endswith(
            expected_preproc_suffix
        )
        assert deform_train.dataset.roots[dataset].endswith(
            expected_preproc_suffix
        )
        assert deform_infer.dataset.roots[dataset].endswith(
            expected_preproc_suffix
        )
        assert deform_eval.dataset.roots[dataset].endswith(
            expected_preproc_suffix
        )

        # Segmentation outputs / InitSurf inputs.
        assert seg_infer.outputs.out_roots[dataset].endswith(
            "/derivatives/sc-seg"
        )
        assert seg_eval.outputs.pred_roots[dataset].endswith(
            "/derivatives/sc-seg"
        )
        assert init.dataset.seg_roots[dataset].endswith(
            "/derivatives/sc-seg"
        )

        # InitSurf outputs / deformation inputs.
        assert init.outputs.out_roots[dataset].endswith(
            "/derivatives/sc-initsurf"
        )
        assert deform_train.dataset.initsurf_roots[dataset].endswith(
            "/derivatives/sc-initsurf"
        )
        assert deform_infer.dataset.initsurf_roots[dataset].endswith(
            "/derivatives/sc-initsurf"
        )

        # Final deformation outputs.
        assert deform_infer.outputs.out_roots[dataset].endswith(
            "/derivatives/sc-deform"
        )
        assert deform_eval.outputs.pred_roots[dataset].endswith(
            "/derivatives/sc-deform"
        )
