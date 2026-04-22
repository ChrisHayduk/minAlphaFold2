import os
from pathlib import Path

import pytest
import torch

from minalphafold.losses import AlphaFoldLoss
from minalphafold.model import AlphaFold2
from tests.test_data_pipeline import write_processed_cache


def _linear(module: object) -> torch.nn.Linear:
    """Narrow ``nn.Module.__getattr__`` → ``Tensor | Module`` down to ``nn.Linear``.

    PyTorch's ``Module.__getattr__`` is typed as ``Tensor | Module`` so
    chained attribute access like ``model.evoformer_blocks[0].msa_row_att
    .linear_output`` comes out as a bare ``Module`` with no ``.weight`` /
    ``.bias`` visible to the type checker. This helper asserts at runtime
    that the target is an ``nn.Linear`` and returns the narrowed type —
    stricter than a bare ``cast`` and no less compact.
    """
    assert isinstance(module, torch.nn.Linear), f"expected nn.Linear, got {type(module).__name__}"
    return module


from minalphafold.trainer import (
    CONFIGS_DIR,
    OptimizerConfig,
    StageConfig,
    TrainingProtocol,
    build_ema_model,
    build_optimizer,
    DataConfig,
    TrainingConfig,
    build_dataloader,
    evaluate,
    fit,
    learning_rate_at_step,
    learning_rate_for_samples,
    learning_rate_for_step,
    list_available_profiles,
    list_available_training_protocols,
    load_checkpoint_for_resume,
    load_model_config,
    load_training_protocol,
    main,
    save_checkpoint,
    train_step,
    use_finetune_loss,
    zero_dropout_model_config,
)


def make_processed_cache_dirs(tmp_path: Path) -> tuple[Path, Path]:
    feature_dir = tmp_path / "processed_features"
    label_dir = tmp_path / "processed_labels"
    feature_dir.mkdir()
    label_dir.mkdir()
    write_processed_cache(feature_dir, label_dir, "1abc_A", "AGAGA", include_templates=True)
    write_processed_cache(feature_dir, label_dir, "2xyz_A", "AGGAA", include_templates=False)
    return feature_dir, label_dir


def test_train_step_updates_model_parameters(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    data_config = DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
    )
    training_config = TrainingConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        seed=0,
        n_cycles=1,
        n_ensemble=1,
    )

    dataloader = build_dataloader(
        "all",
        data_config,
        training=True,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        device=training_config.device,
        seed=training_config.seed,
    )
    batch = next(iter(dataloader))

    model = AlphaFold2(load_model_config("tiny"))
    loss_fn = AlphaFoldLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_step(model, loss_fn, optimizer, batch, training_config)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert any(
        not torch.allclose(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    )


def test_alphafold2_uses_canonical_constructor_initialization():
    # ``get_submodule`` returns ``nn.Module`` directly — bypasses the
    # ``Tensor | Module`` ambiguity of chained ``__getattr__`` on a
    # ``ModuleList`` — so the helper ``_linear`` narrows the leaf cleanly.
    model = AlphaFold2(load_model_config("tiny"))

    row_output = _linear(model.get_submodule("evoformer_blocks.0.msa_row_att.linear_output"))
    assert torch.allclose(row_output.weight, torch.zeros_like(row_output.weight))

    row_gate = _linear(model.get_submodule("evoformer_blocks.0.msa_row_att.linear_gate"))
    assert torch.allclose(row_gate.bias, torch.ones_like(row_gate.bias))

    tmo_out = _linear(model.get_submodule("evoformer_blocks.0.triangle_mult_out.out_linear"))
    assert torch.allclose(tmo_out.weight, torch.zeros_like(tmo_out.weight))

    tmo_gate = _linear(model.get_submodule("evoformer_blocks.0.triangle_mult_out.gate"))
    assert torch.allclose(tmo_gate.bias, torch.ones_like(tmo_gate.bias))

    input_msa = _linear(model.get_submodule("input_embedder.linear_msa"))
    assert torch.allclose(input_msa.bias, torch.zeros_like(input_msa.bias))
    assert not torch.allclose(input_msa.weight, torch.zeros_like(input_msa.weight))

    tm_head = _linear(model.get_submodule("tm_score_head.linear"))
    assert torch.allclose(tm_head.weight, torch.zeros_like(tm_head.weight))


def test_evaluate_returns_finite_mean_loss_without_gradients(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    data_config = DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
    )
    training_config = TrainingConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        seed=0,
        n_cycles=1,
        n_ensemble=1,
    )
    dataloader = build_dataloader(
        "all",
        data_config,
        training=False,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        device=training_config.device,
        seed=training_config.seed,
    )

    model = AlphaFold2(load_model_config("tiny"))
    loss_fn = AlphaFoldLoss()
    metrics = evaluate(model, loss_fn, dataloader, training_config)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fit_runs_and_writes_checkpoints(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    latest_path = tmp_path / "latest.pt"
    best_path = tmp_path / "best.pt"

    model, history = fit(
        model_config=load_model_config("tiny"),
        data_config=DataConfig(
            processed_features_dir=feature_dir,
            processed_labels_dir=label_dir,
            val_fraction=0.5,
            crop_size=8,
            msa_depth=3,
            extra_msa_depth=2,
            max_templates=1,
        ),
        training_config=TrainingConfig(
            epochs=2,
            batch_size=1,
            device="cpu",
            seed=0,
            n_cycles=1,
            n_ensemble=1,
            latest_checkpoint_path=latest_path,
            best_checkpoint_path=best_path,
        ),
    )

    assert isinstance(model, AlphaFold2)
    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
    assert latest_path.exists()
    assert best_path.exists()


def test_main_runs_one_epoch_from_cli_args(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    latest_path = tmp_path / "cli_latest.pt"

    model, history = main(
        [
            "--processed-features-dir",
            str(feature_dir),
            "--processed-labels-dir",
            str(label_dir),
            "--val-fraction",
            "0.5",
            "--crop-size",
            "8",
            "--msa-depth",
            "3",
            "--extra-msa-depth",
            "2",
            "--max-templates",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--n-cycles",
            "1",
            "--n-ensemble",
            "1",
            "--latest-checkpoint-path",
            str(latest_path),
        ]
    )

    assert isinstance(model, AlphaFold2)
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert latest_path.exists()


def test_load_model_config_selects_requested_profile():
    assert load_model_config("tiny").model_profile == "tiny"
    assert load_model_config("medium").model_profile == "medium"
    assert load_model_config("alphafold2").model_profile == "alphafold2"


def test_load_model_config_accepts_an_explicit_toml_path():
    path = CONFIGS_DIR / "tiny.toml"
    assert load_model_config(path).model_profile == "tiny"
    assert load_model_config(str(path)).model_profile == "tiny"


def test_load_model_config_raises_for_missing_profile():
    with pytest.raises(FileNotFoundError):
        load_model_config("does_not_exist")


def test_list_available_profiles_includes_shipped_json_configs():
    profiles = list_available_profiles()
    assert {"tiny", "medium", "alphafold2"}.issubset(set(profiles))


def test_shipped_profiles_have_expected_scales():
    tiny = load_model_config("tiny")
    medium = load_model_config("medium")
    alphafold2 = load_model_config("alphafold2")

    assert medium.model_profile == "medium"
    assert alphafold2.model_profile == "alphafold2"
    assert medium.c_m > tiny.c_m
    assert medium.c_z > tiny.c_z
    assert medium.num_evoformer > tiny.num_evoformer
    # alphafold2 profile locked to the paper (supplement 1.5 / 1.6 / Algorithm 22).
    assert alphafold2.c_m == 256
    assert alphafold2.c_s == 384
    assert alphafold2.c_z == 128
    assert alphafold2.num_evoformer == 48
    assert alphafold2.structure_module_layers == 8
    assert alphafold2.ipa_num_heads == 12
    assert alphafold2.ipa_c == 16
    # Supplement 1.7.1 / Algorithm 16: TemplatePair overrides.
    assert alphafold2.template_triangle_mult_c == 64
    assert alphafold2.template_triangle_attn_c == 64
    assert alphafold2.template_pair_transition_n == 2


def test_zero_dropout_model_config_preserves_dimensions_and_clears_dropout():
    config = load_model_config("alphafold2")
    overfit_config = zero_dropout_model_config(config)

    assert overfit_config.model_profile == "alphafold2_no_dropout"
    assert overfit_config.c_m == config.c_m
    assert overfit_config.c_s == config.c_s
    assert overfit_config.c_z == config.c_z
    assert overfit_config.num_evoformer == config.num_evoformer
    assert overfit_config.template_pair_dropout == 0.0
    assert overfit_config.extra_msa_dropout == 0.0
    assert overfit_config.extra_pair_dropout == 0.0
    assert overfit_config.evoformer_msa_dropout == 0.0
    assert overfit_config.evoformer_pair_dropout == 0.0
    assert overfit_config.structure_module_dropout_ipa == 0.0
    assert overfit_config.structure_module_dropout_transition == 0.0


def test_learning_rate_for_step_supports_warmup_cosine():
    training_config = TrainingConfig(
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        lr_schedule="warmup_cosine",
        warmup_steps=10,
    )

    warmup_lr = learning_rate_for_step(training_config, step=4, total_steps=100)
    after_warmup_lr = learning_rate_for_step(training_config, step=10, total_steps=100)
    late_lr = learning_rate_for_step(training_config, step=99, total_steps=100)

    assert abs(warmup_lr - 5e-4) < 1e-8
    assert after_warmup_lr < training_config.learning_rate
    assert late_lr >= training_config.min_learning_rate
    assert late_lr < after_warmup_lr


def test_build_optimizer_uses_configured_adam_hyperparameters():
    model = AlphaFold2(load_model_config("tiny"))
    training_config = TrainingConfig(
        learning_rate=2e-4,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_eps=1e-6,
        weight_decay=1e-3,
    )

    optimizer = build_optimizer(model, training_config)
    group = optimizer.param_groups[0]

    assert abs(group["lr"] - 2e-4) < 1e-12
    assert group["betas"] == (0.8, 0.95)
    assert abs(group["eps"] - 1e-6) < 1e-12
    assert abs(group["weight_decay"] - 1e-3) < 1e-12


def test_learning_rate_at_step_applies_finetune_scale_without_warmup():
    """Supplement 1.11.3: fine-tuning has no warmup, half base LR."""
    training_config = TrainingConfig(
        learning_rate=1e-3,
        lr_schedule="warmup_cosine",
        warmup_steps=10,
        finetune_lr_scale=0.5,
    )

    # Step 0 during pre-training is inside the linear warmup → tiny LR.
    pretrain_lr = learning_rate_at_step(training_config, step=0, total_steps=100, is_finetune=False)
    assert pretrain_lr < training_config.learning_rate

    # Same step during fine-tuning ignores warmup and returns lr * 0.5.
    finetune_lr = learning_rate_at_step(training_config, step=0, total_steps=100, is_finetune=True)
    assert abs(finetune_lr - 5e-4) < 1e-12


def test_training_config_defaults_match_supplement_1_11_3():
    """Supplement 1.11.3 fixes base lr=1e-3, ε=1e-6, clip=0.1, halving at fine-tune."""
    cfg = TrainingConfig()
    assert cfg.learning_rate == 1e-3
    assert cfg.adam_beta1 == 0.9
    assert cfg.adam_beta2 == 0.999
    assert cfg.adam_eps == 1e-6
    assert cfg.grad_clip_norm == 0.1
    assert cfg.finetune_lr_scale == 0.5


def test_use_finetune_loss_supports_two_phase_schedule():
    always_pretrain = TrainingConfig(finetune=False, finetune_start_step=None)
    always_finetune = TrainingConfig(finetune=True, finetune_start_step=None)
    scheduled = TrainingConfig(finetune=False, finetune_start_step=5)

    assert use_finetune_loss(always_pretrain, global_step=0) is False
    assert use_finetune_loss(always_finetune, global_step=0) is True
    assert use_finetune_loss(scheduled, global_step=4) is False
    assert use_finetune_loss(scheduled, global_step=5) is True


def test_build_dataloader_can_fix_training_features(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    data_config = DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
        block_delete_training_msa=False,
        fixed_feature_seed=11,
    )
    loader = build_dataloader(
        "all",
        data_config,
        training=True,
        batch_size=1,
        num_workers=0,
        device="cpu",
        seed=0,
    )

    first = next(iter(loader))
    second = next(iter(loader))

    assert torch.equal(first["msa_feat"], second["msa_feat"])
    assert torch.equal(first["masked_msa_mask"], second["masked_msa_mask"])


def test_build_dataloader_can_emit_recycling_feature_samples(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    data_config = DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
        fixed_feature_seed=13,
    )
    loader = build_dataloader(
        "all",
        data_config,
        training=True,
        batch_size=1,
        num_workers=0,
        device="cpu",
        seed=0,
        n_cycles=2,
        n_ensemble=1,
    )

    batch = next(iter(loader))

    assert batch["msa_feat"].shape[:3] == (2, 1, 1)
    assert batch["extra_msa_feat"].shape[:3] == (2, 1, 1)


# ---------------------------------------------------------------------
# Two-stage training protocol — supplement Table 4 + §1.11.
# ---------------------------------------------------------------------


def test_load_training_protocol_alphafold2_matches_table4():
    """The shipped TOML must reproduce Supplementary Table 4 verbatim."""
    protocol = load_training_protocol("alphafold2")

    assert protocol.protocol == "alphafold2"

    # Shared optimizer (§1.11.3 / §1.11.7).
    assert protocol.optimizer.adam_beta1 == 0.9
    assert protocol.optimizer.adam_beta2 == 0.999
    assert protocol.optimizer.adam_eps == 1e-6
    assert protocol.optimizer.grad_clip_norm == 0.1
    assert protocol.optimizer.ema_decay == 0.999
    assert protocol.optimizer.mini_batch_size == 128
    assert protocol.optimizer.lr_decay_samples == 6_400_000
    assert protocol.optimizer.lr_decay_factor == 0.95

    # Initial stage (Table 4, "Initial training" column).
    assert protocol.initial.crop_size == 256
    assert protocol.initial.msa_depth == 128
    assert protocol.initial.extra_msa_depth == 1024
    assert protocol.initial.max_templates == 4
    assert protocol.initial.learning_rate == 1e-3
    assert protocol.initial.warmup_samples == 128_000
    assert protocol.initial.violation_loss_weight == 0.0
    assert protocol.initial.total_samples == 10_000_000

    # Fine-tuning stage (Table 4, "Fine-tuning" column).
    assert protocol.finetune.crop_size == 384
    assert protocol.finetune.msa_depth == 512
    assert protocol.finetune.extra_msa_depth == 5120
    assert protocol.finetune.max_templates == 4
    assert protocol.finetune.learning_rate == 5e-4
    assert protocol.finetune.warmup_samples == 0
    assert protocol.finetune.violation_loss_weight == 1.0
    assert protocol.finetune.total_samples == 1_500_000


def test_load_training_protocol_unknown_name_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_training_protocol("definitely_not_a_real_protocol")


def test_load_training_protocol_rejects_unknown_stage_key(tmp_path):
    """Typos in a TOML field must surface at load time, not during training."""
    path = tmp_path / "training_bad.toml"
    path.write_text(
        'protocol = "bad"\n'
        "[optimizer]\n"
        "adam_beta1 = 0.9\nadam_beta2 = 0.999\nadam_eps = 1e-6\n"
        "grad_clip_norm = 0.1\nema_decay = 0.999\nmini_batch_size = 128\n"
        "lr_decay_samples = 6400000\nlr_decay_factor = 0.95\n"
        "[initial]\n"
        "crop_size = 256\nmsa_depth = 128\nextra_msa_depth = 1024\n"
        "max_templates = 4\nlearning_rate = 1e-3\nwarmup_samples = 0\n"
        "violation_loss_weight = 0.0\ntotal_samples = 1000\n"
        'not_a_real_key = "oops"\n'
        "[finetune]\n"
        "crop_size = 384\nmsa_depth = 512\nextra_msa_depth = 5120\n"
        "max_templates = 4\nlearning_rate = 5e-4\nwarmup_samples = 0\n"
        "violation_loss_weight = 1.0\ntotal_samples = 1000\n"
    )
    with pytest.raises(TypeError):
        load_training_protocol(path)


def test_training_protocol_stage_returns_matching_stage():
    protocol = load_training_protocol("alphafold2")
    assert protocol.stage("initial") is protocol.initial
    assert protocol.stage("finetune") is protocol.finetune


def test_training_protocol_stage_rejects_unknown_name():
    protocol = load_training_protocol("alphafold2")
    with pytest.raises(ValueError):
        protocol.stage("pretrain")


def test_list_available_training_protocols_excludes_model_profiles():
    protocols = list_available_training_protocols()
    assert "alphafold2" in protocols
    # Model profiles (tiny/medium/alphafold2 without the ``training_``
    # prefix) must not leak into the training-protocol list.
    assert all(
        not protocol.startswith("training_") for protocol in protocols
    )


def test_list_available_profiles_excludes_training_protocols():
    profiles = list_available_profiles()
    assert all(not p.startswith("training_") for p in profiles)
    # Sanity: the canonical model profiles should still show up.
    assert "alphafold2" in profiles


# ---------------------------------------------------------------------
# Paper samples-based LR schedule — supplement §1.11.3.
# ---------------------------------------------------------------------


def test_learning_rate_for_samples_linear_warmup_range():
    """Linear warmup: LR(0) = 0, LR(warmup/2) = base/2, LR(warmup) = base."""
    base = 1e-3
    warmup = 1000
    assert learning_rate_for_samples(base, 0, warmup, None, 1.0) == 0.0
    assert learning_rate_for_samples(base, warmup // 2, warmup, None, 1.0) == base / 2
    # At exactly warmup_samples we exit the warmup branch — the constant
    # phase returns the full base LR.
    assert learning_rate_for_samples(base, warmup, warmup, None, 1.0) == base


def test_learning_rate_for_samples_constant_phase():
    """Between warmup end and decay trigger, LR stays at base."""
    base = 1e-3
    assert learning_rate_for_samples(base, 2_000_000, 128_000, 6_400_000, 0.95) == base
    # Without any decay configured, same story indefinitely.
    assert learning_rate_for_samples(base, 10_000_000, 128_000, None, 1.0) == base


def test_learning_rate_for_samples_one_shot_decay_at_threshold():
    """At and past lr_decay_samples the LR drops by lr_decay_factor once."""
    base = 1e-3
    pre = learning_rate_for_samples(base, 6_400_000 - 1, 128_000, 6_400_000, 0.95)
    post = learning_rate_for_samples(base, 6_400_000, 128_000, 6_400_000, 0.95)
    later = learning_rate_for_samples(base, 10_000_000, 128_000, 6_400_000, 0.95)
    assert pre == base
    assert post == pytest.approx(base * 0.95)
    assert later == pytest.approx(base * 0.95)


def test_learning_rate_at_step_uses_samples_when_configured():
    """Setting warmup_samples > 0 switches to the samples-based path."""
    cfg = TrainingConfig(learning_rate=1e-3, warmup_samples=1000)
    assert learning_rate_at_step(cfg, step=0, total_steps=100, is_finetune=False, samples_seen=0) == 0.0
    assert learning_rate_at_step(cfg, step=0, total_steps=100, is_finetune=False, samples_seen=500) == 5e-4


def test_learning_rate_at_step_falls_back_to_step_schedule_by_default():
    """With no samples knobs set we keep the old step-based ``lr_schedule``."""
    cfg = TrainingConfig(learning_rate=1e-3)  # default: constant, no warmup
    lr = learning_rate_at_step(
        cfg, step=5, total_steps=10, is_finetune=False, samples_seen=999_999,
    )
    assert lr == 1e-3


def test_learning_rate_at_step_finetune_applies_decay_factor_past_threshold():
    """Fine-tuning LR also respects the one-shot ×0.95 drop."""
    cfg = TrainingConfig(
        learning_rate=1e-3,
        finetune_lr_scale=0.5,
        lr_decay_samples=6_400_000,
        lr_decay_factor=0.95,
    )
    pre = learning_rate_at_step(
        cfg, step=0, total_steps=1, is_finetune=True, samples_seen=1_000_000,
    )
    post = learning_rate_at_step(
        cfg, step=0, total_steps=1, is_finetune=True, samples_seen=7_000_000,
    )
    assert pre == pytest.approx(1e-3 * 0.5)
    assert post == pytest.approx(1e-3 * 0.5 * 0.95)


# ---------------------------------------------------------------------
# EMA — supplement §1.11.7.
# ---------------------------------------------------------------------


def test_build_ema_model_first_update_copies_current_params():
    """At num_averaged=0 the EMA should mirror the current model exactly."""
    linear = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.ones(2, 3))
    ema = build_ema_model(linear, ema_decay=0.9)
    # Training "step": change the live params before updating the EMA.
    with torch.no_grad():
        linear.weight.copy_(torch.zeros(2, 3))
    ema.update_parameters(linear)
    ema_weight = _linear(ema.module).weight
    assert torch.allclose(ema_weight, torch.zeros(2, 3))


def test_build_ema_model_subsequent_updates_apply_decay():
    """After the first sample the EMA blends decay·avg + (1−decay)·current."""
    linear = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.ones(2, 3))
    ema = build_ema_model(linear, ema_decay=0.9)
    ema.update_parameters(linear)  # first call → EMA = ones
    with torch.no_grad():
        linear.weight.copy_(torch.zeros(2, 3))
    ema.update_parameters(linear)  # second call → 0.9*ones + 0.1*zeros = 0.9
    ema_weight = _linear(ema.module).weight
    assert torch.allclose(ema_weight, torch.full((2, 3), 0.9))


# ---------------------------------------------------------------------
# Gradient accumulation + EMA + resume inside fit().
# ---------------------------------------------------------------------


def test_fit_with_grad_accumulation_performs_fewer_optimizer_steps(tmp_path):
    """With grad_accum_steps=2 and 2 training examples, fit should do 1 step.

    Two micro-batches accumulate into one optimizer.step, so global_step
    ends at 1. Without accumulation the same run would do 2 steps.
    """
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    _, history = fit(
        model_config=zero_dropout_model_config(load_model_config("tiny")),
        data_config=DataConfig(
            processed_features_dir=feature_dir,
            processed_labels_dir=label_dir,
            val_fraction=0.0,
            crop_size=5,
            msa_depth=2,
            extra_msa_depth=2,
            max_templates=1,
        ),
        training_config=TrainingConfig(
            epochs=1,
            batch_size=1,
            grad_accum_steps=2,
            device="cpu",
            seed=0,
            n_cycles=1,
            n_ensemble=1,
        ),
    )
    assert len(history) == 1
    # Two micro-batches → one optimizer.step.
    assert history[0]["global_step"] == 1
    assert history[0]["global_samples"] == 2


def test_fit_with_ema_writes_ema_state_in_checkpoint(tmp_path):
    """Enabling ema_decay should persist an ``ema_state_dict`` on disk."""
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    latest_path = tmp_path / "ema.pt"
    fit(
        model_config=zero_dropout_model_config(load_model_config("tiny")),
        data_config=DataConfig(
            processed_features_dir=feature_dir,
            processed_labels_dir=label_dir,
            val_fraction=0.0,
            crop_size=5,
            msa_depth=2,
            extra_msa_depth=2,
            max_templates=1,
        ),
        training_config=TrainingConfig(
            epochs=1,
            batch_size=1,
            ema_decay=0.9,
            device="cpu",
            seed=0,
            n_cycles=1,
            n_ensemble=1,
            latest_checkpoint_path=latest_path,
        ),
    )
    payload = torch.load(latest_path, weights_only=False, map_location="cpu")
    assert "ema_state_dict" in payload
    assert payload["global_step"] >= 1
    assert payload["global_samples"] >= 1


def test_fit_resumes_global_step_and_samples(tmp_path):
    """A resumed run should continue the counters from where it left off."""
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    checkpoint_path = tmp_path / "ck.pt"

    data_config = DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
        crop_size=5,
        msa_depth=2,
        extra_msa_depth=2,
        max_templates=1,
    )
    common_training = dict(
        batch_size=1,
        grad_accum_steps=1,
        device="cpu",
        seed=0,
        n_cycles=1,
        n_ensemble=1,
    )

    _, history_first = fit(
        model_config=zero_dropout_model_config(load_model_config("tiny")),
        data_config=data_config,
        training_config=TrainingConfig(
            epochs=1,
            latest_checkpoint_path=checkpoint_path,
            **common_training,
        ),
    )
    first_samples = int(history_first[0]["global_samples"])
    first_step = int(history_first[0]["global_step"])
    assert first_samples > 0

    _, history_resumed = fit(
        model_config=zero_dropout_model_config(load_model_config("tiny")),
        data_config=data_config,
        training_config=TrainingConfig(
            epochs=2,
            resume_from_checkpoint=checkpoint_path,
            **common_training,
        ),
    )
    # ``fit`` preserves the checkpoint's history list and appends new
    # epochs onto the end, so the resumed run's final entry is epoch 2.
    assert len(history_resumed) == 2
    assert history_resumed[-1]["epoch"] == 2
    assert int(history_resumed[-1]["global_samples"]) > first_samples
    assert int(history_resumed[-1]["global_step"]) > first_step


from train_af2 import (  # noqa: E402 — sys.path fix happens in conftest
    _epochs_for_target_samples,
    data_config_for_stage,
    training_config_for_stage,
)


# ---------------------------------------------------------------------
# scripts/train_af2.py — protocol → (DataConfig, TrainingConfig) plumbing.
# ---------------------------------------------------------------------


def test_epochs_for_target_samples_rounds_up():
    """Enough epochs to cover the sample budget, always rounded up."""
    # Exactly divisible → exact number of epochs.
    assert _epochs_for_target_samples(1000, 100) == 10
    # Not divisible → ceiling.
    assert _epochs_for_target_samples(1001, 100) == 11
    # Smaller target than one epoch → one epoch (never zero).
    assert _epochs_for_target_samples(50, 100) == 1
    # Empty / missing dataset → one epoch (avoid div-by-zero).
    assert _epochs_for_target_samples(1000, 0) == 1


def test_data_config_for_stage_mirrors_table4_columns():
    """Initial + fine-tune stages produce DataConfigs with Table 4 numbers."""
    protocol = load_training_protocol("alphafold2")
    initial_dc = data_config_for_stage(
        protocol.initial,
        processed_features_dir=Path("/f"),
        processed_labels_dir=Path("/l"),
        val_fraction=0.0,
    )
    assert initial_dc.crop_size == 256
    assert initial_dc.msa_depth == 128
    assert initial_dc.extra_msa_depth == 1024
    assert initial_dc.max_templates == 4

    finetune_dc = data_config_for_stage(
        protocol.finetune,
        processed_features_dir=Path("/f"),
        processed_labels_dir=Path("/l"),
        val_fraction=0.0,
    )
    assert finetune_dc.crop_size == 384
    assert finetune_dc.msa_depth == 512
    assert finetune_dc.extra_msa_depth == 5120
    assert finetune_dc.max_templates == 4


def test_training_config_for_stage_uses_stage_lr_without_halving():
    """Fine-tune stage LR comes straight from Table 4 (5e-4), not halved again.

    The trainer's ``finetune_lr_scale`` would normally halve the base LR
    during fine-tuning. ``training_config_for_stage`` sets that scale to
    1.0 because the stage LR field is already the post-halving 5e-4;
    applying 0.5 again would produce 2.5e-4, which isn't the paper value.
    """
    protocol = load_training_protocol("alphafold2")
    cfg = training_config_for_stage(
        protocol, protocol.finetune,
        device="cpu",
        seed=0,
        batch_size=1,
        grad_accum_steps=1,
        num_workers=0,
        n_cycles=1,
        n_ensemble=1,
        epochs=1,
        is_finetune=True,
        latest_checkpoint_path=Path("/tmp/latest.pt"),
        best_checkpoint_path=None,
        resume_from_checkpoint=None,
        init_weights_from_checkpoint=None,
    )
    assert cfg.learning_rate == 5e-4
    assert cfg.finetune_lr_scale == 1.0
    assert cfg.finetune is True


def test_training_config_for_stage_forwards_optimizer_and_ema():
    """Shared optimizer settings (§1.11.3, §1.11.7) flow through unchanged."""
    protocol = load_training_protocol("alphafold2")
    cfg = training_config_for_stage(
        protocol, protocol.initial,
        device="cpu",
        seed=42,
        batch_size=1,
        grad_accum_steps=128,
        num_workers=0,
        n_cycles=4,
        n_ensemble=1,
        epochs=10,
        is_finetune=False,
        latest_checkpoint_path=Path("/tmp/latest.pt"),
        best_checkpoint_path=None,
        resume_from_checkpoint=None,
        init_weights_from_checkpoint=None,
    )
    assert cfg.adam_beta1 == 0.9
    assert cfg.adam_beta2 == 0.999
    assert cfg.adam_eps == 1e-6
    assert cfg.grad_clip_norm == 0.1
    assert cfg.ema_decay == 0.999
    assert cfg.warmup_samples == 128_000
    assert cfg.lr_decay_samples == 6_400_000
    assert cfg.lr_decay_factor == 0.95
    assert cfg.grad_accum_steps == 128


def test_load_checkpoint_for_resume_restores_counters(tmp_path):
    """Direct-call sanity: the loader returns the saved step/sample state."""
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    model_config = zero_dropout_model_config(load_model_config("tiny"))
    model = AlphaFold2(model_config)
    optimizer = build_optimizer(
        model,
        TrainingConfig(batch_size=1, device="cpu"),
    )

    path = tmp_path / "direct.pt"
    save_checkpoint(
        path,
        epoch=3,
        global_step=42,
        global_samples=42 * 128,
        model=model,
        optimizer=optimizer,
        best_val_loss=None,
        history=[{"epoch": 3, "global_step": 42}],
        data_config=DataConfig(
            processed_features_dir=feature_dir,
            processed_labels_dir=label_dir,
        ),
        training_config=TrainingConfig(batch_size=1, device="cpu"),
        model_config=model_config,
    )
    fresh_model = AlphaFold2(model_config)
    fresh_optimizer = build_optimizer(
        fresh_model,
        TrainingConfig(batch_size=1, device="cpu"),
    )
    restored = load_checkpoint_for_resume(path, fresh_model, fresh_optimizer, None)
    assert restored["epoch"] == 4
    assert restored["global_step"] == 42
    assert restored["global_samples"] == 42 * 128
