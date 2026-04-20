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
    build_optimizer,
    DataConfig,
    TrainingConfig,
    build_dataloader,
    evaluate,
    fit,
    learning_rate_at_step,
    learning_rate_for_step,
    list_available_profiles,
    load_model_config,
    main,
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
