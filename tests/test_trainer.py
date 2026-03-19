import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minalphafold"))


from losses import AlphaFoldLoss
from model import AlphaFold2
from tests.test_data_pipeline import write_processed_cache
from trainer import (
    DataConfig,
    TrainingConfig,
    build_dataloader,
    default_model_config,
    evaluate,
    fit,
    main,
    train_step,
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

    model = AlphaFold2(default_model_config())
    loss_fn = AlphaFoldLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_step(model, loss_fn, optimizer, batch, training_config)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert any(
        not torch.allclose(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    )


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

    model = AlphaFold2(default_model_config())
    loss_fn = AlphaFoldLoss()
    metrics = evaluate(model, loss_fn, dataloader, training_config)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fit_runs_and_writes_checkpoints(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    latest_path = tmp_path / "latest.pt"
    best_path = tmp_path / "best.pt"

    model, history = fit(
        model_config=default_model_config(),
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
