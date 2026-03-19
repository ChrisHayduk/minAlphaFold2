from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ProcessedOpenProteinSetDataset, collate_batch
from losses import AlphaFoldLoss
from model import AlphaFold2


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DataConfig:
    processed_features_dir: str | Path = "data/processed_features"
    processed_labels_dir: str | Path = "data/processed_labels"
    val_fraction: float = 0.1
    crop_size: int = 128
    msa_depth: int = 64
    extra_msa_depth: int = 128
    max_templates: int = 1


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    device: str = default_device()
    seed: int = 0
    num_workers: int = 0
    n_cycles: int = 1
    n_ensemble: int = 1
    finetune: bool = False
    latest_checkpoint_path: str | Path | None = None
    best_checkpoint_path: str | Path | None = None


def default_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        c_m=32,
        c_s=32,
        c_z=16,
        c_t=16,
        c_e=24,
        dim=8,
        num_heads=4,
        msa_transition_n=2,
        outer_product_dim=8,
        triangle_mult_c=16,
        triangle_dim=8,
        triangle_num_heads=2,
        pair_transition_n=2,
        template_pair_num_blocks=1,
        template_pair_dropout=0.0,
        template_pointwise_attention_dim=8,
        template_pointwise_num_heads=2,
        extra_msa_dim=8,
        extra_msa_dropout=0.0,
        extra_pair_dropout=0.0,
        msa_column_global_attention_dim=8,
        num_evoformer=1,
        evoformer_msa_dropout=0.0,
        evoformer_pair_dropout=0.0,
        structure_module_c=16,
        structure_module_layers=2,
        structure_module_dropout_ipa=0.0,
        structure_module_dropout_transition=0.0,
        ipa_num_heads=4,
        ipa_c=8,
        ipa_n_query_points=4,
        ipa_n_value_points=4,
        n_dist_bins=64,
        plddt_hidden_dim=32,
        n_plddt_bins=50,
        n_msa_classes=23,
        n_pae_bins=64,
        num_extra_msa=1,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return device


def build_dataloader(
    split: str,
    data_config: DataConfig,
    *,
    training: bool,
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = "cpu",
    seed: int = 0,
) -> DataLoader:
    dataset = ProcessedOpenProteinSetDataset(
        data_config.processed_features_dir,
        data_config.processed_labels_dir,
        split=split,
        val_fraction=data_config.val_fraction,
        seed=seed,
    )
    collate_fn = partial(
        collate_batch,
        crop_size=data_config.crop_size,
        msa_depth=data_config.msa_depth,
        extra_msa_depth=data_config.extra_msa_depth,
        max_templates=data_config.max_templates,
        training=training,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_fn,
        generator=generator,
    )


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def model_inputs_from_batch(batch: dict[str, Any], training_config: TrainingConfig) -> dict[str, torch.Tensor | int]:
    return {
        "target_feat": batch["target_feat"],
        "residue_index": batch["residue_index"],
        "msa_feat": batch["msa_feat"],
        "extra_msa_feat": batch["extra_msa_feat"],
        "template_pair_feat": batch["template_pair_feat"],
        "aatype": batch["aatype"],
        "template_angle_feat": batch["template_angle_feat"],
        "template_mask": batch["template_mask"],
        "seq_mask": batch["seq_mask"],
        "msa_mask": batch["msa_mask"],
        "extra_msa_mask": batch["extra_msa_mask"],
        "n_cycles": training_config.n_cycles,
        "n_ensemble": training_config.n_ensemble,
    }


def loss_inputs_from_batch(batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "structure_model_prediction": outputs,
        "true_rotations": batch["true_rotations"],
        "true_translations": batch["true_translations"],
        "true_atom_positions": batch["true_atom_positions"],
        "true_atom_mask": batch["true_atom_mask"],
        "true_torsion_angles": batch["true_torsion_angles"],
        "true_torsion_mask": batch["true_torsion_mask"],
        "experimentally_resolved_pred": outputs["experimentally_resolved_logits"],
        "experimentally_resolved_true": batch["experimentally_resolved_true"],
        "masked_msa_pred": outputs["masked_msa_logits"],
        "masked_msa_target": batch["masked_msa_target"],
        "masked_msa_mask": batch["masked_msa_mask"],
        "plddt_pred": outputs["plddt_logits"],
        "distogram_pred": outputs["distogram_logits"],
        "res_types": batch["res_types"],
        "seq_mask": batch["seq_mask"],
    }


def train_step(
    model: AlphaFold2,
    loss_fn: AlphaFoldLoss,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    training_config: TrainingConfig,
) -> dict[str, float]:
    device = resolve_device(training_config.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch = move_to_device(batch, device)
    outputs = model(**model_inputs_from_batch(batch, training_config))
    per_example_loss = loss_fn(**loss_inputs_from_batch(batch, outputs))
    loss = per_example_loss.mean()
    loss.backward()

    if training_config.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)

    optimizer.step()
    return {"loss": float(loss.item())}


def evaluate(
    model: AlphaFold2,
    loss_fn: AlphaFoldLoss,
    dataloader: DataLoader,
    training_config: TrainingConfig,
) -> dict[str, float]:
    device = resolve_device(training_config.device)
    model.eval()

    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            outputs = model(**model_inputs_from_batch(batch, training_config))
            per_example_loss = loss_fn(**loss_inputs_from_batch(batch, outputs))
            total_loss += float(per_example_loss.sum().item())
            total_examples += int(per_example_loss.shape[0])

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    return {"loss": total_loss / total_examples}


def config_to_dict(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return config


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: AlphaFold2,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float | None,
    history: list[dict[str, float | int]],
    data_config: DataConfig,
    training_config: TrainingConfig,
    model_config: Any,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "data_config": config_to_dict(data_config),
            "training_config": config_to_dict(training_config),
            "model_config": config_to_dict(model_config),
        },
        checkpoint_path,
    )


def fit(
    model_config: Any | None = None,
    data_config: DataConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> tuple[AlphaFold2, list[dict[str, float | int]]]:
    model_config = default_model_config() if model_config is None else model_config
    data_config = DataConfig() if data_config is None else data_config
    training_config = TrainingConfig() if training_config is None else training_config

    set_seed(training_config.seed)
    device = resolve_device(training_config.device)

    train_loader = build_dataloader(
        "train",
        data_config,
        training=True,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        device=str(device),
        seed=training_config.seed,
    )
    if len(train_loader.dataset) == 0:
        raise ValueError("Training split is empty. Check the processed cache paths and val_fraction.")

    val_loader = build_dataloader(
        "val",
        data_config,
        training=False,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        device=str(device),
        seed=training_config.seed,
    )
    has_validation = data_config.val_fraction > 0.0 and len(val_loader.dataset) > 0

    model = AlphaFold2(model_config).to(device)
    loss_fn = AlphaFoldLoss(finetune=training_config.finetune).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    best_val_loss: float | None = None

    for epoch in range(1, training_config.epochs + 1):
        total_train_loss = 0.0
        total_train_examples = 0

        for batch in train_loader:
            metrics = train_step(model, loss_fn, optimizer, batch, training_config)
            batch_size = int(batch["aatype"].shape[0])
            total_train_loss += metrics["loss"] * batch_size
            total_train_examples += batch_size

        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_train_loss / max(total_train_examples, 1),
        }

        if has_validation:
            val_metrics = evaluate(model, loss_fn, val_loader, training_config)
            epoch_metrics["val_loss"] = val_metrics["loss"]

        history.append(epoch_metrics)

        if "val_loss" in epoch_metrics:
            print(
                f"epoch {epoch}/{training_config.epochs} "
                f"train_loss={epoch_metrics['train_loss']:.4f} "
                f"val_loss={epoch_metrics['val_loss']:.4f}"
            )
        else:
            print(f"epoch {epoch}/{training_config.epochs} train_loss={epoch_metrics['train_loss']:.4f}")

        if training_config.latest_checkpoint_path is not None:
            save_checkpoint(
                training_config.latest_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                history=history,
                data_config=data_config,
                training_config=training_config,
                model_config=model_config,
            )

        if training_config.best_checkpoint_path is not None and "val_loss" in epoch_metrics:
            current_val_loss = float(epoch_metrics["val_loss"])
            if best_val_loss is None or current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_checkpoint(
                    training_config.best_checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_val_loss=best_val_loss,
                    history=history,
                    data_config=data_config,
                    training_config=training_config,
                    model_config=model_config,
                )

    return model, history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pedagogical AlphaFold2 model on processed OpenProteinSet caches.")
    parser.add_argument("--processed-features-dir", type=str, default="data/processed_features")
    parser.add_argument("--processed-labels-dir", type=str, default="data/processed_labels")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--msa-depth", type=int, default=64)
    parser.add_argument("--extra-msa-depth", type=int, default=128)
    parser.add_argument("--max-templates", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-cycles", type=int, default=1)
    parser.add_argument("--n-ensemble", type=int, default=1)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--latest-checkpoint-path", type=str, default=None)
    parser.add_argument("--best-checkpoint-path", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[AlphaFold2, list[dict[str, float | int]]]:
    args = parse_args(argv)

    data_config = DataConfig(
        processed_features_dir=args.processed_features_dir,
        processed_labels_dir=args.processed_labels_dir,
        val_fraction=args.val_fraction,
        crop_size=args.crop_size,
        msa_depth=args.msa_depth,
        extra_msa_depth=args.extra_msa_depth,
        max_templates=args.max_templates,
    )
    grad_clip_norm = None if args.grad_clip_norm is not None and args.grad_clip_norm <= 0 else args.grad_clip_norm
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=grad_clip_norm,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        n_cycles=args.n_cycles,
        n_ensemble=args.n_ensemble,
        finetune=args.finetune,
        latest_checkpoint_path=args.latest_checkpoint_path,
        best_checkpoint_path=args.best_checkpoint_path,
    )

    return fit(
        model_config=default_model_config(),
        data_config=data_config,
        training_config=training_config,
    )


if __name__ == "__main__":
    main()
