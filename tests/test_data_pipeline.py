import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minalphafold"))
sys.path.insert(0, str(ROOT / "scripts"))


from a3m import MSA_ALPHABET_SIZE, sequence_to_ids
from data import (
    ProcessedOpenProteinSetDataset,
    build_processed_example,
    build_template_angle_feat,
    build_template_pair_feat,
    collate_batch,
    discover_chain_ids,
    split_chain_ids,
)
from losses import AlphaFoldLoss
from model import AlphaFold2
from preprocess_openproteinset import preprocess_chain
from residue_constants import restype_1to3


class SmallConfig:
    c_m = 32
    c_s = 32
    c_z = 16
    c_t = 16
    c_e = 24

    dim = 8
    num_heads = 4

    msa_transition_n = 2
    outer_product_dim = 8

    triangle_mult_c = 16
    triangle_dim = 8
    triangle_num_heads = 2
    pair_transition_n = 2

    template_pair_num_blocks = 1
    template_pair_dropout = 0.0
    template_pointwise_attention_dim = 8
    template_pointwise_num_heads = 2

    extra_msa_dim = 8
    extra_msa_dropout = 0.0
    extra_pair_dropout = 0.0
    msa_column_global_attention_dim = 8

    num_evoformer = 1
    evoformer_msa_dropout = 0.0
    evoformer_pair_dropout = 0.0

    structure_module_c = 16
    structure_module_layers = 2
    structure_module_dropout_ipa = 0.0
    structure_module_dropout_transition = 0.0

    ipa_num_heads = 4
    ipa_c = 8
    ipa_n_query_points = 4
    ipa_n_value_points = 4

    n_dist_bins = 64
    plddt_hidden_dim = 32
    n_plddt_bins = 50
    n_msa_classes = 23
    n_pae_bins = 64

    num_extra_msa = 1


def make_atom14_arrays(sequence: str, *, x_shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    positions = np.zeros((len(sequence), 14, 3), dtype=np.float32)
    mask = np.zeros((len(sequence), 14), dtype=np.float32)

    for index, residue in enumerate(sequence):
        offset = x_shift + 3.8 * index
        positions[index, 0] = np.asarray([offset, 1.0, 0.0], dtype=np.float32)
        positions[index, 1] = np.asarray([offset + 1.3, 0.0, 0.0], dtype=np.float32)
        positions[index, 2] = np.asarray([offset + 2.6, 0.2, 0.0], dtype=np.float32)
        positions[index, 3] = np.asarray([offset + 3.0, 0.7, 0.0], dtype=np.float32)
        mask[index, :4] = 1.0
        if residue != "G":
            positions[index, 4] = np.asarray([offset + 1.2, -0.8, 1.1], dtype=np.float32)
            mask[index, 4] = 1.0

    return positions, mask


def write_simple_mmcif(path: Path, sequence: str, *, chain_id: str) -> None:
    positions, mask = make_atom14_arrays(sequence)
    pdb_id = path.stem.upper()

    rows = []
    for residue_index, residue in enumerate(sequence, start=1):
        residue_name = restype_1to3[residue]
        atom_names = ["N", "CA", "C", "O"]
        atom_indices = [0, 1, 2, 3]
        if residue != "G":
            atom_names.append("CB")
            atom_indices.append(4)

        for atom_name, atom_index in zip(atom_names, atom_indices):
            x, y, z = positions[residue_index - 1, atom_index]
            rows.append(
                f"ATOM 1 {chain_id} {chain_id} 1 {residue_index} . "
                f"{residue_name} {residue_name} {atom_name} {atom_name} "
                f"{x:.3f} {y:.3f} {z:.3f} {mask[residue_index - 1, atom_index]:.1f}"
            )

    text = (
        f"data_{pdb_id}\n"
        "_entity_poly.entity_id 1\n"
        f"_entity_poly.pdbx_seq_one_letter_code_can {sequence}\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        + "\n".join(rows)
        + "\n#\n"
    )
    path.write_text(text)


def write_a3m(path: Path, sequences: list[str]) -> None:
    lines = []
    for index, sequence in enumerate(sequences):
        lines.append(f">seq{index}")
        lines.append(sequence)
    path.write_text("\n".join(lines) + "\n")


def make_feature_and_label_example(sequence: str, *, include_templates: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    aatype = sequence_to_ids(sequence).astype(np.int32)
    msa_rows = [
        sequence,
        sequence,
        "G" + sequence[1:],
        sequence[:-1] + "G",
        sequence,
    ]
    msa = np.stack([sequence_to_ids(row) for row in msa_rows]).astype(np.int32)
    deletions = np.zeros_like(msa, dtype=np.int32)
    atom14_positions, atom14_mask = make_atom14_arrays(sequence)

    if include_templates:
        template_aatype = aatype[None]
        template_atom14_positions = atom14_positions[None] + 0.25
        template_atom14_mask = atom14_mask[None]
    else:
        template_aatype = np.zeros((0, len(sequence)), dtype=np.int32)
        template_atom14_positions = np.zeros((0, len(sequence), 14, 3), dtype=np.float32)
        template_atom14_mask = np.zeros((0, len(sequence), 14), dtype=np.float32)

    features = {
        "aatype": aatype,
        "msa": msa,
        "deletions": deletions,
        "template_aatype": template_aatype,
        "template_atom14_positions": template_atom14_positions.astype(np.float32),
        "template_atom14_mask": template_atom14_mask.astype(np.float32),
    }
    labels = {
        "atom14_positions": atom14_positions.astype(np.float32),
        "atom14_mask": atom14_mask.astype(np.float32),
    }
    return features, labels


def write_processed_cache(
    feature_dir: Path,
    label_dir: Path,
    chain_id: str,
    sequence: str,
    *,
    include_templates: bool,
) -> None:
    features, labels = make_feature_and_label_example(sequence, include_templates=include_templates)
    np.savez_compressed(feature_dir / f"{chain_id}.npz", **features)
    np.savez_compressed(label_dir / f"{chain_id}.npz", **labels)


def test_preprocess_chain_projects_query_and_template_features(tmp_path):
    raw_root = tmp_path / "openproteinset"
    chain_dir = raw_root / "roda_pdb" / "1abc_A"
    (chain_dir / "a3m").mkdir(parents=True)
    (chain_dir / "hhr").mkdir(parents=True)
    mmcif_root = raw_root / "pdb_data" / "mmcif_files"
    mmcif_root.mkdir(parents=True)

    write_a3m(chain_dir / "a3m" / "uniref90_hits.a3m", ["AGA", "AGA", "AAA"])
    (chain_dir / "hhr" / "pdb70_hits.hhr").write_text(
        "No 1\n"
        ">2XYZ_A Example template\n"
        "Q query             1 AGA 3 (3)\n"
        "Q Consensus         1 AGA 3 (3)\n"
        "T Consensus         5 AGA 7 (7)\n"
        "T 2XYZ_A            5 AGA 7 (7)\n"
    )
    write_simple_mmcif(mmcif_root / "1abc.cif", "AGA", chain_id="A")
    write_simple_mmcif(mmcif_root / "2xyz.cif", "TTTTAGA", chain_id="A")

    features, labels = preprocess_chain(
        chain_dir,
        mmcif_root=mmcif_root,
        max_msa_seqs=8,
        max_templates=1,
        msa_name="uniref90_hits.a3m",
        template_hhr_name="pdb70_hits.hhr",
        skip_templates=False,
    )

    assert set(features) == {
        "aatype",
        "msa",
        "deletions",
        "template_aatype",
        "template_atom14_positions",
        "template_atom14_mask",
    }
    assert set(labels) == {"atom14_positions", "atom14_mask"}
    assert features["aatype"].tolist() == sequence_to_ids("AGA").tolist()
    assert labels["atom14_positions"].shape == (3, 14, 3)
    assert labels["atom14_mask"].shape == (3, 14)
    assert features["template_aatype"].shape == (1, 3)
    assert features["template_aatype"][0].tolist() == sequence_to_ids("AGA").tolist()


def test_dataset_split_and_collate_build_expected_feature_widths(tmp_path):
    feature_dir = tmp_path / "processed_features"
    label_dir = tmp_path / "processed_labels"
    feature_dir.mkdir()
    label_dir.mkdir()

    write_processed_cache(feature_dir, label_dir, "1abc_A", "AGAGA", include_templates=True)
    write_processed_cache(feature_dir, label_dir, "2xyz_A", "AGGAA", include_templates=False)

    chain_ids = discover_chain_ids(feature_dir, label_dir)
    train_ids = split_chain_ids(chain_ids, split="train", val_fraction=0.5, seed=0)
    val_ids = split_chain_ids(chain_ids, split="val", val_fraction=0.5, seed=0)

    assert chain_ids == ["1abc_A", "2xyz_A"]
    assert sorted(train_ids + val_ids) == chain_ids
    assert set(train_ids).isdisjoint(val_ids)

    dataset = ProcessedOpenProteinSetDataset(feature_dir, label_dir, split="all")
    random.seed(0)
    torch.manual_seed(0)
    batch = collate_batch(
        [dataset[0], dataset[1]],
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
        training=True,
    )

    assert batch["target_feat"].shape[-1] == 21
    assert batch["msa_feat"].shape[-1] == 49
    assert batch["extra_msa_feat"].shape[-1] == 25
    assert batch["template_pair_feat"].shape[-1] == 88
    assert batch["template_angle_feat"].shape[-1] == 57
    assert batch["masked_msa_target"].shape[-1] == MSA_ALPHABET_SIZE
    assert batch["template_mask"][1].sum().item() == 0.0


def test_template_feature_builders_return_canonical_widths():
    features, labels = make_feature_and_label_example("AGAGA", include_templates=True)
    template_aatype = torch.from_numpy(features["template_aatype"]).long()
    template_positions = torch.from_numpy(features["template_atom14_positions"]).float()
    template_mask = torch.from_numpy(features["template_atom14_mask"]).float()

    pair_feat = build_template_pair_feat(template_aatype, template_positions, template_mask)
    angle_feat = build_template_angle_feat(template_aatype, template_positions, template_mask)

    assert pair_feat.shape == (1, 5, 5, 88)
    assert angle_feat.shape == (1, 5, 57)


def test_build_processed_example_emits_loss_supervision_fields():
    features, labels = make_feature_and_label_example("AGAGA", include_templates=True)
    example = {
        "chain_id": "1abc_A",
        "aatype": torch.from_numpy(features["aatype"]).long(),
        "msa": torch.from_numpy(features["msa"]).long(),
        "deletions": torch.from_numpy(features["deletions"]).long(),
        "template_aatype": torch.from_numpy(features["template_aatype"]).long(),
        "template_atom14_positions": torch.from_numpy(features["template_atom14_positions"]).float(),
        "template_atom14_mask": torch.from_numpy(features["template_atom14_mask"]).float(),
        "atom14_positions": torch.from_numpy(labels["atom14_positions"]).float(),
        "atom14_mask": torch.from_numpy(labels["atom14_mask"]).float(),
    }

    processed = build_processed_example(
        example,
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
        training=False,
    )

    assert processed["true_atom_positions"].shape == (5, 14, 3)
    assert processed["true_atom_mask"].shape == (5, 14)
    assert processed["true_torsion_angles"].shape == (5, 7, 2)
    assert processed["true_torsion_mask"].shape == (5, 7)
    assert processed["masked_msa_target"].shape[-1] == MSA_ALPHABET_SIZE


def test_collate_batch_can_make_training_features_deterministic():
    features, labels = make_feature_and_label_example("AGAGA", include_templates=True)
    example = {
        "chain_id": "1abc_A",
        "aatype": torch.from_numpy(features["aatype"]).long(),
        "msa": torch.from_numpy(features["msa"]).long(),
        "deletions": torch.from_numpy(features["deletions"]).long(),
        "template_aatype": torch.from_numpy(features["template_aatype"]).long(),
        "template_atom14_positions": torch.from_numpy(features["template_atom14_positions"]).float(),
        "template_atom14_mask": torch.from_numpy(features["template_atom14_mask"]).float(),
        "atom14_positions": torch.from_numpy(labels["atom14_positions"]).float(),
        "atom14_mask": torch.from_numpy(labels["atom14_mask"]).float(),
    }

    batch_1 = collate_batch(
        [example],
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
        training=True,
        random_seed=7,
    )
    batch_2 = collate_batch(
        [example],
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
        training=True,
        random_seed=7,
    )

    assert torch.equal(batch_1["msa_feat"], batch_2["msa_feat"])
    assert torch.equal(batch_1["masked_msa_mask"], batch_2["masked_msa_mask"])


def test_collate_batch_can_disable_block_deletion():
    features, labels = make_feature_and_label_example("AGAGA", include_templates=False)
    example = {
        "chain_id": "1abc_A",
        "aatype": torch.from_numpy(features["aatype"]).long(),
        "msa": torch.from_numpy(features["msa"]).long(),
        "deletions": torch.from_numpy(features["deletions"]).long(),
        "template_aatype": torch.from_numpy(features["template_aatype"]).long(),
        "template_atom14_positions": torch.from_numpy(features["template_atom14_positions"]).float(),
        "template_atom14_mask": torch.from_numpy(features["template_atom14_mask"]).float(),
        "atom14_positions": torch.from_numpy(labels["atom14_positions"]).float(),
        "atom14_mask": torch.from_numpy(labels["atom14_mask"]).float(),
    }

    batch = collate_batch(
        [example],
        crop_size=8,
        msa_depth=8,
        extra_msa_depth=0,
        max_templates=1,
        training=True,
        block_delete_training_msa=False,
        random_seed=3,
    )

    assert batch["msa_feat"].shape[1] == example["msa"].shape[0]


def test_end_to_end_processed_batch_runs_model_and_loss(tmp_path):
    feature_dir = tmp_path / "processed_features"
    label_dir = tmp_path / "processed_labels"
    feature_dir.mkdir()
    label_dir.mkdir()

    write_processed_cache(feature_dir, label_dir, "1abc_A", "AGAGA", include_templates=True)
    dataset = ProcessedOpenProteinSetDataset(feature_dir, label_dir, split="all")
    batch = collate_batch(
        [dataset[0]],
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
        training=False,
    )

    model = AlphaFold2(SmallConfig())
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch["target_feat"],
            batch["residue_index"],
            batch["msa_feat"],
            batch["extra_msa_feat"],
            batch["template_pair_feat"],
            batch["aatype"],
            template_angle_feat=batch["template_angle_feat"],
            template_mask=batch["template_mask"],
            seq_mask=batch["seq_mask"],
            msa_mask=batch["msa_mask"],
            extra_msa_mask=batch["extra_msa_mask"],
            n_cycles=1,
            n_ensemble=1,
        )

        loss = AlphaFoldLoss()(
            structure_model_prediction=outputs,
            true_rotations=batch["true_rotations"],
            true_translations=batch["true_translations"],
            true_atom_positions=batch["true_atom_positions"],
            true_atom_mask=batch["true_atom_mask"],
            true_torsion_angles=batch["true_torsion_angles"],
            true_torsion_mask=batch["true_torsion_mask"],
            experimentally_resolved_pred=outputs["experimentally_resolved_logits"],
            experimentally_resolved_true=batch["experimentally_resolved_true"],
            masked_msa_pred=outputs["masked_msa_logits"],
            masked_msa_target=batch["masked_msa_target"],
            masked_msa_mask=batch["masked_msa_mask"],
            plddt_pred=outputs["plddt_logits"],
            distogram_pred=outputs["distogram_logits"],
            res_types=batch["res_types"],
            seq_mask=batch["seq_mask"],
        )

    assert loss.shape == (1,)
    assert torch.isfinite(loss).all()
