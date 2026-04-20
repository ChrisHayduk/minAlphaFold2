import os
import random
from pathlib import Path

import numpy as np
import torch

from minalphafold import data as data_module
from minalphafold.a3m import MASK_ID, MSA_ALPHABET_SIZE, sequence_to_ids
from minalphafold.data import (
    ProcessedOpenProteinSetDataset,
    TARGET_FEAT_DIM,
    build_msa_features,
    build_processed_example,
    sample_cluster_and_extra,
    build_target_feat,
    build_supervision,
    build_template_angle_feat,
    build_template_pair_feat,
    collate_batch,
    discover_chain_ids,
    split_chain_ids,
)
from minalphafold.losses import AlphaFoldLoss, select_best_atom14_ground_truth
from minalphafold.model import AlphaFold2
from minalphafold.residue_constants import (
    atom_type_num,
    restype_1to3,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from minalphafold.structure_module import compute_all_atom_coordinates
from preprocess_openproteinset import preprocess_chain

ROOT = Path(__file__).resolve().parents[1]


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
    # Supplement 1.7.1 / Algorithm 16: template-pair-stack-specific triangle dims.
    template_triangle_mult_c = 16
    template_triangle_attn_c = 8
    template_triangle_attn_num_heads = 2
    template_pair_transition_n = 2

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
    position_scale = 10.0
    zero_init = True

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


def write_simple_mmcif(
    path: Path,
    sequence: str,
    *,
    chain_id: str,
    resolution: float = 1.5,
    auth_seq_ids: list[int] | None = None,
) -> None:
    positions, mask = make_atom14_arrays(sequence)
    pdb_id = path.stem.upper()
    if auth_seq_ids is None:
        auth_seq_ids = list(range(1, len(sequence) + 1))
    if len(auth_seq_ids) != len(sequence):
        raise ValueError("auth_seq_ids must match the sequence length.")

    rows = []
    for residue_index, residue in enumerate(sequence, start=1):
        residue_name = restype_1to3[residue]
        auth_seq_id = auth_seq_ids[residue_index - 1]
        atom_names = ["N", "CA", "C", "O"]
        atom_indices = [0, 1, 2, 3]
        if residue != "G":
            atom_names.append("CB")
            atom_indices.append(4)

        for atom_name, atom_index in zip(atom_names, atom_indices):
            x, y, z = positions[residue_index - 1, atom_index]
            rows.append(
                f"ATOM 1 {chain_id} {chain_id} 1 {residue_index} . {auth_seq_id} "
                f"{residue_name} {residue_name} {atom_name} {atom_name} "
                f"{x:.3f} {y:.3f} {z:.3f} {mask[residue_index - 1, atom_index]:.1f}"
            )

    text = (
        f"data_{pdb_id}\n"
        "_entity_poly.entity_id 1\n"
        f"_entity_poly.pdbx_seq_one_letter_code_can {sequence}\n"
        f"_refine.ls_d_res_high {resolution:.2f}\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.auth_seq_id\n"
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
        "between_segment_residues": np.zeros((len(sequence),), dtype=np.int32),
        "residue_index": np.arange(len(sequence), dtype=np.int32),
        "template_aatype": template_aatype,
        "template_atom14_positions": template_atom14_positions.astype(np.float32),
        "template_atom14_mask": template_atom14_mask.astype(np.float32),
    }
    labels = {
        "atom14_positions": atom14_positions.astype(np.float32),
        "atom14_mask": atom14_mask.astype(np.float32),
        "resolution": np.asarray(1.5, dtype=np.float32),
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
        "between_segment_residues",
        "residue_index",
        "template_aatype",
        "template_atom14_positions",
        "template_atom14_mask",
    }
    assert set(labels) == {"atom14_positions", "atom14_mask", "resolution"}
    assert features["aatype"].tolist() == sequence_to_ids("AGA").tolist()
    assert features["residue_index"].tolist() == [0, 1, 2]
    assert labels["atom14_positions"].shape == (3, 14, 3)
    assert labels["atom14_mask"].shape == (3, 14)
    assert labels["resolution"].item() == 1.5
    assert "between_segment_residues" in features
    assert np.array_equal(features["between_segment_residues"], np.zeros((3,), dtype=np.int32))
    assert features["template_aatype"].shape == (1, 3)
    assert features["template_aatype"][0].tolist() == sequence_to_ids("AGA").tolist()


def test_preprocess_chain_merges_multiple_msa_sources(tmp_path):
    raw_root = tmp_path / "openproteinset"
    chain_dir = raw_root / "roda_pdb" / "1abc_A"
    (chain_dir / "a3m").mkdir(parents=True)
    mmcif_root = raw_root / "pdb_data" / "mmcif_files"
    mmcif_root.mkdir(parents=True)

    write_a3m(chain_dir / "a3m" / "uniref90_hits.a3m", ["AGA", "AAA"])
    write_a3m(chain_dir / "a3m" / "bfd_uniclust_hits.a3m", ["AGA", "GGA", "CCC"])
    write_simple_mmcif(mmcif_root / "1abc.cif", "AGA", chain_id="A")

    features, labels = preprocess_chain(
        chain_dir,
        mmcif_root=mmcif_root,
        max_msa_seqs=8,
        max_templates=0,
        msa_name="uniref90_hits.a3m",
        msa_names=("uniref90_hits.a3m", "bfd_uniclust_hits.a3m"),
        template_hhr_name="pdb70_hits.hhr",
        skip_templates=True,
    )

    assert features["msa"].shape == (4, 3)
    assert features["deletions"].shape == (4, 3)
    assert features["residue_index"].tolist() == [0, 1, 2]
    assert features["msa"][0].tolist() == sequence_to_ids("AGA").tolist()
    assert features["msa"][1].tolist() == sequence_to_ids("AAA").tolist()
    assert features["msa"][2].tolist() == sequence_to_ids("GGA").tolist()
    assert features["msa"][3].tolist() == sequence_to_ids("CCC").tolist()
    assert labels["resolution"].item() == 1.5


def test_preprocess_chain_uses_canonical_sequence_indices_not_author_numbering(tmp_path):
    raw_root = tmp_path / "openproteinset"
    chain_dir = raw_root / "roda_pdb" / "1abc_A"
    (chain_dir / "a3m").mkdir(parents=True)
    mmcif_root = raw_root / "pdb_data" / "mmcif_files"
    mmcif_root.mkdir(parents=True)

    write_a3m(chain_dir / "a3m" / "uniref90_hits.a3m", ["AGA", "AAA"])
    write_simple_mmcif(mmcif_root / "1abc.cif", "AGA", chain_id="A", auth_seq_ids=[10, 11, 13])

    features, _labels = preprocess_chain(
        chain_dir,
        mmcif_root=mmcif_root,
        max_msa_seqs=8,
        max_templates=0,
        msa_name="uniref90_hits.a3m",
        template_hhr_name="pdb70_hits.hhr",
        skip_templates=True,
    )

    assert features["residue_index"].tolist() == [0, 1, 2]


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

    assert batch["target_feat"].shape[-1] == TARGET_FEAT_DIM
    assert batch["msa_feat"].shape[-1] == 49
    assert batch["extra_msa_feat"].shape[-1] == 25
    assert batch["template_pair_feat"].shape[-1] == 88
    assert batch["template_angle_feat"].shape[-1] == 57
    assert batch["masked_msa_target"].shape[-1] == MSA_ALPHABET_SIZE
    assert batch["resolution"].shape == (2,)
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


def test_build_target_feat_includes_canonical_break_channel():
    aatype = torch.tensor([0, 5, 2], dtype=torch.long)
    between_segment = torch.tensor([0, 1, 0], dtype=torch.long)

    target_feat = build_target_feat(aatype, between_segment)

    assert target_feat.shape == (3, TARGET_FEAT_DIM)
    assert torch.equal(target_feat[:, 0], between_segment.float())
    assert torch.equal(target_feat[1, 1:], torch.nn.functional.one_hot(aatype[1], 21).float())


def test_sample_cluster_and_extra_preserves_canonical_sample_order():
    msa = torch.tensor(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
        ],
        dtype=torch.long,
    )
    deletions = torch.zeros_like(msa)

    expected_generator = torch.Generator()
    expected_generator.manual_seed(11)
    remaining = torch.arange(1, msa.shape[0], dtype=torch.long)
    expected_remaining = remaining[torch.randperm(remaining.numel(), generator=expected_generator)]

    generator = torch.Generator()
    generator.manual_seed(11)
    cluster_msa, _, _, _ = sample_cluster_and_extra(
        msa,
        deletions,
        msa_depth=4,
        extra_msa_depth=0,
        training=True,
        torch_generator=generator,
    )

    expected_indices = torch.cat([torch.zeros(1, dtype=torch.long), expected_remaining[:3]])
    assert torch.equal(cluster_msa, msa[expected_indices])


def test_build_msa_features_clusters_on_masked_centers(monkeypatch):
    monkeypatch.setattr(
        data_module,
        "_sample_categorical",
        lambda probabilities, **_: torch.full(probabilities.shape[:-1], MASK_ID, dtype=torch.long),
    )

    features = build_msa_features(
        {
            "msa": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "deletions": torch.zeros((1, 3), dtype=torch.float32),
        },
        msa_depth=1,
        extra_msa_depth=0,
        training=True,
        masked_msa_probability=1.0,
        random_seed=0,
    )

    cluster_profile = features["msa_feat"][0, :, 25:48]
    assert torch.allclose(cluster_profile[:, MASK_ID], torch.ones(3))


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
    assert processed["true_atom_positions_alt"].shape == (5, 14, 3)
    assert processed["true_atom_mask_alt"].shape == (5, 14)
    assert processed["true_atom_is_ambiguous"].shape == (5, 14)
    assert processed["true_torsion_angles"].shape == (5, 7, 2)
    assert processed["true_torsion_angles_alt"].shape == (5, 7, 2)
    assert processed["true_torsion_mask"].shape == (5, 7)
    assert processed["true_rigid_group_frames_R"].shape == (5, 8, 3, 3)
    assert processed["true_rigid_group_frames_t"].shape == (5, 8, 3)
    assert processed["true_rigid_group_frames_R_alt"].shape == (5, 8, 3, 3)
    assert processed["true_rigid_group_frames_t_alt"].shape == (5, 8, 3)
    assert processed["true_rigid_group_exists"].shape == (5, 8)
    assert processed["atom37_exists"].shape == (5, atom_type_num)
    assert processed["experimentally_resolved_true"].shape == (5, atom_type_num)
    assert processed["resolution"].shape == ()
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


def test_template_pair_features_include_canonical_unit_vectors():
    features, _ = make_feature_and_label_example("AGAGA", include_templates=True)
    template_aatype = torch.from_numpy(features["template_aatype"]).long()
    template_positions = torch.from_numpy(features["template_atom14_positions"]).float()
    template_mask = torch.from_numpy(features["template_atom14_mask"]).float()

    pair_feat = build_template_pair_feat(template_aatype, template_positions, template_mask)

    assert torch.any(torch.abs(pair_feat[..., 84:87]) > 1e-6)


def test_build_processed_example_preserves_explicit_residue_indices():
    features, labels = make_feature_and_label_example("AGAGA", include_templates=False)
    example = {
        "chain_id": "1abc_A",
        "aatype": torch.from_numpy(features["aatype"]).long(),
        "msa": torch.from_numpy(features["msa"]).long(),
        "deletions": torch.from_numpy(features["deletions"]).long(),
        "residue_index": torch.tensor([10, 11, 12, 14, 15], dtype=torch.long),
        "template_aatype": torch.from_numpy(features["template_aatype"]).long(),
        "template_atom14_positions": torch.from_numpy(features["template_atom14_positions"]).float(),
        "template_atom14_mask": torch.from_numpy(features["template_atom14_mask"]).float(),
        "atom14_positions": torch.from_numpy(labels["atom14_positions"]).float(),
        "atom14_mask": torch.from_numpy(labels["atom14_mask"]).float(),
        "crop_start": 0,
    }

    processed = build_processed_example(
        example,
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=0,
        training=False,
    )

    assert torch.equal(processed["residue_index"], torch.tensor([10, 11, 12, 14, 15], dtype=torch.long))


def test_collate_batch_can_emit_recycling_feature_samples():
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

    batch = collate_batch(
        [example],
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
        training=True,
        random_seed=5,
        num_recycling_samples=2,
        num_ensemble_samples=1,
    )

    assert batch["msa_feat"].shape[:3] == (2, 1, 1)
    assert batch["extra_msa_feat"].shape[:3] == (2, 1, 1)
    assert batch["masked_msa_mask"].shape[:3] == (2, 1, 1)
    assert torch.equal(batch["residue_index"], torch.tensor([[0, 1, 2, 3, 4]]))


def test_alphafold_loss_uses_per_example_mask_normalization():
    loss_fn = AlphaFoldLoss()
    logits = torch.zeros((2, 1, 2, 23), dtype=torch.float32)
    target = torch.zeros_like(logits)
    target[..., 0] = 1.0
    mask = torch.tensor([[[1.0, 1.0]], [[1.0, 0.0]]], dtype=torch.float32)

    loss = loss_fn.msa_loss(logits, target, mask)
    # `torch.full(fill_value=...)` wants a Python scalar, not a 0-d tensor.
    expected = torch.full((2,), float(torch.log(torch.tensor(23.0)).item()))

    assert torch.allclose(loss, expected, atol=1e-6)


def test_select_best_atom14_ground_truth_can_choose_alternative_naming():
    predicted = torch.zeros((1, 2, 14, 3), dtype=torch.float32)
    true_positions = torch.zeros_like(predicted)
    true_mask = torch.zeros((1, 2, 14), dtype=torch.float32)
    true_alt_positions = torch.zeros_like(predicted)
    true_alt_mask = torch.zeros_like(true_mask)
    true_atom_is_ambiguous = torch.zeros_like(true_mask)

    predicted[0, 0, 6] = torch.tensor([2.0, 0.0, 0.0])
    predicted[0, 0, 7] = torch.tensor([1.0, 0.0, 0.0])
    predicted[0, 1, 0] = torch.tensor([2.0, 0.0, 0.0])

    true_positions[0, 0, 6] = torch.tensor([1.0, 0.0, 0.0])
    true_positions[0, 0, 7] = torch.tensor([2.0, 0.0, 0.0])
    true_alt_positions[0, 0, 6] = torch.tensor([2.0, 0.0, 0.0])
    true_alt_positions[0, 0, 7] = torch.tensor([1.0, 0.0, 0.0])
    true_positions[0, 1, 0] = torch.tensor([2.0, 0.0, 0.0])
    true_alt_positions[0, 1, 0] = torch.tensor([2.0, 0.0, 0.0])
    true_mask[0, 0, 6:8] = 1.0
    true_alt_mask[0, 0, 6:8] = 1.0
    true_mask[0, 1, 0] = 1.0
    true_alt_mask[0, 1, 0] = 1.0
    true_atom_is_ambiguous[0, 0, 6:8] = 1.0

    chosen_positions, chosen_mask, _ = select_best_atom14_ground_truth(
        predicted,
        true_positions,
        true_mask,
        true_alt_positions,
        true_alt_mask,
        true_atom_is_ambiguous,
    )

    assert torch.equal(chosen_positions[0, 0], true_alt_positions[0, 0])
    assert torch.equal(chosen_mask[0, 0], true_alt_mask[0, 0])


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

        loss, breakdown = AlphaFoldLoss()(
            structure_model_prediction=outputs,
            true_rotations=batch["true_rotations"],
            true_translations=batch["true_translations"],
            true_atom_positions=batch["true_atom_positions"],
            true_atom_mask=batch["true_atom_mask"],
            true_atom_positions_alt=batch["true_atom_positions_alt"],
            true_atom_mask_alt=batch["true_atom_mask_alt"],
            true_atom_is_ambiguous=batch["true_atom_is_ambiguous"],
            true_torsion_angles=batch["true_torsion_angles"],
            true_torsion_angles_alt=batch["true_torsion_angles_alt"],
            true_torsion_mask=batch["true_torsion_mask"],
            true_rigid_group_frames_R=batch["true_rigid_group_frames_R"],
            true_rigid_group_frames_t=batch["true_rigid_group_frames_t"],
            true_rigid_group_frames_R_alt=batch["true_rigid_group_frames_R_alt"],
            true_rigid_group_frames_t_alt=batch["true_rigid_group_frames_t_alt"],
            true_rigid_group_exists=batch["true_rigid_group_exists"],
            experimentally_resolved_pred=outputs["experimentally_resolved_logits"],
            experimentally_resolved_true=batch["experimentally_resolved_true"],
            experimentally_resolved_exists=batch["atom37_exists"],
            masked_msa_pred=outputs["masked_msa_logits"],
            masked_msa_target=batch["masked_msa_target"],
            masked_msa_mask=batch["masked_msa_mask"],
            plddt_pred=outputs["plddt_logits"],
            distogram_pred=outputs["distogram_logits"],
            res_types=batch["res_types"],
            residue_index=batch["residue_index"],
            seq_mask=batch["seq_mask"],
            return_breakdown=True,
        )

    assert loss.shape == (1,)
    assert torch.isfinite(loss).all()
    assert torch.allclose(loss, breakdown["loss"])
    assert "weighted_structural_violation_loss" not in breakdown
    expected = (
        breakdown["structure_loss"]
        + breakdown["weighted_distogram_loss"]
        + breakdown["weighted_msa_loss"]
        + breakdown["weighted_plddt_loss"]
    )
    assert torch.allclose(loss, expected, atol=1e-6)


def test_build_supervision_round_trips_into_exact_structure_targets():
    sequence = "RKYD"
    aatype = torch.from_numpy(sequence_to_ids(sequence)).long().unsqueeze(0)

    translations = torch.tensor(
        [[[0.0, 0.0, 0.0], [3.8, 0.5, 0.2], [7.6, -0.4, 0.1], [11.4, 0.2, -0.3]]],
        dtype=torch.float32,
    )
    rotations = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, len(sequence), 1, 1)
    torsions = torch.tensor(
        [
            [
                [[0.0, 1.0], [0.0, 1.0], [0.6, 0.8], [0.5, 0.8660254], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.8, 0.6], [0.70710677, 0.70710677], [0.5, 0.8660254], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [-0.5, 0.8660254], [0.8660254, 0.5], [0.70710677, 0.70710677], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.25881904, 0.9659258], [0.5, 0.8660254], [0.25881904, 0.9659258], [0.70710677, 0.70710677], [0.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )

    default_frames = torch.tensor(restype_rigid_group_default_frame, dtype=torch.float32)
    lit_positions = torch.tensor(restype_atom14_rigid_group_positions, dtype=torch.float32)
    atom_frame_idx_table = torch.tensor(restype_atom14_to_rigid_group, dtype=torch.long)
    atom_mask_table = torch.tensor(restype_atom14_mask, dtype=torch.float32)

    _, _, atom_positions, atom_mask = compute_all_atom_coordinates(
        translations,
        rotations,
        torsions,
        aatype,
        default_frames,
        lit_positions,
        atom_frame_idx_table,
        atom_mask_table,
    )
    supervision = build_supervision(aatype[0], atom_positions[0], atom_mask[0])
    assert supervision["atom37_exists"].shape == (len(sequence), atom_type_num)
    assert supervision["experimentally_resolved_true"].shape == (len(sequence), atom_type_num)
    assert torch.all(supervision["experimentally_resolved_true"] <= supervision["atom37_exists"])

    reconstructed_frames_R, reconstructed_frames_t, reconstructed_positions, reconstructed_mask = compute_all_atom_coordinates(
        supervision["true_translations"].unsqueeze(0),
        supervision["true_rotations"].unsqueeze(0),
        supervision["true_torsion_angles"].unsqueeze(0),
        aatype,
        default_frames,
        lit_positions,
        atom_frame_idx_table,
        atom_mask_table,
    )

    common_mask = (atom_mask[0] > 0.5) & (reconstructed_mask[0] > 0.5)
    assert torch.allclose(reconstructed_positions[0][common_mask], atom_positions[0][common_mask], atol=1e-3)

    structure_prediction = {
        "traj_rotations": supervision["true_rotations"].unsqueeze(0).repeat(2, 1, 1, 1, 1),
        "traj_translations": supervision["true_translations"].unsqueeze(0).repeat(2, 1, 1, 1),
        "traj_torsion_angles": supervision["true_torsion_angles"].unsqueeze(0).repeat(2, 1, 1, 1, 1),
        "traj_torsion_angles_unnormalized": supervision["true_torsion_angles"].unsqueeze(0).repeat(2, 1, 1, 1, 1),
        "all_frames_R": reconstructed_frames_R,
        "all_frames_t": reconstructed_frames_t,
        "atom14_coords": reconstructed_positions,
        "atom14_mask": reconstructed_mask,
    }

    loss_module = AlphaFoldLoss(finetune=False)
    loss_module.distogram_weight = 0.0
    loss_module.msa_weight = 0.0
    loss_module.confidence_weight = 0.0
    loss_terms = loss_module.compute_loss_terms(
        structure_model_prediction=structure_prediction,
        true_rotations=supervision["true_rotations"].unsqueeze(0),
        true_translations=supervision["true_translations"].unsqueeze(0),
        true_atom_positions=supervision["true_atom_positions"].unsqueeze(0),
        true_atom_mask=supervision["true_atom_mask"].unsqueeze(0),
        true_atom_positions_alt=supervision["true_atom_positions_alt"].unsqueeze(0),
        true_atom_mask_alt=supervision["true_atom_mask_alt"].unsqueeze(0),
        true_atom_is_ambiguous=supervision["true_atom_is_ambiguous"].unsqueeze(0),
        true_torsion_angles=supervision["true_torsion_angles"].unsqueeze(0),
        true_torsion_angles_alt=supervision["true_torsion_angles_alt"].unsqueeze(0),
        true_torsion_mask=supervision["true_torsion_mask"].unsqueeze(0),
        true_rigid_group_frames_R=supervision["true_rigid_group_frames_R"].unsqueeze(0),
        true_rigid_group_frames_t=supervision["true_rigid_group_frames_t"].unsqueeze(0),
        true_rigid_group_frames_R_alt=supervision["true_rigid_group_frames_R_alt"].unsqueeze(0),
        true_rigid_group_frames_t_alt=supervision["true_rigid_group_frames_t_alt"].unsqueeze(0),
        true_rigid_group_exists=supervision["true_rigid_group_exists"].unsqueeze(0),
        experimentally_resolved_pred=torch.zeros((1, len(sequence), atom_type_num), dtype=torch.float32),
        experimentally_resolved_true=supervision["experimentally_resolved_true"].unsqueeze(0),
        experimentally_resolved_exists=supervision["atom37_exists"].unsqueeze(0),
        masked_msa_pred=torch.zeros((1, 1, len(sequence), 23), dtype=torch.float32),
        masked_msa_target=torch.zeros((1, 1, len(sequence), 23), dtype=torch.float32),
        masked_msa_mask=torch.zeros((1, 1, len(sequence)), dtype=torch.float32),
        plddt_pred=torch.zeros((1, len(sequence), 50), dtype=torch.float32),
        distogram_pred=torch.zeros((1, len(sequence), len(sequence), 64), dtype=torch.float32),
        res_types=supervision["res_types"].unsqueeze(0),
        residue_index=torch.arange(len(sequence), dtype=torch.long).unsqueeze(0),
        seq_mask=torch.ones((1, len(sequence)), dtype=torch.float32),
    )

    # With predictions matching the supervision targets, each FAPE d_ij
    # collapses to sqrt(eps) / Z. BackboneFAPE uses eps=1e-12 per Algorithm
    # 20 line 17 (sqrt ≈ 1e-6, /Z=10 ≈ 1e-7); AllAtomFAPE uses eps=1e-4 per
    # Algorithm 20 line 28 (sqrt=1e-2, /Z=10 ≈ 1e-3).
    assert torch.allclose(
        loss_terms["backbone_loss"],
        torch.tensor([1e-7], dtype=torch.float32),
        atol=5e-7,
    )
    assert torch.allclose(
        loss_terms["sidechain_fape_loss"],
        torch.tensor([0.0010], dtype=torch.float32),
        atol=5e-4,
    )
    assert float(loss_terms["structure_loss"].item()) < 0.01


def test_build_supervision_gives_near_zero_loss_on_non_idealised_atoms():
    """Regression test for the parametric-GT-frame fix.

    Before the fix, ``build_supervision`` built sidechain rigid-group frames
    via Gram-Schmidt on the real atoms (``atom14_to_rigid_group_frames``).
    The Structure Module builds its *predicted* sidechain frames
    parametrically (backbone ∘ T^lit ∘ makeRotX(torsion), Algorithm 24),
    using literature bond geometry. On real PDB atoms the two paths
    disagree by a few tenths of an Å because real bond lengths are not
    exactly the literature values — so the sidechain FAPE loss acquired a
    non-zero floor (~0.021 on 1a0m_A) that the model could never clear,
    producing the classic "loss low but RMSD 3-4 Å and ribbon mode fails"
    overfit pathology.

    After the fix, ``build_supervision`` builds GT sidechain frames the
    *same way* as the prediction path, so the loss floor collapses down to
    the atom-idealisation-only level (sub-0.01).

    This test simulates the problematic scenario: take synthetic atoms,
    apply bond-length perturbations that mimic real-atom non-idealisation,
    run them through the full supervision pipeline, feed the supervision
    through as a "perfect prediction", and verify the sidechain FAPE floor
    is near zero — asserting below the pre-fix 0.021 value that caused the
    bug.
    """
    sequence = "RKYD"
    aatype = torch.from_numpy(sequence_to_ids(sequence)).long().unsqueeze(0)
    translations = torch.tensor(
        [[[0.0, 0.0, 0.0], [3.8, 0.5, 0.2], [7.6, -0.4, 0.1], [11.4, 0.2, -0.3]]],
        dtype=torch.float32,
    )
    rotations = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, len(sequence), 1, 1)
    torsions = torch.tensor(
        [
            [
                [[0.0, 1.0], [0.0, 1.0], [0.6, 0.8], [0.5, 0.8660254], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.8, 0.6], [0.70710677, 0.70710677], [0.5, 0.8660254], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [-0.5, 0.8660254], [0.8660254, 0.5], [0.70710677, 0.70710677], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.25881904, 0.9659258], [0.5, 0.8660254], [0.25881904, 0.9659258], [0.70710677, 0.70710677], [0.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    default_frames = torch.tensor(restype_rigid_group_default_frame, dtype=torch.float32)
    lit_positions = torch.tensor(restype_atom14_rigid_group_positions, dtype=torch.float32)
    atom_frame_idx_table = torch.tensor(restype_atom14_to_rigid_group, dtype=torch.long)
    atom_mask_table = torch.tensor(restype_atom14_mask, dtype=torch.float32)

    _, _, atom_positions, atom_mask = compute_all_atom_coordinates(
        translations, rotations, torsions, aatype,
        default_frames, lit_positions, atom_frame_idx_table, atom_mask_table,
    )
    # Perturb each atom by ~0.02 Å, typical of real-vs-literature bond-length
    # drift in X-ray structures. Large enough that the pre-fix Gram-Schmidt
    # sidechain frames would noticeably disagree with the parametric prediction
    # frames (driving sidechain FAPE ≳ 0.02 on this scale); small enough that,
    # post-fix, both paths agree and sidechain FAPE collapses to the atom
    # idealisation term.
    torch.manual_seed(42)
    perturbation = 0.02 * torch.randn_like(atom_positions) * atom_mask.unsqueeze(-1)
    real_atoms = atom_positions + perturbation

    supervision = build_supervision(aatype[0], real_atoms[0], atom_mask[0])

    recon_R, recon_t, recon_pos, recon_mask = compute_all_atom_coordinates(
        supervision["true_translations"].unsqueeze(0),
        supervision["true_rotations"].unsqueeze(0),
        supervision["true_torsion_angles"].unsqueeze(0),
        aatype,
        default_frames, lit_positions, atom_frame_idx_table, atom_mask_table,
    )
    structure_prediction = {
        "traj_rotations": supervision["true_rotations"].unsqueeze(0).unsqueeze(0),
        "traj_translations": supervision["true_translations"].unsqueeze(0).unsqueeze(0),
        "traj_torsion_angles": supervision["true_torsion_angles"].unsqueeze(0).unsqueeze(0),
        "traj_torsion_angles_unnormalized": supervision["true_torsion_angles"].unsqueeze(0).unsqueeze(0),
        "all_frames_R": recon_R,
        "all_frames_t": recon_t,
        "atom14_coords": recon_pos,
        "atom14_mask": recon_mask,
    }

    loss_module = AlphaFoldLoss(finetune=False)
    loss_module.distogram_weight = 0.0
    loss_module.msa_weight = 0.0
    loss_module.confidence_weight = 0.0
    loss_terms = loss_module.compute_loss_terms(
        structure_model_prediction=structure_prediction,
        true_rotations=supervision["true_rotations"].unsqueeze(0),
        true_translations=supervision["true_translations"].unsqueeze(0),
        true_atom_positions=supervision["true_atom_positions"].unsqueeze(0),
        true_atom_mask=supervision["true_atom_mask"].unsqueeze(0),
        true_atom_positions_alt=supervision["true_atom_positions_alt"].unsqueeze(0),
        true_atom_mask_alt=supervision["true_atom_mask_alt"].unsqueeze(0),
        true_atom_is_ambiguous=supervision["true_atom_is_ambiguous"].unsqueeze(0),
        true_torsion_angles=supervision["true_torsion_angles"].unsqueeze(0),
        true_torsion_angles_alt=supervision["true_torsion_angles_alt"].unsqueeze(0),
        true_torsion_mask=supervision["true_torsion_mask"].unsqueeze(0),
        true_rigid_group_frames_R=supervision["true_rigid_group_frames_R"].unsqueeze(0),
        true_rigid_group_frames_t=supervision["true_rigid_group_frames_t"].unsqueeze(0),
        true_rigid_group_frames_R_alt=supervision["true_rigid_group_frames_R_alt"].unsqueeze(0),
        true_rigid_group_frames_t_alt=supervision["true_rigid_group_frames_t_alt"].unsqueeze(0),
        true_rigid_group_exists=supervision["true_rigid_group_exists"].unsqueeze(0),
        experimentally_resolved_pred=torch.zeros((1, len(sequence), atom_type_num), dtype=torch.float32),
        experimentally_resolved_true=supervision["experimentally_resolved_true"].unsqueeze(0),
        experimentally_resolved_exists=supervision["atom37_exists"].unsqueeze(0),
        masked_msa_pred=torch.zeros((1, 1, len(sequence), 23), dtype=torch.float32),
        masked_msa_target=torch.zeros((1, 1, len(sequence), 23), dtype=torch.float32),
        masked_msa_mask=torch.zeros((1, 1, len(sequence)), dtype=torch.float32),
        plddt_pred=torch.zeros((1, len(sequence), 50), dtype=torch.float32),
        distogram_pred=torch.zeros((1, len(sequence), len(sequence), 64), dtype=torch.float32),
        res_types=supervision["res_types"].unsqueeze(0),
        residue_index=torch.arange(len(sequence), dtype=torch.long).unsqueeze(0),
        seq_mask=torch.ones((1, len(sequence)), dtype=torch.float32),
    )

    # Pre-fix: sidechain FAPE had a floor dominated by the Gram-Schmidt vs
    # parametric frame mismatch — on 1a0m_A (a real 16-residue PDB chain) we
    # measured 0.0210 even when the prediction exactly matched the
    # supervision. Post-fix: the floor collapses to atom-idealisation only,
    # 3× lower on 1a0m_A (0.007) and below 0.012 here. Threshold < 0.015
    # comfortably separates post-fix from pre-fix.
    sidechain = float(loss_terms["sidechain_fape_loss"].item())
    assert sidechain < 0.015, (
        f"sidechain_fape_loss={sidechain:.4f} — Gram-Schmidt vs parametric "
        f"frame mismatch has returned; the pre-fix floor on this scale of "
        f"perturbation exceeded this threshold."
    )
    assert float(loss_terms["backbone_loss"].item()) < 1e-5
    assert float(loss_terms["structure_loss"].item()) < 0.02
