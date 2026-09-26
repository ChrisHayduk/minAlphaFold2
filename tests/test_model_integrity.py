import pytest
import torch

from minalphafold.data import build_supervision, collate_batch
from minalphafold.model import AlphaFold2
from tests.test_data_pipeline import (
    make_feature_and_label_example,
    torch_example_from_npz_parts,
)
from tests.test_shapes import MockConfig


def _batch(sequence="AGAGA", *, cycles=2, ensembles=2):
    features, labels = make_feature_and_label_example(sequence, include_templates=False)
    example = torch_example_from_npz_parts("test_A", features, labels)
    return collate_batch(
        [example],
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=0,
        training=False,
        random_seed=11,
        num_recycling_samples=cycles,
        num_ensemble_samples=ensembles,
    )


def _inputs(batch):
    return {
        key: batch[key]
        for key in (
            "target_feat",
            "residue_index",
            "msa_feat",
            "extra_msa_feat",
            "template_pair_feat",
            "aatype",
            "seq_mask",
        )
    }


def test_sampled_inputs_default_masks_match_explicit_masks_and_backpropagate():
    torch.manual_seed(31)
    model = AlphaFold2(MockConfig()).train()
    batch = _batch()
    default = model(**_inputs(batch), n_cycles=2, n_ensemble=2, sample_recycles=False)
    explicit = model(
        **_inputs(batch),
        msa_mask=batch["msa_mask"],
        extra_msa_mask=batch["extra_msa_mask"],
        n_cycles=2,
        n_ensemble=2,
        sample_recycles=False,
    )
    torch.testing.assert_close(default["atom14_coords"], explicit["atom14_coords"])
    assert model.last_n_cycles == 2
    assert model.last_n_ensemble == 2
    default["atom14_coords"].square().sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.count_nonzero() > 0 for g in gradients)


def test_structure_output_masks_padding_and_absent_atoms():
    batch = _batch()
    batch["seq_mask"][:, -2:] = 0
    model = AlphaFold2(MockConfig()).eval()
    with torch.no_grad():
        output = model(**_inputs(batch), n_cycles=1, n_ensemble=1)
    mask = output["atom14_mask"]
    assert mask[:, -2:].count_nonzero() == 0
    assert output["atom14_coords"][~mask.bool()].count_nonzero() == 0


@pytest.mark.parametrize(
    "options", [{"n_cycles": True}, {"n_ensemble": 1.5}, {"sample_recycles": 1}]
)
def test_invalid_recycle_controls_fail_before_forward(options):
    with pytest.raises(ValueError):
        AlphaFold2(MockConfig())(**_inputs(_batch()), **options)


def test_gap_and_segment_torsions_do_not_supervise_nonbonded_residues():
    features, labels = make_feature_and_label_example("AAAAA", include_templates=False)
    example = torch_example_from_npz_parts("gap_A", features, labels)
    example["residue_index"] = torch.tensor([10, 11, 14, 15, 16])
    example["between_segment_residues"] = torch.tensor([0, 0, 0, 1, 0])
    uninterrupted = build_supervision(
        example["aatype"], example["atom14_positions"], example["atom14_mask"]
    )
    batch = collate_batch(
        [example],
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=0,
        training=False,
    )
    assert batch["true_torsion_mask"][0, 2:4, :2].count_nonzero() == 0
    torch.testing.assert_close(
        batch["true_torsion_mask"][0, :, 2:], uninterrupted["true_torsion_mask"][:, 2:]
    )
    torch.testing.assert_close(batch["residue_index"][0], example["residue_index"])
    assert torch.equal(
        torch.diff(batch["loss_residue_index"])[0] == 1,
        torch.tensor([True, False, False, True]),
    )


def test_absent_nonfinite_coordinates_are_neutralized_before_geometry():
    features, labels = make_feature_and_label_example("AGAGA", include_templates=False)
    example = torch_example_from_npz_parts("mask_A", features, labels)
    positions = example["atom14_positions"].clone()
    mask = example["atom14_mask"]
    positions[~mask.bool()] = float("nan")
    expected = build_supervision(example["aatype"], example["atom14_positions"], mask)
    actual = build_supervision(example["aatype"], positions, mask)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name])
        assert torch.isfinite(actual[name]).all()
    positions[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Observed atom"):
        build_supervision(example["aatype"], positions, mask)


def test_default_msa_masks_exclude_padding_after_learning_nonzero_projections():
    torch.manual_seed(121)
    batch = _batch()
    batch["seq_mask"][:, -2:] = 0
    model = AlphaFold2(MockConfig()).eval()
    with torch.no_grad():
        # Identity/zero output initialization can hide padding contamination;
        # exercise nonzero learned projections on the actual model path.
        for parameter in model.parameters():
            if parameter.ndim >= 2 and parameter.count_nonzero() == 0:
                parameter.normal_(std=0.03)
        actual = model(**_inputs(batch), n_cycles=2, n_ensemble=2)
        expected = model(
            **_inputs(batch),
            msa_mask=batch["msa_mask"] * batch["seq_mask"],
            extra_msa_mask=batch["extra_msa_mask"] * batch["seq_mask"],
            n_cycles=2, n_ensemble=2,
        )
    torch.testing.assert_close(actual["atom14_coords"], expected["atom14_coords"])
    torch.testing.assert_close(actual["pair_representation"], expected["pair_representation"])
