import math

import torch

from minalphafold.losses import (
    AlphaFoldLoss,
    AllAtomFAPE,
    BackboneFAPE,
    ExperimentallyResolvedLoss,
    PLDDTLoss,
    StructuralViolationLoss,
    TMScoreLoss,
    TorsionAngleLoss,
    select_best_atom14_ground_truth,
)
from minalphafold.residue_constants import (
    restype_atom14_distance_lower_bound,
    restype_atom14_distance_upper_bound,
)


def test_all_atom_fape_matches_joint_masked_mean():
    loss_fn = AllAtomFAPE(d_clamp=10.0, eps=1e-4, Z=10.0)

    predicted_frames_R = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 2, 8, 1, 1)
    true_frames_R = predicted_frames_R.clone()
    predicted_frames_t = torch.zeros((1, 2, 8, 3), dtype=torch.float32)
    true_frames_t = torch.zeros((1, 2, 8, 3), dtype=torch.float32)

    predicted_atom_positions = torch.zeros((1, 2, 14, 3), dtype=torch.float32)
    true_atom_positions = torch.zeros_like(predicted_atom_positions)
    atom_mask = torch.zeros((1, 2, 14), dtype=torch.float32)
    true_atom_mask = torch.zeros_like(atom_mask)
    seq_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    # Rigid-group existence: only frames 0 (backbone) and 1 (ω→bb) active.
    rigid_group_existence = torch.tensor(
        [
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    atom_mask[0, 0, 0] = 1.0
    atom_mask[0, 1, 0] = 1.0
    true_atom_mask.copy_(atom_mask)
    predicted_atom_positions[0, 0, 0] = torch.tensor([2.0, 0.0, 0.0])
    predicted_atom_positions[0, 1, 0] = torch.tensor([4.0, 0.0, 0.0])

    loss = loss_fn(
        predicted_frames_R,
        predicted_frames_t,
        predicted_atom_positions,
        atom_mask,
        true_frames_R,
        true_frames_t,
        true_atom_positions,
        true_atom_mask=true_atom_mask,
        seq_mask=seq_mask,
        frame_mask=rigid_group_existence,
    )

    flat_atom_mask = atom_mask.reshape(1, -1)
    frame_mask = (seq_mask[:, :, None] * rigid_group_existence).reshape(1, -1)
    pred_R = predicted_frames_R.reshape(1, -1, 3, 3)
    pred_t = predicted_frames_t.reshape(1, -1, 3)
    true_R = true_frames_R.reshape(1, -1, 3, 3)
    true_t = true_frames_t.reshape(1, -1, 3)
    pred_pos = predicted_atom_positions.reshape(1, -1, 3)
    true_pos = true_atom_positions.reshape(1, -1, 3)
    local_pred = torch.einsum("bfij,baj->bfai", pred_R.transpose(-1, -2), pred_pos)
    local_true = torch.einsum("bfij,baj->bfai", true_R.transpose(-1, -2), true_pos)
    error = torch.sqrt(((local_pred - local_true) ** 2).sum(dim=-1) + 1e-4).clamp(max=10.0) / 10.0
    joint_mask = frame_mask[:, :, None] * flat_atom_mask[:, None, :]
    expected = (error * joint_mask).sum(dim=(-1, -2)) / joint_mask.sum(dim=(-1, -2)).clamp(min=1.0)

    assert torch.allclose(loss, expected, atol=1e-6)


def test_backbone_fape_matches_af2_frame_aligned_point_error():
    loss_fn = BackboneFAPE(d_clamp=10.0, eps=1e-4, Z=10.0)

    predicted_rotations = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    true_rotations = predicted_rotations.clone()
    predicted_translations = torch.tensor([[[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]], dtype=torch.float32)
    true_translations = torch.zeros_like(predicted_translations)
    mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    loss = loss_fn(
        predicted_rotations,
        predicted_translations,
        true_rotations,
        true_translations,
        frame_mask=mask,
        position_mask=mask,
        l1_clamp_distance=10.0,
    )

    pred_rot_inv = predicted_rotations.transpose(-1, -2)
    true_rot_inv = true_rotations.transpose(-1, -2)
    pred_trans_inv = -torch.einsum("bfij,bfj->bfi", pred_rot_inv, predicted_translations)
    true_trans_inv = -torch.einsum("bfij,bfj->bfi", true_rot_inv, true_translations)
    local_pred = torch.einsum("bfij,baj->bfai", pred_rot_inv, predicted_translations) + pred_trans_inv[:, :, None, :]
    local_true = torch.einsum("bfij,baj->bfai", true_rot_inv, true_translations) + true_trans_inv[:, :, None, :]
    error = torch.sqrt(((local_pred - local_true) ** 2).sum(dim=-1) + 1e-4).clamp(max=10.0) / 10.0
    joint_mask = mask[:, :, None] * mask[:, None, :]
    expected = (error * joint_mask).sum(dim=(-1, -2)) / joint_mask.sum(dim=(-1, -2)).clamp(min=1.0)

    assert torch.allclose(loss, expected, atol=1e-6)


def test_tm_score_loss_uniform_logits_match_log_nbins_when_prediction_matches_truth():
    """Supplement 1.9.7: 64-bin cross-entropy on the pairwise aligned error
    e_ij = ||T_i^{-1} x_j − T_i^{true,-1} x_j^{true}||.

    When prediction equals truth, e_ij = 0 everywhere → target bin 0; with
    uniform logits, every distribution is uniform over the 64 bins, so the
    per-pair CE is log(64)."""
    loss_fn = TMScoreLoss(n_bins=64, filter_by_resolution=False)

    b, n = 2, 5
    torch.manual_seed(0)
    R = torch.eye(3).reshape(1, 1, 3, 3).expand(b, n, -1, -1).contiguous()
    t = torch.randn((b, n, 3), dtype=torch.float32)
    pae_pred = torch.zeros((b, n, n, 64))  # uniform softmax

    loss = loss_fn(
        pae_pred,
        predicted_rotations=R,
        predicted_translations=t,
        true_rotations=R,
        true_translations=t,
    )

    assert loss.shape == (b,)
    assert torch.allclose(loss, torch.full((b,), math.log(64.0)), atol=1e-5)


def test_tm_score_loss_bucketizes_pairwise_error_into_correct_bin():
    """Supplement 1.9.7 discretises e_ij into bins of 0.5 Å. A 5 Å shift
    between predicted and true Cα places residue pair (0, 1) in bin 10
    ([5.0, 5.5) Å). A one-hot logit at that bin should give near-zero CE
    for the pair — and only for that pair."""
    loss_fn = TMScoreLoss(n_bins=64, filter_by_resolution=False)

    b, n = 1, 2
    R = torch.eye(3).reshape(1, 1, 3, 3).expand(b, n, -1, -1).contiguous()
    pred_t = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]], dtype=torch.float32)
    true_t = torch.zeros_like(pred_t)

    # Make bin 10 one-hot for pair (0, 1); all other pairs use uniform logits.
    pae_pred = torch.zeros((b, n, n, 64))
    pae_pred[0, 0, 1, 10] = 1e9

    # By symmetry of translations, pair (1, 0) has the same |t_j − t_i| = 5;
    # give it the same spike so only the diagonal pairs incur log(n_bins).
    pae_pred[0, 1, 0, 10] = 1e9

    loss = loss_fn(
        pae_pred,
        predicted_rotations=R,
        predicted_translations=pred_t,
        true_rotations=R,
        true_translations=true_t,
    )

    # Diagonals (0, 0) and (1, 1): e_ii = 0 → bin 0, uniform logits → CE = log(64).
    # Off-diagonals (0, 1) and (1, 0): correct bin → CE ≈ 0.
    expected = torch.tensor([2.0 * math.log(64.0) / 4.0])
    assert torch.allclose(loss, expected, atol=1e-4)


def test_select_best_atom14_ground_truth_uses_pairwise_distance_rule():
    predicted = torch.zeros((1, 2, 14, 3), dtype=torch.float32)
    true_positions = torch.zeros_like(predicted)
    true_alt_positions = torch.zeros_like(predicted)
    true_mask = torch.zeros((1, 2, 14), dtype=torch.float32)
    true_alt_mask = torch.zeros_like(true_mask)
    true_atom_is_ambiguous = torch.zeros_like(true_mask)

    predicted[0, 0, 6] = torch.tensor([10.0, 0.0, 0.0])
    predicted[0, 0, 7] = torch.tensor([0.0, 0.0, 0.0])
    predicted[0, 1, 0] = torch.tensor([10.0, 0.0, 0.0])

    true_positions[0, 0, 6] = torch.tensor([0.0, 0.0, 0.0])
    true_positions[0, 0, 7] = torch.tensor([10.0, 0.0, 0.0])
    true_alt_positions[0, 0, 6] = torch.tensor([10.0, 0.0, 0.0])
    true_alt_positions[0, 0, 7] = torch.tensor([0.0, 0.0, 0.0])
    true_positions[0, 1, 0] = torch.tensor([10.0, 0.0, 0.0])
    true_alt_positions[0, 1, 0] = torch.tensor([10.0, 0.0, 0.0])

    true_mask[0, 0, 6:8] = 1.0
    true_alt_mask[0, 0, 6:8] = 1.0
    true_mask[0, 1, 0] = 1.0
    true_alt_mask[0, 1, 0] = 1.0
    true_atom_is_ambiguous[0, 0, 6:8] = 1.0

    chosen_positions, chosen_mask, use_alt = select_best_atom14_ground_truth(
        predicted,
        true_positions,
        true_mask,
        true_alt_positions,
        true_alt_mask,
        true_atom_is_ambiguous,
    )

    assert torch.equal(use_alt, torch.tensor([[1.0, 0.0]]))
    assert torch.equal(chosen_positions[0, 0], true_alt_positions[0, 0])
    assert torch.equal(chosen_mask[0, 0], true_alt_mask[0, 0])
    assert torch.equal(chosen_positions[0, 1], true_positions[0, 1])
    assert torch.equal(chosen_mask[0, 1], true_mask[0, 1])


def test_torsion_angle_loss_scores_all_seven_torsions():
    """Algorithm 27 scores f ∈ {ω, φ, ψ, χ1..χ4} — backbone torsions included."""
    loss_fn = TorsionAngleLoss()

    # Predicted matches true everywhere → baseline torsion loss is 0.
    predicted = torch.zeros((1, 1, 7, 2), dtype=torch.float32)
    predicted[..., :, 1] = 1.0  # unit [sin=0, cos=1] for every angle
    true_angles = predicted.clone()
    true_alt = true_angles.clone()

    torsion_mask = torch.ones((1, 1, 7), dtype=torch.float32)
    baseline_loss = loss_fn(predicted, predicted, true_angles, true_alt, torsion_mask)

    # Perturbing a backbone torsion (ω) must increase the loss per Algorithm 27.
    changed_omega = predicted.clone()
    changed_omega[..., 0, :] = torch.tensor([1.0, 0.0])
    changed_omega_loss = loss_fn(changed_omega, changed_omega, true_angles, true_alt, torsion_mask)
    assert changed_omega_loss.item() > baseline_loss.item()

    # Perturbing a side-chain torsion (χ1) must also increase the loss.
    changed_chi = predicted.clone()
    changed_chi[..., 3, :] = torch.tensor([1.0, 0.0])
    changed_chi_loss = loss_fn(changed_chi, changed_chi, true_angles, true_alt, torsion_mask)
    assert changed_chi_loss.item() > baseline_loss.item()

    # A perturbation to a torsion whose mask is 0 must be ignored — torsion_mask
    # is how geometry.torsion_angles propagates residue-type and atom-presence
    # validity (e.g. χ2 for SER or ω at the first residue is zero).
    masked_out = torsion_mask.clone()
    masked_out[..., 0] = 0.0
    changed_omega_masked_loss = loss_fn(changed_omega, changed_omega, true_angles, true_alt, masked_out)
    aligned_masked_loss = loss_fn(predicted, predicted, true_angles, true_alt, masked_out)
    assert torch.allclose(changed_omega_masked_loss, aligned_masked_loss, atol=1e-5)


def test_structural_violation_loss_respects_residue_index_gaps():
    loss_fn = StructuralViolationLoss()
    positions = torch.zeros((1, 2, 14, 3), dtype=torch.float32)
    mask = torch.zeros((1, 2, 14), dtype=torch.float32)
    residue_types = torch.zeros((1, 2), dtype=torch.long)

    positions[0, 0, 1] = torch.tensor([0.0, 0.0, 0.0])
    positions[0, 0, 2] = torch.tensor([1.5, 0.0, 0.0])
    positions[0, 1, 0] = torch.tensor([8.0, 0.0, 0.0])
    positions[0, 1, 1] = torch.tensor([9.3, 0.0, 0.0])
    mask[0, 0, 1:3] = 1.0
    mask[0, 1, 0:2] = 1.0

    adjacent_index = torch.tensor([[0, 1]], dtype=torch.long)
    gapped_index = torch.tensor([[0, 5]], dtype=torch.long)

    adjacent_loss = loss_fn(positions, mask, residue_types, adjacent_index)
    gapped_loss = loss_fn(positions, mask, residue_types, gapped_index)

    assert adjacent_loss.item() > gapped_loss.item()


def test_between_residue_bond_loss_uses_flat_bottom_tolerance():
    loss_fn = StructuralViolationLoss()
    positions = torch.zeros((1, 2, 14, 3), dtype=torch.float32)
    mask = torch.zeros((1, 2, 14), dtype=torch.float32)
    residue_types = torch.zeros((1, 2), dtype=torch.long)
    residue_index = torch.tensor([[0, 1]], dtype=torch.long)

    positions[0, 0, 2] = torch.tensor([0.0, 0.0, 0.0])   # C(i)
    positions[0, 1, 0] = torch.tensor([1.43, 0.0, 0.0])  # N(i+1), near ideal
    mask[0, 0, 2] = 1.0
    mask[0, 1, 0] = 1.0

    small_losses = loss_fn.between_residue_bond_and_angle_loss(
        positions,
        mask,
        residue_types,
        residue_index,
    )

    positions[0, 1, 0] = torch.tensor([2.5, 0.0, 0.0])   # Large bond violation
    large_losses = loss_fn.between_residue_bond_and_angle_loss(
        positions,
        mask,
        residue_types,
        residue_index,
    )

    assert small_losses["c_n_loss_mean"].item() < 0.1
    assert large_losses["c_n_loss_mean"].item() > small_losses["c_n_loss_mean"].item()
    assert small_losses["ca_c_n_loss_mean"].item() == 0.0
    assert large_losses["ca_c_n_loss_mean"].item() == 0.0


def test_structural_violation_loss_matches_canonical_clash_aggregation():
    loss_fn = StructuralViolationLoss()
    positions = torch.zeros((1, 2, 14, 3), dtype=torch.float32)
    mask = torch.zeros((1, 2, 14), dtype=torch.float32)
    residue_types = torch.zeros((1, 2), dtype=torch.long)
    residue_index = torch.tensor([[0, 1]], dtype=torch.long)

    positions[0, 0, 1] = torch.tensor([0.0, 0.0, 0.0])
    positions[0, 0, 2] = torch.tensor([1.5, 0.0, 0.0])
    positions[0, 1, 0] = torch.tensor([2.8, 0.0, 0.0])
    positions[0, 1, 1] = torch.tensor([4.1, 0.0, 0.0])
    mask[0, 0, 1:3] = 1.0
    mask[0, 1, 0:2] = 1.0

    connection = loss_fn.between_residue_bond_and_angle_loss(positions, mask, residue_types, residue_index)
    clashes = loss_fn.between_residue_clash_loss(positions, mask, residue_types, residue_index)
    within = loss_fn.within_residue_violation_loss(positions, mask, residue_types)
    num_atoms = mask.sum(dim=(1, 2)).clamp(min=1e-6)
    expected = (
        connection["c_n_loss_mean"]
        + connection["ca_c_n_loss_mean"]
        + connection["c_n_ca_loss_mean"]
        + (clashes["per_atom_loss_sum"] + within["per_atom_loss_sum"]).sum(dim=(1, 2)) / num_atoms
    )

    loss = loss_fn(positions, mask, residue_types, residue_index)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_structural_violation_loss_uses_runtime_within_residue_tolerance():
    strict_loss = StructuralViolationLoss(violation_tolerance_factor=12.0)
    loose_loss = StructuralViolationLoss(violation_tolerance_factor=15.0)

    positions = torch.zeros((1, 1, 14, 3), dtype=torch.float32)
    mask = torch.zeros((1, 1, 14), dtype=torch.float32)
    residue_types = torch.zeros((1, 1), dtype=torch.long)

    positions[0, 0, 0] = torch.tensor([0.0, 0.0, 0.0])
    positions[0, 0, 1] = torch.tensor([1.73, 0.0, 0.0])
    mask[0, 0, 0] = 1.0
    mask[0, 0, 1] = 1.0

    strict_within = strict_loss.within_residue_violation_loss(positions, mask, residue_types)
    loose_within = loose_loss.within_residue_violation_loss(positions, mask, residue_types)

    assert strict_within["per_atom_loss_sum"].sum().item() > 0.0
    assert torch.allclose(loose_within["per_atom_loss_sum"], torch.zeros_like(loose_within["per_atom_loss_sum"]))


def test_experimentally_resolved_loss_masks_nonexistent_atom37_logits():
    loss_fn = ExperimentallyResolvedLoss()
    logits = torch.zeros((1, 1, 37), dtype=torch.float32)
    targets = torch.zeros_like(logits)
    atom37_exists = torch.zeros_like(logits)
    atom37_exists[..., 0] = 1.0
    atom37_exists[..., 1] = 1.0

    loss = loss_fn(logits, targets, atom37_exists)

    assert torch.allclose(loss, torch.full((1,), math.log(2.0)), atol=1e-6)


def test_experimentally_resolved_loss_can_filter_by_resolution():
    loss_fn = ExperimentallyResolvedLoss(filter_by_resolution=True, min_resolution=0.1, max_resolution=3.0)
    logits = torch.zeros((1, 1, 37), dtype=torch.float32)
    targets = torch.zeros_like(logits)
    atom37_exists = torch.zeros_like(logits)
    atom37_exists[..., 0] = 1.0
    atom37_exists[..., 1] = 1.0

    in_range = loss_fn(logits, targets, atom37_exists, resolution=torch.tensor([1.5]))
    out_of_range = loss_fn(logits, targets, atom37_exists, resolution=torch.tensor([10.0]))

    assert torch.allclose(in_range, torch.full((1,), math.log(2.0)), atol=1e-6)
    assert torch.allclose(out_of_range, torch.zeros((1,), dtype=torch.float32), atol=1e-6)


def test_plddt_loss_can_filter_by_resolution():
    loss_fn = PLDDTLoss(filter_by_resolution=True, min_resolution=0.1, max_resolution=3.0)
    logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    targets = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    seq_mask = torch.ones((1, 2), dtype=torch.float32)

    in_range = loss_fn(logits, targets, seq_mask=seq_mask, resolution=torch.tensor([1.5]))
    out_of_range = loss_fn(logits, targets, seq_mask=seq_mask, resolution=torch.tensor([10.0]))

    assert torch.allclose(in_range, torch.full((1,), math.log(3.0)), atol=1e-6)
    assert torch.allclose(out_of_range, torch.zeros((1,), dtype=torch.float32), atol=1e-6)


def test_atom14_distance_bounds_use_canonical_stereochemistry():
    ala_index = 0
    n_index = 0
    ca_index = 1
    c_index = 2
    o_index = 3

    assert math.isclose(
        float(restype_atom14_distance_lower_bound[ala_index, n_index, ca_index]),
        1.459 - 15.0 * 0.020,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(restype_atom14_distance_upper_bound[ala_index, n_index, ca_index]),
        1.459 + 15.0 * 0.020,
        rel_tol=0.0,
        abs_tol=1e-6,
    )

    angle = math.radians(111.0)
    bond1 = 1.459
    bond2 = 1.525
    expected_length = math.sqrt(bond1 ** 2 + bond2 ** 2 - 2.0 * bond1 * bond2 * math.cos(angle))
    dl_outer = 0.5 / expected_length
    angle_stddev = math.radians(2.7)
    dl_dgamma = (2.0 * bond1 * bond2 * math.sin(angle)) * dl_outer
    dl_db1 = (2.0 * bond1 - 2.0 * bond2 * math.cos(angle)) * dl_outer
    dl_db2 = (2.0 * bond2 - 2.0 * bond1 * math.cos(angle)) * dl_outer
    expected_stddev = math.sqrt(
        (dl_dgamma * angle_stddev) ** 2
        + (dl_db1 * 0.020) ** 2
        + (dl_db2 * 0.026) ** 2
    )
    assert math.isclose(
        float(restype_atom14_distance_lower_bound[ala_index, n_index, c_index]),
        expected_length - 15.0 * expected_stddev,
        rel_tol=0.0,
        abs_tol=1e-4,
    )
    assert math.isclose(
        float(restype_atom14_distance_lower_bound[ala_index, n_index, o_index]),
        1.55 + 1.52 - 1.5,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_alphafold_loss_matches_paper_top_level_weights():
    pretrain = AlphaFoldLoss(finetune=False)
    finetune = AlphaFoldLoss(finetune=True)

    assert pretrain.sidechain_weight_frac == 0.5
    assert pretrain.distogram_weight == 0.3
    assert pretrain.msa_weight == 2.0
    assert pretrain.confidence_weight == 0.01
    assert finetune.experimentally_resolved_weight == 0.01
    assert finetune.structural_violation_weight == 1.0
    assert finetune.tm_score_weight == 0.1  # supplement 1.9.7 (paragraph after eq 38)
