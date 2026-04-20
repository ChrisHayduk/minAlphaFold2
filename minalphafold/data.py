from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .a3m import GAP_ID, MASK_ID, MSA_ALPHABET_SIZE, SEQ_ALPHABET_SIZE
    from .geometry import (
        atom14_to_rigid_group_frames,
        alternative_atom14_ground_truth,
        alternative_torsion_angles,
        backbone_frames,
        pseudo_beta_positions,
        torsion_angles,
    )
    from .residue_constants import STANDARD_ATOM_MASK, atom_type_num, restype_atom14_to_atom37
except ImportError:  # pragma: no cover - compatibility for direct module imports in tests/scripts.
    from a3m import GAP_ID, MASK_ID, MSA_ALPHABET_SIZE, SEQ_ALPHABET_SIZE
    from geometry import (
        atom14_to_rigid_group_frames,
        alternative_atom14_ground_truth,
        alternative_torsion_angles,
        backbone_frames,
        pseudo_beta_positions,
        torsion_angles,
    )
    from residue_constants import STANDARD_ATOM_MASK, atom_type_num, restype_atom14_to_atom37
TEMPLATE_PAIR_BINS = 39
TEMPLATE_PAIR_DIM = 88
TEMPLATE_ANGLE_DIM = 57
TARGET_FEAT_DIM = SEQ_ALPHABET_SIZE + 1
HHBLITS_AA_ALPHABET_SIZE = GAP_ID + 1
MASKED_MSA_PROFILE_PROB = 0.1
MASKED_MSA_SAME_PROB = 0.1
MASKED_MSA_UNIFORM_PROB = 0.1
USE_TEMPLATE_UNIT_VECTOR = True


def _example_seed(base_seed: int, example_index: int) -> int:
    return base_seed + example_index


def _make_torch_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _torch_randint(low: int, high: int, shape: Sequence[int], generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.randint(low, high, tuple(shape))
    return torch.randint(low, high, tuple(shape), generator=generator)


def _torch_randperm(length: int, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.randperm(length)
    return torch.randperm(length, generator=generator)


def _torch_rand(shape: Sequence[int], generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
    if generator is None:
        return torch.rand(tuple(shape), device=device)
    return torch.rand(tuple(shape), generator=generator, device=device)


def transformed_deletions(deletions: torch.Tensor) -> torch.Tensor:
    return torch.atan(deletions.float() / 3.0) * (2.0 / math.pi)


def hhblits_profile(msa: torch.Tensor) -> torch.Tensor:
    msa_one_hot = F.one_hot(
        msa.clamp(min=0, max=HHBLITS_AA_ALPHABET_SIZE - 1),
        num_classes=HHBLITS_AA_ALPHABET_SIZE,
    )
    return msa_one_hot.float().mean(dim=0)


def discover_chain_ids(processed_features_dir: str | Path, processed_labels_dir: str | Path | None = None) -> List[str]:
    feature_dir = Path(processed_features_dir)
    label_dir = Path(processed_labels_dir) if processed_labels_dir is not None else None

    chain_ids = sorted(path.stem for path in feature_dir.glob("*.npz"))
    if label_dir is None:
        return chain_ids

    return [chain_id for chain_id in chain_ids if (label_dir / f"{chain_id}.npz").exists()]


def split_chain_ids(chain_ids: Sequence[str], split: str, val_fraction: float, seed: int) -> List[str]:
    if split == "all":
        return list(chain_ids)

    shuffled = list(chain_ids)
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    val_ids = set(shuffled[:n_val])

    if split == "val":
        return [chain_id for chain_id in chain_ids if chain_id in val_ids]
    if split == "train":
        return [chain_id for chain_id in chain_ids if chain_id not in val_ids]

    raise ValueError(f"Unsupported split {split!r}. Expected 'train', 'val', or 'all'.")


class ProcessedOpenProteinSetDataset(Dataset):
    def __init__(
        self,
        processed_features_dir: str | Path,
        processed_labels_dir: str | Path,
        *,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 0,
    ):
        self.processed_features_dir = Path(processed_features_dir)
        self.processed_labels_dir = Path(processed_labels_dir)
        chain_ids = discover_chain_ids(self.processed_features_dir, self.processed_labels_dir)
        self.chain_ids = split_chain_ids(chain_ids, split=split, val_fraction=val_fraction, seed=seed)
        self.split = split

    def __len__(self) -> int:
        return len(self.chain_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        chain_id = self.chain_ids[index]

        with np.load(self.processed_features_dir / f"{chain_id}.npz") as feature_data:
            aatype = torch.from_numpy(feature_data["aatype"]).long()
            msa = torch.from_numpy(feature_data["msa"]).long()
            deletions = torch.from_numpy(feature_data["deletions"]).long()
            if "between_segment_residues" in feature_data.files:
                between_segment_residues = torch.from_numpy(feature_data["between_segment_residues"]).long()
            else:
                between_segment_residues = torch.zeros_like(aatype)
            if "residue_index" in feature_data.files:
                residue_index = torch.from_numpy(feature_data["residue_index"]).long()
            else:
                residue_index = torch.arange(aatype.shape[0], dtype=torch.long)
            template_aatype = torch.from_numpy(feature_data["template_aatype"]).long()
            template_atom14_positions = torch.from_numpy(feature_data["template_atom14_positions"]).float()
            template_atom14_mask = torch.from_numpy(feature_data["template_atom14_mask"]).float()

        with np.load(self.processed_labels_dir / f"{chain_id}.npz") as label_data:
            atom14_positions = torch.from_numpy(label_data["atom14_positions"]).float()
            atom14_mask = torch.from_numpy(label_data["atom14_mask"]).float()
            if "resolution" in label_data.files:
                resolution = torch.as_tensor(label_data["resolution"]).float()
            else:
                resolution = torch.tensor(0.0, dtype=torch.float32)

        return {
            "chain_id": chain_id,
            "aatype": aatype,
            "msa": msa,
            "deletions": deletions,
            "between_segment_residues": between_segment_residues,
            "residue_index": residue_index,
            "template_aatype": template_aatype,
            "template_atom14_positions": template_atom14_positions,
            "template_atom14_mask": template_atom14_mask,
            "atom14_positions": atom14_positions,
            "atom14_mask": atom14_mask,
            "resolution": resolution,
        }


def _crop_start(
    length: int,
    crop_size: int,
    training: bool,
    *,
    torch_generator: torch.Generator | None = None,
) -> int:
    if length <= crop_size:
        return 0
    if training:
        return int(_torch_randint(0, length - crop_size + 1, (1,), torch_generator).item())
    return (length - crop_size) // 2


def crop_example(
    example: Dict[str, Any],
    crop_size: int,
    training: bool,
    *,
    torch_generator: torch.Generator | None = None,
) -> Dict[str, Any]:
    length = int(example["aatype"].shape[0])
    if length <= crop_size:
        cropped = dict(example)
        cropped["crop_start"] = 0
        return cropped

    start = _crop_start(length, crop_size=crop_size, training=training, torch_generator=torch_generator)
    end = start + crop_size

    cropped = dict(example)
    cropped["crop_start"] = start
    cropped["aatype"] = example["aatype"][start:end]
    cropped["msa"] = example["msa"][:, start:end]
    cropped["deletions"] = example["deletions"][:, start:end]
    if "between_segment_residues" in example:
        cropped["between_segment_residues"] = example["between_segment_residues"][start:end]
    if "residue_index" in example:
        cropped["residue_index"] = example["residue_index"][start:end]
    cropped["template_aatype"] = example["template_aatype"][:, start:end]
    cropped["template_atom14_positions"] = example["template_atom14_positions"][:, start:end]
    cropped["template_atom14_mask"] = example["template_atom14_mask"][:, start:end]
    cropped["atom14_positions"] = example["atom14_positions"][start:end]
    cropped["atom14_mask"] = example["atom14_mask"][start:end]
    return cropped


def block_delete_msa(
    msa: torch.Tensor,
    deletions: torch.Tensor,
    training: bool,
    *,
    enabled: bool = True,
    msa_fraction_per_block: float = 0.3,
    randomize_num_blocks: bool = False,
    num_blocks: int = 5,
    torch_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not training or not enabled or msa.shape[0] <= 2:
        return msa, deletions

    non_query = msa.shape[0] - 1
    block_size = int(math.floor(non_query * msa_fraction_per_block))
    if block_size <= 0:
        return msa, deletions

    effective_num_blocks = num_blocks
    if randomize_num_blocks:
        effective_num_blocks = int(_torch_randint(0, num_blocks + 1, (1,), torch_generator).item())
    if effective_num_blocks <= 0:
        return msa, deletions

    keep_mask = torch.ones(msa.shape[0], dtype=torch.bool, device=msa.device)
    for _ in range(effective_num_blocks):
        block_start = int(_torch_randint(1, msa.shape[0], (1,), torch_generator).item())
        block_end = min(msa.shape[0], block_start + block_size)
        keep_mask[block_start:block_end] = False

    keep_mask[0] = True
    return msa[keep_mask], deletions[keep_mask]


def sample_cluster_and_extra(
    msa: torch.Tensor,
    deletions: torch.Tensor,
    msa_depth: int,
    extra_msa_depth: int,
    training: bool,
    *,
    torch_generator: torch.Generator | None = None,
    python_random: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_rows = msa.shape[0]
    if total_rows == 0:
        raise ValueError("MSA must contain at least the query row.")

    remaining = torch.arange(1, total_rows, dtype=torch.long)
    if training and remaining.numel() > 0:
        remaining = remaining[_torch_randperm(remaining.numel(), torch_generator)]

    n_cluster_other = max(0, min(msa_depth - 1, remaining.numel()))
    cluster_indices = torch.cat([torch.zeros(1, dtype=torch.long), remaining[:n_cluster_other]])

    cluster_msa = msa[cluster_indices]
    cluster_deletions = deletions[cluster_indices]

    chosen = set(cluster_indices.tolist())
    extra_candidates = [index for index in range(total_rows) if index not in chosen]
    if training:
        if python_random is None:
            random.shuffle(extra_candidates)
        else:
            python_random.shuffle(extra_candidates)
    extra_candidates = extra_candidates[:extra_msa_depth]

    if extra_candidates:
        extra_index_tensor = torch.tensor(extra_candidates, dtype=torch.long)
        extra_msa = msa[extra_index_tensor]
        extra_deletions = deletions[extra_index_tensor]
    else:
        extra_msa = msa.new_zeros((0, msa.shape[1]))
        extra_deletions = deletions.new_zeros((0, deletions.shape[1]))

    return cluster_msa, cluster_deletions, extra_msa, extra_deletions


def cluster_statistics(
    cluster_msa: torch.Tensor,
    cluster_deletions: torch.Tensor,
    extra_msa: torch.Tensor,
    extra_deletions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_cluster, n_res = cluster_msa.shape
    del n_res  # Length is implicit in the tensors below.
    cluster_profile = F.one_hot(
        cluster_msa.clamp(min=0, max=MSA_ALPHABET_SIZE - 1),
        num_classes=MSA_ALPHABET_SIZE,
    ).float()
    cluster_deletion_mean = cluster_deletions.float()
    counts = torch.ones((n_cluster, 1), dtype=cluster_profile.dtype, device=cluster_profile.device)

    if extra_msa.shape[0] > 0:
        weights = cluster_profile.new_tensor([1.0] * (HHBLITS_AA_ALPHABET_SIZE - 1) + [0.0, 0.0])
        extra_one_hot = F.one_hot(
            extra_msa.clamp(min=0, max=MSA_ALPHABET_SIZE - 1),
            num_classes=MSA_ALPHABET_SIZE,
        ).float()
        agreement = torch.matmul(
            extra_one_hot.reshape(extra_msa.shape[0], -1),
            (cluster_profile * weights).reshape(n_cluster, -1).transpose(0, 1),
        )
        assignments = agreement.argmax(dim=-1)

        cluster_profile.scatter_add_(
            0,
            assignments[:, None, None].expand(-1, extra_msa.shape[1], MSA_ALPHABET_SIZE),
            extra_one_hot,
        )
        cluster_deletion_mean.scatter_add_(
            0,
            assignments[:, None].expand(-1, extra_deletions.shape[1]),
            extra_deletions.float(),
        )
        counts.scatter_add_(
            0,
            assignments[:, None],
            torch.ones((extra_msa.shape[0], 1), dtype=counts.dtype, device=counts.device),
        )

    cluster_profile = cluster_profile / counts.unsqueeze(-1)
    cluster_deletion_mean = cluster_deletion_mean / counts
    return cluster_profile, cluster_deletion_mean


def _sample_categorical(
    probabilities: torch.Tensor,
    *,
    torch_generator: torch.Generator | None = None,
) -> torch.Tensor:
    flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    sampled = torch.multinomial(flat_probabilities, 1, generator=torch_generator).squeeze(-1)
    return sampled.reshape(probabilities.shape[:-1])


def masked_msa_inputs(
    cluster_msa: torch.Tensor,
    hhblits_profile_values: torch.Tensor,
    training: bool,
    mask_probability: float = 0.15,
    *,
    torch_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = F.one_hot(cluster_msa.clamp(min=0, max=MSA_ALPHABET_SIZE - 1), num_classes=MSA_ALPHABET_SIZE).float()
    if not training:
        return cluster_msa.clone(), target, torch.zeros_like(cluster_msa, dtype=torch.float32)

    mask = (_torch_rand(cluster_msa.shape, torch_generator, cluster_msa.device) < mask_probability).float()
    corrupted = cluster_msa.clone()

    mask_token_prob = 1.0 - (
        MASKED_MSA_PROFILE_PROB + MASKED_MSA_SAME_PROB + MASKED_MSA_UNIFORM_PROB
    )
    random_aa = cluster_msa.new_tensor(([0.05] * 20) + [0.0, 0.0], dtype=torch.float32)
    replacement_probs = hhblits_profile_values.unsqueeze(0) * MASKED_MSA_PROFILE_PROB
    replacement_probs = replacement_probs + (
        F.one_hot(
            cluster_msa.clamp(min=0, max=HHBLITS_AA_ALPHABET_SIZE - 1),
            num_classes=HHBLITS_AA_ALPHABET_SIZE,
        ).float()
        * MASKED_MSA_SAME_PROB
    )
    replacement_probs = replacement_probs + (random_aa * MASKED_MSA_UNIFORM_PROB)
    replacement_probs = F.pad(replacement_probs, (0, 1), value=mask_token_prob)
    replacement_probs = replacement_probs / replacement_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    if mask.any():
        corrupted[mask.bool()] = _sample_categorical(
            replacement_probs[mask.bool()],
            torch_generator=torch_generator,
        )
    return corrupted, target, mask


def build_msa_feat(
    masked_cluster_msa: torch.Tensor,
    cluster_deletions: torch.Tensor,
    cluster_profile: torch.Tensor,
    cluster_deletion_mean: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            F.one_hot(masked_cluster_msa.clamp(min=0, max=MSA_ALPHABET_SIZE - 1), num_classes=MSA_ALPHABET_SIZE).float(),
            (cluster_deletions > 0).float().unsqueeze(-1),
            transformed_deletions(cluster_deletions).unsqueeze(-1),
            cluster_profile,
            transformed_deletions(cluster_deletion_mean).unsqueeze(-1),
        ],
        dim=-1,
    )


def build_extra_msa_feat(extra_msa: torch.Tensor, extra_deletions: torch.Tensor) -> torch.Tensor:
    if extra_msa.shape[0] == 0:
        return extra_msa.new_zeros((0, extra_msa.shape[1], 25), dtype=torch.float32)

    return torch.cat(
        [
            F.one_hot(extra_msa.clamp(min=0, max=MSA_ALPHABET_SIZE - 1), num_classes=MSA_ALPHABET_SIZE).float(),
            (extra_deletions > 0).float().unsqueeze(-1),
            transformed_deletions(extra_deletions).unsqueeze(-1),
        ],
        dim=-1,
    )


def build_target_feat(
    aatype: torch.Tensor,
    between_segment_residues: torch.Tensor | None = None,
) -> torch.Tensor:
    if between_segment_residues is None:
        between_segment_residues = torch.zeros_like(aatype)
    has_break = between_segment_residues.float().clamp(min=0.0, max=1.0).unsqueeze(-1)
    aatype_one_hot = F.one_hot(
        aatype.clamp(min=0, max=SEQ_ALPHABET_SIZE - 1),
        num_classes=SEQ_ALPHABET_SIZE,
    ).float()
    return torch.cat([has_break, aatype_one_hot], dim=-1)


def build_atom37_masks(
    aatype: torch.Tensor,
    atom14_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    atom37_exists_table = torch.as_tensor(
        STANDARD_ATOM_MASK,
        device=aatype.device,
        dtype=atom14_mask.dtype,
    )
    atom14_to_atom37_table = torch.as_tensor(
        restype_atom14_to_atom37,
        device=aatype.device,
        dtype=torch.long,
    )

    atom37_exists = atom37_exists_table[aatype.long()]
    atom14_to_atom37_index = atom14_to_atom37_table[aatype.long()]
    valid_atom14 = atom14_to_atom37_index >= 0
    atom37_index = atom14_to_atom37_index.clamp(min=0)

    atom37_resolved = atom14_mask.new_zeros((*atom14_mask.shape[:-1], atom_type_num))
    atom37_resolved.scatter_add_(
        dim=-1,
        index=atom37_index,
        src=atom14_mask * valid_atom14.to(atom14_mask.dtype),
    )
    return atom37_exists, atom37_resolved


def _pair_distance_bins(distances: torch.Tensor, num_bins: int = TEMPLATE_PAIR_BINS) -> torch.Tensor:
    bin_edges = torch.linspace(3.25, 50.75, num_bins - 1, device=distances.device, dtype=distances.dtype)
    indices = torch.bucketize(distances, bin_edges)
    return F.one_hot(indices, num_classes=num_bins).float()


def build_template_pair_feat(
    template_aatype: torch.Tensor,
    template_atom14_positions: torch.Tensor,
    template_atom14_mask: torch.Tensor,
) -> torch.Tensor:
    if template_aatype.shape[0] == 0:
        length = template_aatype.shape[1]
        return template_atom14_positions.new_zeros((0, length, length, TEMPLATE_PAIR_DIM))

    pseudo_beta, pseudo_beta_mask = pseudo_beta_positions(template_atom14_positions, template_atom14_mask, template_aatype)
    rotations, translations, frame_mask = backbone_frames(
        template_atom14_positions,
        template_atom14_mask,
        template_aatype,
    )

    pair_mask = pseudo_beta_mask[:, :, None] * pseudo_beta_mask[:, None, :]
    distances = torch.cdist(pseudo_beta, pseudo_beta)
    distogram = _pair_distance_bins(distances) * pair_mask[..., None]

    template_one_hot = F.one_hot(template_aatype.clamp(min=0, max=21), num_classes=22).float()
    aatype_i = template_one_hot[:, :, None, :].expand(-1, -1, template_aatype.shape[1], -1)
    aatype_j = template_one_hot[:, None, :, :].expand(-1, template_aatype.shape[1], -1, -1)

    ca_positions = template_atom14_positions[..., 1, :]
    delta = ca_positions[:, None, :, :] - translations[:, :, None, :]
    local_vectors = torch.einsum("tlia,tlja->tlji", rotations.transpose(-1, -2), delta)
    local_unit_vectors = local_vectors / torch.sqrt(torch.sum(local_vectors ** 2, dim=-1, keepdim=True) + 1e-8)
    backbone_pair_mask = frame_mask[:, :, None] * frame_mask[:, None, :]
    if not USE_TEMPLATE_UNIT_VECTOR:
        local_unit_vectors = torch.zeros_like(local_unit_vectors)
    local_unit_vectors = local_unit_vectors * backbone_pair_mask[..., None]

    return torch.cat(
        [
            distogram,
            pair_mask[..., None],
            aatype_i,
            aatype_j,
            local_unit_vectors,
            backbone_pair_mask[..., None],
        ],
        dim=-1,
    )


def build_template_angle_feat(
    template_aatype: torch.Tensor,
    template_atom14_positions: torch.Tensor,
    template_atom14_mask: torch.Tensor,
) -> torch.Tensor:
    if template_aatype.shape[0] == 0:
        length = template_aatype.shape[1]
        return template_atom14_positions.new_zeros((0, length, TEMPLATE_ANGLE_DIM))

    torsion_values, torsion_mask = torsion_angles(template_atom14_positions, template_atom14_mask, template_aatype)
    torsion_alt = alternative_torsion_angles(torsion_values, template_aatype)
    one_hot = F.one_hot(template_aatype.clamp(min=0, max=21), num_classes=22).float()

    return torch.cat(
        [
            one_hot,
            torsion_values.reshape(template_aatype.shape[0], template_aatype.shape[1], -1),
            torsion_alt.reshape(template_aatype.shape[0], template_aatype.shape[1], -1),
            torsion_mask,
        ],
        dim=-1,
    )


def build_supervision(
    aatype: torch.Tensor,
    atom14_positions: torch.Tensor,
    atom14_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    (
        true_rigid_group_frames_R,
        true_rigid_group_frames_t,
        true_rigid_group_exists,
        true_rigid_group_frames_R_alt,
        true_rigid_group_frames_t_alt,
    ) = atom14_to_rigid_group_frames(atom14_positions, atom14_mask, aatype)
    true_rotations = true_rigid_group_frames_R[:, 0]
    true_translations = true_rigid_group_frames_t[:, 0]
    backbone_mask = true_rigid_group_exists[:, 0]
    true_torsion_angles, true_torsion_mask = torsion_angles(atom14_positions, atom14_mask, aatype)
    true_torsion_angles_alt = alternative_torsion_angles(true_torsion_angles, aatype)
    pseudo_beta, pseudo_beta_mask = pseudo_beta_positions(atom14_positions, atom14_mask, aatype)
    true_atom_positions_alt, true_atom_mask_alt, true_atom_is_ambiguous = alternative_atom14_ground_truth(
        aatype,
        atom14_positions,
        atom14_mask,
    )
    atom37_exists, experimentally_resolved_true = build_atom37_masks(aatype, atom14_mask)

    return {
        "true_rotations": true_rotations,
        "true_translations": true_translations,
        "true_atom_positions": atom14_positions,
        "true_atom_mask": atom14_mask,
        "true_atom_positions_alt": true_atom_positions_alt,
        "true_atom_mask_alt": true_atom_mask_alt,
        "true_atom_is_ambiguous": true_atom_is_ambiguous,
        "true_torsion_angles": true_torsion_angles,
        "true_torsion_angles_alt": true_torsion_angles_alt,
        "true_torsion_mask": true_torsion_mask,
        "true_rigid_group_frames_R": true_rigid_group_frames_R,
        "true_rigid_group_frames_t": true_rigid_group_frames_t,
        "true_rigid_group_frames_R_alt": true_rigid_group_frames_R_alt,
        "true_rigid_group_frames_t_alt": true_rigid_group_frames_t_alt,
        "true_rigid_group_exists": true_rigid_group_exists,
        "atom37_exists": atom37_exists,
        "experimentally_resolved_true": experimentally_resolved_true,
        "res_types": aatype,
        "backbone_mask": backbone_mask,
        "pseudo_beta_mask": pseudo_beta_mask,
        "pseudo_beta_positions": pseudo_beta,
    }


def pad_tensor(tensor: torch.Tensor, target_shape: Sequence[int], value: float = 0.0) -> torch.Tensor:
    output = tensor.new_full(tuple(target_shape), value)
    slices = tuple(slice(0, size) for size in tensor.shape)
    output[slices] = tensor
    return output


def build_msa_features(
    cropped: Dict[str, Any],
    *,
    msa_depth: int,
    extra_msa_depth: int,
    training: bool,
    block_delete_training_msa: bool = True,
    block_delete_msa_fraction: float = 0.3,
    block_delete_msa_randomize_num_blocks: bool = False,
    block_delete_msa_num_blocks: int = 5,
    masked_msa_probability: float = 0.15,
    random_seed: int | None = None,
) -> Dict[str, Any]:
    torch_generator = _make_torch_generator(random_seed)
    python_random = random.Random(random_seed) if random_seed is not None else None
    msa_profile = hhblits_profile(cropped["msa"])

    msa, deletions = block_delete_msa(
        cropped["msa"],
        cropped["deletions"],
        training=training,
        enabled=block_delete_training_msa,
        msa_fraction_per_block=block_delete_msa_fraction,
        randomize_num_blocks=block_delete_msa_randomize_num_blocks,
        num_blocks=block_delete_msa_num_blocks,
        torch_generator=torch_generator,
    )
    cluster_msa, cluster_deletions, extra_msa, extra_deletions = sample_cluster_and_extra(
        msa,
        deletions,
        msa_depth=msa_depth,
        extra_msa_depth=extra_msa_depth,
        training=training,
        torch_generator=torch_generator,
        python_random=python_random,
    )

    masked_cluster_msa, masked_msa_target, masked_msa_mask = masked_msa_inputs(
        cluster_msa,
        msa_profile,
        training=training,
        mask_probability=masked_msa_probability,
        torch_generator=torch_generator,
    )
    cluster_profile, cluster_deletion_mean = cluster_statistics(
        masked_cluster_msa,
        cluster_deletions,
        extra_msa,
        extra_deletions,
    )
    return {
        "msa_feat": build_msa_feat(masked_cluster_msa, cluster_deletions, cluster_profile, cluster_deletion_mean),
        "extra_msa_feat": build_extra_msa_feat(extra_msa, extra_deletions),
        "msa_mask": torch.ones(cluster_msa.shape, dtype=torch.float32),
        "extra_msa_mask": torch.ones(extra_msa.shape, dtype=torch.float32),
        "masked_msa_target": masked_msa_target,
        "masked_msa_mask": masked_msa_mask.float(),
    }


def build_processed_example(
    example: Dict[str, Any],
    *,
    crop_size: int,
    msa_depth: int,
    extra_msa_depth: int,
    max_templates: int,
    training: bool,
    block_delete_training_msa: bool = True,
    block_delete_msa_fraction: float = 0.3,
    block_delete_msa_randomize_num_blocks: bool = False,
    block_delete_msa_num_blocks: int = 5,
    masked_msa_probability: float = 0.15,
    random_seed: int | None = None,
) -> Dict[str, Any]:
    torch_generator = _make_torch_generator(random_seed)
    cropped = crop_example(example, crop_size=crop_size, training=training, torch_generator=torch_generator)
    return build_processed_example_from_cropped(
        cropped,
        msa_depth=msa_depth,
        extra_msa_depth=extra_msa_depth,
        max_templates=max_templates,
        training=training,
        block_delete_training_msa=block_delete_training_msa,
        block_delete_msa_fraction=block_delete_msa_fraction,
        block_delete_msa_randomize_num_blocks=block_delete_msa_randomize_num_blocks,
        block_delete_msa_num_blocks=block_delete_msa_num_blocks,
        masked_msa_probability=masked_msa_probability,
        random_seed=random_seed,
    )


def build_processed_example_from_cropped(
    example: Dict[str, Any],
    *,
    msa_depth: int,
    extra_msa_depth: int,
    max_templates: int,
    training: bool,
    block_delete_training_msa: bool = True,
    block_delete_msa_fraction: float = 0.3,
    block_delete_msa_randomize_num_blocks: bool = False,
    block_delete_msa_num_blocks: int = 5,
    masked_msa_probability: float = 0.15,
    random_seed: int | None = None,
) -> Dict[str, Any]:
    residue_index = example.get("residue_index")
    if residue_index is None:
        residue_index = torch.arange(
            example["crop_start"],
            example["crop_start"] + example["aatype"].shape[0],
            dtype=torch.long,
        )

    template_aatype = example["template_aatype"][:max_templates]
    template_positions = example["template_atom14_positions"][:max_templates]
    template_atom14_mask = example["template_atom14_mask"][:max_templates]
    template_residue_mask = template_atom14_mask.amax(dim=-1)
    template_mask = (template_residue_mask.sum(dim=-1) > 0).float()

    processed = {
        "chain_id": example["chain_id"],
        "aatype": example["aatype"],
        "resolution": torch.as_tensor(example.get("resolution", 0.0)).float(),
        "target_feat": build_target_feat(
            example["aatype"],
            example.get("between_segment_residues"),
        ),
        "residue_index": residue_index.long(),
        "template_pair_feat": build_template_pair_feat(template_aatype, template_positions, template_atom14_mask),
        "template_angle_feat": build_template_angle_feat(template_aatype, template_positions, template_atom14_mask),
        "template_mask": template_mask,
        "template_residue_mask": template_residue_mask,
        "seq_mask": torch.ones(example["aatype"].shape[0], dtype=torch.float32),
    }
    processed.update(
        build_msa_features(
            example,
            msa_depth=msa_depth,
            extra_msa_depth=extra_msa_depth,
            training=training,
            block_delete_training_msa=block_delete_training_msa,
            block_delete_msa_fraction=block_delete_msa_fraction,
            block_delete_msa_randomize_num_blocks=block_delete_msa_randomize_num_blocks,
            block_delete_msa_num_blocks=block_delete_msa_num_blocks,
            masked_msa_probability=masked_msa_probability,
            random_seed=random_seed,
        )
    )
    processed.update(build_supervision(example["aatype"], example["atom14_positions"], example["atom14_mask"]))
    return processed


def collate_batch(
    examples: List[Dict[str, Any]],
    *,
    crop_size: int,
    msa_depth: int,
    extra_msa_depth: int,
    max_templates: int,
    training: bool,
    block_delete_training_msa: bool = True,
    block_delete_msa_fraction: float = 0.3,
    block_delete_msa_randomize_num_blocks: bool = False,
    block_delete_msa_num_blocks: int = 5,
    masked_msa_probability: float = 0.15,
    random_seed: int | None = None,
    num_recycling_samples: int = 1,
    num_ensemble_samples: int = 1,
) -> Dict[str, Any]:
    cropped_examples = [
        crop_example(
            example,
            crop_size=crop_size,
            training=training,
            torch_generator=_make_torch_generator(None if random_seed is None else _example_seed(random_seed, index)),
        )
        for index, example in enumerate(examples)
    ]
    processed = [
        build_processed_example_from_cropped(
            cropped,
            msa_depth=msa_depth,
            extra_msa_depth=extra_msa_depth,
            max_templates=max_templates,
            training=training,
            block_delete_training_msa=block_delete_training_msa,
            block_delete_msa_fraction=block_delete_msa_fraction,
            block_delete_msa_randomize_num_blocks=block_delete_msa_randomize_num_blocks,
            block_delete_msa_num_blocks=block_delete_msa_num_blocks,
            masked_msa_probability=masked_msa_probability,
            random_seed=None if random_seed is None else _example_seed(random_seed, index),
        )
        for index, cropped in enumerate(cropped_examples)
    ]

    sampled_msa_features: list[list[list[Dict[str, Any]]]] = []
    if num_recycling_samples > 1 or num_ensemble_samples > 1:
        for recycle_index in range(num_recycling_samples):
            recycle_samples: list[list[Dict[str, Any]]] = []
            for ensemble_index in range(num_ensemble_samples):
                sample_index = recycle_index * num_ensemble_samples + ensemble_index
                recycle_samples.append(
                    [
                        build_msa_features(
                            cropped,
                            msa_depth=msa_depth,
                            extra_msa_depth=extra_msa_depth,
                            training=training,
                            block_delete_training_msa=block_delete_training_msa,
                            block_delete_msa_fraction=block_delete_msa_fraction,
                            block_delete_msa_randomize_num_blocks=block_delete_msa_randomize_num_blocks,
                            block_delete_msa_num_blocks=block_delete_msa_num_blocks,
                            masked_msa_probability=masked_msa_probability,
                            random_seed=None
                            if random_seed is None
                            else _example_seed(random_seed + 1000 * sample_index, example_index),
                        )
                        for example_index, cropped in enumerate(cropped_examples)
                    ]
                )
            sampled_msa_features.append(recycle_samples)

    max_length = max(item["aatype"].shape[0] for item in processed)
    max_cluster = max(item["msa_feat"].shape[0] for item in processed)
    max_extra = max(item["extra_msa_feat"].shape[0] for item in processed)
    if sampled_msa_features:
        max_cluster = max(
            max_cluster,
            max(
                sample["msa_feat"].shape[0]
                for recycle_samples in sampled_msa_features
                for ensemble_samples in recycle_samples
                for sample in ensemble_samples
            ),
        )
        max_extra = max(
            max_extra,
            max(
                sample["extra_msa_feat"].shape[0]
                for recycle_samples in sampled_msa_features
                for ensemble_samples in recycle_samples
                for sample in ensemble_samples
            ),
        )
    max_templates_in_batch = max(item["template_pair_feat"].shape[0] for item in processed)

    batch: Dict[str, Any] = {"chain_id": [item["chain_id"] for item in processed]}

    def stack(key: str, *, fill_value: float = 0.0, target_shape: Sequence[int]) -> None:
        batch[key] = torch.stack(
            [pad_tensor(item[key], target_shape=target_shape, value=fill_value) for item in processed],
            dim=0,
        )

    stack("aatype", target_shape=(max_length,))
    stack("resolution", target_shape=())
    stack("target_feat", target_shape=(max_length, TARGET_FEAT_DIM))
    stack("residue_index", target_shape=(max_length,))
    stack("seq_mask", target_shape=(max_length,))
    stack("msa_feat", target_shape=(max_cluster, max_length, 49))
    stack("msa_mask", target_shape=(max_cluster, max_length))
    stack("extra_msa_feat", target_shape=(max_extra, max_length, 25))
    stack("extra_msa_mask", target_shape=(max_extra, max_length))
    stack("template_pair_feat", target_shape=(max_templates_in_batch, max_length, max_length, TEMPLATE_PAIR_DIM))
    stack("template_angle_feat", target_shape=(max_templates_in_batch, max_length, TEMPLATE_ANGLE_DIM))
    stack("template_mask", target_shape=(max_templates_in_batch,))
    stack("template_residue_mask", target_shape=(max_templates_in_batch, max_length))
    stack("true_rotations", target_shape=(max_length, 3, 3))
    stack("true_translations", target_shape=(max_length, 3))
    stack("true_atom_positions", target_shape=(max_length, 14, 3))
    stack("true_atom_mask", target_shape=(max_length, 14))
    stack("true_atom_positions_alt", target_shape=(max_length, 14, 3))
    stack("true_atom_mask_alt", target_shape=(max_length, 14))
    stack("true_atom_is_ambiguous", target_shape=(max_length, 14))
    stack("true_torsion_angles", target_shape=(max_length, 7, 2))
    stack("true_torsion_angles_alt", target_shape=(max_length, 7, 2))
    stack("true_torsion_mask", target_shape=(max_length, 7))
    stack("true_rigid_group_frames_R", target_shape=(max_length, 8, 3, 3))
    stack("true_rigid_group_frames_t", target_shape=(max_length, 8, 3))
    stack("true_rigid_group_frames_R_alt", target_shape=(max_length, 8, 3, 3))
    stack("true_rigid_group_frames_t_alt", target_shape=(max_length, 8, 3))
    stack("true_rigid_group_exists", target_shape=(max_length, 8))
    stack("atom37_exists", target_shape=(max_length, atom_type_num))
    stack("experimentally_resolved_true", target_shape=(max_length, atom_type_num))
    stack("res_types", target_shape=(max_length,))
    stack("backbone_mask", target_shape=(max_length,))
    stack("pseudo_beta_mask", target_shape=(max_length,))
    stack("pseudo_beta_positions", target_shape=(max_length, 3))
    stack("masked_msa_target", target_shape=(max_cluster, max_length, MSA_ALPHABET_SIZE))
    stack("masked_msa_mask", target_shape=(max_cluster, max_length))

    if sampled_msa_features:
        def stack_sampled(key: str, *, fill_value: float = 0.0, target_shape: Sequence[int]) -> None:
            recycle_batches = []
            for recycle_samples in sampled_msa_features:
                ensemble_batches = []
                for ensemble_samples in recycle_samples:
                    ensemble_batches.append(
                        torch.stack(
                            [
                                pad_tensor(sample[key], target_shape=target_shape, value=fill_value)
                                for sample in ensemble_samples
                            ],
                            dim=0,
                        )
                    )
                recycle_batches.append(torch.stack(ensemble_batches, dim=0))
            batch[key] = torch.stack(recycle_batches, dim=0)

        stack_sampled("msa_feat", target_shape=(max_cluster, max_length, 49))
        stack_sampled("msa_mask", target_shape=(max_cluster, max_length))
        stack_sampled("extra_msa_feat", target_shape=(max_extra, max_length, 25))
        stack_sampled("extra_msa_mask", target_shape=(max_extra, max_length))
        stack_sampled("masked_msa_target", target_shape=(max_cluster, max_length, MSA_ALPHABET_SIZE))
        stack_sampled("masked_msa_mask", target_shape=(max_cluster, max_length))

    return batch
