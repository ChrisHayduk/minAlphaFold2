"""Auxiliary prediction heads.

Each head is a thin projection from an Evoformer/Structure-Module output
to a per-position logit tensor. All five logit layers are zero-initialised
(supplement 1.11.4 "final weight layers"), so at step 0 every head emits
a uniform distribution and only FAPE carries gradient signal into the
trunk.

Algorithm-table mapping:

* :class:`DistogramHead`           — supplement 1.9.8 eq 41.
* :class:`PLDDTHead`               — Algorithm 29 (supplement 1.9.6).
* :class:`MaskedMSAHead`           — supplement 1.9.9 eq 42.
* :class:`TMScoreHead`             — supplement 1.9.7 eqs 38-40 (PAE/pTM).
* :class:`ExperimentallyResolvedHead` — supplement 1.9.10 eq 43.
"""

import torch


def _zero_init_linear(linear: torch.nn.Linear):
    """Zero-initialize weights and bias of a Linear layer (Supplement 1.11.4)."""
    torch.nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)


class DistogramHead(torch.nn.Module):
    """Distogram head (supplement 1.9.8).

    Projects the pair representation ``z_ij`` to ``n_dist_bins`` bin
    logits and symmetrises across (i, j). The loss target is one-hot
    binned Cβ-Cβ (or Cα for glycine) distance between ground-truth
    residue pairs, cross-entropy averaged (eq 41).
    """

    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(config.c_z, config.n_dist_bins)
        # Supplement 1.11.4: zero-init residue distance prediction logits
        _zero_init_linear(self.linear)

    def forward(self, pair_representation: torch.Tensor):
        # pair_representation: (batch, N_res, N_res, c_z)
        logits = self.linear(pair_representation)  # (batch, N_res, N_res, n_dist_bins)
        # Distograms of an unordered pair are symmetric by definition; average
        # the (i, j) and (j, i) logits so predictions match that invariance.
        logits = (logits + logits.transpose(1, 2)) / 2
        return logits


class PLDDTHead(torch.nn.Module):
    """Per-residue confidence head (Algorithm 29 / supplement 1.9.6).

    Takes the post-Structure-Module single representation ``s_i``, normal-
    ises it, passes it through a two-layer ReLU MLP, and projects to
    ``n_plddt_bins`` (default 50). The scalar pLDDT for residue i is
    ``sum_k p_i^k · v_k`` where ``v_k = k + 0.5`` scaled into [1, 99] —
    that bin-centre transform lives in ``pdbio`` (B-factor write path) and
    ``losses.PLDDTLoss`` (LDDT-Cα supervision).
    """

    def __init__(self, config):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(config.c_s),
            torch.nn.Linear(config.c_s, config.plddt_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(config.plddt_hidden_dim, config.plddt_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(config.plddt_hidden_dim, config.n_plddt_bins),
        )
        # Supplement 1.11.4: zero-init model confidence prediction logits
        final_linear: torch.nn.Linear = self.net[-1]  # type: ignore[assignment]
        _zero_init_linear(final_linear)

    def forward(self, single_representation: torch.Tensor):
        # single_representation: (batch, N_res, c_s)
        return self.net(single_representation)  # (batch, N_res, n_plddt_bins)


class MaskedMSAHead(torch.nn.Module):
    """Masked MSA prediction head (supplement 1.9.9 eq 42).

    Projects each MSA cell ``m_{si}`` to a 23-way class logit (20 AAs +
    unknown + gap + mask). The loss is cross-entropy averaged over the
    positions the MSA-masking step in ``data.masked_msa_inputs`` selected
    (supplement 1.2.7).
    """

    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(config.c_m, config.n_msa_classes)
        # Supplement 1.11.4: zero-init masked MSA prediction logits
        _zero_init_linear(self.linear)

    def forward(self, msa_representation: torch.Tensor):
        # msa_representation: (batch, N_seq, N_res, c_m)
        return self.linear(msa_representation)  # (batch, N_seq, N_res, n_msa_classes)


class TMScoreHead(torch.nn.Module):
    """Predicted aligned error head (supplement 1.9.7).

    Linearly projects the pair representation z_ij to a distribution over
    ``n_pae_bins`` aligned-error bins, used by ``TMScoreLoss`` during
    fine-tuning to produce the pTM estimate of equation 39. The bins are
    non-symmetric: PAE(i, j) != PAE(j, i). Default config uses 64 bins of
    width 0.5 Å covering [0, 31.5 Å] with the final bin open-ended.
    """

    def __init__(self, config):
        super().__init__()
        self.n_pae_bins = config.n_pae_bins  # 64 = supplement 1.9.7 default
        self.linear = torch.nn.Linear(config.c_z, config.n_pae_bins)
        _zero_init_linear(self.linear)

    def forward(self, pair_representation: torch.Tensor):
        # pair_representation: (batch, N_res, N_res, c_z)
        logits = self.linear(pair_representation)  # (batch, N_res, N_res, n_pae_bins)
        # No symmetrization — supplement 1.9.7 defines PAE as non-symmetric.
        return logits


class ExperimentallyResolvedHead(torch.nn.Module):
    """Atom-resolution-status head (supplement 1.9.10 eq 43).

    Per-residue 37-way binary logits (one per atom in the atom37
    ordering) predicting whether each atom was experimentally resolved
    in the target structure. Only trained during fine-tuning, on high-
    resolution X-ray/cryo-EM targets (resolution < 3 Å).
    """

    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(config.c_s, 37)
        _zero_init_linear(self.linear)

    def forward(self, single_representation: torch.Tensor):
        # single_representation: (batch, N_res, c_s)
        return self.linear(single_representation)  # (batch, N_res, 37)
