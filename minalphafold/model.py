import torch
import torch.utils.checkpoint as torch_checkpoint
import math
from typing import cast

from .evoformer import Evoformer
from .structure_module import StructureModule
from .initialization import init_gate_linear, init_linear, zero_linear
from .embedders import InputEmbedder, TemplatePair, TemplatePointwiseAttention, ExtraMsaStack
from .heads import DistogramHead, PLDDTHead, MaskedMSAHead, TMScoreHead, ExperimentallyResolvedHead
from .utils import recycling_distance_bin

class AlphaFold2(torch.nn.Module):
    """Top-level AlphaFold2 model (Algorithm 2).

    Wires together the pipeline described in Supplementary Figure 1 and
    Algorithm 2:

    1. **Input embedding** (Algorithm 3): ``target_feat``/``residue_index``/
       ``msa_feat`` → initial ``m_si`` and ``z_ij``.
    2. **Recycling embedding** (Algorithm 32): ``LayerNorm(m_1i^prev)`` and
       ``LayerNorm(z_ij^prev) + Linear(one_hot(d_ij^prev))`` injected back
       into the first row of ``m_si`` and into ``z_ij``.
    3. **Template embedding** (supplement 1.7.1): template torsion angles are
       concatenated onto ``m_si``; template pair features pass through
       ``TemplatePair`` and are pooled via ``TemplatePointwiseAttention``
       into ``z_ij``.
    4. **Extra MSA stack** (supplement 1.7.2): large-depth auxiliary MSA rep
       updates ``z_ij`` through a shallow Evoformer-like stack.
    5. **Evoformer trunk** (Algorithm 6): the main stack. ``s_i = Linear(m_1i)``
       (Algorithm 6 line 12) is computed after ensemble averaging here.
    6. **Structure Module** (Algorithm 20): iterative IPA + backbone update →
       all-atom coordinates, per-layer frame/torsion trajectories, and the
       post-IPA single representation used by pLDDT.
    7. **Auxiliary heads** (supplement 1.9): distogram (eq 41), masked MSA
       (eq 42), experimentally-resolved (eq 43), pLDDT (Algorithm 29), and
       TM-score / PAE (supplement 1.9.7).

    Recycling (supplement 1.10 / Algorithm 31): the forward pass runs
    ``n_cycles`` times, re-embedding the previous cycle's ``m_1i``, ``z_ij``,
    and pseudo-β positions. During training ``n_cycles`` is sampled uniformly
    from ``{1, ..., n_cycles}`` and only the final cycle carries gradients.

    Ensembling (supplement 1.11.2 / Algorithm 2 lines 4, 18, 20): the
    pre-Structure-Module pipeline is run ``n_ensemble`` times per cycle and
    the averaged ``m̂_1i`` and ``ẑ_ij`` feed the Structure Module and heads.
    Per the supplement, ensembling is only used at inference; training uses
    ``n_ensemble = 1``.
    """

    def __init__(self, config):
        super().__init__()
        self.evoformer_blocks = torch.nn.ModuleList([Evoformer(config) for _ in range(config.num_evoformer)])
        self.structure_model = StructureModule(config)

        self.input_embedder = InputEmbedder(config)

        # Recycling embedders (Algorithm 32): LN-only for single/pair reps; distance bins use a learned linear
        self.recycle_norm_s = torch.nn.LayerNorm(config.c_m)
        self.recycle_norm_z = torch.nn.LayerNorm(config.c_z)
        self.recycle_linear_d = torch.nn.Linear(15, config.c_z)
        init_linear(self.recycle_linear_d, init="default")

        # Project from MSA channel dim (c_m) to single rep dim (c_s)
        self.single_rep_proj = torch.nn.Linear(config.c_m, config.c_s)
        init_linear(self.single_rep_proj, init="default")

        # Template processing
        self.template_pair_feat_linear = torch.nn.Linear(88, config.c_t)
        self.template_pair_stack = TemplatePair(config)
        self.template_pointwise_att = TemplatePointwiseAttention(config)
        init_linear(self.template_pair_feat_linear, init="relu")

        self.template_angle_linear_1 = torch.nn.Linear(57, config.c_m)
        self.template_angle_linear_2 = torch.nn.Linear(config.c_m, config.c_m)
        init_linear(self.template_angle_linear_1, init="relu")
        init_linear(self.template_angle_linear_2, init="relu")

        # Extra MSA processing
        self.extra_msa_feat_linear = torch.nn.Linear(25, config.c_e)
        self.extra_msa_blocks = torch.nn.ModuleList(
            [ExtraMsaStack(config) for _ in range(config.num_extra_msa)]
        )
        init_linear(self.extra_msa_feat_linear, init="default")

        # Prediction heads
        self.distogram_head = DistogramHead(config)
        self.plddt_head = PLDDTHead(config)
        self.masked_msa_head = MaskedMSAHead(config)
        self.tm_score_head = TMScoreHead(config)
        self.experimentally_resolved_head = ExperimentallyResolvedHead(config)

        self.config = config
        self._initialize_alphafold_parameters()

    @staticmethod
    def _zero_linear(linear: torch.nn.Linear):
        zero_linear(linear)

    @staticmethod
    def _init_gate_linear(linear: torch.nn.Linear):
        init_gate_linear(linear)

    def _initialize_alphafold_parameters(self):
        zero_init = bool(getattr(self.config, "zero_init", True))
        output_zero_init_classes = {
            "MSARowAttentionWithPairBias",
            "MSAColumnAttention",
            "MSAColumnGlobalAttention",
            "TemplatePointwiseAttention",
            "ExtraMsaStack",
            "TriangleAttentionStartingNode",
            "TriangleAttentionEndingNode",
            "InvariantPointAttention",
        }
        transition_zero_init_classes = {"MSATransition", "PairTransition"}

        for module in self.modules():
            class_name = module.__class__.__name__

            if hasattr(module, "linear_gate") and isinstance(module.linear_gate, torch.nn.Linear):
                self._init_gate_linear(module.linear_gate)

            if zero_init and class_name in output_zero_init_classes and hasattr(module, "linear_output"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_output))

            if zero_init and class_name in transition_zero_init_classes and hasattr(module, "linear_down"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_down))

            if zero_init and class_name == "OuterProductMean" and hasattr(module, "linear_out"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_out))

            if class_name in {"TriangleMultiplicationOutgoing", "TriangleMultiplicationIncoming"}:
                self._init_gate_linear(cast(torch.nn.Linear, module.gate1))
                self._init_gate_linear(cast(torch.nn.Linear, module.gate2))
                self._init_gate_linear(cast(torch.nn.Linear, module.gate))
                self._zero_linear(cast(torch.nn.Linear, module.out_linear))

            if zero_init and class_name == "StructureModule":
                self._zero_linear(cast(torch.nn.Linear, module.transition_linear_3))

            if zero_init and class_name == "BackboneUpdate":
                self._zero_linear(cast(torch.nn.Linear, module.linear))

            if zero_init and class_name == "AngleResnetBlock" and hasattr(module, "linear_2"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_2))

            if class_name == "InvariantPointAttention":
                # Set head weights so softplus(head_weights) = 1 at init.
                cast(torch.nn.Parameter, module.head_weights).data.fill_(math.log(math.e - 1.0))

    @staticmethod
    def _sampled_feature_slice(
        tensor: torch.Tensor,
        cycle_index: int,
        ensemble_index: int,
        *,
        base_ndim: int,
    ) -> torch.Tensor:
        """Index into a feature tensor by recycling cycle and ensemble sample.

        Supplement 1.11.2 / Algorithm 2 draws ``N_cycle × N_ensemble`` random
        samples of ``msa_feat``/``extra_msa_feat``. The data pipeline may
        materialise those samples ahead of time by prepending ``[cycle,
        ensemble, ...]`` axes to the feature, or just a ``[cycle, ...]`` axis,
        or neither. ``base_ndim`` is the ndim of a *single* sample (e.g. 4 for
        ``msa_feat = (batch, N_seq, N_res, 49)``); this helper strips whichever
        outer sampling axes are present so the model always sees one slice.
        """
        if tensor.ndim == base_ndim + 2:
            return tensor[cycle_index, ensemble_index]
        if tensor.ndim == base_ndim + 1:
            return tensor[cycle_index]
        return tensor

    def forward(
            self,
            target_feat: torch.Tensor,
            residue_index: torch.Tensor,
            msa_feat: torch.Tensor,
            extra_msa_feat: torch.Tensor,
            template_pair_feat: torch.Tensor,
            aatype: torch.Tensor,
            template_angle_feat: torch.Tensor | None = None,
            template_mask: torch.Tensor | None = None,
            template_residue_mask: torch.Tensor | None = None,
            seq_mask: torch.Tensor | None = None,
            msa_mask: torch.Tensor | None = None,
            extra_msa_mask: torch.Tensor | None = None,
            n_cycles: int = 3,
            n_ensemble: int = 1,
            detach_rotations: bool = True,
        ):
        """Algorithm 2 forward pass. See the class docstring for the full map."""
        # seq_mask: (batch, N_res) — 1 for valid residues, 0 for padding
        # msa_mask: (batch, N_seq, N_res) — 1 for valid, 0 for padding
        # extra_msa_mask: (batch, N_extra, N_res) — 1 for valid, 0 for padding
        assert n_ensemble > 0
        assert n_cycles > 0

        if self.training:
            # Algorithm 31 line 1: N' ~ Uniform(1, N_cycle). Only iteration
            # N' carries gradients; earlier iterations are stop-grad'd via the
            # detach() calls at the bottom of the loop (Algorithm 31 line 4).
            n_cycles = int(torch.randint(1, n_cycles + 1, (1,), device=target_feat.device).item())
        self.last_n_cycles = int(n_cycles)
        self.last_n_ensemble = int(n_ensemble)

        outer_grad = torch.is_grad_enabled()

        N_res = target_feat.shape[1]
        c_m = self.config.c_m
        c_z = self.config.c_z
        batch_size = target_feat.shape[0]

        # Default masks: all ones (no padding).
        if seq_mask is None:
            seq_mask = target_feat.new_ones(batch_size, N_res)
        pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]  # (batch, N_res, N_res)
        if msa_mask is None:
            msa_mask = target_feat.new_ones(batch_size, msa_feat.shape[1], N_res)
        if extra_msa_mask is None:
            extra_msa_mask = target_feat.new_ones(batch_size, extra_msa_feat.shape[1], N_res)

        # Algorithm 2 line 1: m̂_1i^prev, ẑ_ij^prev, x̄_i^{prev,Cβ} ← 0, 0, 0.
        single_rep_prev = torch.zeros(batch_size, N_res, c_m, device=msa_feat.device)
        z_prev = torch.zeros(batch_size, N_res, N_res, c_z, device=msa_feat.device)
        x_prev = torch.zeros(batch_size, N_res, 3, device=msa_feat.device)

        # Algorithm 2 line 2: recycling loop.
        for i in range(n_cycles):
            is_last = (i == n_cycles - 1)

            with torch.set_grad_enabled(is_last and outer_grad):
                # Algorithm 2 line 3: zero the ensemble accumulators. Per line 18
                # we accumulate only m_1i (the first MSA row) and z_ij — the
                # single rep s_i = Linear(m_1i) is obtained after averaging via
                # single_rep_proj, which is equivalent by linearity. The full
                # MSA representation is NOT averaged (Algorithm 2 line 18 only
                # averages m_1i, not m_si); the masked MSA head instead consumes
                # the last ensemble sample's full rep. In practice this only
                # matters at inference — supplement 1.11.2 uses N_ensemble = 1
                # at training, and the masked MSA head is training-only.
                single_rep_accum = torch.zeros(batch_size, N_res, c_m, device=msa_feat.device)
                pair_repr_accum = torch.zeros(batch_size, N_res, N_res, c_z, device=msa_feat.device)
                msa_repr_last: torch.Tensor | None = None

                # Algorithm 2 line 4: ensemble loop.
                for ensemble_index in range(n_ensemble):
                    msa_feat_current = self._sampled_feature_slice(msa_feat, i, ensemble_index, base_ndim=4)
                    extra_msa_feat_current = self._sampled_feature_slice(extra_msa_feat, i, ensemble_index, base_ndim=4)
                    msa_mask_current = self._sampled_feature_slice(msa_mask, i, ensemble_index, base_ndim=3)
                    extra_msa_mask_current = self._sampled_feature_slice(extra_msa_mask, i, ensemble_index, base_ndim=3)

                    # Algorithm 2 line 5 (= Algorithm 3 / InputEmbedder).
                    msa_representation, pair_representation = self.input_embedder(
                        target_feat,
                        residue_index,
                        msa_feat_current,
                    )

                    # Algorithm 2 line 6 (= Algorithm 32 / RecyclingEmbedder):
                    #   m_1i  += LayerNorm(m_1i^prev)
                    #   z_ij  += LayerNorm(z_ij^prev) + Linear(one_hot(d_ij^prev))
                    # On the first cycle the prev tensors are zero, so these
                    # additions vanish. Clone before the in-place write to the
                    # first MSA row so the embedder's output tensor is untouched.
                    msa_repr = msa_representation.clone()
                    pair_repr = pair_representation.clone()
                    msa_repr[:, 0, :, :] += self.recycle_norm_s(single_rep_prev)
                    pair_repr += self.recycle_norm_z(z_prev)
                    pair_repr += self.recycle_linear_d(recycling_distance_bin(x_prev, n_bins=15))

                    # Algorithm 2 lines 7-8: template torsion-angle embedding,
                    # concatenated onto the MSA representation as extra rows so
                    # the Evoformer attends over real MSA + template angles
                    # uniformly. msa_mask is extended to cover the new rows.
                    evo_msa_mask = msa_mask_current
                    if template_angle_feat is not None and template_angle_feat.shape[1] > 0:
                        template_angle_repr = self.template_angle_linear_2(
                            torch.relu(self.template_angle_linear_1(template_angle_feat))
                        )
                        msa_repr = torch.cat([msa_repr, template_angle_repr], dim=1)
                        n_templ = template_angle_repr.shape[1]
                        if template_mask is not None:
                            templ_mask = template_mask[:, :, None].to(msa_mask_current.dtype) * seq_mask[:, None, :]
                        else:
                            templ_mask = msa_mask_current.new_ones(batch_size, n_templ, N_res)
                        evo_msa_mask = torch.cat([msa_mask_current, templ_mask], dim=1)

                    # Algorithm 2 lines 9-13: template pair stack + pointwise
                    # attention pool into z_ij. Template pair/angle paths touch
                    # disjoint tensors (z vs m), so the two blocks commute;
                    # we follow the supplement's ordering.
                    if template_pair_feat.shape[1] > 0:
                        template_pair_mask = None
                        if template_residue_mask is not None:
                            template_pair_mask = (
                                template_residue_mask[:, :, :, None]
                                * template_residue_mask[:, :, None, :]
                            )
                        template_pair = self.template_pair_feat_linear(template_pair_feat)
                        template_pair = self.template_pair_stack(template_pair, pair_mask=template_pair_mask)
                        pair_repr = pair_repr + self.template_pointwise_att(
                            template_pair,
                            pair_repr,
                            template_mask=template_mask,
                        )

                    # Algorithm 2 lines 14-16: extra MSA stack updates z_ij
                    # (shallow Evoformer-like blocks, supplement 1.7.2).
                    #
                    # Supplement 1.11.8: "we store the activations that are
                    # passed between the N_block = 48 Evoformer blocks.
                    # During the backward pass, we recompute all activations
                    # within the blocks." That's exactly what
                    # torch.utils.checkpoint does. We apply it only when
                    # gradients are required — during eval the checkpointed
                    # path would just add overhead for no memory benefit.
                    if extra_msa_feat_current.shape[1] > 0:
                        extra_msa_repr = self.extra_msa_feat_linear(extra_msa_feat_current)
                        for extra_block in self.extra_msa_blocks:
                            if self.training:
                                extra_msa_repr, pair_repr = torch_checkpoint.checkpoint(
                                    extra_block,
                                    extra_msa_repr, pair_repr,
                                    extra_msa_mask=extra_msa_mask_current, pair_mask=pair_mask,
                                    use_reentrant=False,
                                )
                            else:
                                extra_msa_repr, pair_repr = extra_block(
                                    extra_msa_repr, pair_repr,
                                    extra_msa_mask=extra_msa_mask_current, pair_mask=pair_mask,
                                )

                    # Algorithm 2 line 17 (= Algorithm 6 / EvoformerStack).
                    for block in self.evoformer_blocks:
                        if self.training:
                            msa_repr, pair_repr = torch_checkpoint.checkpoint(
                                block,
                                msa_repr, pair_repr,
                                msa_mask=evo_msa_mask, pair_mask=pair_mask,
                                use_reentrant=False,
                            )
                        else:
                            msa_repr, pair_repr = block(
                                msa_repr, pair_repr,
                                msa_mask=evo_msa_mask, pair_mask=pair_mask,
                            )

                    # Algorithm 2 line 18: accumulate m_1i and z_ij only.
                    # Keep the last sample's full MSA rep (real-MSA rows only,
                    # dropping the appended template-angle rows) for the
                    # masked MSA head — see the comment above the accumulators.
                    single_rep_accum = single_rep_accum + msa_repr[:, 0, :, :]
                    pair_repr_accum = pair_repr_accum + pair_repr
                    msa_repr_last = msa_repr[:, :msa_feat_current.shape[1], :, :]

                # Algorithm 2 line 20: m̂_1i, ẑ_ij /= N_ensemble.
                msa_first_row = single_rep_accum / n_ensemble
                pair_repr = pair_repr_accum / n_ensemble
                assert msa_repr_last is not None  # n_ensemble > 0 guarantees this
                msa_repr = msa_repr_last

                # Algorithm 6 line 12: s_i = Linear(m_1i). By linearity this is
                # equivalent to averaging s_i itself across ensemble members.
                single_rep = self.single_rep_proj(msa_first_row)

                # Algorithm 2 line 21: StructureModule consumes (ŝ_i, ẑ_ij).
                structure_predictions = self.structure_model(
                    single_rep, pair_repr, aatype,
                    seq_mask=seq_mask, detach_rotations=detach_rotations,
                )

                if is_last:
                    # Auxiliary prediction heads (supplement 1.9). Each head's
                    # input is fixed by the paper:
                    #   - distogram (eq 41): averaged pair rep
                    #   - masked MSA (eq 42): last ensemble's full MSA rep
                    #   - experimentally resolved (1.9.10): Evoformer single
                    #     rep (s_i from Algorithm 6 line 12), NOT the SM single
                    #   - pLDDT (Algorithm 29 line 1): post-IPA single rep
                    #     from the Structure Module
                    #   - TM-score / PAE (1.9.7): averaged pair rep
                    distogram_logits = self.distogram_head(pair_repr)
                    masked_msa_logits = self.masked_msa_head(msa_repr)
                    experimentally_resolved_logits = self.experimentally_resolved_head(single_rep)
                    plddt_logits = self.plddt_head(structure_predictions["single"])
                    tm_logits = self.tm_score_head(pair_repr)

                    return {
                        **structure_predictions,
                        "distogram_logits": distogram_logits,
                        "masked_msa_logits": masked_msa_logits,
                        "experimentally_resolved_logits": experimentally_resolved_logits,
                        "plddt_logits": plddt_logits,
                        "tm_logits": tm_logits,
                        "pair_representation": pair_repr,
                        "msa_representation": msa_repr,
                        "single_representation": single_rep,
                        "sampled_n_cycles": n_cycles,
                        "sampled_n_ensemble": n_ensemble,
                    }

                # Algorithm 2 line 22: store averaged m̂_1i, ẑ_ij (pre-projection,
                # in c_m not c_s) and pseudo-β positions for the next cycle.
                # detach() realises Algorithm 31 line 4 (no gradients between
                # iterations) — required since only the last cycle is unrolled
                # for backward.
                single_rep_prev = msa_first_row.detach()
                z_prev = pair_repr.detach()

                # Pseudo-β: Cα for glycine (atom14 index 1, since GLY has no Cβ),
                # Cβ otherwise (atom14 index 4). Matches the pseudo-β convention
                # used throughout AF2 for pairwise distances (supplement 1.9.8,
                # Algorithm 32 line 1). aatype==7 is glycine in AF2's alphabet.
                is_gly = (aatype == 7)
                cb_idx = torch.where(is_gly, 1, 4)
                atom_coords = structure_predictions["atom14_coords"]
                x_prev = torch.gather(
                    atom_coords, 2,
                    cb_idx[:, :, None, None].expand(-1, -1, 1, 3),
                ).squeeze(2).detach()

        raise ValueError("n_cycles and n_ensemble must be > 0")
