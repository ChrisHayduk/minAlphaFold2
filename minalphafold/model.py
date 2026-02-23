import torch
import math
from typing import cast
from evoformer import Evoformer
from structure_module import StructureModule
from embedders import InputEmbedder, TemplatePair, TemplatePointwiseAttention, ExtraMsaStack
from utils import recycling_distance_bin

class AlphaFold2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.evoformer_blocks = torch.nn.ModuleList([Evoformer(config) for _ in range(config.num_evoformer)])
        self.structure_model = StructureModule(config)

        self.input_embedder = InputEmbedder(config)

        # Recycling embedders (Algorithm 32): LayerNorm only, no learned projections
        self.recycle_norm_s = torch.nn.LayerNorm(config.c_m)
        self.recycle_norm_z = torch.nn.LayerNorm(config.c_z)
        self.recycle_linear_d = torch.nn.Linear(15, config.c_z)

        # Project from MSA channel dim (c_m) to single rep dim (c_s)
        self.single_rep_proj = torch.nn.Linear(config.c_m, config.c_s)

        # Template processing
        self.template_pair_feat_linear = torch.nn.Linear(88, config.c_t)
        self.template_pair_stack = TemplatePair(config)
        self.template_pointwise_att = TemplatePointwiseAttention(config)

        self.template_angle_linear_1 = torch.nn.Linear(51, config.c_m)
        self.template_angle_linear_2 = torch.nn.Linear(config.c_m, config.c_m)

        # Extra MSA processing
        self.extra_msa_feat_linear = torch.nn.Linear(25, config.c_e)
        self.extra_msa_blocks = torch.nn.ModuleList(
            [ExtraMsaStack(config) for _ in range(config.num_extra_msa)]
        )

        self.config = config
        self._initialize_alphafold_parameters()

    @staticmethod
    def _zero_linear(linear: torch.nn.Linear):
        torch.nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            torch.nn.init.zeros_(linear.bias)

    @staticmethod
    def _init_gate_linear(linear: torch.nn.Linear):
        torch.nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            torch.nn.init.ones_(linear.bias)

    def _initialize_alphafold_parameters(self):
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

            if class_name in output_zero_init_classes and hasattr(module, "linear_output"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_output))

            if class_name in transition_zero_init_classes and hasattr(module, "linear_down"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_down))

            if class_name == "OuterProductMean" and hasattr(module, "linear_out"):
                self._zero_linear(cast(torch.nn.Linear, module.linear_out))

            if class_name in {"TriangleMultiplicationOutgoing", "TriangleMultiplicationIncoming"}:
                self._init_gate_linear(cast(torch.nn.Linear, module.gate1))
                self._init_gate_linear(cast(torch.nn.Linear, module.gate2))
                self._init_gate_linear(cast(torch.nn.Linear, module.gate))
                self._zero_linear(cast(torch.nn.Linear, module.out_linear))

            if class_name == "StructureModule":
                self._zero_linear(cast(torch.nn.Linear, module.transition_linear_3))

            if class_name == "BackboneUpdate":
                self._zero_linear(cast(torch.nn.Linear, module.linear))

            if class_name == "InvariantPointAttention":
                # Set head weights so softplus(head_weights) = 1 at init.
                cast(torch.nn.Parameter, module.head_weights).data.fill_(math.log(math.e - 1.0))

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
            seq_mask: torch.Tensor | None = None,
            msa_mask: torch.Tensor | None = None,
            extra_msa_mask: torch.Tensor | None = None,
            n_cycles: int = 3,
            n_ensemble: int = 1,
        ):
        # seq_mask: (batch, N_res) — 1 for valid residues, 0 for padding
        # msa_mask: (batch, N_seq, N_res) — 1 for valid, 0 for padding
        # extra_msa_mask: (batch, N_extra, N_res) — 1 for valid, 0 for padding
        assert(n_ensemble > 0)
        assert(n_cycles > 0)

        if self.training:
            # Algorithm 31: sample uniformly from {1, ..., n_cycles}
            n_cycles = int(torch.randint(1, n_cycles + 1, (1,), device=target_feat.device).item())

        outer_grad = torch.is_grad_enabled()

        N_res = target_feat.shape[1]
        c_m = self.config.c_m
        c_z = self.config.c_z
        batch_size = target_feat.shape[0]

        # Default masks: all ones (no padding)
        if seq_mask is None:
            seq_mask = target_feat.new_ones(batch_size, N_res)
        pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]  # (batch, N_res, N_res)
        if msa_mask is None:
            msa_mask = target_feat.new_ones(batch_size, msa_feat.shape[1], N_res)
        if extra_msa_mask is None:
            extra_msa_mask = target_feat.new_ones(batch_size, extra_msa_feat.shape[1], N_res)

        # Initialize recycling tensors (only once, before the loop)
        single_rep_prev = torch.zeros(batch_size, N_res, c_m, device=msa_feat.device)
        z_prev = torch.zeros(batch_size, N_res, N_res, c_z, device=msa_feat.device)
        x_prev = torch.zeros(batch_size, N_res, 3, device=msa_feat.device)

        for i in range(n_cycles):
            is_last = (i == n_cycles-1)

            with torch.set_grad_enabled(is_last and outer_grad):
                # Ensemble: accumulate non-MSA representations and average
                single_rep_accum = torch.zeros(batch_size, N_res, c_m, device=msa_feat.device)
                pair_repr_accum = torch.zeros(batch_size, N_res, N_res, c_z, device=msa_feat.device)
                msa_repr_accum = torch.zeros(
                    batch_size,
                    msa_feat.shape[1],
                    N_res,
                    c_m,
                    device=msa_feat.device,
                )

                msa_repr = None  # will be set by ensemble loop (n_ensemble > 0)
                for _ in range(n_ensemble):
                    msa_representation, pair_representation = self.input_embedder(target_feat, residue_index, msa_feat)

                    msa_repr = msa_representation.clone()
                    pair_repr = pair_representation.clone()

                    # Algorithm 32: LayerNorm only (no learned projection)
                    msa_repr[:, 0, :, :] += self.recycle_norm_s(single_rep_prev)
                    pair_repr += self.recycle_norm_z(z_prev)
                    pair_repr += self.recycle_linear_d(recycling_distance_bin(x_prev, n_bins=15))

                    # Template processing
                    template_pair = self.template_pair_feat_linear(template_pair_feat)
                    template_pair = self.template_pair_stack(template_pair)
                    pair_repr = pair_repr + self.template_pointwise_att(
                        template_pair,
                        pair_repr,
                        template_mask=template_mask,
                    )

                    # Template torsion-angle features are appended as extra MSA rows.
                    evo_msa_mask = msa_mask
                    if template_angle_feat is not None:
                        template_angle_repr = self.template_angle_linear_2(
                            torch.relu(self.template_angle_linear_1(template_angle_feat))
                        )
                        msa_repr = torch.cat([msa_repr, template_angle_repr], dim=1)
                        # Extend msa_mask for appended template rows (all valid)
                        n_templ = template_angle_repr.shape[1]
                        templ_mask = msa_mask.new_ones(batch_size, n_templ, N_res)
                        evo_msa_mask = torch.cat([msa_mask, templ_mask], dim=1)

                    # Extra MSA processing
                    extra_msa_repr = self.extra_msa_feat_linear(extra_msa_feat)
                    for extra_block in self.extra_msa_blocks:
                        extra_msa_repr, pair_repr = extra_block(
                            extra_msa_repr, pair_repr,
                            extra_msa_mask=extra_msa_mask, pair_mask=pair_mask,
                        )

                    for block in self.evoformer_blocks:
                        msa_repr, pair_repr = block(
                            msa_repr, pair_repr,
                            msa_mask=evo_msa_mask, pair_mask=pair_mask,
                        )

                    single_rep_accum += msa_repr[:, 0, :, :]
                    pair_repr_accum += pair_repr
                    msa_repr_accum += msa_repr[:, :msa_feat.shape[1], :, :]

                # Average across ensemble members
                msa_first_row = single_rep_accum / n_ensemble
                pair_repr = pair_repr_accum / n_ensemble
                msa_repr = msa_repr_accum / n_ensemble

                single_rep = self.single_rep_proj(msa_first_row)

                structure_predictions = self.structure_model(single_rep, pair_repr, aatype, seq_mask=seq_mask)

                if is_last:
                    return structure_predictions, pair_repr, msa_repr, {
                        "single_pre_sm": single_rep,
                        "single_post_sm": structure_predictions["single"],
                    }

                # Recycle: store pre-projection single rep (c_m) for next cycle
                single_rep_prev = msa_first_row.detach()
                z_prev = pair_repr.detach()

                # Use pseudo-beta positions for recycling distance features
                is_gly = (aatype == 7)
                cb_idx = torch.where(is_gly, 1, 4)  # CA=1 for GLY, CB=4 otherwise
                atom_coords = structure_predictions["atom14_coords"]
                x_prev = torch.gather(
                    atom_coords, 2,
                    cb_idx[:, :, None, None].expand(-1, -1, 1, 3),
                ).squeeze(2).detach()

        raise ValueError("n_cycles and n_ensemble must be > 0")
