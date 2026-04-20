"""Embedding and pair-update modules used by the Evoformer trunk.

This module collects every shared Evoformer sub-block plus the pre-trunk
embedders. The class-to-algorithm mapping mirrors the supplement:

* :class:`InputEmbedder`                      — Algorithm 3
* :class:`RelPos`                             — Algorithm 4
* :class:`MSAColumnAttention`                 — Algorithm 8
* :class:`MSATransition`                      — Algorithm 9
* :class:`OuterProductMean`                   — Algorithm 10
* :class:`TriangleMultiplicationOutgoing`     — Algorithm 11
* :class:`TriangleMultiplicationIncoming`     — Algorithm 12
* :class:`TriangleAttentionStartingNode`      — Algorithm 13
* :class:`TriangleAttentionEndingNode`        — Algorithm 14
* :class:`PairTransition`                     — Algorithm 15
* :class:`TemplatePair`                       — Algorithm 16 (1.7.1)
* :class:`TemplatePointwiseAttention`         — Algorithm 17 (1.7.1)
* :class:`ExtraMsaStack`                      — Algorithm 18 (1.7.2)
* :class:`MSAColumnGlobalAttention`           — Algorithm 19 (1.7.2)

``MSARowAttentionWithPairBias`` (Algorithm 7) lives in
:mod:`minalphafold.evoformer` alongside the full Evoformer block.
"""

import torch
import math
from typing import Optional

from .a3m import SEQ_ALPHABET_SIZE
from .initialization import init_gate_linear, init_linear
from .utils import dropout_columnwise, dropout_rowwise


TARGET_FEAT_DIM = SEQ_ALPHABET_SIZE + 1

class InputEmbedder(torch.nn.Module):
    """Initial MSA + pair embedding (Algorithm 3).

    Produces the starting ``m_si`` and ``z_ij`` for the Evoformer by
    combining:

    * three linear projections of ``target_feat`` (shape
      ``(batch, N_res, TARGET_FEAT_DIM)``) — two broadcast into the
      outer-sum for ``z`` and one added to the query row of ``m``;
    * a relative-positional encoding ``RelPos(residue_index)`` added to
      ``z`` (Algorithm 4);
    * a linear projection of ``msa_feat`` (49 channels — cluster profile
      + deletion features, per Table 1) added to every MSA row in ``m``.

    Output shapes: ``m`` ``(batch, N_cluster, N_res, c_m)``, ``z``
    ``(batch, N_res, N_res, c_z)``.
    """

    def __init__(self, config):
        super().__init__()

        self.linear_target_feat_1 = torch.nn.Linear(in_features=TARGET_FEAT_DIM, out_features=config.c_z)
        self.linear_target_feat_2 = torch.nn.Linear(in_features=TARGET_FEAT_DIM, out_features=config.c_z)
        self.linear_target_feat_3 = torch.nn.Linear(in_features=TARGET_FEAT_DIM, out_features=config.c_m)


        self.linear_msa = torch.nn.Linear(in_features=49, out_features=config.c_m)
        init_linear(self.linear_target_feat_1, init="default")
        init_linear(self.linear_target_feat_2, init="default")
        init_linear(self.linear_target_feat_3, init="default")
        init_linear(self.linear_msa, init="default")

        self.rel_pos = RelPos(config)

    def forward(self, target_feat: torch.Tensor, residue_index: torch.Tensor, msa_feat: torch.Tensor):
        # target_feat shape: (batch, N_res, 22)
        # residue_index shape: (batch, N_res)
        # msa_feat shape: (batch, N_cluster, N_res, 49)
        assert target_feat.ndim == 3 and target_feat.shape[-1] == TARGET_FEAT_DIM, \
            f"target_feat must be (batch, N_res, {TARGET_FEAT_DIM}), got {target_feat.shape}"
        assert residue_index.ndim == 2, \
            f"residue_index must be (batch, N_res), got {residue_index.shape}"
        assert msa_feat.ndim == 4 and msa_feat.shape[-1] == 49, \
            f"msa_feat must be (batch, N_cluster, N_res, 49), got {msa_feat.shape}"

        # Output shape: (batch, N_res, c_z)
        a = self.linear_target_feat_1(target_feat)
        b = self.linear_target_feat_2(target_feat)

        # Output shape: (batch, N_res, N_res, c_z)
        # Row i should use element i from a, and col j should use element j from b
        z = a.unsqueeze(-2) + b.unsqueeze(-3)

        z += self.rel_pos(residue_index)

        # Output shape: (batch, N_cluster, N_res, c_m)
        m = self.linear_target_feat_3(target_feat).unsqueeze(1) + self.linear_msa(msa_feat)

        return m, z

class RelPos(torch.nn.Module):
    """Relative-position encoding (Algorithm 4).

    One-hots the clipped residue-index difference
    ``clamp(r_i - r_j, -max_rel, max_rel)`` into ``2·max_rel+1`` bins
    (default ``max_rel = 32`` → 65 bins) and projects to ``c_z``. The
    output is added to the pair representation by :class:`InputEmbedder`
    so the Evoformer trunk has a learned sense of residue adjacency from
    the very first block. Clipping at ±32 matches the supplement.
    """

    def __init__(self, config, max_rel=32):
        super().__init__()
        self.max_rel = max_rel
        self.linear = torch.nn.Linear(2 * max_rel + 1, config.c_z)
        init_linear(self.linear, init="default")

    def forward(self, residue_index: torch.Tensor):
        # residue_index shape: (batch, N_res)
        d = residue_index[:, :, None] - residue_index[:, None, :]  # (batch, N_res, N_res)
        d = d.clamp(-self.max_rel, self.max_rel) + self.max_rel
        oh = torch.nn.functional.one_hot(d.long(), 2 * self.max_rel + 1).float()
        return self.linear(oh)  # (batch, N_res, N_res, c_z)

class TemplatePair(torch.nn.Module):
    """Template pair stack (Algorithm 16, supplement 1.7.1).

    Per-template shallow Evoformer-like pair stack: each of
    ``config.template_pair_num_blocks`` blocks applies triangle
    self-attention (start + end), triangle multiplication
    (outgoing + incoming), and a pair transition. Dropout matches the
    supplement — row-wise on starting / multiplicative updates, column-
    wise on ending. Batch and template dims are flattened for each
    block so templates evolve independently; the final LayerNorm
    happens once before the pointwise attention pool in
    :class:`TemplatePointwiseAttention`.
    """

    def __init__(self, config):
        super().__init__()

        self.num_blocks = config.template_pair_num_blocks
        self.dropout_p = config.template_pair_dropout

        # Supplement 1.7.1 / Algorithm 16: the template pair stack overrides
        # the main-Evoformer triangle dims (paper default c=64 for both the
        # multiplicative and attention updates, and n=2 on the pair transition)
        # instead of inheriting triangle_mult_c / triangle_dim / pair_transition_n.
        template_tri_mult_c = config.template_triangle_mult_c
        template_tri_attn_c = config.template_triangle_attn_c
        template_tri_attn_heads = config.template_triangle_attn_num_heads
        template_pair_trans_n = config.template_pair_transition_n

        self.layer_norm = torch.nn.LayerNorm(config.c_t)
        self.linear_in = torch.nn.Linear(in_features=config.c_t, out_features=config.c_z)
        init_linear(self.linear_in, init="default")

        self.triangle_mult_out = torch.nn.ModuleList(
            [TriangleMultiplicationOutgoing(config, c=template_tri_mult_c) for _ in range(self.num_blocks)]
        )
        self.triangle_mult_in = torch.nn.ModuleList(
            [TriangleMultiplicationIncoming(config, c=template_tri_mult_c) for _ in range(self.num_blocks)]
        )
        self.triangle_att_start = torch.nn.ModuleList(
            [
                TriangleAttentionStartingNode(config, c=template_tri_attn_c, num_heads=template_tri_attn_heads)
                for _ in range(self.num_blocks)
            ]
        )
        self.triangle_att_end = torch.nn.ModuleList(
            [
                TriangleAttentionEndingNode(config, c=template_tri_attn_c, num_heads=template_tri_attn_heads)
                for _ in range(self.num_blocks)
            ]
        )
        self.pair_transition = torch.nn.ModuleList(
            [PairTransition(config, n=template_pair_trans_n) for _ in range(self.num_blocks)]
        )
        self.final_layer_norm = torch.nn.LayerNorm(config.c_z)

    def forward(self, template_feat: torch.Tensor, pair_mask: Optional[torch.Tensor] = None):
        # template_feat shape: (batch, N_templates, N_res, N_res, c_t)

        # Project from template feature space to pair representation space
        # Output shape: (batch, N_templates, N_res, N_res, c_z)
        template_feat = self.linear_in(self.layer_norm(template_feat))

        b, t, n_i, n_j, c = template_feat.shape

        # Merge batch and template dims to process each template independently
        # Shape: (batch * N_templates, N_res, N_res, c_z)
        pair_representation = template_feat.reshape(b * t, n_i, n_j, c)
        flat_pair_mask = None
        if pair_mask is not None:
            flat_pair_mask = pair_mask.reshape(b * t, n_i, n_j)
            pair_representation = pair_representation * flat_pair_mask[..., None]

        for block_idx in range(self.num_blocks):
            if flat_pair_mask is not None:
                pair_representation = pair_representation * flat_pair_mask[..., None]
            pair_representation = pair_representation + dropout_rowwise(
                self.triangle_att_start[block_idx](pair_representation, pair_mask=flat_pair_mask),
                p=self.dropout_p,
                training=self.training,
            )
            pair_representation = pair_representation + dropout_columnwise(
                self.triangle_att_end[block_idx](pair_representation, pair_mask=flat_pair_mask),
                p=self.dropout_p,
                training=self.training,
            )
            pair_representation = pair_representation + dropout_rowwise(
                self.triangle_mult_out[block_idx](pair_representation, pair_mask=flat_pair_mask),
                p=self.dropout_p,
                training=self.training,
            )
            pair_representation = pair_representation + dropout_rowwise(
                self.triangle_mult_in[block_idx](pair_representation, pair_mask=flat_pair_mask),
                p=self.dropout_p,
                training=self.training,
            )
            pair_representation = pair_representation + self.pair_transition[block_idx](pair_representation)
            if flat_pair_mask is not None:
                pair_representation = pair_representation * flat_pair_mask[..., None]

        pair_representation = self.final_layer_norm(pair_representation)
        if flat_pair_mask is not None:
            pair_representation = pair_representation * flat_pair_mask[..., None]

        # Restore batch and template dims
        # Output shape: (batch, N_templates, N_res, N_res, c_z)
        pair_representation = pair_representation.reshape(b, t, n_i, n_j, c)

        return pair_representation

class TemplatePointwiseAttention(torch.nn.Module):
    """Template pointwise attention (Algorithm 17, supplement 1.7.1).

    Pool ``N_templates`` per-pair features into a single pair update by
    attending, for each residue pair (i, j) independently, across the
    template dimension. The query is the current pair representation
    ``z_ij``; keys and values are the per-template embeddings produced
    by :class:`TemplatePair`. No spatial mixing — the softmax is only
    over templates. Padding templates are masked before the softmax
    and the attention rows are renormalised over the surviving
    templates.
    """

    def __init__(self, config):
        super().__init__()

        self.head_dim = config.template_pointwise_attention_dim
        self.num_heads = config.template_pointwise_num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_z)
        init_linear(self.linear_q, init="default")
        init_linear(self.linear_k, init="default")
        init_linear(self.linear_v, init="default")
        init_linear(self.linear_output, init="final")

    def forward(
        self,
        template_feat: torch.Tensor,
        pair_representation: torch.Tensor,
        template_mask: Optional[torch.Tensor] = None,
    ):
        # template_feat shape: (batch, N_templates, N_res, N_res, c_z)
        # pair_representation shape: (batch, N_res, N_res, c_z)

        # Query from pair representation
        # Shape: (batch, N_res, N_res, total_dim)
        Q = self.linear_q(pair_representation)

        # Keys and values from template features
        # Shape: (batch, N_templates, N_res, N_res, total_dim)
        K = self.linear_k(template_feat)
        V = self.linear_v(template_feat)

        # Reshape to (batch, N_res, N_res, num_heads, head_dim)
        Q = Q.reshape((Q.shape[0], Q.shape[1], Q.shape[2], self.num_heads, self.head_dim))

        # Reshape to (batch, N_templates, N_res, N_res, num_heads, head_dim)
        K = K.reshape((K.shape[0], K.shape[1], K.shape[2], K.shape[3], self.num_heads, self.head_dim))
        V = V.reshape((V.shape[0], V.shape[1], V.shape[2], V.shape[3], self.num_heads, self.head_dim))

        # Attention over templates: for each residue pair (i,j), attend across templates
        # Q shape (batch, N_res_i, N_res_j, num_heads, head_dim)
        # K shape (batch, N_templates, N_res_i, N_res_j, num_heads, head_dim)
        # Output shape: (batch, N_templates, N_res, N_res, num_heads)
        scores = torch.einsum('bijhd, btijhd -> btijh', Q, K)
        scores = scores / math.sqrt(self.head_dim)

        # template_mask shape: (batch, N_templates), 1.0 for valid templates
        if template_mask is not None:
            mask = template_mask.to(scores.dtype)
            scores = scores + (1.0 - mask[:, :, None, None, None]) * (-1e9)

        # Softmax over templates dimension
        attention = torch.nn.functional.softmax(scores, dim=1)
        if template_mask is not None:
            attention = attention * template_mask[:, :, None, None, None].to(attention.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Weighted sum over templates
        # Output shape: (batch, N_res, N_res, num_heads, head_dim)
        values = torch.einsum('btijh, btijhd -> bijhd', attention, V)

        # Reshape to (batch, N_res, N_res, total_dim)
        values = values.reshape((values.shape[0], values.shape[1], values.shape[2], -1))

        output = self.linear_output(values)

        return output

class ExtraMsaStack(torch.nn.Module):
    """Extra MSA stack (Algorithm 18, supplement 1.7.2).

    Lightweight Evoformer-like block for the unclustered "extra" MSA.
    The extra MSA is much deeper (default ``N_extra_seq = 1024`` vs
    ``N_cluster = 128``) but compressed to a smaller channel dim
    ``c_e`` to stay cheap. Two differences from the main Evoformer:

    * MSA column attention is replaced by
      :class:`MSAColumnGlobalAttention` (Algorithm 19) — across
      thousands of sequences, per-head K/V sharing is what keeps
      the column step tractable.
    * Row attention with pair bias is inlined here rather than
      reusing :class:`~minalphafold.evoformer.MSARowAttentionWithPairBias`
      so ``c_e ≠ c_m`` projections stay self-contained.

    Consumes the extra MSA representation and the pair representation;
    writes updates back to both (triangle updates + pair transition
    apply after the OPM consumes the updated extra MSA).
    """

    def __init__(self, config):
        super().__init__()

        self.layer_norm_msa = torch.nn.LayerNorm(config.c_e)
        self.layer_norm_pair = torch.nn.LayerNorm(config.c_z)

        self.head_dim = config.extra_msa_dim
        self.num_heads = config.num_heads

        self.total_dim = self.head_dim * self.num_heads

        # MSA row attention with pair bias (inline, same as Algorithm 7)
        self.linear_q = torch.nn.Linear(in_features=config.c_e, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_e, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_e, out_features=self.total_dim, bias=False)

        self.linear_pair = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=config.c_e, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_e)
        init_linear(self.linear_q, init="default")
        init_linear(self.linear_k, init="default")
        init_linear(self.linear_v, init="default")
        init_linear(self.linear_pair, init="default")
        init_gate_linear(self.linear_gate)
        init_linear(self.linear_output, init="final")

        self.msa_col_att = MSAColumnGlobalAttention(config, c_in=config.c_e)
        self.msa_transition = MSATransition(
            config,
            c_in=config.c_e,
            n=getattr(config, "extra_msa_transition_n", config.msa_transition_n),
        )
        self.outer_mean = OuterProductMean(
            config,
            c_in=config.c_e,
            c_hidden=getattr(config, "extra_msa_outer_product_dim", config.outer_product_dim),
        )

        self.triangle_mult_out = TriangleMultiplicationOutgoing(config)
        self.triangle_mult_in = TriangleMultiplicationIncoming(config)
        self.triangle_att_start = TriangleAttentionStartingNode(config)
        self.triangle_att_end = TriangleAttentionEndingNode(config)
        self.pair_transition = PairTransition(config)

        self.msa_dropout_p = config.extra_msa_dropout
        self.pair_dropout_p = config.extra_pair_dropout

    def forward(self, extra_msa_representation: torch.Tensor, pair_representation: torch.Tensor,
                extra_msa_mask: Optional[torch.Tensor] = None, pair_mask: Optional[torch.Tensor] = None):
        # extra_msa_representation shape: (batch, N_extra_seq, N_res, c_e)
        # pair_representation shape: (batch, N_res, N_res, c_z)
        # extra_msa_mask: (batch, N_extra_seq, N_res) — 1 for valid, 0 for padding
        # pair_mask: (batch, N_res, N_res) — 1 for valid, 0 for padding

        msa_representation = self.layer_norm_msa(extra_msa_representation)
        pair_norm = self.layer_norm_pair(pair_representation)

        # --- MSA row attention with pair bias ---

        # Shape (batch, N_extra_seq, N_res, total_dim)
        Q = self.linear_q(msa_representation)
        K = self.linear_k(msa_representation)
        V = self.linear_v(msa_representation)

        # Reshape to (batch, N_extra_seq, N_res, num_heads, head_dim)
        Q = Q.reshape((Q.shape[0], Q.shape[1], Q.shape[2], self.num_heads, self.head_dim))
        K = K.reshape((K.shape[0], K.shape[1], K.shape[2], self.num_heads, self.head_dim))
        V = V.reshape((V.shape[0], V.shape[1], V.shape[2], self.num_heads, self.head_dim))

        G = self.linear_gate(msa_representation)
        G = G.reshape((G.shape[0], G.shape[1], G.shape[2], self.num_heads, self.head_dim))

        # Squash values in range 0 to 1 to act as gating mechanism
        G = torch.sigmoid(G)

        # Pair bias: project pair representation to per-head bias
        # Shape (batch, N_res, N_res, num_heads) -> (batch, num_heads, N_res, N_res)
        B = self.linear_pair(pair_norm)
        B = B.permute(0, 3, 1, 2)

        # Add sequence dim for broadcast: (batch, 1, num_heads, N_res, N_res)
        B = B.unsqueeze(1)

        # Q shape (batch, N_extra_seq, N_res_i, num_heads, head_dim)
        # K shape (batch, N_extra_seq, N_res_j, num_heads, head_dim)
        # Output shape (batch, N_extra_seq, num_heads, N_res, N_res)
        scores = torch.einsum('bsihd, bsjhd -> bshij', Q, K)
        scores = scores / math.sqrt(self.head_dim) + B

        # Apply extra MSA mask to key positions (j dimension)
        if extra_msa_mask is not None:
            mask_bias = (1.0 - extra_msa_mask[:, :, None, None, :]) * (-1e9)
            scores = scores + mask_bias

        attention = torch.nn.functional.softmax(scores, dim=-1)

        # Shape (batch, N_extra_seq, N_res, num_heads, head_dim)
        values = torch.einsum('bshij, bsjhd -> bsihd', attention, V)

        values = G * values

        # Reshape to (batch, N_extra_seq, N_res, total_dim)
        values = values.reshape((values.shape[0], values.shape[1], values.shape[2], -1))

        row_update = self.linear_output(values)

        # Zero out padded query positions
        if extra_msa_mask is not None:
            row_update = row_update * extra_msa_mask[..., None]

        # --- MSA representation updates ---

        extra_msa_representation = extra_msa_representation + dropout_rowwise(
            row_update,
            p=self.msa_dropout_p,
            training=self.training,
        )
        extra_msa_representation = extra_msa_representation + self.msa_col_att(
            extra_msa_representation, msa_mask=extra_msa_mask)
        extra_msa_representation = extra_msa_representation + self.msa_transition(extra_msa_representation)

        # --- Pair representation updates ---

        pair_representation = pair_representation + self.outer_mean(
            extra_msa_representation, msa_mask=extra_msa_mask)
        pair_representation = pair_representation + dropout_rowwise(
            self.triangle_mult_out(pair_representation, pair_mask=pair_mask),
            p=self.pair_dropout_p,
            training=self.training,
        )
        pair_representation = pair_representation + dropout_rowwise(
            self.triangle_mult_in(pair_representation, pair_mask=pair_mask),
            p=self.pair_dropout_p,
            training=self.training,
        )
        pair_representation = pair_representation + dropout_rowwise(
            self.triangle_att_start(pair_representation, pair_mask=pair_mask),
            p=self.pair_dropout_p,
            training=self.training,
        )
        pair_representation = pair_representation + dropout_columnwise(
            self.triangle_att_end(pair_representation, pair_mask=pair_mask),
            p=self.pair_dropout_p,
            training=self.training,
        )
        pair_representation = pair_representation + self.pair_transition(pair_representation)

        return extra_msa_representation, pair_representation

class MSAColumnGlobalAttention(torch.nn.Module):
    """MSA global column-wise gated self-attention (Algorithm 19).

    Per the supplement, only the query is per-head (q^h_si); the key and value
    projections produce a single c-dim vector that is shared across heads
    (k_si, v_si ∈ R^c, no h superscript). Sharing k/v across heads is what
    makes this module "global" — sequences contribute the same K/V to every
    head, and each head only differs in the mean-pooled query direction.
    """

    def __init__(self, config, c_in: Optional[int] = None):
        super().__init__()
        self.c_in = config.c_s if c_in is None else c_in
        self.layer_norm_msa = torch.nn.LayerNorm(self.c_in)

        self.head_dim = config.msa_column_global_attention_dim
        self.num_heads = config.num_heads

        self.total_dim = self.head_dim * self.num_heads

        # q is per-head (total = num_heads * c); k and v are shared (total = c).
        self.linear_q = torch.nn.Linear(in_features=self.c_in, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=self.c_in, out_features=self.head_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=self.c_in, out_features=self.head_dim, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=self.c_in, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=self.c_in)
        init_linear(self.linear_q, init="default")
        init_linear(self.linear_k, init="default")
        init_linear(self.linear_v, init="default")
        init_gate_linear(self.linear_gate)
        init_linear(self.linear_output, init="final")

    def forward(self, msa_representation: torch.Tensor, msa_mask: Optional[torch.Tensor] = None):
        # msa_representation: (batch, N_seq, N_res, c_in)
        # msa_mask: (batch, N_seq, N_res) — 1 for valid, 0 for padding
        b, s, i, _ = msa_representation.shape

        # Alg 19 line 1: LayerNorm the MSA, then all downstream projections read from x_ln.
        x_ln = self.layer_norm_msa(msa_representation)  # (b, s, i, c_in)

        # Alg 19 line 2: k_si, v_si ∈ R^c (shared across heads — no num_heads dim).
        K = self.linear_k(x_ln)  # (b, s, i, c)
        V = self.linear_v(x_ln)  # (b, s, i, c)

        # Alg 19 line 3: q^h_si then q^h_i = mean_s q^h_si. Use a mask-aware
        # mean so padded sequences don't dilute the query.
        Q_si = self.linear_q(x_ln).reshape(b, s, i, self.num_heads, self.head_dim)
        if msa_mask is not None:
            query_mask = msa_mask[..., None, None].to(Q_si.dtype)
            Q = (Q_si * query_mask).sum(dim=1) / query_mask.sum(dim=1).clamp(min=1.0)
        else:
            Q = Q_si.mean(dim=1)  # (b, i, h, c)

        # Alg 19 line 4: g^h_si ∈ R^c (per-head gate).
        G = torch.sigmoid(self.linear_gate(x_ln)).reshape(b, s, i, self.num_heads, self.head_dim)

        # Alg 19 line 5: a^h_ti = softmax_t(1/sqrt(c) q^h_i^T k_ti). Since k is
        # shared across heads, the einsum contracts only the channel dim.
        # scores: (b, t, i, h) with t indexing sequences.
        scores = torch.einsum('bihc, btic -> btih', Q, K)
        scores = scores / math.sqrt(self.head_dim)

        # Mask key positions (sequence dim t).
        if msa_mask is not None:
            # msa_mask: (batch, N_seq, N_res) -> (batch, N_seq, N_res, 1)
            mask_bias = (1.0 - msa_mask[:, :, :, None]) * (-1e9)
            scores = scores + mask_bias

        # Softmax over sequences (dim=1).
        attention = torch.nn.functional.softmax(scores, dim=1)

        # Alg 19 line 6: o^h_si = g^h_si ⊙ sum_t a^h_ti * v_ti. The weighted sum
        # over t is the same for every s (it doesn't depend on s), so we compute
        # it once and broadcast.
        weighted = torch.einsum('btih, btic -> bihc', attention, V)  # (b, i, h, c)
        weighted = weighted.unsqueeze(1)                              # (b, 1, i, h, c)

        # Apply the gate (broadcasting weighted across the sequence dim).
        output = self.linear_output((G * weighted).reshape(b, s, i, -1))

        # Zero out padded positions.
        if msa_mask is not None:
            output = output * msa_mask[..., None]

        return output

class MSAColumnAttention(torch.nn.Module):
    """MSA column-wise gated self-attention (Algorithm 8).

    For each residue column i, attend across MSA sequences
    ``s = 1, ..., N_seq`` with standard multi-head attention on ``m_{si}``
    (no pair bias, unlike the row variant). Gated by
    ``sigmoid(Linear(m))`` and projected back to ``c_m``. No dropout
    per Algorithm 6. Used only in the main Evoformer; the extra MSA
    stack uses :class:`MSAColumnGlobalAttention` instead.
    """

    def __init__(self, config):
        super().__init__()
        self.layer_norm_msa = torch.nn.LayerNorm(config.c_m)

        self.head_dim = config.dim
        self.num_heads = config.num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_m, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_m, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_m, out_features=self.total_dim, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=config.c_m, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_m)
        init_linear(self.linear_q, init="default")
        init_linear(self.linear_k, init="default")
        init_linear(self.linear_v, init="default")
        init_gate_linear(self.linear_gate)
        init_linear(self.linear_output, init="final")

    def forward(self, msa_representation: torch.Tensor, msa_mask: Optional[torch.Tensor] = None):
        msa_representation = self.layer_norm_msa(msa_representation)

        # Shape (batch, N_seq, N_res, self.total_dim)
        Q = self.linear_q(msa_representation)
        K = self.linear_k(msa_representation)
        V = self.linear_v(msa_representation)

        # Reshape to (batch, N_seq, N_res, self.num_heads, self.head_dim)
        Q = Q.reshape((Q.shape[0], Q.shape[1], Q.shape[2], self.num_heads, self.head_dim))
        K = K.reshape((K.shape[0], K.shape[1], K.shape[2], self.num_heads, self.head_dim))
        V = V.reshape((V.shape[0], V.shape[1], V.shape[2], self.num_heads, self.head_dim))

        G = self.linear_gate(msa_representation)
        G = G.reshape((G.shape[0], G.shape[1], G.shape[2], self.num_heads, self.head_dim))

        # Squash values in range 0 to 1 to act as gating mechanism
        G = torch.sigmoid(G)

        # Shape (batch, N_res, self.num_heads, N_seq, N_seq)
        scores = torch.einsum('bsihd, btihd -> bihst', Q, K)
        scores = scores / math.sqrt(self.head_dim)

        # Apply MSA mask to key positions (t dimension = sequences)
        if msa_mask is not None:
            # msa_mask: (batch, N_seq, N_res) -> (batch, N_res, 1, 1, N_seq)
            mask_bias = (1.0 - msa_mask.permute(0, 2, 1)[:, :, None, None, :]) * (-1e9)
            scores = scores + mask_bias

        attention = torch.nn.functional.softmax(scores, dim=-1)

        # Shape (batch, N_seq, N_res, self.num_heads, self.head_dim)
        values = torch.einsum('bihst, btihd -> bsihd', attention, V)

        values = G * values

        values = values.reshape((Q.shape[0], Q.shape[1], Q.shape[2], -1))

        output = self.linear_output(values)

        # Zero out padded query positions
        if msa_mask is not None:
            output = output * msa_mask[..., None]

        return output


class MSATransition(torch.nn.Module):
    """MSA transition (Algorithm 9).

    Two-layer feed-forward applied per MSA cell: ``LayerNorm → Linear(c
    → n·c) → ReLU → Linear(n·c → c)`` with ``n = 4`` by default. The
    widening factor ``n`` is the supplement's ``n`` parameter, kept
    configurable so the extra MSA stack can choose its own (``c_in``
    also overrideable so the same module serves both ``c_m`` and
    ``c_e`` MSA reps). No dropout — the residual connection is purely
    additive per Algorithm 6.
    """

    def __init__(self, config, c_in: Optional[int] = None, n: Optional[int] = None):
        super().__init__()
        self.c_in = config.c_m if c_in is None else c_in
        self.n = config.msa_transition_n if n is None else n

        self.layer_norm = torch.nn.LayerNorm(self.c_in)

        self.linear_up = torch.nn.Linear(in_features=self.c_in, out_features=self.n * self.c_in)
        self.linear_down = torch.nn.Linear(in_features=self.c_in * self.n, out_features=self.c_in)
        init_linear(self.linear_up, init="relu")
        init_linear(self.linear_down, init="final")

    def forward(self, msa_representation: torch.Tensor):
        msa_representation = self.layer_norm(msa_representation)

        activations = self.linear_up(msa_representation)

        return self.linear_down(torch.nn.functional.relu(activations))

class OuterProductMean(torch.nn.Module):
    """Outer product mean (Algorithm 10).

    Symmetric MSA → pair update: project each MSA cell to two hidden
    vectors ``a_{si}, b_{si} ∈ R^{c_hidden}``, take the MSA mean of
    their outer product ``mean_s (a_{si} ⊗ b_{sj})``, flatten to
    ``c_hidden^2`` channels, and project back to ``c_z``. This is the
    only channel in the Evoformer where the MSA rep writes into the
    pair rep; the reverse direction (pair → MSA) goes through the
    pair-biased row attention.
    ``c_in``/``c_hidden`` are configurable so the extra MSA stack can
    run a narrower OPM.
    """

    def __init__(self, config, c_in: Optional[int] = None, c_hidden: Optional[int] = None):
        super().__init__()
        self.c_in = config.c_m if c_in is None else c_in
        self.layer_norm = torch.nn.LayerNorm(self.c_in)

        self.c = config.outer_product_dim if c_hidden is None else c_hidden

        self.linear_left = torch.nn.Linear(self.c_in, self.c)
        self.linear_right = torch.nn.Linear(self.c_in, self.c)

        self.linear_out = torch.nn.Linear(in_features=self.c*self.c, out_features=config.c_z)
        init_linear(self.linear_left, init="default")
        init_linear(self.linear_right, init="default")
        init_linear(self.linear_out, init="final")

    def forward(self, msa_representation: torch.Tensor, msa_mask: Optional[torch.Tensor] = None):
        # msa_mask: (batch, N_seq, N_res) — 1 for valid, 0 for padding
        msa_representation = self.layer_norm(msa_representation)

        # Shape (batch, N_seq, N_res, self.c)
        A = self.linear_left(msa_representation)
        B = self.linear_right(msa_representation)

        if msa_mask is not None:
            # Zero out padded MSA rows before outer product
            m = msa_mask.to(A.dtype)              # (batch, N_seq, N_res)
            A = A * m[..., None]                   # (batch, N_seq, N_res, c)
            B = B * m[..., None]

        # Sum over N_seq: (batch, N_res_i, N_res_j, c, c)
        outer = torch.einsum('bsic, bsjd -> bijcd', A, B)

        if msa_mask is not None:
            # Mask-aware normalization: count valid (s,i)*(s,j) pairs
            m = msa_mask.to(A.dtype)
            norm = torch.einsum('bsi, bsj -> bij', m, m).clamp(min=1.0)  # (batch, N_res, N_res)
            mean_val = outer / norm[..., None, None]
        else:
            mean_val = outer / msa_representation.shape[1]

        # Shape (batch, N_res, N_res, self.c*self.c)
        mean_val = mean_val.reshape(mean_val.shape[0], mean_val.shape[1], mean_val.shape[2], -1)

        return self.linear_out(mean_val)

class TriangleMultiplicationOutgoing(torch.nn.Module):
    """Triangle multiplicative update, outgoing edges (Algorithm 11).

    Update ``z_ij`` from the two *outgoing* edges of the triangle
    ``(i, j, k)``: ``z_ij ← g_ij ⊙ Linear(LayerNorm(sum_k a_ik ⊙
    b_jk))`` where ``a = gate_a ⊙ projection_a(z)`` and likewise for
    ``b``. Enforces the triangle-inequality structure across the pair
    rep. Algorithm 11 pools over intermediate node ``k`` via the
    outgoing edges ``z_{ik}`` and ``z_{jk}``; the incoming-edge
    counterpart is :class:`TriangleMultiplicationIncoming`.
    """

    def __init__(self, config, c: Optional[int] = None):
        super().__init__()
        mult_c = config.triangle_mult_c if c is None else c
        self.layer_norm_pair = torch.nn.LayerNorm(config.c_z)
        self.layer_norm_out = torch.nn.LayerNorm(mult_c)

        self.gate1 = torch.nn.Linear(in_features=config.c_z, out_features=mult_c)
        self.gate2 = torch.nn.Linear(in_features=config.c_z, out_features=mult_c)

        self.linear1 = torch.nn.Linear(in_features=config.c_z, out_features=mult_c)
        self.linear2 = torch.nn.Linear(in_features=config.c_z, out_features=mult_c)

        self.gate = torch.nn.Linear(in_features=config.c_z, out_features=config.c_z)

        self.out_linear = torch.nn.Linear(in_features=mult_c, out_features=config.c_z)
        init_gate_linear(self.gate1)
        init_gate_linear(self.gate2)
        init_linear(self.linear1, init="default")
        init_linear(self.linear2, init="default")
        init_gate_linear(self.gate)
        init_linear(self.out_linear, init="final")

    def forward(self, pair_representation: torch.Tensor, pair_mask: Optional[torch.Tensor] = None):
        pair_representation = self.layer_norm_pair(pair_representation)

        # Shape (batch, N_res, N_res, c)
        A = torch.sigmoid(self.gate1(pair_representation)) * self.linear1(pair_representation)
        B = torch.sigmoid(self.gate2(pair_representation)) * self.linear2(pair_representation)

        # Mask out padded positions before contraction
        if pair_mask is not None:
            A = A * pair_mask[..., None]
            B = B * pair_mask[..., None]

        # Shape (batch, N_res, N_res, c_z)
        G = torch.sigmoid(self.gate(pair_representation))

        # A: (batch, N_res_i, N_res_k, c)
        # B: (batch, N_res_j, N_res_k, c)
        # Result: (batch, N_res_i, N_res_j, c)
        vals = torch.einsum('bikc, bjkc -> bijc', A, B)

        # Shape (batch, N_res, N_res, c_z)
        out = G * self.out_linear(self.layer_norm_out(vals))

        if pair_mask is not None:
            out = out * pair_mask[..., None]

        return out


class TriangleMultiplicationIncoming(torch.nn.Module):
    """Triangle multiplicative update, incoming edges (Algorithm 12).

    Symmetric partner of :class:`TriangleMultiplicationOutgoing`: pool
    over intermediate node ``k`` using the *incoming* edges
    ``z_{ki}`` and ``z_{kj}`` (i.e. ``sum_k a_ki ⊙ b_kj``). Outgoing and
    incoming variants fire back-to-back in every Evoformer block so the
    pair rep sees both triangle orientations per iteration.
    """

    def __init__(self, config, c: Optional[int] = None):
        super().__init__()
        mult_c = config.triangle_mult_c if c is None else c
        self.layer_norm_pair = torch.nn.LayerNorm(config.c_z)
        self.layer_norm_out = torch.nn.LayerNorm(mult_c)

        self.gate1 = torch.nn.Linear(in_features=config.c_z, out_features=mult_c)
        self.gate2 = torch.nn.Linear(in_features=config.c_z, out_features=mult_c)

        self.linear1 = torch.nn.Linear(in_features=config.c_z, out_features=mult_c)
        self.linear2 = torch.nn.Linear(in_features=config.c_z, out_features=mult_c)

        self.gate = torch.nn.Linear(in_features=config.c_z, out_features=config.c_z)

        self.out_linear = torch.nn.Linear(in_features=mult_c, out_features=config.c_z)
        init_gate_linear(self.gate1)
        init_gate_linear(self.gate2)
        init_linear(self.linear1, init="default")
        init_linear(self.linear2, init="default")
        init_gate_linear(self.gate)
        init_linear(self.out_linear, init="final")

    def forward(self, pair_representation: torch.Tensor, pair_mask: Optional[torch.Tensor] = None):
        pair_representation = self.layer_norm_pair(pair_representation)

        # Shape (batch, N_res, N_res, c)
        A = torch.sigmoid(self.gate1(pair_representation)) * self.linear1(pair_representation)
        B = torch.sigmoid(self.gate2(pair_representation)) * self.linear2(pair_representation)

        # Mask out padded positions before contraction
        if pair_mask is not None:
            A = A * pair_mask[..., None]
            B = B * pair_mask[..., None]

        # Shape (batch, N_res, N_res, c_z)
        G = torch.sigmoid(self.gate(pair_representation))

        # A: (batch, N_res_k, N_res_i, c)
        # B: (batch, N_res_k, N_res_j, c)
        # Result: (batch, N_res_i, N_res_j, c)
        vals = torch.einsum('bkic, bkjc -> bijc', A, B)

        # Shape (batch, N_res, N_res, c_z)
        out = G * self.out_linear(self.layer_norm_out(vals))

        if pair_mask is not None:
            out = out * pair_mask[..., None]

        return out

class TriangleAttentionStartingNode(torch.nn.Module):
    """Triangle self-attention around the starting node (Algorithm 13).

    Gated multi-head self-attention over the pair rep with a
    triangle-consistency bias: for fixed starting node i, attend over
    ending nodes j with keys from ``z_{ij}`` and values from
    ``z_{ik}``, plus a pair bias ``b_{jk} = LinearNoBias(LayerNorm(
    z_{jk}))``. Row-wise dropout (supplement 1.11.6) matches Algorithm 6.
    """

    def __init__(self, config, c: Optional[int] = None, num_heads: Optional[int] = None):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(config.c_z)

        self.head_dim = config.triangle_dim if c is None else c
        self.num_heads = config.triangle_num_heads if num_heads is None else num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)

        self.linear_bias = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_z)
        init_linear(self.linear_q, init="default")
        init_linear(self.linear_k, init="default")
        init_linear(self.linear_v, init="default")
        init_linear(self.linear_bias, init="default")
        init_gate_linear(self.linear_gate)
        init_linear(self.linear_output, init="final")

    def forward(self, pair_representation: torch.Tensor, pair_mask: Optional[torch.Tensor] = None):
        pair_representation = self.layer_norm(pair_representation)

        # Shape (batch, N_res, N_res, self.total_dim)
        Q = self.linear_q(pair_representation)
        K = self.linear_k(pair_representation)
        V = self.linear_v(pair_representation)

        # Reshape to (batch, N_res, N_res, self.num_heads, self.head_dim)
        Q = Q.reshape((Q.shape[0], Q.shape[1], Q.shape[2], self.num_heads, self.head_dim))
        K = K.reshape((K.shape[0], K.shape[1], K.shape[2], self.num_heads, self.head_dim))
        V = V.reshape((V.shape[0], V.shape[1], V.shape[2], self.num_heads, self.head_dim))

        G = self.linear_gate(pair_representation)
        G = G.reshape((G.shape[0], G.shape[1], G.shape[2], self.num_heads, self.head_dim))

        # Squash values in range 0 to 1 to act as gating mechanism
        G = torch.sigmoid(G)

        # Shape (batch, N_res, N_res, self.num_heads)
        B = self.linear_bias(pair_representation)

        # Q shape (batch, N_res_i, N_res_j, self.num_heads, self.head_dim)
        # K shape (batch, N_res_i, N_res_k, self.num_heads, self.head_dim)
        # B shape (batch, N_res_j, N_res_k, self.num_heads)
        # Output shape (batch, N_res_i, N_res_j, N_res_k, self.num_heads)
        scores = torch.einsum('bijhd, bikhd -> bijkh', Q, K)
        scores = scores / math.sqrt(self.head_dim) + B.unsqueeze(1)

        # Apply pair mask to key positions (k dimension, for a given i)
        if pair_mask is not None:
            # pair_mask: (batch, N_res, N_res) -> (batch, N_res_i, 1, N_res_k, 1)
            mask_bias = (1.0 - pair_mask[:, :, None, :, None]) * (-1e9)
            scores = scores + mask_bias

        attention = torch.nn.functional.softmax(scores, dim=3)

        # Shape (batch, N_res, N_res, self.num_heads, self.head_dim)
        values = torch.einsum('bijkh, bikhd -> bijhd', attention, V)

        values = G * values

        values = values.reshape((Q.shape[0], Q.shape[1], Q.shape[2], -1))

        output = self.linear_output(values)

        # Zero out padded query positions
        if pair_mask is not None:
            output = output * pair_mask[..., None]

        return output

class TriangleAttentionEndingNode(torch.nn.Module):
    """Triangle self-attention around the ending node (Algorithm 14).

    Mirror image of :class:`TriangleAttentionStartingNode`: fix the
    ending node j and attend over starting nodes i. The pair bias is
    ``b_{ki} = LinearNoBias(LayerNorm(z_{ki}))``. The supplement
    prescribes column-wise dropout (not row-wise) on this output — the
    Evoformer block applies it accordingly.
    """

    def __init__(self, config, c: Optional[int] = None, num_heads: Optional[int] = None):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(config.c_z)

        self.head_dim = config.triangle_dim if c is None else c
        self.num_heads = config.triangle_num_heads if num_heads is None else num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)

        self.linear_bias = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_z)
        init_linear(self.linear_q, init="default")
        init_linear(self.linear_k, init="default")
        init_linear(self.linear_v, init="default")
        init_linear(self.linear_bias, init="default")
        init_gate_linear(self.linear_gate)
        init_linear(self.linear_output, init="final")

    def forward(self, pair_representation: torch.Tensor, pair_mask: Optional[torch.Tensor] = None):
        pair_representation = self.layer_norm(pair_representation)

        # Shape (batch, N_res, N_res, self.total_dim)
        Q = self.linear_q(pair_representation)
        K = self.linear_k(pair_representation)
        V = self.linear_v(pair_representation)

        # Reshape to (batch, N_res, N_res, self.num_heads, self.head_dim)
        Q = Q.reshape((Q.shape[0], Q.shape[1], Q.shape[2], self.num_heads, self.head_dim))
        K = K.reshape((K.shape[0], K.shape[1], K.shape[2], self.num_heads, self.head_dim))
        V = V.reshape((V.shape[0], V.shape[1], V.shape[2], self.num_heads, self.head_dim))

        G = self.linear_gate(pair_representation)
        G = G.reshape((G.shape[0], G.shape[1], G.shape[2], self.num_heads, self.head_dim))

        # Squash values in range 0 to 1 to act as gating mechanism
        G = torch.sigmoid(G)

        # Shape (batch, N_res, N_res, self.num_heads)
        B = self.linear_bias(pair_representation)

        # Algorithm 14 line 5: a_ijk^h = softmax_k(1/sqrt(c) q_ij^h . k_kj^h + b_ki^h).
        # The highlighted differences from the starting-node version (Algorithm 13)
        # are that keys/values are indexed (k, j) instead of (i, k), and the bias
        # is b_{k,i} instead of b_{j,k}.
        #
        # Q shape (batch, N_res_i, N_res_j, num_heads, head_dim)
        # K shape (batch, N_res_k, N_res_j, num_heads, head_dim)
        # B shape (batch, N_res, N_res, num_heads), with B[b, i, j, h] coming from z_{ij}.
        # Output shape (batch, N_res_i, N_res_j, N_res_k, num_heads)

        scores = torch.einsum('bijhd, bkjhd -> bijkh', Q, K)
        # We need B indexed as b^h_{k,i} at score position (b, i, j, k, h), i.e.
        # B[b, k, i, h]. Swapping axes 1 and 2 of B yields a view whose indexing
        # is B_t[b, x, y, h] = B[b, y, x, h], so B_t[b, i, k, h] = B[b, k, i, h].
        # Inserting a new j-axis with unsqueeze(2) broadcasts that bias to every
        # (i, j, k, h) score.
        scores = scores / math.sqrt(self.head_dim) + B.transpose(1, 2).unsqueeze(2)

        # Apply pair mask to key positions: an attention score at (b, i, j, k, h)
        # should be masked when the *key* pair z_{k,j} is padding, i.e. when
        # pair_mask[b, k, j] == 0. Permute swaps the k/j axes so the view exposes
        # [b, j, k], then broadcast over i and h.
        if pair_mask is not None:
            mask_bias = (1.0 - pair_mask.permute(0, 2, 1)[:, None, :, :, None]) * (-1e9)
            scores = scores + mask_bias

        attention = torch.nn.functional.softmax(scores, dim=3)

        # Shape (batch, N_res, N_res, self.num_heads, self.head_dim)
        values = torch.einsum('bijkh, bkjhd -> bijhd', attention, V)

        values = G * values

        values = values.reshape((Q.shape[0], Q.shape[1], Q.shape[2], -1))

        output = self.linear_output(values)

        # Zero out padded query positions
        if pair_mask is not None:
            output = output * pair_mask[..., None]

        return output

class PairTransition(torch.nn.Module):
    """Pair transition (Algorithm 15).

    Per-pair feed-forward: ``LayerNorm → Linear(c_z → n·c_z) → ReLU →
    Linear(n·c_z → c_z)`` with widening factor ``n = 4``. Same shape as
    :class:`MSATransition` but over the pair rep instead of the MSA
    rep. No dropout per Algorithm 6.
    """

    def __init__(self, config, n: Optional[int] = None):
        super().__init__()
        self.n = config.pair_transition_n if n is None else n

        self.layer_norm = torch.nn.LayerNorm(config.c_z)

        self.linear_up = torch.nn.Linear(in_features=config.c_z, out_features=self.n*config.c_z)
        self.linear_down = torch.nn.Linear(in_features=config.c_z*self.n, out_features=config.c_z)
        init_linear(self.linear_up, init="relu")
        init_linear(self.linear_down, init="final")

    def forward(self, pair_representation: torch.Tensor):
        pair_representation = self.layer_norm(pair_representation)

        activations = self.linear_up(pair_representation)

        return self.linear_down(torch.nn.functional.relu(activations))
