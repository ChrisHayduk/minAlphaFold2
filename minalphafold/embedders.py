import torch
import math
from typing import Optional
from utils import dropout_columnwise, dropout_rowwise

class InputEmbedder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_target_feat_1 = torch.nn.Linear(in_features=21, out_features=config.c_z)
        self.linear_target_feat_2 = torch.nn.Linear(in_features=21, out_features=config.c_z)
        self.linear_target_feat_3 = torch.nn.Linear(in_features=21, out_features=config.c_m)


        self.linear_msa = torch.nn.Linear(in_features=49, out_features=config.c_m)

        self.rel_pos = RelPos(config)

    def forward(self, target_feat: torch.Tensor, residue_index: torch.Tensor, msa_feat: torch.Tensor):
        # target_feat shape: (batch, N_res, 21)
        # residue_index shape: (batch, N_res)
        # msa_feat shape: (batch, N_cluster, N_res, 49)
        assert target_feat.ndim == 3 and target_feat.shape[-1] == 21, \
            f"target_feat must be (batch, N_res, 21), got {target_feat.shape}"
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
    def __init__(self, config, max_rel=32):
        super().__init__()
        self.max_rel = max_rel
        self.linear = torch.nn.Linear(2 * max_rel + 1, config.c_z)

    def forward(self, residue_index: torch.Tensor):
        # residue_index shape: (batch, N_res)
        d = residue_index[:, :, None] - residue_index[:, None, :]  # (batch, N_res, N_res)
        d = d.clamp(-self.max_rel, self.max_rel) + self.max_rel
        oh = torch.nn.functional.one_hot(d.long(), 2 * self.max_rel + 1).float()
        return self.linear(oh)  # (batch, N_res, N_res, c_z)

class TemplatePair(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_blocks = config.template_pair_num_blocks
        self.dropout_p = config.template_pair_dropout

        self.layer_norm = torch.nn.LayerNorm(config.c_t)
        self.linear_in = torch.nn.Linear(in_features=config.c_t, out_features=config.c_z)

        self.triangle_mult_out = torch.nn.ModuleList(
            [TriangleMultiplicationOutgoing(config) for _ in range(self.num_blocks)]
        )
        self.triangle_mult_in = torch.nn.ModuleList(
            [TriangleMultiplicationIncoming(config) for _ in range(self.num_blocks)]
        )
        self.triangle_att_start = torch.nn.ModuleList(
            [TriangleAttentionStartingNode(config) for _ in range(self.num_blocks)]
        )
        self.triangle_att_end = torch.nn.ModuleList(
            [TriangleAttentionEndingNode(config) for _ in range(self.num_blocks)]
        )
        self.pair_transition = torch.nn.ModuleList(
            [PairTransition(config) for _ in range(self.num_blocks)]
        )
        self.final_layer_norm = torch.nn.LayerNorm(config.c_z)

    def forward(self, template_feat: torch.Tensor):
        # template_feat shape: (batch, N_templates, N_res, N_res, c_t)

        # Project from template feature space to pair representation space
        # Output shape: (batch, N_templates, N_res, N_res, c_z)
        template_feat = self.linear_in(self.layer_norm(template_feat))

        b, t, n_i, n_j, c = template_feat.shape

        # Merge batch and template dims to process each template independently
        # Shape: (batch * N_templates, N_res, N_res, c_z)
        pair_representation = template_feat.reshape(b * t, n_i, n_j, c)

        for block_idx in range(self.num_blocks):
            pair_representation = pair_representation + dropout_rowwise(
                self.triangle_att_start[block_idx](pair_representation),
                p=self.dropout_p,
                training=self.training,
            )
            pair_representation = pair_representation + dropout_columnwise(
                self.triangle_att_end[block_idx](pair_representation),
                p=self.dropout_p,
                training=self.training,
            )
            pair_representation = pair_representation + dropout_rowwise(
                self.triangle_mult_out[block_idx](pair_representation),
                p=self.dropout_p,
                training=self.training,
            )
            pair_representation = pair_representation + dropout_rowwise(
                self.triangle_mult_in[block_idx](pair_representation),
                p=self.dropout_p,
                training=self.training,
            )
            pair_representation = pair_representation + self.pair_transition[block_idx](pair_representation)

        pair_representation = self.final_layer_norm(pair_representation)

        # Restore batch and template dims
        # Output shape: (batch, N_templates, N_res, N_res, c_z)
        pair_representation = pair_representation.reshape(b, t, n_i, n_j, c)

        return pair_representation

class TemplatePointwiseAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.template_pointwise_attention_dim
        self.num_heads = config.template_pointwise_num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_z)

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

        # Weighted sum over templates
        # Output shape: (batch, N_res, N_res, num_heads, head_dim)
        values = torch.einsum('btijh, btijhd -> bijhd', attention, V)

        # Reshape to (batch, N_res, N_res, total_dim)
        values = values.reshape((values.shape[0], values.shape[1], values.shape[2], -1))

        output = self.linear_output(values)

        return output

class ExtraMsaStack(torch.nn.Module):
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
    def __init__(self, config, c_in: Optional[int] = None):
        super().__init__()
        self.c_in = config.c_s if c_in is None else c_in
        self.layer_norm_msa = torch.nn.LayerNorm(self.c_in)

        self.head_dim = config.msa_column_global_attention_dim
        self.num_heads = config.num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=self.c_in, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=self.c_in, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=self.c_in, out_features=self.total_dim, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=self.c_in, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=self.c_in)

    def forward(self, msa_representation: torch.Tensor, msa_mask: Optional[torch.Tensor] = None):
        # msa_representation: (batch, N_seq, N_res, c_in)
        # msa_mask: (batch, N_seq, N_res) — 1 for valid, 0 for padding
        b, s, i, _ = msa_representation.shape

        # K, V, G from per-sequence LN(x_si) (Algorithm 19)
        x_ln = self.layer_norm_msa(msa_representation)  # (b, s, i, c_in)
        K = self.linear_k(x_ln).reshape(b, s, i, self.num_heads, self.head_dim)
        V = self.linear_v(x_ln).reshape(b, s, i, self.num_heads, self.head_dim)
        G = torch.sigmoid(self.linear_gate(x_ln)).reshape(b, s, i, self.num_heads, self.head_dim)

        # Q from LN(mean_s(x_si)) — mean raw input first, then normalize, then project
        x_mean = msa_representation.mean(dim=1)          # (b, i, c_in)
        x_mean_ln = self.layer_norm_msa(x_mean)          # (b, i, c_in)
        Q = self.linear_q(x_mean_ln).reshape(b, i, self.num_heads, self.head_dim)

        # Attention scores: (batch, N_seq, num_heads, N_res)
        scores = torch.einsum('bihd, bsihd -> bshi', Q, K)
        scores = scores / math.sqrt(self.head_dim)

        # Mask key positions (sequence dim)
        if msa_mask is not None:
            # msa_mask: (batch, N_seq, N_res) -> (batch, N_seq, 1, N_res)
            mask_bias = (1.0 - msa_mask[:, :, None, :]) * (-1e9)
            scores = scores + mask_bias

        # Softmax over sequences (dim=1)
        attention = torch.nn.functional.softmax(scores, dim=1)

        # Weighted sum over sequences: (batch, N_res, num_heads, head_dim)
        weighted = torch.einsum('bshi, bsihd -> bihd', attention, V)

        # Broadcast to all sequences: (batch, 1, N_res, num_heads, head_dim)
        weighted = weighted.unsqueeze(1)

        # Gate and project: G is (batch, N_seq, N_res, num_heads, head_dim)
        output = self.linear_output((G * weighted).reshape(b, s, i, -1))

        # Zero out padded positions
        if msa_mask is not None:
            output = output * msa_mask[..., None]

        return output

class MSAColumnAttention(torch.nn.Module):
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
    def __init__(self, config, c_in: Optional[int] = None, n: Optional[int] = None):
        super().__init__()
        self.c_in = config.c_m if c_in is None else c_in
        self.n = config.msa_transition_n if n is None else n

        self.layer_norm = torch.nn.LayerNorm(self.c_in)

        self.linear_up = torch.nn.Linear(in_features=self.c_in, out_features=self.n * self.c_in)
        self.linear_down = torch.nn.Linear(in_features=self.c_in * self.n, out_features=self.c_in)

    def forward(self, msa_representation: torch.Tensor):
        msa_representation = self.layer_norm(msa_representation)

        activations = self.linear_up(msa_representation)

        return self.linear_down(torch.nn.functional.relu(activations))

class OuterProductMean(torch.nn.Module):
    def __init__(self, config, c_in: Optional[int] = None, c_hidden: Optional[int] = None):
        super().__init__()
        self.c_in = config.c_m if c_in is None else c_in
        self.layer_norm = torch.nn.LayerNorm(self.c_in)

        self.c = config.outer_product_dim if c_hidden is None else c_hidden

        self.linear_left = torch.nn.Linear(self.c_in, self.c)
        self.linear_right = torch.nn.Linear(self.c_in, self.c)

        self.linear_out = torch.nn.Linear(in_features=self.c*self.c, out_features=config.c_z)

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
    def __init__(self, config):
        super().__init__()
        self.layer_norm_pair = torch.nn.LayerNorm(config.c_z)
        self.layer_norm_out = torch.nn.LayerNorm(config.triangle_mult_c)

        self.gate1 = torch.nn.Linear(in_features=config.c_z, out_features=config.triangle_mult_c)
        self.gate2 = torch.nn.Linear(in_features=config.c_z, out_features=config.triangle_mult_c)

        self.linear1 = torch.nn.Linear(in_features=config.c_z, out_features=config.triangle_mult_c)
        self.linear2 = torch.nn.Linear(in_features=config.c_z, out_features=config.triangle_mult_c)

        self.gate = torch.nn.Linear(in_features=config.c_z, out_features=config.c_z)

        self.out_linear = torch.nn.Linear(in_features=config.triangle_mult_c, out_features=config.c_z)

    def forward(self, pair_representation: torch.Tensor, pair_mask: Optional[torch.Tensor] = None):
        pair_representation = self.layer_norm_pair(pair_representation)

        # Shape (batch, N_res, N_res, config.triangle_mult_c)
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
    def __init__(self, config):
        super().__init__()
        self.layer_norm_pair = torch.nn.LayerNorm(config.c_z)
        self.layer_norm_out = torch.nn.LayerNorm(config.triangle_mult_c)

        self.gate1 = torch.nn.Linear(in_features=config.c_z, out_features=config.triangle_mult_c)
        self.gate2 = torch.nn.Linear(in_features=config.c_z, out_features=config.triangle_mult_c)

        self.linear1 = torch.nn.Linear(in_features=config.c_z, out_features=config.triangle_mult_c)
        self.linear2 = torch.nn.Linear(in_features=config.c_z, out_features=config.triangle_mult_c)

        self.gate = torch.nn.Linear(in_features=config.c_z, out_features=config.c_z)

        self.out_linear = torch.nn.Linear(in_features=config.triangle_mult_c, out_features=config.c_z)

    def forward(self, pair_representation: torch.Tensor, pair_mask: Optional[torch.Tensor] = None):
        pair_representation = self.layer_norm_pair(pair_representation)

        # Shape (batch, N_res, N_res, config.triangle_mult_c)
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
    def __init__(self, config):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(config.c_z)

        self.head_dim = config.triangle_dim
        self.num_heads = config.triangle_num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)

        self.linear_bias = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_z)

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
    def __init__(self, config):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(config.c_z)

        self.head_dim = config.triangle_dim
        self.num_heads = config.triangle_num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim, bias=False)

        self.linear_bias = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=config.c_z, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_z)

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
        # K shape (batch, N_res_k, N_res_j, self.num_heads, self.head_dim)
        # B shape (batch, N_res, N_res, self.num_heads) — Algorithm 14 requires b_{k,i}
        # Output shape (batch, N_res_i, N_res_j, N_res_k, self.num_heads)

        scores = torch.einsum('bijhd, bkjhd -> bijkh', Q, K)
        # B.transpose(1,2): (b, k, i, h) -> unsqueeze(2): (b, k, 1, i, h) -> broadcasts to (b, i, j, k, h)
        scores = scores / math.sqrt(self.head_dim) + B.transpose(1, 2).unsqueeze(2)

        # Apply pair mask to key positions (k dimension, for a given j)
        if pair_mask is not None:
            # pair_mask: (batch, N_res, N_res) -> (batch, 1, N_res_j, N_res_k, 1)
            # pair_mask[b,k,j] = valid -> permute to (batch, j, k) then reshape
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
    def __init__(self, config):
        super().__init__()
        self.n = config.pair_transition_n

        self.layer_norm = torch.nn.LayerNorm(config.c_z)

        self.linear_up = torch.nn.Linear(in_features=config.c_z, out_features=self.n*config.c_z)
        self.linear_down = torch.nn.Linear(in_features=config.c_z*self.n, out_features=config.c_z)

    def forward(self, pair_representation: torch.Tensor):
        pair_representation = self.layer_norm(pair_representation)

        activations = self.linear_up(pair_representation)

        return self.linear_down(torch.nn.functional.relu(activations))
