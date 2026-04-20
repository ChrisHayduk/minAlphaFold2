import torch
import math
from typing import Optional

try:
    from .initialization import init_gate_linear, init_linear
    from .utils import dropout_columnwise, dropout_rowwise
    from .embedders import (
        MSATransition,
        MSAColumnAttention,
        OuterProductMean,
        TriangleAttentionEndingNode,
        TriangleAttentionStartingNode,
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
        PairTransition,
    )
except ImportError:  # pragma: no cover - compatibility for direct module imports in tests/scripts.
    from initialization import init_gate_linear, init_linear
    from utils import dropout_columnwise, dropout_rowwise
    from embedders import (
        MSATransition,
        MSAColumnAttention,
        OuterProductMean,
        TriangleAttentionEndingNode,
        TriangleAttentionStartingNode,
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
        PairTransition,
    )

class Evoformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.msa_row_att = MSARowAttentionWithPairBias(config)
        self.msa_col_att = MSAColumnAttention(config)
        self.msa_transition = MSATransition(config)
        self.outer_mean = OuterProductMean(config)

        self.triangle_mult_out = TriangleMultiplicationOutgoing(config)
        self.triangle_mult_in = TriangleMultiplicationIncoming(config)
        self.triangle_att_start = TriangleAttentionStartingNode(config)
        self.triangle_att_end = TriangleAttentionEndingNode(config)

        self.pair_transition = PairTransition(config)

        # Dropout rates from config
        self.msa_dropout = config.evoformer_msa_dropout
        self.pair_dropout = config.evoformer_pair_dropout

    def forward(self, msa_representation: torch.Tensor, pair_representation: torch.Tensor,
                msa_mask: Optional[torch.Tensor] = None, pair_mask: Optional[torch.Tensor] = None):
        # msa_mask: (batch, N_seq, N_res) — 1 for valid, 0 for padding
        # pair_mask: (batch, N_res, N_res) — 1 for valid, 0 for padding
        assert msa_representation.ndim == 4, \
            f"msa_representation must be (batch, N_seq, N_res, c_m), got {msa_representation.shape}"
        assert pair_representation.ndim == 4, \
            f"pair_representation must be (batch, N_res, N_res, c_z), got {pair_representation.shape}"
        # Shape (batch, N_seq, N_res, c_m)
        z = self.msa_row_att(msa_representation, pair_representation, msa_mask=msa_mask)
        msa_representation = msa_representation + dropout_rowwise(z, p=self.msa_dropout, training=self.training)

        # No dropout on column attention or MSA transition per Algorithm 6
        msa_representation = msa_representation + self.msa_col_att(msa_representation, msa_mask=msa_mask)
        msa_representation = msa_representation + self.msa_transition(msa_representation)

        pair_representation = pair_representation + self.outer_mean(msa_representation, msa_mask=msa_mask)

        pair_representation = pair_representation + dropout_rowwise(self.triangle_mult_out(pair_representation, pair_mask=pair_mask), p=self.pair_dropout, training=self.training)
        pair_representation = pair_representation + dropout_rowwise(self.triangle_mult_in(pair_representation, pair_mask=pair_mask), p=self.pair_dropout, training=self.training)
        pair_representation = pair_representation + dropout_rowwise(self.triangle_att_start(pair_representation, pair_mask=pair_mask), p=self.pair_dropout, training=self.training)
        pair_representation = pair_representation + dropout_columnwise(self.triangle_att_end(pair_representation, pair_mask=pair_mask), p=self.pair_dropout, training=self.training)
        # No dropout on pair transition per Algorithm 6
        pair_representation = pair_representation + self.pair_transition(pair_representation)

        return msa_representation, pair_representation

class MSARowAttentionWithPairBias(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_msa = torch.nn.LayerNorm(config.c_m)
        self.layer_norm_pair = torch.nn.LayerNorm(config.c_z)

        self.head_dim = config.dim
        self.num_heads = config.num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_m, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_m, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_m, out_features=self.total_dim, bias=False)

        self.linear_pair = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=config.c_m, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_m)
        init_linear(self.linear_q, init="default")
        init_linear(self.linear_k, init="default")
        init_linear(self.linear_v, init="default")
        init_linear(self.linear_pair, init="default")
        init_gate_linear(self.linear_gate)
        init_linear(self.linear_output, init="final")

    def forward(self, msa_representation: torch.Tensor, pair_representation: torch.Tensor,
                msa_mask: Optional[torch.Tensor] = None):
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

        pair_representation = self.layer_norm_pair(pair_representation)

        # Algorithm 7 line 3: b_ij^h = LinearNoBias(LayerNorm(z_ij)).
        # Shape (batch, N_res, N_res, self.num_heads)
        B = self.linear_pair(pair_representation)

        # Align B's axes with the score tensor below: (batch, num_heads, i, j),
        # then broadcast across the MSA sequence dim.
        B = B.permute(0, 3, 1, 2)            # (batch, num_heads, N_res_i, N_res_j)
        B = B.unsqueeze(1)                    # (batch, 1, num_heads, N_res_i, N_res_j)

        # Algorithm 7 line 5: a_sij^h = softmax_j(1/sqrt(c) q_si^h . k_sj^h + b_ij^h)
        # scores shape: (batch, N_seq, num_heads, N_res_i, N_res_j)
        scores = torch.einsum('bsihd, bsjhd -> bshij', Q, K)
        scores = scores / math.sqrt(self.head_dim) + B

        # Apply MSA mask to key positions (j dimension)
        if msa_mask is not None:
            # msa_mask: (batch, N_seq, N_res) -> (batch, N_seq, 1, 1, N_res)
            mask_bias = (1.0 - msa_mask[:, :, None, None, :]) * (-1e9)
            scores = scores + mask_bias

        attention = torch.nn.functional.softmax(scores, dim=-1)

        # Shape (batch, N_seq, N_res, self.num_heads, self.head_dim)
        values = torch.einsum('bshij, bsjhd -> bsihd', attention, V)

        values = G * values

        values = values.reshape((Q.shape[0], Q.shape[1], Q.shape[2], -1))

        output = self.linear_output(values)

        # Zero out padded query positions
        if msa_mask is not None:
            output = output * msa_mask[..., None]

        return output
