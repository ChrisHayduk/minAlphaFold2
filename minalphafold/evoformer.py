import torch
import math
from utils import dropout_columnwise, dropout_rowwise
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

    def forward(self, msa_representation: torch.Tensor, pair_representation: torch.Tensor):
        # Shape (batch, N_seq, N_res, c_s)
        z = self.msa_row_att(msa_representation, pair_representation)
        msa_representation += dropout_rowwise(z, p=0.15, training=self.training)

        msa_representation += self.msa_col_att(msa_representation)
        msa_representation += self.msa_transition(msa_representation)
        
        pair_representation += self.outer_mean(msa_representation)

        pair_representation += dropout_rowwise(self.triangle_mult_out(pair_representation), p=0.25, training=self.training)
        pair_representation += dropout_rowwise(self.triangle_mult_in(pair_representation), p=0.25, training=self.training)
        pair_representation += dropout_rowwise(self.triangle_att_start(pair_representation), p=0.25, training=self.training)
        pair_representation += dropout_columnwise(self.triangle_att_end(pair_representation), p=0.25, training=self.training)
        pair_representation += self.pair_transition(pair_representation)

        return msa_representation, pair_representation

class MSARowAttentionWithPairBias(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_msa = torch.nn.LayerNorm(config.c_s)
        self.layer_norm_pair = torch.nn.LayerNorm(config.c_z)

        self.head_dim = config.dim
        self.num_heads = config.num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)

        self.linear_pair = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_s)

    def forward(self, msa_representation: torch.Tensor, pair_representation: torch.Tensor):
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

        # Shape (batch, N_res, N_res, self.num_heads)
        B = self.linear_pair(pair_representation)

        # Shape (batch, self.num_heads, N_res, N_res)
        B = B.permute(0, 3, 1, 2)

        # Shape (batch, 1, N_res, N_res, self.num_heads)
        B = B.unsqueeze(1)

        # Shape (batch, N_seq, self.num_heads, N_res, N_res)
        scores = torch.einsum('bsihd, bsjhd -> bshij', Q, K)
        scores = scores / math.sqrt(self.head_dim) + B

        attention = torch.nn.functional.softmax(scores, dim=-1)

        # Shape (batch, N_seq, N_res, self.num_heads, self.head_dim)
        values = torch.einsum('bshij, bsjhd -> bsihd', attention, V)

        values = G * values

        values = values.reshape((Q.shape[0], Q.shape[1], Q.shape[2], -1))

        output = self.linear_output(values)

        return output

class MSAColumnAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_msa = torch.nn.LayerNorm(config.c_s)

        self.head_dim = config.dim
        self.num_heads = config.num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_s)

    def forward(self, msa_representation: torch.Tensor):
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

        attention = torch.nn.functional.softmax(scores, dim=-1)

        # Shape (batch, N_seq, N_res, self.num_heads, self.head_dim)
        values = torch.einsum('bihst, btihd -> bsihd', attention, V)

        values = G * values

        values = values.reshape((Q.shape[0], Q.shape[1], Q.shape[2], -1))

        output = self.linear_output(values)

        return output


class MSATransition(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n = config.msa_transition_n

        self.layer_norm = torch.nn.LayerNorm(config.c_s)

        self.linear_up = torch.nn.Linear(in_features=config.c_s, out_features=self.n*config.c_s)
        self.linear_down = torch.nn.Linear(in_features=config.c_s*self.n, out_features=config.c_s)

    def forward(self, msa_representation: torch.Tensor):
        msa_representation = self.layer_norm(msa_representation)

        activations = self.linear_up(msa_representation)

        return self.linear_down(torch.nn.functional.relu(activations))

class OuterProductMean(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(config.c_s)

        self.c = config.outer_product_dim

        self.linear_left = torch.nn.Linear(config.c_s, self.c)
        self.linear_right = torch.nn.Linear(config.c_s, self.c)

        self.linear_out = torch.nn.Linear(in_features=self.c*self.c, out_features=config.c_z)

    def forward(self, msa_representation: torch.Tensor):
        msa_representation = self.layer_norm(msa_representation)

        # Shape (batch, N_seq, N_res, self.c)
        A = self.linear_left(msa_representation)
        B = self.linear_right(msa_representation)

        # Shape (batch, N_seq, N_res, N_res, self.c, self.c)
        # We sum over N_seq implicitly by not including s in the output
        # This reduces the tensor size that we need to store
        outer = torch.einsum('bsic, bsjd -> bijcd', A, B)
        
        # Shape (batch, N_res, N_res, self.c, self.c)
        # Now divide by N_seq to get mean
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

    def forward(self, pair_representation: torch.Tensor):
        pair_representation = self.layer_norm_pair(pair_representation)

        # Shape (batch, N_res, N_res, config.triangle_mult_c)
        A = torch.sigmoid(self.gate1(pair_representation)) * self.linear1(pair_representation)
        B = torch.sigmoid(self.gate2(pair_representation)) * self.linear2(pair_representation)
        
        # Shape (batch, N_res, N_res, c_z)
        G = torch.sigmoid(self.gate(pair_representation))

        # A: (batch, N_res_i, N_res_k, c)
        # B: (batch, N_res_j, N_res_k, c)
        # Result: (batch, N_res_i, N_res_j, c)
        vals = torch.einsum('bikc, bjkc -> bijc', A, B)

        # Shape (batch, N_res, N_res, c_z)
        out = G * self.out_linear(self.layer_norm_out(vals))
        
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

    def forward(self, pair_representation: torch.Tensor):
        pair_representation = self.layer_norm_pair(pair_representation)

        # Shape (batch, N_res, N_res, config.triangle_mult_c)
        A = torch.sigmoid(self.gate1(pair_representation)) * self.linear1(pair_representation)
        B = torch.sigmoid(self.gate2(pair_representation)) * self.linear2(pair_representation)
        
        # Shape (batch, N_res, N_res, c_z)
        G = torch.sigmoid(self.gate(pair_representation))

        # A: (batch, N_res_i, N_res_k, c)
        # B: (batch, N_res_j, N_res_k, c)
        # Result: (batch, N_res_i, N_res_j, c)
        vals = torch.einsum('bkic, bkjc -> bijc', A, B)

        # Shape (batch, N_res, N_res, c_z)
        out = G * self.out_linear(self.layer_norm_out(vals))
        
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

    def forward(self, pair_representation: torch.Tensor):
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

        attention = torch.nn.functional.softmax(scores, dim=3)

        # Shape (batch, N_res, N_res, self.num_heads, self.head_dim)
        values = torch.einsum('bijkh, bikhd -> bijhd', attention, V)

        values = G * values

        values = values.reshape((Q.shape[0], Q.shape[1], Q.shape[2], -1))

        output = self.linear_output(values)

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

    def forward(self, pair_representation: torch.Tensor):
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
        # B shape (batch, N_res_i, N_res_k, self.num_heads)
        # Output shape (batch, N_res_i, N_res_j, N_res_k, self.num_heads)

        scores = torch.einsum('bijhd, bkjhd -> bijkh', Q, K)
        scores = scores / math.sqrt(self.head_dim) + B.unsqueeze(2)

        attention = torch.nn.functional.softmax(scores, dim=3)

        # Shape (batch, N_res, N_res, self.num_heads, self.head_dim)
        values = torch.einsum('bijkh, bkjhd -> bijhd', attention, V)

        values = G * values

        values = values.reshape((Q.shape[0], Q.shape[1], Q.shape[2], -1))

        output = self.linear_output(values)

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