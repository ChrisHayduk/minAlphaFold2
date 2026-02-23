import torch


def _zero_init_linear(linear: torch.nn.Linear):
    """Zero-initialize weights and bias of a Linear layer (Supplement 1.11.4)."""
    torch.nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)


class DistogramHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(config.c_z, config.n_dist_bins)
        # Supplement 1.11.4: zero-init residue distance prediction logits
        _zero_init_linear(self.linear)

    def forward(self, pair_representation: torch.Tensor):
        # pair_representation: (batch, N_res, N_res, c_z)
        logits = self.linear(pair_representation)  # (batch, N_res, N_res, n_dist_bins)
        logits = (logits + logits.transpose(1, 2)) / 2
        return logits


class PLDDTHead(torch.nn.Module):
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
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(config.c_m, config.n_msa_classes)
        # Supplement 1.11.4: zero-init masked MSA prediction logits
        _zero_init_linear(self.linear)

    def forward(self, msa_representation: torch.Tensor):
        # msa_representation: (batch, N_seq, N_res, c_m)
        return self.linear(msa_representation)  # (batch, N_seq, N_res, n_msa_classes)


class TMScoreHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_pae_bins = config.n_pae_bins  # 64, covering 0–31.75 Å in 0.5 Å bins
        self.linear = torch.nn.Linear(config.c_z, config.n_pae_bins)

    def forward(self, pair_representation: torch.Tensor):
        # pair_representation: (batch, N_res, N_res, c_z)
        logits = self.linear(pair_representation)  # (batch, N_res, N_res, n_pae_bins)
        # No symmetrization — PAE(i, j) != PAE(j, i)
        return logits


class ExperimentallyResolvedHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(config.c_s, 14)

    def forward(self, single_representation: torch.Tensor):
        # single_representation: (batch, N_res, c_s)
        return self.linear(single_representation)  # (batch, N_res, 14)
