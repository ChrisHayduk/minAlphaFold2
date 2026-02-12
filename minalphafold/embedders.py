import torch

class InputEmbedder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_target_feat_1 = torch.nn.Linear(in_features=21, out_features=config.c_z)
        self.linear_target_feat_2 = torch.nn.Linear(in_features=21, out_features=config.c_z)
        self.linear_target_feat_3 = torch.nn.Linear(in_features=21, out_features=config.c_m)


        self.linear_msa = torch.nn.Linear(in_features=49, out_features=config.c_m)

        self.rel_pos = RelPos(config)

    def forward(self, target_feat: torch.Tensor, residue_index: torch.Tensor, msa_feat: torch.Tensor):
        # target_feat shape: (N_res, 21)
        # residue_index shape: (N_res)
        # msa_feat shape: (N_cluster, N_res, 49)

        # Output shape: (N_res, c_z)
        a = self.linear_target_feat_1(target_feat)
        b = self.linear_target_feat_2(target_feat)

        # Output shape: (N_res, N_res, c_z)
        # Row i should use element i from a, and col j should use element j from b
        # Unsqueeze lets us do this
        z = a.unsqueeze(1) + b.unsqueeze(0)

        z += self.rel_pos(residue_index)

        # Output shape: (N_cluster, N_res, c_m)
        m = self.linear_target_feat_3(target_feat).unsqueeze(0) + self.linear_msa(msa_feat)

        return m, z

class RelPos(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear = torch.nn.Linear(in_features=config.n_res, out_features=config.c_z)

    def forward(self, residue_index: torch.Tensor):
        v_bins = torch.arange(-32,33)
        d = residue_index.unsqueeze(0) - residue_index.unsqueeze(1)

        d.fill_diagonal_(float('inf'))

        p = self.linear(one_hot(d, v_bins))

        return p

def one_hot(x: torch.Tensor, v_bins: torch.Tensor):
    p = torch.zeros_like(x)

    indices = torch.argmin(torch.abs(x-v_bins),dim=-1)

    p[torch.arange(x.shape[0]), indices] = 1

    return p