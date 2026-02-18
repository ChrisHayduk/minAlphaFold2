import torch
from evoformer import Evoformer
from structure_module import StructureModule
from utils import distance_bin
from random import randint

class AlphaFold2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.evoformer_blocks = torch.nn.ModuleList([Evoformer(config) for _ in range(config.num_evoformer)])
        self.structure_model = StructureModule(config)

        self.recycle_norm_s = torch.nn.LayerNorm(config.c_s)
        self.recycle_norm_z = torch.nn.LayerNorm(config.c_z)
        self.recycle_linear_s = torch.nn.Linear(config.c_s, config.c_s)
        self.recycle_linear_z = torch.nn.Linear(config.c_z, config.c_z)
        self.recycle_linear_d = torch.nn.Linear(config.n_dist_bins, config.c_z)

        self.config = config

    def forward(self, msa_representation: torch.Tensor, pair_representation: torch.Tensor, n_recycles: int = 3):
        if self.training:
            n_recycles = randint(0, 3)

        N_seq = msa_representation.shape[1]
        N_res = msa_representation.shape[2]
        c_s = msa_representation.shape[-1]
        c_z = pair_representation.shape[-1]
        batch_size = msa_representation.shape[0]

        single_rep_prev = torch.zeros(batch_size, N_res, c_s, device=msa_representation.device)   # previous single repr
        z_prev = torch.zeros_like(pair_representation)  # previous pair repr
        x_prev = torch.zeros(batch_size, N_res, 3, device=msa_representation.device)    # previous Ca positions

        # Run n_recycles with no gradients
        for i in range(n_recycles):
            with torch.no_grad():
                msa_repr = msa_representation.clone()
                pair_repr = pair_representation.clone()
                
                msa_repr[:, 0, :, :] += self.recycle_linear_s(self.recycle_norm_s(single_rep_prev))
                pair_repr += self.recycle_linear_z(self.recycle_norm_z(z_prev))
                pair_repr += self.recycle_linear_d(distance_bin(x_prev, self.config.n_dist_bins))

                for block in self.evoformer_blocks:
                    msa_repr, pair_repr = block(msa_repr, pair_repr)

                # MSA rep has shape (batch, N_seq, N_res, channels)
                single_rep = msa_repr[:, 0,:,:]

                structure_predictions = self.structure_model(single_rep, pair_repr)

                single_rep_prev = msa_repr[:, 0, :, :].detach()
                z_prev = pair_repr.detach()
                x_prev = structure_predictions["final_translations"].detach()
        
        # Now pass with gradients
        msa_repr = msa_representation.clone()
        pair_repr = pair_representation.clone()
                
        msa_repr[:, 0, :, :] += self.recycle_linear_s(self.recycle_norm_s(single_rep_prev))
        pair_repr += self.recycle_linear_z(self.recycle_norm_z(z_prev))
        pair_repr += self.recycle_linear_d(distance_bin(x_prev, self.config.n_dist_bins))

        for block in self.evoformer_blocks:
            msa_repr, pair_repr = block(msa_repr, pair_repr)

        # MSA rep has shape (batch, N_seq, N_res, channels)
        single_rep = msa_repr[:, 0, :, :]

        structure_predictions = self.structure_model(single_rep, pair_repr)

        return structure_predictions, pair_repr, msa_repr, single_rep