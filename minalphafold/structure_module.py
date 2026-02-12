import torch    
import math
from residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
)

class StructureModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c = config.structure_module_c
        self.num_layers = config.structure_module_layers

        # Layer Norms
        self.layer_norm_single_rep_1 = torch.nn.LayerNorm(config.c_s)
        self.layer_norm_single_rep_2 = torch.nn.LayerNorm(config.c_s)
        self.layer_norm_single_rep_3 = torch.nn.LayerNorm(config.c_s)

        self.layer_norm_pair_rep = torch.nn.LayerNorm(config.c_z)

        # Dropouts
        self.dropout_1 = torch.nn.Dropout(p=0.1)
        self.dropout_2 = torch.nn.Dropout(p=0.2)

        # Linear layers
        self.single_rep_proj = torch.nn.Linear(in_features=config.c_s, out_features=config.c_s)
        self.transition_linear_1 = torch.nn.Linear(in_features=config.c_s, out_features=config.c_s)
        self.transition_linear_2 = torch.nn.Linear(in_features=config.c_s, out_features=config.c_s)
        self.transition_linear_3 = torch.nn.Linear(in_features=config.c_s, out_features=config.c_s)

        self.angle_linear_1 = torch.nn.Linear(in_features=config.c_s, out_features=self.c)
        self.angle_linear_2 = torch.nn.Linear(in_features=config.c_s, out_features=self.c)

        self.angle_linear_3 = torch.nn.Linear(in_features=self.c, out_features=self.c)
        self.angle_linear_4 = torch.nn.Linear(in_features=self.c, out_features=self.c)
        self.angle_linear_5 = torch.nn.Linear(in_features=self.c, out_features=self.c)
        self.angle_linear_6 = torch.nn.Linear(in_features=self.c, out_features=self.c)

        self.angle_linear_7 = torch.nn.Linear(in_features=self.c, out_features=2)

        # Core blocks
        self.IPA = InvariantPointAttention(config)
        self.backbone_update = BackboneUpdate(config)

        self.relu = torch.nn.ReLU()

    def forward(self, single_representation: torch.Tensor, pair_representation: torch.Tensor, aatype: torch.Tensor):
        single_representation = self.layer_norm_single_rep_1(single_representation)

        pair_representation = self.layer_norm_pair_rep(pair_representation)

        s = self.single_rep_proj(single_representation)

        rotations = torch.eye((3))

        # Reshape to (batch, N_res, 3, 3)
        rotations = rotations.view(1, 1, 3, 3).expand(s.shape[0], s.shape[1], 3, 3)

        translations = torch.zeros((s.shape[0], s.shape[1], 3))

        # Collect intermediates for auxiliary losses
        all_rotations = []
        all_translations = []
        all_torsion_angles = []
        alpha = torch.zeros((s.shape[0], s.shape[1], 7, 2), device=single_representation.device)

        for l in range(self.num_layers):
            # Stop rotation gradients between iterations (Alg. 20, lines 19-20)
            if l < self.num_layers - 1:
                rotations = rotations.detach()

            s += self.IPA(s, pair_representation, rotations, translations)

            s = self.layer_norm_single_rep_1(self.dropout_1(s))

            # Transition
            s += self.transition_linear_3(self.relu(self.transition_linear_2(self.relu(self.transition_linear_1(s)))))
            s = self.layer_norm_single_rep_2(self.dropout_2(s))

            # Update backbone
            new_rotations, new_translations = self.backbone_update(s)

            rotations = torch.einsum('bsij, bsjk -> bsik', rotations, new_rotations)

            temp = torch.einsum('bsij, bsi -> bsj', new_rotations, translations)

            translations = temp + translations

            # Predict side chain and backbone torsion angles
            a = self.angle_linear_1(s) + self.angle_linear_2(single_representation)

            a += self.angle_linear_4(self.relu(self.angle_linear_3(self.relu(a))))

            a += self.angle_linear_6(self.relu(self.angle_linear_5(self.relu(a))))

            alpha = self.angle_linear_7(self.relu(a))

            all_rotations.append(rotations)
            all_translations.append(translations)
            all_torsion_angles.append(alpha)

        all_rotations = torch.stack(all_rotations)
        all_translations = torch.stack(all_translations)
        all_torsion_angles = torch.stack(all_torsion_angles)

        # Alg. 20, line 24: compute all-atom coordinates from final backbone + torsions
        all_frames_R, all_frames_t, atom_coords, mask = compute_all_atom_coordinates(
            translations,
            rotations,
            alpha,
            aatype
        )

        # Alg. 20, line 25: concat backbone frame T_i with per-group frames T_i^f
        # all_frames already has backbone as frame 0, but we prepend the standalone
        # backbone frame so the FAPE loss can address it separately (9 frames total)
        backbone_R = rotations.unsqueeze(2)      # (batch, N_res, 1, 3, 3)
        backbone_t = translations.unsqueeze(2)    # (batch, N_res, 1, 3)
        all_frames_R = torch.cat([backbone_R, all_frames_R], dim=2)  # (batch, N_res, 9, 3, 3)
        all_frames_t = torch.cat([backbone_t, all_frames_t], dim=2)  # (batch, N_res, 9, 3)

        predictions = {
            # Per-layer backbone frames for auxiliary FAPE loss
            "traj_rotations": all_rotations,          # (L, batch, N_res, 3, 3)
            "traj_translations": all_translations,     # (L, batch, N_res, 3)

            # Per-layer torsion angles for torsion angle loss
            "traj_torsion_angles": all_torsion_angles, # (L, batch, N_res, 7, 2)

            # Final backbone frames
            "final_rotations": rotations,              # (batch, N_res, 3, 3)
            "final_translations": translations,        # (batch, N_res, 3)

            # Final all-atom outputs (frames include backbone + 8 groups = 9 total)
            "all_frames_R": all_frames_R,              # (batch, N_res, 9, 3, 3)
            "all_frames_t": all_frames_t,              # (batch, N_res, 9, 3)
            "atom14_coords": atom_coords,            # (batch, N_res, 14, 3)
            "atom14_mask": mask,                # (batch, N_res, 14)

            # Final single representation (for distogram, pLDDT, etc.)
            "single": s,                               # (batch, N_res, c_s)
        }

        return predictions


class InvariantPointAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.ipa_num_heads
        self.head_dim = config.ipa_c
        self.total_dim = self.head_dim * self.num_heads
        self.n_query_points = config.ipa_n_query_points
        self.n_value_points = config.ipa_n_value_points

        self.linear_q_rep = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)
        self.linear_k_rep = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)
        self.linear_v_rep = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)

        self.linear_q_frames = torch.nn.Linear(in_features=config.c_s, out_features=3*self.num_heads*self.n_query_points, bias=False)
        self.linear_k_frames = torch.nn.Linear(in_features=config.c_s, out_features=3*self.num_heads*self.n_query_points, bias=False)
        self.linear_v_frames = torch.nn.Linear(in_features=config.c_s, out_features=3*self.num_heads*self.n_value_points, bias=False)

        self.linear_bias = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=False)

        self.linear_output = torch.nn.Linear(
            in_features=self.total_dim 
                + self.num_heads * self.n_value_points * 3 
                + self.num_heads * self.n_value_points      # norms
                + self.num_heads * config.c_z,
            out_features=config.c_s
        )

        self.w_c = math.sqrt(2/(9 * self.n_query_points))
        self.w_l = math.sqrt(1/3)

        self.head_weights = torch.nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, single_representation: torch.Tensor, pair_representation: torch.Tensor, rotations: torch.Tensor, translation: torch.Tensor):
        # single_rep shape: (batch, N_res, c_s)
        # pair_rep shape: (batch, N_res, N_res, c_z)
        # rotations shape: (batch, N_res, 3, 3)
        # translations shape: (batch, N_res, 3)

        batch_size = single_representation.shape[0]
        N_res = single_representation.shape[1]

        # Shapes (batch, N_res, self.total_dim)
        Q_rep = self.linear_q_rep(single_representation)
        K_rep = self.linear_k_rep(single_representation)
        V_rep = self.linear_v_rep(single_representation)

        Q_rep = Q_rep.reshape((Q_rep.shape[0], Q_rep.shape[1], self.num_heads, self.head_dim))
        K_rep = K_rep.reshape((K_rep.shape[0], K_rep.shape[1], self.num_heads, self.head_dim))
        V_rep = V_rep.reshape((V_rep.shape[0], V_rep.shape[1], self.num_heads, self.head_dim))

        # Shapes (batch, N_res, 3*self.num_heads*self.n_query_points)
        Q_frames = self.linear_q_frames(single_representation)
        K_frames = self.linear_k_frames(single_representation)

        Q_frames = Q_frames.reshape((Q_frames.shape[0], Q_frames.shape[1], self.num_heads, self.n_query_points, -1))
        K_frames = K_frames.reshape((K_frames.shape[0], K_frames.shape[1], self.num_heads, self.n_query_points, -1))

        # Shape (batch, N_res, 3*self.num_heads*self.n_value_points)
        V_frames = self.linear_v_frames(single_representation)
        V_frames = V_frames.reshape((V_frames.shape[0], V_frames.shape[1], self.num_heads, self.n_value_points, -1))

        # Shape (batch, N_res, N_res, self.num_heads)
        B = self.linear_bias(pair_representation)

        # Query frames shape: (batch, N_res, self.num_heads, self.n_query_points, 3)
        # Rotations shape: (batch, N_res, 3, 3)
        # Translation shape: (batch, N_res, 3)
        # Output shape: (batch, N_res, self.num_heads, self.n_query_points, 3)
        global_frame_q = torch.einsum('biop, bihqp -> bihqo', rotations, Q_frames) + translation[:, :, None, None, :]

        global_frame_k = torch.einsum('biop, bihqp -> bihqo', rotations, K_frames) + translation[:, :, None, None, :]

        # Difference: (batch, N_res_i, N_res_j, num_heads, n_query_points, 3)
        diff = global_frame_q[:, :, None, :, :, :] - global_frame_k[:, None, :, :, :, :]

        gamma = torch.nn.functional.softplus(self.head_weights)

        # Squared distances summed over points and xyz: (batch, i, j, h)
        point_scores = -0.5 * gamma[None, None, None, :] * self.w_c * torch.sum(diff ** 2, dim=(-1, -2))

        # Shape (batch, N_res, N_res, self.num_heads)
        rep_scores = torch.einsum('bihd, bjhd -> bijh', Q_rep, K_rep) / math.sqrt(self.head_dim) + B

        # Shape (batch, N_res, N_res, self.num_heads)
        scores = self.w_l * (rep_scores + point_scores)

        # Shape (batch, N_res, N_res, self.num_heads)
        attention = torch.nn.functional.softmax(scores, dim=-2)

        # Output shape: (batch, N_res self.num_heads, c_z)
        output_pair = torch.einsum('bijh, bijd -> bihd', attention, pair_representation)

        # Output shape: (batch, self.num_heads, N_res, self.head_dim)
        output_rep = torch.einsum('bijh, bjhd -> bhid', attention, V_rep)

        # Output shape: (batch, N_res, self.num_heads, self.n_value_points, 3)
        global_values = torch.einsum('biop, bihqp -> bihqo', rotations, V_frames) + translation[:, :, None, None, :]

        temp = torch.einsum('bijh, bjhqo->bhiqo', attention, global_values)

        # R^T
        R_inv = rotations.transpose(-1, -2)                          
        t_inv = -torch.einsum('bira, bia -> bir', R_inv, translation)
        
        output_values = torch.einsum('biop, bhiqp -> bhiqo', R_inv, temp) + t_inv[:, None, :, None, :]

        # output_rep: (batch, h, N_res, head_dim) → (batch, N_res, h * head_dim)
        output_rep = output_rep.permute(0, 2, 1, 3).reshape(batch_size, N_res, -1)

        # output_values: (batch, h, N_res, n_value_points, 3) → (batch, N_res, h, n_value_points, 3)
        output_values = output_values.permute(0, 2, 1, 3, 4)

        # (batch, N_res, h * n_value_points * 3)
        output_values = output_values.reshape(batch_size, N_res, -1)

        # (batch, N_res, h, n_value_points)
        output_norms = torch.sqrt(torch.sum(output_values ** 2, dim=-1) + 1e-8)  

        # (batch, N_res, h * n_value_points)
        output_norms = output_norms.reshape(batch_size, N_res, -1)

        # output_pair: (batch, N_res, h, c_z) → (batch, N_res, h * c_z)
        output_pair = output_pair.reshape(batch_size, N_res, -1)

        # Final concatenation and projection
        output = torch.cat([output_rep, output_values, output_norms, output_pair], dim=-1)
        output = self.linear_output(output)

        return output
    
class BackboneUpdate(torch.nn.Module):
    def __init__(self, config):
        self.linear = torch.nn.Linear(in_features=config.c_s, out_features=6)

    def forward(self, single_representation: torch.Tensor):
        # output shape; (batch, N_res, 6)
        vals = self.linear(single_representation)

        # Rotation quaternions
        b = vals[:, :, 0]
        c = vals[:, :, 1]
        d = vals[:, :, 2]

        a = torch.ones_like(b)
        norm = torch.sqrt(1 + b**2 + c**2 + d**2)

        a = a / norm
        b = b / norm
        c = c / norm
        d = d / norm

        # Construct pairwise multiplications for rotation matrix
        aa = a*a
        bb = b*b
        cc = c*c
        dd = d*d

        ab = a*b
        ac = a*c
        ad = a*d

        bc = b*c
        bd = b*d

        cd = c*d

        # Construct rotation matrix entries
        r11 = aa + bb - cc - dd
        r12 = 2*bc - 2*ad
        r13 = 2*bd + 2*ac

        r21 = 2*bc + 2*ad
        r22 = aa - bb + cc - dd
        r23 = 2*cd - 2*ab

        r31 = 2*bd - 2*ac
        r32 = 2*cd + 2*ab
        r33 = aa - bb - cc + dd

        # Output shape: (batch, N_res, 3, 3)
        R = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=-1).reshape((single_representation.shape[0], single_representation.shape[1], 3, 3))

        t = vals[:, :, 3:]

        return R, t
    
def make_rot_x(alpha: torch.Tensor):
    a1 = alpha[..., 0]
    a2 = alpha[..., 1]

    zeros = torch.zeros_like(a1)
    ones = torch.ones_like(a1)

    R = torch.stack([
        torch.stack([ones,  zeros, zeros], dim=-1),
        torch.stack([zeros, a1,    -a2],   dim=-1),
        torch.stack([zeros, a2,     a1],   dim=-1),
    ], dim=-2)

    t = torch.zeros(*alpha.shape[:-1], 3, device=alpha.device, dtype=alpha.dtype)

    return R, t

def compose_transforms(R1, t1, R2, t2):
    """Compose two rigid transforms: T1 ∘ T2 = (R1 @ R2, R1 @ t2 + t1)"""
    R = R1 @ R2
    t = (R1 @ t2.unsqueeze(-1)).squeeze(-1) + t1
    return R, t


def compute_all_atom_coordinates(
    translations: torch.Tensor,   # (batch, N_res, 3)
    rotations: torch.Tensor,      # (batch, N_res, 3, 3)
    torsion_angles: torch.Tensor, # (batch, N_res, 7, 2) — [ω, φ, ψ, χ1, χ2, χ3, χ4]
    aatype: torch.Tensor,         # (batch, N_res) — integer residue type indices
):
    device = translations.device
    dtype = translations.dtype

    # --- Step 1: Normalize torsion angles to unit vectors ---
    torsion_angles = torsion_angles / (torch.norm(torsion_angles, dim=-1, keepdim=True) + 1e-8)

    # --- Step 2: Get literature constants, indexed by residue type ---
    default_frames = torch.tensor(
        restype_rigid_group_default_frame, device=device, dtype=dtype
    )  # (21, 8, 4, 4)
    lit_all = default_frames[aatype]          # (batch, N_res, 8, 4, 4)
    lit_R = lit_all[..., :3, :3]              # (batch, N_res, 8, 3, 3)
    lit_t = lit_all[..., :3, 3]               # (batch, N_res, 8, 3)

    # --- Step 3: Build torsion rotations via makeRotX ---
    torsion_R, torsion_t = make_rot_x(torsion_angles)  # (batch, N_res, 7, 3, 3), (batch, N_res, 7, 3)

    # --- Step 4: Build all 8 frames ---
    frames_R = [rotations]   # Frame 0: backbone
    frames_t = [translations]

    # Frames 1–4: T_i ∘ T_lit[f] ∘ makeRotX(angle_f)
    # Each branches independently from the backbone frame
    for f in range(4):
        mid_R, mid_t = compose_transforms(
            lit_R[:, :, f + 1], lit_t[:, :, f + 1],
            torsion_R[:, :, f], torsion_t[:, :, f],
        )
        frame_R, frame_t = compose_transforms(rotations, translations, mid_R, mid_t)
        frames_R.append(frame_R)
        frames_t.append(frame_t)

    # Frames 5–7: chain sequentially from previous sidechain frame
    # Frame 5 = T_i4 ∘ T_lit[5] ∘ makeRotX(χ2)
    # Frame 6 = T_i5 ∘ T_lit[6] ∘ makeRotX(χ3)
    # Frame 7 = T_i6 ∘ T_lit[7] ∘ makeRotX(χ4)
    for f in range(3):
        prev_R = frames_R[f + 4]
        prev_t = frames_t[f + 4]
        mid_R, mid_t = compose_transforms(
            lit_R[:, :, f + 5], lit_t[:, :, f + 5],
            torsion_R[:, :, f + 4], torsion_t[:, :, f + 4],
        )
        frame_R, frame_t = compose_transforms(prev_R, prev_t, mid_R, mid_t)
        frames_R.append(frame_R)
        frames_t.append(frame_t)

    all_frames_R = torch.stack(frames_R, dim=2)  # (batch, N_res, 8, 3, 3)
    all_frames_t = torch.stack(frames_t, dim=2)  # (batch, N_res, 8, 3)

    # --- Step 5: Place atoms using their frame assignments ---
    lit_pos = torch.tensor(
        restype_atom14_rigid_group_positions, device=device, dtype=dtype
    )[aatype]  # (batch, N_res, 14, 3)

    atom_frame_idx = torch.tensor(
        restype_atom14_to_rigid_group, device=device, dtype=torch.long
    )[aatype]  # (batch, N_res, 14)

    mask = torch.tensor(
        restype_atom14_mask, device=device, dtype=dtype
    )[aatype]  # (batch, N_res, 14)

    # Gather the correct frame for each atom
    idx_R = atom_frame_idx[:, :, :, None, None].expand(-1, -1, -1, 3, 3)
    atom_R = torch.gather(all_frames_R, 2, idx_R)  # (batch, N_res, 14, 3, 3)

    idx_t = atom_frame_idx[:, :, :, None].expand(-1, -1, -1, 3)
    atom_t = torch.gather(all_frames_t, 2, idx_t)  # (batch, N_res, 14, 3)

    # x_global = R_frame @ x_lit + t_frame
    atom_coords = torch.einsum('bnaji, bnaj -> bnai', atom_R, lit_pos) + atom_t

    return all_frames_R, all_frames_t, atom_coords, mask