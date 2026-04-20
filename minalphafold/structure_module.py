import torch
import math
from typing import Optional

try:
    from .residue_constants import (
        restype_rigid_group_default_frame,
        restype_atom14_rigid_group_positions,
        restype_atom14_to_rigid_group,
        restype_atom14_mask,
    )
except ImportError:  # pragma: no cover - compatibility for direct module imports in tests/scripts.
    from residue_constants import (
        restype_rigid_group_default_frame,
        restype_atom14_rigid_group_positions,
        restype_atom14_to_rigid_group,
        restype_atom14_mask,
    )


def _truncated_normal_(tensor: torch.Tensor, std: float) -> None:
    torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def _init_linear(linear: torch.nn.Linear, init: str = "default") -> None:
    fan_in = linear.weight.shape[1]
    with torch.no_grad():
        if linear.bias is not None:
            linear.bias.zero_()
        if init == "default":
            _truncated_normal_(linear.weight, std=math.sqrt(1.0 / fan_in))
        elif init == "relu":
            _truncated_normal_(linear.weight, std=math.sqrt(2.0 / fan_in))
        elif init == "glorot":
            torch.nn.init.xavier_uniform_(linear.weight, gain=1.0)
        elif init == "final":
            linear.weight.zero_()
        else:
            raise ValueError(f"Unknown linear init: {init}")


class AngleResnetBlock(torch.nn.Module):
    def __init__(self, c_hidden: int):
        super().__init__()
        self.linear_1 = torch.nn.Linear(c_hidden, c_hidden)
        self.linear_2 = torch.nn.Linear(c_hidden, c_hidden)
        _init_linear(self.linear_1, init="relu")
        _init_linear(self.linear_2, init="final")
        self.relu = torch.nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        residual = a
        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)
        return a + residual


class AngleResnet(torch.nn.Module):
    """Minimal AF2/OpenFold-style torsion angle head."""

    def __init__(self, c_in: int, c_hidden: int, num_blocks: int, num_angles: int, epsilon: float = 1e-8):
        super().__init__()
        self.linear_in = torch.nn.Linear(c_in, c_hidden)
        self.linear_initial = torch.nn.Linear(c_in, c_hidden)
        _init_linear(self.linear_in, init="default")
        _init_linear(self.linear_initial, init="default")

        self.blocks = torch.nn.ModuleList([AngleResnetBlock(c_hidden) for _ in range(num_blocks)])
        self.linear_out = torch.nn.Linear(c_hidden, num_angles * 2)
        _init_linear(self.linear_out, init="default")

        self.relu = torch.nn.ReLU()
        self.epsilon = epsilon

    def forward(self, single_representation: torch.Tensor, initial_single_representation: torch.Tensor):
        single_act = self.linear_in(self.relu(single_representation))
        initial_act = self.linear_initial(self.relu(initial_single_representation))
        sidechain_act = single_act + initial_act

        for block in self.blocks:
            sidechain_act = block(sidechain_act)

        unnormalized_angles = self.linear_out(self.relu(sidechain_act)).reshape(
            *sidechain_act.shape[:-1],
            -1,
            2,
        )
        norm_denom = torch.sqrt(
            torch.clamp(torch.sum(unnormalized_angles ** 2, dim=-1, keepdim=True), min=self.epsilon)
        )
        angles = unnormalized_angles / norm_denom
        return unnormalized_angles, angles

class StructureModule(torch.nn.Module):
    default_frames: torch.Tensor
    lit_positions: torch.Tensor
    atom_frame_idx_table: torch.Tensor
    atom_mask_table: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.c = config.structure_module_c
        self.num_layers = config.structure_module_layers
        self.position_scale = float(getattr(config, "position_scale", 10.0))

        # Layer Norms
        self.layer_norm_single_rep_1 = torch.nn.LayerNorm(config.c_s)
        self.layer_norm_single_rep_2 = torch.nn.LayerNorm(config.c_s)
        self.layer_norm_single_rep_3 = torch.nn.LayerNorm(config.c_s)

        self.layer_norm_pair_rep = torch.nn.LayerNorm(config.c_z)

        # Dropouts (rates from config)
        self.dropout_1 = torch.nn.Dropout(p=config.structure_module_dropout_ipa)
        self.dropout_2 = torch.nn.Dropout(p=config.structure_module_dropout_transition)

        # Register residue constant tensors as buffers (avoid device bugs, improve speed)
        self.register_buffer('default_frames',
            torch.tensor(restype_rigid_group_default_frame))   # (21, 8, 4, 4)
        self.register_buffer('lit_positions',
            torch.tensor(restype_atom14_rigid_group_positions)) # (21, 14, 3)
        self.register_buffer('atom_frame_idx_table',
            torch.tensor(restype_atom14_to_rigid_group))        # (21, 14)
        self.register_buffer('atom_mask_table',
            torch.tensor(restype_atom14_mask))                  # (21, 14)

        # Linear layers
        self.single_rep_proj = torch.nn.Linear(in_features=config.c_s, out_features=config.c_s)
        self.transition_linear_1 = torch.nn.Linear(in_features=config.c_s, out_features=config.c_s)
        self.transition_linear_2 = torch.nn.Linear(in_features=config.c_s, out_features=config.c_s)
        self.transition_linear_3 = torch.nn.Linear(in_features=config.c_s, out_features=config.c_s)
        _init_linear(self.single_rep_proj, init="default")
        _init_linear(self.transition_linear_1, init="relu")
        _init_linear(self.transition_linear_2, init="relu")
        _init_linear(self.transition_linear_3, init="final")

        # Core blocks
        self.IPA = InvariantPointAttention(config)
        self.backbone_update = BackboneUpdate(config)
        self.sidechain_module = MultiRigidSidechain(config)

        self.relu = torch.nn.ReLU()

        # AF2 keeps structure-module translations in internal units and rescales
        # them by position_scale when materializing coordinates and losses.
        internal_scale = 1.0 / self.position_scale
        self.default_frames[..., :3, 3] *= internal_scale
        self.lit_positions *= internal_scale

    def forward(self, single_representation: torch.Tensor, pair_representation: torch.Tensor,
                aatype: torch.Tensor, seq_mask: Optional[torch.Tensor] = None,
                detach_rotations: bool = True):
        # seq_mask: (batch, N_res) — 1 for valid residues, 0 for padding
        # detach_rotations: if True (default, AF2 standard), apply stopgrad to
        #   the rotation component of T_i between iterations (Algorithm 20
        #   lines 19-21). The detach is placed at the *end* of each non-final
        #   iteration so that the next iteration's IPA sees T_i with no
        #   rotation-gradient path, preventing lever effects through the
        #   chained composition of frames. Set to False to allow full gradient
        #   flow (useful for memorization/debugging).
        assert single_representation.ndim == 3, \
            f"single_representation must be (batch, N_res, c_s), got {single_representation.shape}"
        assert pair_representation.ndim == 4, \
            f"pair_representation must be (batch, N_res, N_res, c_z), got {pair_representation.shape}"
        assert aatype.ndim == 2, \
            f"aatype must be (batch, N_res), got {aatype.shape}"
        assert single_representation.shape[1] == pair_representation.shape[1] == pair_representation.shape[2] == aatype.shape[1], \
            f"N_res mismatch: single={single_representation.shape[1]}, pair={pair_representation.shape[1:3]}, aatype={aatype.shape[1]}"

        single_representation = self.layer_norm_single_rep_1(single_representation)
        initial_single_representation = single_representation

        pair_representation = self.layer_norm_pair_rep(pair_representation)

        s = self.single_rep_proj(single_representation)

        rotations = torch.eye(3, device=s.device, dtype=s.dtype).view(1, 1, 3, 3).expand(s.shape[0], s.shape[1], 3, 3)

        translations = torch.zeros(s.shape[0], s.shape[1], 3, device=s.device, dtype=s.dtype)

        # Collect intermediates for auxiliary losses
        all_rotations = []
        all_translations = []
        all_torsion_angles = []
        all_torsion_angles_unnormalized = []
        sidechain_outputs = None

        for l in range(self.num_layers):
            # Algorithm 20, line 6-7: s += IPA(s); s = LN(Dropout(s))
            s = s + self.IPA(s, pair_representation, rotations, translations, seq_mask)
            s = self.layer_norm_single_rep_3(self.dropout_1(s))

            # Algorithm 20, line 8-9: s += Transition(s); s = LN(Dropout(s))
            s = s + self.transition_linear_3(self.relu(self.transition_linear_2(self.relu(self.transition_linear_1(s)))))
            s = self.layer_norm_single_rep_2(self.dropout_2(s))

            # Algorithm 20, line 10: T_i ← T_i ∘ BackboneUpdate(s_i)
            new_rotations, new_translations = self.backbone_update(s)
            translations = torch.einsum('bsij, bsj -> bsi', rotations, new_translations) + translations
            rotations = torch.einsum('bsij, bsjk -> bsik', rotations, new_rotations)

            # Algorithm 20, lines 11-18: side-chain torsions and auxiliary losses use
            # the updated (un-detached) T_i so gradients reach this iteration's s.
            sidechain_outputs = self.sidechain_module(
                s,
                initial_single_representation,
                rotations,
                translations,
                aatype,
                self.default_frames,
                self.lit_positions,
                self.atom_frame_idx_table,
                self.atom_mask_table,
            )

            all_rotations.append(rotations)
            all_translations.append(translations)
            all_torsion_angles.append(sidechain_outputs["angles_sin_cos"])
            all_torsion_angles_unnormalized.append(sidechain_outputs["unnormalized_angles_sin_cos"])

            # Algorithm 20, lines 19-21: stopgrad on rotations between iterations
            # (but not after the final one). Detaching *after* the backbone update
            # means iteration l+1 receives the rotation without a gradient path,
            # exactly as the supplement prescribes.
            if detach_rotations and l < self.num_layers - 1:
                rotations = rotations.detach()

        all_rotations = torch.stack(all_rotations)
        all_translations = torch.stack(all_translations)
        all_torsion_angles = torch.stack(all_torsion_angles)
        all_torsion_angles_unnormalized = torch.stack(all_torsion_angles_unnormalized)
        if sidechain_outputs is None:
            raise RuntimeError("StructureModule produced no sidechain outputs.")

        all_frames_R = sidechain_outputs["frames_R"]
        all_frames_t = sidechain_outputs["frames_t"]
        atom_coords = sidechain_outputs["atom_pos"]
        mask = sidechain_outputs["atom_mask"]

        # Convert internal structure-module units back to angstroms.
        # Rotations are unitless — no conversion needed
        predictions = {
            # Per-layer backbone frames for auxiliary FAPE loss
            "traj_rotations": all_rotations,               # (num_layers, batch, N_res, 3, 3)
            "traj_translations": all_translations * self.position_scale,

            # Per-layer torsion angles for torsion angle loss
            "traj_torsion_angles": all_torsion_angles,     # (num_layers, batch, N_res, 7, 2)
            "traj_torsion_angles_unnormalized": all_torsion_angles_unnormalized,

            # Final backbone frames
            "final_rotations": rotations,                  # (batch, N_res, 3, 3)
            "final_translations": translations * self.position_scale,

            # Final all-atom outputs (8 rigid-group frames including backbone frame 0)
            "all_frames_R": all_frames_R,                  # (batch, N_res, 8, 3, 3)
            "all_frames_t": all_frames_t * self.position_scale,
            "atom14_coords": atom_coords * self.position_scale,
            "atom14_mask": mask,                           # (batch, N_res, 14)

            # Final single representation (for distogram, pLDDT, etc.)
            "single": s,                                   # (batch, N_res, c_s)
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
        self.inf = 1e5
        self.eps = 1e-8

        # Canonical AF2/OpenFold monomer IPA uses a biased query projection and
        # combined key/value projections for both scalar and point features.
        self.linear_q = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=True)
        self.linear_kv = torch.nn.Linear(in_features=config.c_s, out_features=2 * self.total_dim, bias=True)

        self.linear_q_points = torch.nn.Linear(
            in_features=config.c_s,
            out_features=3 * self.num_heads * self.n_query_points,
            bias=True,
        )
        self.linear_kv_points = torch.nn.Linear(
            in_features=config.c_s,
            out_features=3 * self.num_heads * (self.n_query_points + self.n_value_points),
            bias=True,
        )

        self.linear_bias = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=True)

        self.linear_output = torch.nn.Linear(
            in_features=self.total_dim
            + self.num_heads * self.n_value_points * 4
            + self.num_heads * config.c_z,
            out_features=config.c_s,
        )

        _init_linear(self.linear_q, init="default")
        _init_linear(self.linear_kv, init="default")
        _init_linear(self.linear_q_points, init="default")
        _init_linear(self.linear_kv_points, init="default")
        _init_linear(self.linear_bias, init="default")
        _init_linear(self.linear_output, init="final")

        self.head_weights = torch.nn.Parameter(torch.zeros(self.num_heads))

    def _project_points(
        self,
        linear: torch.nn.Linear,
        single_representation: torch.Tensor,
        num_points: int,
    ) -> torch.Tensor:
        raw_points = linear(single_representation)
        x_coords, y_coords, z_coords = torch.chunk(raw_points, 3, dim=-1)
        point_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        return point_coords.reshape(
            single_representation.shape[0],
            single_representation.shape[1],
            self.num_heads,
            num_points,
            3,
        )

    def _assemble_output_features(
        self,
        attention: torch.Tensor,
        value_scalar: torch.Tensor,
        value_points_global: torch.Tensor,
        pair_representation: torch.Tensor,
        rotations: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rotations.shape[0]
        num_residues = rotations.shape[1]

        output_scalar = torch.matmul(
            attention,
            value_scalar.permute(0, 2, 1, 3).to(dtype=attention.dtype),
        ).permute(0, 2, 1, 3)
        output_scalar = output_scalar.reshape(batch_size, num_residues, -1)

        result_point_global = torch.einsum(
            "bhij,bjhpc->bihpc",
            attention,
            value_points_global.to(dtype=attention.dtype),
        )
        result_point_local = torch.einsum(
            "biop,bihqp->bihqo",
            rotations.transpose(-1, -2),
            result_point_global - translation[:, :, None, None, :],
        )
        result_point_norms = torch.sqrt(torch.sum(result_point_local ** 2, dim=-1) + self.eps)
        result_point_norms = result_point_norms.reshape(batch_size, num_residues, -1)

        # Canonical monomer IPA concatenates x/y/z point channels separately.
        result_point_local = result_point_local.reshape(batch_size, num_residues, -1, 3)
        result_point_x, result_point_y, result_point_z = result_point_local.unbind(dim=-1)

        output_pair = torch.einsum(
            "bhij,bijd->bihd",
            attention,
            pair_representation.to(dtype=attention.dtype),
        )
        output_pair = output_pair.reshape(batch_size, num_residues, -1)

        return torch.cat(
            [
                output_scalar,
                result_point_x,
                result_point_y,
                result_point_z,
                result_point_norms,
                output_pair,
            ],
            dim=-1,
        )

    def _forward_output_features(
        self,
        single_representation: torch.Tensor,
        pair_representation: torch.Tensor,
        rotations: torch.Tensor,
        translation: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = single_representation.shape[0]
        num_residues = single_representation.shape[1]

        q = self.linear_q(single_representation).reshape(batch_size, num_residues, self.num_heads, self.head_dim)
        kv = self.linear_kv(single_representation).reshape(batch_size, num_residues, self.num_heads, 2 * self.head_dim)
        k, v = torch.split(kv, self.head_dim, dim=-1)

        q_points = self._project_points(self.linear_q_points, single_representation, self.n_query_points)
        kv_points = self._project_points(
            self.linear_kv_points,
            single_representation,
            self.n_query_points + self.n_value_points,
        )
        k_points, v_points = torch.split(
            kv_points,
            [self.n_query_points, self.n_value_points],
            dim=-2,
        )

        query_points_global = torch.einsum("biop,bihqp->bihqo", rotations, q_points) + translation[:, :, None, None, :]
        key_points_global = torch.einsum("biop,bihqp->bihqo", rotations, k_points) + translation[:, :, None, None, :]
        value_points_global = torch.einsum("biop,bihqp->bihqo", rotations, v_points) + translation[:, :, None, None, :]

        bias = self.linear_bias(pair_representation)

        attention_logits = torch.matmul(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 3, 1),
        )
        attention_logits *= math.sqrt(1.0 / (3.0 * self.head_dim))
        attention_logits += math.sqrt(1.0 / 3.0) * bias.permute(0, 3, 1, 2)

        point_attention = query_points_global[:, :, None, :, :, :] - key_points_global[:, None, :, :, :, :]
        point_attention = torch.sum(point_attention ** 2, dim=-1)
        head_weights = torch.nn.functional.softplus(self.head_weights).view(1, 1, 1, self.num_heads, 1)
        head_weights = head_weights * math.sqrt(1.0 / (3.0 * (self.n_query_points * 9.0 / 2.0)))
        point_attention = torch.sum(point_attention * head_weights, dim=-1) * (-0.5)
        attention_logits += point_attention.permute(0, 3, 1, 2)

        if seq_mask is not None:
            square_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)
            attention_logits = attention_logits + self.inf * (square_mask[:, None, :, :] - 1.0)

        attention = torch.nn.functional.softmax(attention_logits, dim=-1)
        return self._assemble_output_features(
            attention=attention,
            value_scalar=v,
            value_points_global=value_points_global,
            pair_representation=pair_representation,
            rotations=rotations,
            translation=translation,
        )

    def forward(self, single_representation: torch.Tensor, pair_representation: torch.Tensor,
                rotations: torch.Tensor, translation: torch.Tensor,
                seq_mask: Optional[torch.Tensor] = None):
        # single_rep shape: (batch, N_res, c_s)
        # pair_rep shape: (batch, N_res, N_res, c_z)
        # rotations shape: (batch, N_res, 3, 3)
        # translations shape: (batch, N_res, 3)
        # seq_mask shape: (batch, N_res) — 1 for valid, 0 for padding
        assert rotations.shape[-2:] == (3, 3), \
            f"rotations must end with (3, 3), got {rotations.shape}"
        assert translation.shape[-1] == 3, \
            f"translation must end with (3,), got {translation.shape}"

        output_features = self._forward_output_features(
            single_representation,
            pair_representation,
            rotations,
            translation,
            seq_mask=seq_mask,
        )
        output = self.linear_output(output_features)

        # Zero out padded query positions
        if seq_mask is not None:
            output = output * seq_mask[:, :, None]

        return output
    
class BackboneUpdate(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=config.c_s, out_features=6)
        _init_linear(self.linear, init="final")

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


class MultiRigidSidechain(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c = getattr(config, "sidechain_num_channel", config.structure_module_c)
        self.num_residual_blocks = getattr(config, "sidechain_num_residual_block", 2)
        self.angle_resnet = AngleResnet(
            c_in=config.c_s,
            c_hidden=self.c,
            num_blocks=self.num_residual_blocks,
            num_angles=7,
        )

    def forward(
        self,
        single_representation: torch.Tensor,
        initial_single_representation: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        aatype: torch.Tensor,
        default_frames: torch.Tensor,
        lit_positions: torch.Tensor,
        atom_frame_idx_table: torch.Tensor,
        atom_mask_table: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        unnormalized_angles, angles = self.angle_resnet(
            single_representation,
            initial_single_representation,
        )

        frames_R, frames_t, atom_pos, atom_mask = compute_all_atom_coordinates(
            translations,
            rotations,
            angles,
            aatype,
            default_frames,
            lit_positions,
            atom_frame_idx_table,
            atom_mask_table,
        )

        return {
            "angles_sin_cos": angles,
            "unnormalized_angles_sin_cos": unnormalized_angles,
            "frames_R": frames_R,
            "frames_t": frames_t,
            "atom_pos": atom_pos,
            "atom_mask": atom_mask,
        }
    
def make_rot_x(alpha: torch.Tensor):
    # alpha is (sin, cos) pairs; extract cos and sin for rotation matrix
    a1 = alpha[..., 1]  # cos
    a2 = alpha[..., 0]  # sin

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
    default_frames: torch.Tensor, # (21, 8, 4, 4) — registered buffer
    lit_positions: torch.Tensor,  # (21, 14, 3) — registered buffer
    atom_frame_idx_table: torch.Tensor,  # (21, 14) — registered buffer
    atom_mask_table: torch.Tensor,       # (21, 14) — registered buffer
):
    dtype = translations.dtype
    device = translations.device

    # --- Step 1: Normalize torsion angles to unit vectors ---
    torsion_angles = torsion_angles / (torch.norm(torsion_angles, dim=-1, keepdim=True) + 1e-8)

    # --- Step 2: Get literature constants, indexed by residue type ---
    lit_all = default_frames.to(device=device, dtype=dtype)[aatype]  # (batch, N_res, 8, 4, 4)
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
    lit_pos = lit_positions.to(device=device, dtype=dtype)[aatype]          # (batch, N_res, 14, 3)
    atom_frame_idx = atom_frame_idx_table.to(device=aatype.device)[aatype]  # (batch, N_res, 14)
    mask = atom_mask_table.to(device=device, dtype=dtype)[aatype]           # (batch, N_res, 14)

    # Gather the correct frame for each atom
    idx_R = atom_frame_idx[:, :, :, None, None].expand(-1, -1, -1, 3, 3)
    atom_R = torch.gather(all_frames_R, 2, idx_R)  # (batch, N_res, 14, 3, 3)

    idx_t = atom_frame_idx[:, :, :, None].expand(-1, -1, -1, 3)
    atom_t = torch.gather(all_frames_t, 2, idx_t)  # (batch, N_res, 14, 3)

    # x_global = R_frame @ x_lit + t_frame
    atom_coords = torch.einsum('bnaij, bnaj -> bnai', atom_R, lit_pos) + atom_t

    return all_frames_R, all_frames_t, atom_coords, mask
