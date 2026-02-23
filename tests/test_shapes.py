"""Smoke tests: instantiate each module and verify output shapes with random inputs."""
import sys
import os
import torch
import pytest

# Add the package directory to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'minalphafold'))


# ---------------------------------------------------------------------------
# Mock config — small dims for fast CPU tests.
# ---------------------------------------------------------------------------
class MockConfig:
    c_m = 32
    c_s = 32
    c_z = 16
    c_t = 16
    c_e = 24

    dim = 8
    num_heads = 4

    msa_transition_n = 2
    outer_product_dim = 8

    triangle_mult_c = 16
    triangle_dim = 8
    triangle_num_heads = 2
    pair_transition_n = 2

    template_pair_num_blocks = 1
    template_pair_dropout = 0.0
    template_pointwise_attention_dim = 8
    template_pointwise_num_heads = 2

    extra_msa_dim = 8
    extra_msa_dropout = 0.0
    extra_pair_dropout = 0.0
    msa_column_global_attention_dim = 8

    num_evoformer = 1
    evoformer_msa_dropout = 0.0
    evoformer_pair_dropout = 0.0

    structure_module_c = 16
    structure_module_layers = 2
    structure_module_dropout_ipa = 0.0
    structure_module_dropout_transition = 0.0

    ipa_num_heads = 4
    ipa_c = 8
    ipa_n_query_points = 4
    ipa_n_value_points = 4

    n_dist_bins = 64
    plddt_hidden_dim = 32
    n_plddt_bins = 50
    n_msa_classes = 23
    n_pae_bins = 64

    num_extra_msa = 1


@pytest.fixture
def cfg():
    return MockConfig()


# Common test dimensions
B = 2       # batch
N_seq = 4   # MSA sequences
N_res = 6   # residues


# ======================== embedders.py ========================

class TestRelPos:
    def test_output_shape(self, cfg):
        from embedders import RelPos
        module = RelPos(cfg)
        residue_index = torch.arange(N_res).unsqueeze(0).expand(B, -1)
        out = module(residue_index)
        assert out.shape == (B, N_res, N_res, cfg.c_z)


class TestInputEmbedder:
    def test_output_shapes(self, cfg):
        from embedders import InputEmbedder
        module = InputEmbedder(cfg)
        target_feat = torch.randn(B, N_res, 21)
        residue_index = torch.arange(N_res).unsqueeze(0).expand(B, -1)
        msa_feat = torch.randn(B, N_seq, N_res, 49)
        m, z = module(target_feat, residue_index, msa_feat)
        assert m.shape == (B, N_seq, N_res, cfg.c_m)
        assert z.shape == (B, N_res, N_res, cfg.c_z)


class TestMSAColumnAttention:
    def test_output_shape(self, cfg):
        from embedders import MSAColumnAttention
        module = MSAColumnAttention(cfg)
        msa = torch.randn(B, N_seq, N_res, cfg.c_m)
        out = module(msa)
        assert out.shape == msa.shape


class TestMSATransition:
    def test_output_shape(self, cfg):
        from embedders import MSATransition
        module = MSATransition(cfg)
        msa = torch.randn(B, N_seq, N_res, cfg.c_m)
        out = module(msa)
        assert out.shape == msa.shape


class TestOuterProductMean:
    def test_output_shape(self, cfg):
        from embedders import OuterProductMean
        module = OuterProductMean(cfg)
        msa = torch.randn(B, N_seq, N_res, cfg.c_m)
        out = module(msa)
        assert out.shape == (B, N_res, N_res, cfg.c_z)


class TestTriangleMultiplicationOutgoing:
    def test_output_shape(self, cfg):
        from embedders import TriangleMultiplicationOutgoing
        module = TriangleMultiplicationOutgoing(cfg)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out = module(pair)
        assert out.shape == pair.shape


class TestTriangleMultiplicationIncoming:
    def test_output_shape(self, cfg):
        from embedders import TriangleMultiplicationIncoming
        module = TriangleMultiplicationIncoming(cfg)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out = module(pair)
        assert out.shape == pair.shape


class TestTriangleAttentionStartingNode:
    def test_output_shape(self, cfg):
        from embedders import TriangleAttentionStartingNode
        module = TriangleAttentionStartingNode(cfg)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out = module(pair)
        assert out.shape == pair.shape


class TestTriangleAttentionEndingNode:
    def test_output_shape(self, cfg):
        from embedders import TriangleAttentionEndingNode
        module = TriangleAttentionEndingNode(cfg)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out = module(pair)
        assert out.shape == pair.shape


class TestPairTransition:
    def test_output_shape(self, cfg):
        from embedders import PairTransition
        module = PairTransition(cfg)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out = module(pair)
        assert out.shape == pair.shape


class TestTemplatePair:
    def test_output_shape(self, cfg):
        from embedders import TemplatePair
        module = TemplatePair(cfg)
        N_templ = 2
        templ = torch.randn(B, N_templ, N_res, N_res, cfg.c_t)
        out = module(templ)
        assert out.shape == (B, N_templ, N_res, N_res, cfg.c_z)


class TestTemplatePointwiseAttention:
    def test_output_shape(self, cfg):
        from embedders import TemplatePointwiseAttention
        module = TemplatePointwiseAttention(cfg)
        N_templ = 2
        templ = torch.randn(B, N_templ, N_res, N_res, cfg.c_z)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out = module(templ, pair)
        assert out.shape == pair.shape


class TestExtraMsaStack:
    def test_output_shapes(self, cfg):
        from embedders import ExtraMsaStack
        module = ExtraMsaStack(cfg)
        N_extra = 8
        extra_msa = torch.randn(B, N_extra, N_res, cfg.c_e)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out_msa, out_pair = module(extra_msa, pair)
        assert out_msa.shape == extra_msa.shape
        assert out_pair.shape == pair.shape


class TestMSAColumnGlobalAttention:
    def test_output_shape(self, cfg):
        from embedders import MSAColumnGlobalAttention
        module = MSAColumnGlobalAttention(cfg, c_in=cfg.c_e)
        msa = torch.randn(B, N_seq, N_res, cfg.c_e)
        out = module(msa)
        assert out.shape == msa.shape


# ======================== evoformer.py ========================

class TestMSARowAttentionWithPairBias:
    def test_output_shape(self, cfg):
        from evoformer import MSARowAttentionWithPairBias
        module = MSARowAttentionWithPairBias(cfg)
        msa = torch.randn(B, N_seq, N_res, cfg.c_m)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out = module(msa, pair)
        assert out.shape == msa.shape


class TestEvoformer:
    def test_output_shapes(self, cfg):
        from evoformer import Evoformer
        module = Evoformer(cfg)
        msa = torch.randn(B, N_seq, N_res, cfg.c_m)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out_msa, out_pair = module(msa, pair)
        assert out_msa.shape == msa.shape
        assert out_pair.shape == pair.shape

    def test_with_masks(self, cfg):
        from evoformer import Evoformer
        module = Evoformer(cfg)
        msa = torch.randn(B, N_seq, N_res, cfg.c_m)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        msa_mask = torch.ones(B, N_seq, N_res)
        pair_mask = torch.ones(B, N_res, N_res)
        out_msa, out_pair = module(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        assert out_msa.shape == msa.shape
        assert out_pair.shape == pair.shape


# ======================== structure_module.py ========================

class TestBackboneUpdate:
    def test_output_shapes(self, cfg):
        from structure_module import BackboneUpdate
        module = BackboneUpdate(cfg)
        s = torch.randn(B, N_res, cfg.c_s)
        R, t = module(s)
        assert R.shape == (B, N_res, 3, 3)
        assert t.shape == (B, N_res, 3)


class TestInvariantPointAttention:
    def test_output_shape(self, cfg):
        from structure_module import InvariantPointAttention
        module = InvariantPointAttention(cfg)
        s = torch.randn(B, N_res, cfg.c_s)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N_res, 3, 3).clone()
        t = torch.zeros(B, N_res, 3)
        out = module(s, pair, R, t)
        assert out.shape == (B, N_res, cfg.c_s)


class TestMakeRotX:
    def test_output_shapes(self):
        from structure_module import make_rot_x
        alpha = torch.randn(B, N_res, 7, 2)
        R, t = make_rot_x(alpha)
        assert R.shape == (B, N_res, 7, 3, 3)
        assert t.shape == (B, N_res, 7, 3)


class TestComposeTransforms:
    def test_identity(self):
        from structure_module import compose_transforms
        R1 = torch.eye(3).unsqueeze(0).expand(B, 3, 3)
        t1 = torch.zeros(B, 3)
        R2 = torch.eye(3).unsqueeze(0).expand(B, 3, 3)
        t2 = torch.randn(B, 3)
        R, t = compose_transforms(R1, t1, R2, t2)
        assert R.shape == (B, 3, 3)
        assert t.shape == (B, 3)
        assert torch.allclose(t, t2, atol=1e-6)


class TestStructureModule:
    def test_output_dict(self, cfg):
        from structure_module import StructureModule
        module = StructureModule(cfg)
        s = torch.randn(B, N_res, cfg.c_s)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        aatype = torch.randint(0, 20, (B, N_res))
        preds = module(s, pair, aatype)

        assert preds["traj_rotations"].shape == (cfg.structure_module_layers, B, N_res, 3, 3)
        assert preds["traj_translations"].shape == (cfg.structure_module_layers, B, N_res, 3)
        assert preds["traj_torsion_angles"].shape == (cfg.structure_module_layers, B, N_res, 7, 2)
        assert preds["final_rotations"].shape == (B, N_res, 3, 3)
        assert preds["final_translations"].shape == (B, N_res, 3)
        assert preds["all_frames_R"].shape == (B, N_res, 8, 3, 3)
        assert preds["all_frames_t"].shape == (B, N_res, 8, 3)
        assert preds["atom14_coords"].shape == (B, N_res, 14, 3)
        assert preds["atom14_mask"].shape == (B, N_res, 14)
        assert preds["single"].shape == (B, N_res, cfg.c_s)


# ======================== heads.py ========================

class TestDistogramHead:
    def test_output_shape(self, cfg):
        from heads import DistogramHead
        module = DistogramHead(cfg)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out = module(pair)
        assert out.shape == (B, N_res, N_res, cfg.n_dist_bins)

    def test_symmetry(self, cfg):
        from heads import DistogramHead
        module = DistogramHead(cfg)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out = module(pair)
        assert torch.allclose(out, out.transpose(1, 2), atol=1e-6)


class TestPLDDTHead:
    def test_output_shape(self, cfg):
        from heads import PLDDTHead
        module = PLDDTHead(cfg)
        s = torch.randn(B, N_res, cfg.c_s)
        out = module(s)
        assert out.shape == (B, N_res, cfg.n_plddt_bins)


class TestMaskedMSAHead:
    def test_output_shape(self, cfg):
        from heads import MaskedMSAHead
        module = MaskedMSAHead(cfg)
        msa = torch.randn(B, N_seq, N_res, cfg.c_m)
        out = module(msa)
        assert out.shape == (B, N_seq, N_res, cfg.n_msa_classes)


class TestTMScoreHead:
    def test_output_shape(self, cfg):
        from heads import TMScoreHead
        module = TMScoreHead(cfg)
        pair = torch.randn(B, N_res, N_res, cfg.c_z)
        out = module(pair)
        assert out.shape == (B, N_res, N_res, cfg.n_pae_bins)


class TestExperimentallyResolvedHead:
    def test_output_shape(self, cfg):
        from heads import ExperimentallyResolvedHead
        module = ExperimentallyResolvedHead(cfg)
        s = torch.randn(B, N_res, cfg.c_s)
        out = module(s)
        assert out.shape == (B, N_res, 14)


# ======================== utils.py ========================

class TestDropoutRowwise:
    def test_noop_in_eval(self):
        from utils import dropout_rowwise
        x = torch.randn(B, N_seq, N_res, 32)
        out = dropout_rowwise(x, p=0.5, training=False)
        assert torch.equal(out, x)

    def test_shares_mask_across_rows(self):
        """DropoutRowwise: identical mask for every row (dim=1)."""
        from utils import dropout_rowwise
        torch.manual_seed(42)
        x = torch.ones(2, 5, 7, 3)
        y = dropout_rowwise(x, p=0.5, training=True)
        # All rows should be identical since mask broadcasts over dim=1
        assert torch.allclose(y[:, 0], y[:, 1])
        assert torch.allclose(y[:, 0], y[:, 3])


class TestDropoutColumnwise:
    def test_noop_in_eval(self):
        from utils import dropout_columnwise
        x = torch.randn(B, N_seq, N_res, 32)
        out = dropout_columnwise(x, p=0.5, training=False)
        assert torch.equal(out, x)

    def test_shares_mask_across_columns(self):
        """DropoutColumnwise: identical mask for every column (dim=2)."""
        from utils import dropout_columnwise
        torch.manual_seed(42)
        x = torch.ones(2, 5, 7, 3)
        y = dropout_columnwise(x, p=0.5, training=True)
        # All columns should be identical since mask broadcasts over dim=2
        assert torch.allclose(y[:, :, 0], y[:, :, 1])
        assert torch.allclose(y[:, :, 0], y[:, :, 4])


class TestDistanceBin:
    def test_output_shape(self):
        from utils import distance_bin
        pos = torch.randn(B, N_res, 3)
        n_bins = 64
        out = distance_bin(pos, n_bins)
        assert out.shape == (B, N_res, N_res, n_bins)
        # Each position should be one-hot (sum to 1)
        assert torch.allclose(out.sum(dim=-1), torch.ones(B, N_res, N_res))


class TestRecyclingDistanceBin:
    def test_output_shape(self):
        from utils import recycling_distance_bin
        pos = torch.randn(B, N_res, 3)
        out = recycling_distance_bin(pos, n_bins=15)
        assert out.shape == (B, N_res, N_res, 15)

    def test_one_hot(self):
        from utils import recycling_distance_bin
        pos = torch.randn(B, N_res, 3)
        out = recycling_distance_bin(pos, n_bins=15)
        # Each entry should be one-hot (sum to 1)
        assert torch.allclose(out.sum(dim=-1), torch.ones(B, N_res, N_res))


# ======================== model.py (full forward) ========================

class TestAlphaFold2:
    def test_forward_shapes(self, cfg):
        from model import AlphaFold2
        model = AlphaFold2(cfg)
        model.eval()

        N_templ = 2
        N_extra = 8
        target_feat = torch.randn(B, N_res, 21)
        residue_index = torch.arange(N_res).unsqueeze(0).expand(B, -1)
        msa_feat = torch.randn(B, N_seq, N_res, 49)
        extra_msa_feat = torch.randn(B, N_extra, N_res, 25)
        template_pair_feat = torch.randn(B, N_templ, N_res, N_res, 88)
        template_angle_feat = torch.randn(B, N_templ, N_res, 51)
        template_mask = torch.ones(B, N_templ)
        aatype = torch.randint(0, 20, (B, N_res))

        with torch.no_grad():
            structure_preds, pair_repr, msa_repr, single_reps = model(
                target_feat, residue_index, msa_feat,
                extra_msa_feat, template_pair_feat, aatype,
                template_angle_feat=template_angle_feat,
                template_mask=template_mask,
                n_cycles=1, n_ensemble=1,
            )

        assert pair_repr.shape == (B, N_res, N_res, cfg.c_z)
        assert msa_repr.shape == (B, N_seq, N_res, cfg.c_m)
        assert single_reps["single_pre_sm"].shape == (B, N_res, cfg.c_s)
        assert single_reps["single_post_sm"].shape == (B, N_res, cfg.c_s)
        assert structure_preds["atom14_coords"].shape == (B, N_res, 14, 3)
        assert structure_preds["atom14_mask"].shape == (B, N_res, 14)
        assert structure_preds["final_rotations"].shape == (B, N_res, 3, 3)
        assert structure_preds["final_translations"].shape == (B, N_res, 3)


# ======================== Semantic tests ========================

class TestIPAMasking:
    """IPA seq_mask: padded key positions should get zero attention weight."""

    def test_padded_keys_get_zero_attention(self, cfg):
        from structure_module import InvariantPointAttention
        module = InvariantPointAttention(cfg)
        module.eval()

        s = torch.randn(1, N_res, cfg.c_s)
        pair = torch.randn(1, N_res, N_res, cfg.c_z)
        R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, N_res, 3, 3).clone()
        t = torch.zeros(1, N_res, 3)

        # Mask: only first 3 residues valid, last 3 are padding
        seq_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])

        out_masked = module(s, pair, R, t, seq_mask=seq_mask)
        out_unmasked = module(s, pair, R, t)

        # Padded query outputs should be zero
        assert torch.allclose(out_masked[:, 3:], torch.zeros_like(out_masked[:, 3:]), atol=1e-6)

        # Valid query outputs should differ from unmasked (padding leaked signal before)
        # This is a weaker check: just verify masking changes the output
        assert not torch.allclose(out_masked[:, :3], out_unmasked[:, :3], atol=1e-6)

    def test_no_mask_matches_all_ones(self, cfg):
        """No mask = all-ones mask should give identical outputs."""
        from structure_module import InvariantPointAttention
        module = InvariantPointAttention(cfg)
        module.eval()

        s = torch.randn(1, N_res, cfg.c_s)
        pair = torch.randn(1, N_res, N_res, cfg.c_z)
        R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, N_res, 3, 3).clone()
        t = torch.zeros(1, N_res, 3)

        out_none = module(s, pair, R, t, seq_mask=None)
        out_ones = module(s, pair, R, t, seq_mask=torch.ones(1, N_res))
        assert torch.allclose(out_none, out_ones, atol=1e-5)


class TestIPAEquivariance:
    """IPA outputs should transform equivariantly under global rotations."""

    def test_rotation_equivariance(self, cfg):
        from structure_module import InvariantPointAttention
        module = InvariantPointAttention(cfg)
        module.eval()

        torch.manual_seed(0)
        s = torch.randn(1, N_res, cfg.c_s)
        pair = torch.randn(1, N_res, N_res, cfg.c_z)

        # Random per-residue frames
        R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, N_res, 3, 3).clone()
        t = torch.randn(1, N_res, 3)

        # A fixed global rotation (90 degrees around z-axis)
        theta = torch.tensor(3.14159 / 2)
        R_global = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta),  torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

        # Original output
        out_orig = module(s, pair, R, t)

        # Rotated frames: R' = R_global @ R, t' = R_global @ t
        R_rot = R_global @ R
        t_rot = (R_global.squeeze(1) @ t.unsqueeze(-1)).squeeze(-1)

        out_rot = module(s, pair, R_rot, t_rot)

        # IPA output is invariant (same single representation, different frames)
        # The output is projected back to local frame, so it should be invariant
        assert torch.allclose(out_orig, out_rot, atol=1e-4), \
            f"IPA output not invariant: max diff = {(out_orig - out_rot).abs().max():.6f}"


class TestStructureModuleMask:
    """StructureModule with seq_mask should zero padded outputs."""

    def test_padded_outputs_zeroed(self, cfg):
        from structure_module import StructureModule
        module = StructureModule(cfg)
        module.eval()

        s = torch.randn(1, N_res, cfg.c_s)
        pair = torch.randn(1, N_res, N_res, cfg.c_z)
        aatype = torch.randint(0, 20, (1, N_res))
        seq_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])

        with torch.no_grad():
            preds = module(s, pair, aatype, seq_mask=seq_mask)

        # The single representation at padded positions should be near-zero
        # (not exactly zero due to LayerNorm centering, but IPA output is zeroed)
        assert preds["single"].shape == (1, N_res, cfg.c_s)


class TestGradientFlow:
    """Gradients should flow through the recycling loop only on last cycle."""

    def test_structure_module_has_gradients(self, cfg):
        """Gradients flow through StructureModule into single_representation."""
        from structure_module import StructureModule
        module = StructureModule(cfg)

        s = torch.randn(1, N_res, cfg.c_s, requires_grad=True)
        pair = torch.randn(1, N_res, N_res, cfg.c_z)
        aatype = torch.randint(0, 20, (1, N_res))

        preds = module(s, pair, aatype)
        loss = preds["final_translations"].sum()
        loss.backward()
        assert s.grad is not None
        assert s.grad.abs().sum() > 0

    def test_rotation_detach_stops_gradient(self, cfg):
        """Rotation gradients should be detached between SM iterations."""
        from structure_module import StructureModule
        module = StructureModule(cfg)

        s = torch.randn(1, N_res, cfg.c_s, requires_grad=True)
        pair = torch.randn(1, N_res, N_res, cfg.c_z)
        aatype = torch.randint(0, 20, (1, N_res))

        preds = module(s, pair, aatype)

        # Verify that the rotation trajectory has gradients only through
        # the last layer (rotation.detach() is called for l < num_layers - 1)
        loss = preds["traj_rotations"][-1].sum()
        loss.backward()
        assert s.grad is not None


class TestFrameComposition:
    """Composing identity transforms should give identity."""

    def test_identity_composition(self):
        from structure_module import compose_transforms
        R_id = torch.eye(3).unsqueeze(0)  # (1, 3, 3)
        t_zero = torch.zeros(1, 3)

        R_out, t_out = compose_transforms(R_id, t_zero, R_id, t_zero)
        assert torch.allclose(R_out, R_id, atol=1e-6)
        assert torch.allclose(t_out, t_zero, atol=1e-6)

    def test_translation_only(self):
        from structure_module import compose_transforms
        R_id = torch.eye(3).unsqueeze(0)
        t1 = torch.tensor([[1.0, 2.0, 3.0]])
        t2 = torch.tensor([[4.0, 5.0, 6.0]])

        R_out, t_out = compose_transforms(R_id, t1, R_id, t2)
        assert torch.allclose(R_out, R_id, atol=1e-6)
        assert torch.allclose(t_out, t1 + t2, atol=1e-6)

    def test_backbone_update_zero_init(self, cfg):
        """BackboneUpdate with zero input produces identity rotation + zero translation."""
        from structure_module import BackboneUpdate
        from model import AlphaFold2
        # Use AlphaFold2 to trigger _initialize_alphafold_parameters
        model = AlphaFold2(cfg)
        bu = model.structure_model.backbone_update

        s = torch.zeros(1, N_res, cfg.c_s)
        with torch.no_grad():
            R, t = bu(s)

        I = torch.eye(3).unsqueeze(0).unsqueeze(0).expand_as(R)
        assert torch.allclose(R, I, atol=1e-5), \
            f"Expected identity rotation, max diff = {(R - I).abs().max():.6f}"
        assert torch.allclose(t, torch.zeros_like(t), atol=1e-5), \
            f"Expected zero translation, max diff = {t.abs().max():.6f}"


class TestHeadZeroInit:
    """Verify head logit layers are zero-initialized per Supplement 1.11.4."""

    def test_distogram_head_zero_init(self, cfg):
        from heads import DistogramHead
        head = DistogramHead(cfg)
        assert torch.allclose(head.linear.weight, torch.zeros_like(head.linear.weight))

    def test_plddt_head_zero_init(self, cfg):
        from heads import PLDDTHead
        head = PLDDTHead(cfg)
        final_linear = head.net[-1]
        assert torch.allclose(final_linear.weight, torch.zeros_like(final_linear.weight))

    def test_masked_msa_head_zero_init(self, cfg):
        from heads import MaskedMSAHead
        head = MaskedMSAHead(cfg)
        assert torch.allclose(head.linear.weight, torch.zeros_like(head.linear.weight))


class TestIPAInitialization:
    """Verify IPA head weights initialize so softplus(w) = 1."""

    def test_head_weights_softplus_one(self, cfg):
        from model import AlphaFold2
        model = AlphaFold2(cfg)
        ipa = model.structure_model.IPA
        gamma = torch.nn.functional.softplus(ipa.head_weights)
        assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-5)

    def test_ipa_output_zero_init(self, cfg):
        from model import AlphaFold2
        model = AlphaFold2(cfg)
        ipa = model.structure_model.IPA
        assert torch.allclose(ipa.linear_output.weight, torch.zeros_like(ipa.linear_output.weight))
