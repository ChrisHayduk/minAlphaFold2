"""Typed schema for every model-profile hyperparameter.

One :class:`ModelConfig` instance carries every knob the model reads from
``config.<name>``, grouped by supplement section. Each field lands in the
dataclass with a Python type, so:

* ``load_model_config`` can parse ``configs/<name>.toml`` and have a typo
  in either the file or the schema error out at load time (instead of at
  first ``AttributeError`` during a forward pass);
* pyright type-checks every ``config.c_m`` access in the rest of the code;
* ``dataclasses.replace`` gives a clean copy-with-overrides API for
  :func:`minalphafold.trainer.zero_dropout_model_config`.

The dataclass is intentionally *not* frozen — profiles are normally loaded
once and threaded through the model read-only, but the occasional helper
(e.g. the zero-dropout overfit variant) benefits from the flexibility of
``replace`` without juggling ``object.__setattr__``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """All AlphaFold2 architectural hyperparameters."""

    # Human-readable tag; propagates into checkpoint metadata and artifact
    # directory names. ``zero_dropout_model_config`` appends "_no_dropout".
    model_profile: str

    # Channel dimensions (supplement 1.5).
    c_m: int      # MSA representation
    c_s: int      # Single representation (fed to structure module)
    c_z: int      # Pair representation
    c_t: int      # Template pair
    c_e: int      # Extra MSA

    # Main MSA row / column attention head dims (Algorithm 7, 8).
    dim: int
    num_heads: int

    # MSA transition widening factor (Algorithm 9, default n=4 in paper).
    msa_transition_n: int

    # Outer product mean hidden dim (Algorithm 10, paper default c=32).
    outer_product_dim: int

    # Triangle multiplicative updates (Algorithms 11, 12; paper default c=128).
    triangle_mult_c: int

    # Triangle self-attention (Algorithms 13, 14; paper default c=32, N_head=4).
    triangle_dim: int
    triangle_num_heads: int

    # Pair transition widening factor (Algorithm 15, paper default n=4).
    pair_transition_n: int

    # Template pair stack (supplement 1.7.1, Algorithms 16, 17). The template
    # variants of the triangle modules use *different* dims than the main
    # Evoformer per Algorithm 16 (c=64 for both mult and attn, n=2 on the pair
    # transition), hence the separate ``template_*`` keys.
    template_pair_num_blocks: int
    template_pair_dropout: float
    template_pointwise_attention_dim: int
    template_pointwise_num_heads: int
    template_triangle_mult_c: int
    template_triangle_attn_c: int
    template_triangle_attn_num_heads: int
    template_pair_transition_n: int

    # Extra MSA stack (supplement 1.7.2, Algorithms 18, 19).
    extra_msa_dim: int                   # row-attn head dim (paper c=8)
    extra_msa_dropout: float
    extra_pair_dropout: float
    msa_column_global_attention_dim: int # global col-attn head dim (paper c=8)
    num_extra_msa: int                   # number of blocks (paper N_block=4)

    # Main Evoformer (Algorithm 6).
    num_evoformer: int
    evoformer_msa_dropout: float
    evoformer_pair_dropout: float

    # Structure Module (supplement 1.8, Algorithm 20).
    structure_module_c: int
    structure_module_layers: int
    structure_module_dropout_ipa: float
    structure_module_dropout_transition: float
    sidechain_num_channel: int
    sidechain_num_residual_block: int
    position_scale: float
    zero_init: bool

    # Invariant Point Attention (supplement 1.8.2, Algorithm 22).
    ipa_num_heads: int
    ipa_c: int
    ipa_n_query_points: int
    ipa_n_value_points: int

    # Auxiliary heads.
    n_dist_bins: int       # distogram head (supplement 1.9.8)
    plddt_hidden_dim: int  # pLDDT MLP hidden dim (supplement 1.9.6)
    n_plddt_bins: int      # pLDDT output bins (supplement 1.9.6)
    n_msa_classes: int     # masked MSA head (supplement 1.9.9)
    n_pae_bins: int        # PAE / pTM head (supplement 1.9.7)
