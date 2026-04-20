"""Linear-layer initialisers (supplement 1.11.4).

The supplement spells out exactly how every ``nn.Linear`` is meant to be
initialised; this module is the single source of truth. Biases always
start at zero, weights follow one of five recipes:

* ``default``/``linear`` — LeCun (fan-in) truncated normal,
  ``std = sqrt(1 / fan_in)``. The repo-wide default for ordinary Linears.
* ``relu``                 — He truncated normal, ``std = sqrt(2 / fan_in)``,
  for Linears immediately followed by ReLU.
* ``glorot``               — Glorot/Xavier uniform, for query/key/value
  projections inside self-attention (the supplement's "fan-average
  Glorot" scheme).
* ``final``                — zero weight, zero bias. Used for the final
  output projection of every residual block so each layer starts as an
  identity operation, and for the logit layer of every prediction head.
* ``gate``                 — zero weight, bias = 1. Applied to gating
  Linears that feed into a sigmoid so the gate opens at
  ``sigmoid(1) ≈ 0.73`` at initialisation (supplement 1.11.4 "Gating
  Linear layers").

``AlphaFold2._initialize_alphafold_parameters`` enforces the zero-init
sweep across the whole network using the helpers below.
"""

from __future__ import annotations

import math

import torch


def truncated_normal_(tensor: torch.Tensor, std: float) -> None:
    """Truncated normal with the supplement's ``±2σ`` clip (1.11.4)."""
    torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def init_linear(linear: torch.nn.Linear, init: str = "default") -> None:
    """Initialise ``linear`` per supplement 1.11.4.

    ``init`` selects the scheme: ``default``/``linear`` (LeCun fan-in
    truncated normal), ``relu`` (He truncated normal), ``glorot``
    (Xavier uniform), or ``final`` (zeros — identity residual start).
    Bias is always zero. See the module docstring for when to use each.
    """
    fan_in = linear.weight.shape[1]
    with torch.no_grad():
        if linear.bias is not None:
            linear.bias.zero_()
        if init in {"default", "linear"}:
            truncated_normal_(linear.weight, std=math.sqrt(1.0 / fan_in))
        elif init == "relu":
            truncated_normal_(linear.weight, std=math.sqrt(2.0 / fan_in))
        elif init == "glorot":
            torch.nn.init.xavier_uniform_(linear.weight, gain=1.0)
        elif init == "final":
            linear.weight.zero_()
        else:
            raise ValueError(f"Unknown linear init: {init}")


def zero_linear(linear: torch.nn.Linear) -> None:
    """Zero weight and zero bias (supplement 1.11.4 "final weight layers")."""
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()


def init_gate_linear(linear: torch.nn.Linear) -> None:
    """Zero weight, bias = 1 for gating Linears (supplement 1.11.4).

    With ``W = 0`` and ``b = 1`` the pre-sigmoid activation is 1 for
    every input, so the gate opens at ``sigmoid(1) ≈ 0.73`` and the
    layer starts mostly pass-through — as required by the supplement.
    """
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.fill_(1.0)
