from __future__ import annotations

import math

import torch


def truncated_normal_(tensor: torch.Tensor, std: float) -> None:
    torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def init_linear(linear: torch.nn.Linear, init: str = "default") -> None:
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
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()


def init_gate_linear(linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.fill_(1.0)
