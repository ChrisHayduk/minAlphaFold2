import torch

def dropout_rowwise(x, p, training):
    """Mask shared across dim 2 (columns)."""
    if not training or p == 0.0:
        return x
    mask_shape = (x.shape[0], x.shape[1], 1, x.shape[3])
    mask = x.new_ones(mask_shape)
    mask = torch.nn.functional.dropout(mask, p=p, training=True)
    return x * mask

def dropout_columnwise(x, p, training):
    """Mask shared across dim 1 (rows)."""
    if not training or p == 0.0:
        return x
    mask_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
    mask = x.new_ones(mask_shape)
    mask = torch.nn.functional.dropout(mask, p=p, training=True)
    return x * mask

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