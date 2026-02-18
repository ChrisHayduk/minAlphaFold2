import torch

def dropout_rowwise(x, p, training):
    """Mask shared across dim 2 (columns)."""
    if not training or p == 0.0:
        return x
    mask_shape = (x.shape[0], x.shape[1], 1, x.shape[3])
    mask = x.new_ones(mask_shape)
    mask = torch.nn.functional.dropout(mask, p=p, training=True)
    return x * mask

def distance_bin(positions, n_bins, d_min=2.0, d_max=22.0):
    dists = torch.cdist(positions, positions)
    bin_edges = torch.linspace(d_min, d_max, n_bins - 1, device=positions.device)
    bin_idx = torch.bucketize(dists, bin_edges)
    return torch.nn.functional.one_hot(bin_idx, n_bins).float()

def dropout_columnwise(x, p, training):
    """Mask shared across dim 1 (rows)."""
    if not training or p == 0.0:
        return x
    mask_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
    mask = x.new_ones(mask_shape)
    mask = torch.nn.functional.dropout(mask, p=p, training=True)
    return x * mask