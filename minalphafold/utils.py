import torch
import torch.nn.functional as F


def dropout_rowwise(x, p, training):
    """
    DropoutRowwise (Supplement): mask is shared across rows (dim=1),
    i.e. the same mask applies to every row, varying across columns (dim=2).
    Input shape: (B, R, C, D).
    """
    if not training or p == 0.0:
        return x
    assert x.ndim == 4, f"expected (B, R, C, D), got {x.shape}"
    # Broadcast mask over rows (dim=1)
    mask = x.new_ones((x.shape[0], 1, x.shape[2], x.shape[3]))
    mask = F.dropout(mask, p=p, training=True)
    return x * mask


def dropout_columnwise(x, p, training):
    """
    DropoutColumnwise (Supplement): mask is shared across columns (dim=2),
    i.e. the same mask applies to every column, varying across rows (dim=1).
    Input shape: (B, R, C, D).
    """
    if not training or p == 0.0:
        return x
    assert x.ndim == 4, f"expected (B, R, C, D), got {x.shape}"
    # Broadcast mask over columns (dim=2)
    mask = x.new_ones((x.shape[0], x.shape[1], 1, x.shape[3]))
    mask = F.dropout(mask, p=p, training=True)
    return x * mask


def distance_bin(positions, n_bins, d_min=2.0, d_max=22.0):
    """Distogram binning for distogram head / loss (not recycling)."""
    dists = torch.cdist(positions, positions)
    step = (d_max - d_min) / n_bins
    bin_edges = d_min + step * torch.arange(1, n_bins, device=positions.device, dtype=dists.dtype)
    bin_idx = torch.bucketize(dists, bin_edges)
    return F.one_hot(bin_idx, n_bins).to(dtype=positions.dtype)


def recycling_distance_bin(positions, n_bins=15, d_min=3.375, bin_width=1.25):
    """
    Recycling distogram embedding (Algorithm 32).
    15-bin one-hot of C-beta distances used for recycling, distinct from
    the 64-bin distogram head.
    Returns: (B, N, N, n_bins)
    """
    dists = torch.cdist(positions, positions)
    edges = d_min + bin_width * torch.arange(1, n_bins, device=positions.device, dtype=dists.dtype)
    idx = torch.bucketize(dists, edges)
    return F.one_hot(idx, n_bins).to(dtype=positions.dtype)