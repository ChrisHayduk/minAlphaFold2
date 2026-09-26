import torch

from minalphafold.losses import TorsionAngleLoss


def test_torsion_trajectory_replication_preserves_loss_and_total_gradient():
    """Repeated identical refinement layers must not rescale the objective."""
    torch.manual_seed(19)
    prediction = torch.randn(2, 5, 7, 2, dtype=torch.float64, requires_grad=True)
    truth = torch.randn_like(prediction)
    mask = torch.randint(0, 2, prediction.shape[:-1]).to(torch.float64)
    sequence_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float64
    )
    loss_fn = TorsionAngleLoss()
    single = loss_fn(prediction, prediction, truth, truth, mask, sequence_mask)
    single_gradient = torch.autograd.grad(single.sum(), prediction)[0]

    repeated = prediction.detach().unsqueeze(0).repeat(4, 1, 1, 1, 1).requires_grad_()
    trajectory = loss_fn(repeated, repeated, truth, truth, mask, sequence_mask)
    gradient = torch.autograd.grad(trajectory.sum(), repeated)[0]
    torch.testing.assert_close(trajectory, single)
    torch.testing.assert_close(gradient.sum(dim=0), single_gradient)
    assert torch.isfinite(gradient).all()
    assert gradient[:, 0, 3:].count_nonzero() == 0


def test_unsupervised_torsion_trajectory_has_finite_zero_gradient():
    prediction = torch.ones(3, 1, 2, 7, 2, requires_grad=True)
    loss_fn = TorsionAngleLoss()
    result = loss_fn(
        prediction,
        prediction,
        torch.zeros(1, 2, 7, 2),
        torch.zeros(1, 2, 7, 2),
        torch.zeros(1, 2, 7),
        torch.zeros(1, 2),
    )
    result.sum().backward()
    assert torch.equal(result, torch.zeros_like(result))
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
