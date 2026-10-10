'''Numerical checks and an idealised operation count for affine flows.'''

from __future__ import annotations

import math

import torch
from torch import Tensor

from moonflow.models import NormalisingFlow, _positive_integer


def invertibility_errors(flow: NormalisingFlow, points: Tensor) -> dict[str, float]:
    '''Check both compositions of the forward and inverse maps.
    Args
        flow (NormalisingFlow): Flow to check.
        points (Tensor): Nonempty points with shape (..., dim), used in both spaces.
    Returns
        dict: Maximum absolute reconstruction and log-determinant cancellation errors.
    '''

    if points.numel() == 0:
        raise ValueError('At least one point is required.')
    with torch.no_grad():
        latent, inverse_log_det = flow.inverse(points)
        recovered_points, forward_log_det = flow(latent)
        transformed, latent_forward_log_det = flow(points)
        recovered_latent, latent_inverse_log_det = flow.inverse(transformed)
    return {
        'data_roundtrip_max': (points - recovered_points).abs().max().item(),
        'latent_roundtrip_max': (points - recovered_latent).abs().max().item(),
        'data_logdet_sum_max': (inverse_log_det + forward_log_det).abs().max().item(),
        'latent_logdet_sum_max': (latent_forward_log_det + latent_inverse_log_det).abs().max().item(),
    }


def log_det_check(
    flow: NormalisingFlow,
    points: Tensor,
    *,
    eps: float = 1e-5,
) -> dict[str, Tensor]:
    '''Compare inverse log-determinants with two numerical Jacobians.
    Args
        flow (NormalisingFlow): Flow to check; float64 is recommended.
        points (Tensor): Nonempty matrix of shape (n_points, dim), matching the model dtype.
        eps (float): Positive central-difference step in the input coordinate units.
    Returns
        dict: Analytic, autograd and finite-difference log-determinants of shape (n_points,).
    '''

    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != flow.dim:
        raise ValueError('points must have nonempty shape (n_points, flow.dim).')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive.')
    points = points.detach()
    with torch.no_grad():
        analytic = flow.inverse(points)[1]
        columns = []
        for index in range(flow.dim):
            offset = torch.zeros_like(points)
            offset[:, index] = eps
            plus = flow.inverse(points + offset)[0]
            minus = flow.inverse(points - offset)[0]
            columns.append((plus - minus) / (2 * eps))
        finite_difference = torch.linalg.slogdet(torch.stack(columns, dim=-1)).logabsdet
    autograd = []
    with torch.enable_grad():
        for point in points:
            jacobian = torch.autograd.functional.jacobian(lambda value: flow.inverse(value)[0], point)
            autograd.append(torch.linalg.slogdet(jacobian).logabsdet.detach())
    return {
        'analytic': analytic.detach(),
        'autograd': torch.stack(autograd),
        'finite_difference': finite_difference.detach(),
    }


def count_flops(*, dim: int, n_layers: int, hidden: int, batch_size: int) -> int:
    '''Estimate density-evaluation cost under a fixed operation-count convention.
    Args
        dim (int): Number of features.
        n_layers (int): Number of coupling layers.
        hidden (int): Width of each scale and translation network.
        batch_size (int): Number of points.
    Returns
        int: Idealised operation count, excluding mask construction and runtime overhead.
    '''

    for name, value in (
        ('dim', dim), ('n_layers', n_layers), ('hidden', hidden), ('batch_size', batch_size),
    ):
        _positive_integer(value, name)
    # Count exp/log/tanh as ten operations each, ReLU as one and Gaussian log density as twenty.
    per_layer = 6 * dim * hidden + hidden + 30 * dim
    return batch_size * (n_layers * (per_layer + 1) + 21)
