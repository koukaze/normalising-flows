'''Volume-preserving shear maps and the induced density family.'''

from __future__ import annotations

import math

import torch
from torch import Tensor

from moonflow.models import NormalisingFlow, _check_shape


def _check_alpha(alpha: float) -> None:
    '''Validate a finite scalar shear coefficient.'''

    if not math.isfinite(alpha):
        raise ValueError('alpha must be finite.')


def g_alpha(points: Tensor, alpha: float) -> tuple[Tensor, Tensor]:
    '''Apply the shear (x1, x2) -> (x1 + alpha * x2, x2).
    Args
        points (Tensor): Data points with shape (..., 2).
        alpha (float): Finite, dimensionless shear coefficient.
    Returns
        tuple: Sheared points and zero log-determinants of shape (...,).
    '''

    _check_shape(points, 2)
    _check_alpha(alpha)
    transformed = torch.stack((points[..., 0] + alpha * points[..., 1], points[..., 1]), dim=-1)
    return transformed, points.new_zeros(points.shape[:-1])


def g_alpha_inverse(points: Tensor, alpha: float) -> tuple[Tensor, Tensor]:
    '''Invert the shear (y1, y2) -> (y1 - alpha * y2, y2).
    Args
        points (Tensor): Sheared data points with shape (..., 2).
        alpha (float): Finite, dimensionless shear coefficient.
    Returns
        tuple: Unsheared points and zero log-determinants of shape (...,).
    '''

    _check_shape(points, 2)
    _check_alpha(alpha)
    transformed = torch.stack((points[..., 0] - alpha * points[..., 1], points[..., 1]), dim=-1)
    return transformed, points.new_zeros(points.shape[:-1])


def sample_family(
    flow: NormalisingFlow,
    alpha: float,
    n_samples: int = 1000,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    '''Sample the distribution obtained by shearing the flow output.
    Args
        flow (NormalisingFlow): Two-dimensional base flow.
        alpha (float): Finite shear coefficient.
        n_samples (int): Number of samples.
        generator (Generator or None): Random generator on the model's device.
    Returns
        Tensor: Samples from the sheared distribution, with shape (n_samples, 2).
    '''

    if flow.dim != 2:
        raise ValueError('The shear family requires a two-dimensional flow.')
    _check_alpha(alpha)
    return g_alpha(flow.sample(n_samples, generator=generator), alpha)[0]


def family_log_prob(flow: NormalisingFlow, points: Tensor, alpha: float) -> Tensor:
    '''Evaluate the exact density of a sheared flow.
    Args
        flow (NormalisingFlow): Two-dimensional base flow.
        points (Tensor): Sheared data points with shape (..., 2).
        alpha (float): Finite shear coefficient.
    Returns
        Tensor: Log probability densities in nats, with shape (...,).
    '''

    if flow.dim != 2:
        raise ValueError('The shear family requires a two-dimensional flow.')
    unsheared, log_det = g_alpha_inverse(points, alpha)
    return flow.log_prob(unsheared) + log_det
