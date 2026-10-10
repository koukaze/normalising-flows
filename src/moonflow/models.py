'''Affine coupling layers and a Gaussian-base normalising flow.'''

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _positive_integer(value: int, name: str) -> None:
    '''Validate a positive integer architecture parameter.'''

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')


def _check_shape(points: Tensor, dim: int) -> None:
    '''Check the final feature axis of a floating-point tensor.'''

    if points.ndim < 1 or points.shape[-1] != dim:
        raise ValueError(f'Expected a tensor with final dimension {dim}.')
    if not points.is_floating_point():
        raise TypeError('Input points must use a floating-point dtype.')


class CouplingLayer(nn.Module):
    '''Transform unmasked features using an affine coupling map.
    Args
        mask (Tensor): Binary vector of shape (dim,); one keeps a feature fixed.
        dim (int): Number of input features.
        hidden (int): Width of the shared scale and translation network.
    '''

    def __init__(self, mask: Tensor, dim: int, hidden: int) -> None:
        super().__init__()
        _positive_integer(dim, 'dim')
        _positive_integer(hidden, 'hidden')
        if mask.shape != (dim,) or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('mask must be a binary vector with shape (dim,).')
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )
        weight = self.net[0].weight
        self.register_buffer('mask', mask.detach().to(weight).clone())

    def _st(self, points: Tensor) -> tuple[Tensor, Tensor]:
        '''Evaluate the masked scale and translation networks.
        Args
            points (Tensor): Features with shape (..., dim).
        Returns
            tuple: Scale and translation with shape (..., dim); scale is bounded to [-1, 1].
        '''

        _check_shape(points, self.dim)
        scale, translation = self.net(points * self.mask).chunk(2, dim=-1)
        scale = torch.tanh(scale) * (1 - self.mask)
        translation = translation * (1 - self.mask)
        return scale, translation

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        '''Map latent features to data features.
        Args
            latent (Tensor): Features with shape (..., dim).
        Returns
            tuple: Transformed features and forward log-determinants of shape (...,).
        '''

        scale, translation = self._st(latent)
        points = latent * self.mask + (
            latent * torch.exp(scale) + translation
        ) * (1 - self.mask)
        return points, scale.sum(dim=-1)

    def inverse(self, points: Tensor) -> tuple[Tensor, Tensor]:
        '''Map data features to latent features.
        Args
            points (Tensor): Features with shape (..., dim).
        Returns
            tuple: Latent features and inverse log-determinants of shape (...,).
        '''

        scale, translation = self._st(points)
        latent = points * self.mask + (
            (points - translation) * torch.exp(-scale)
        ) * (1 - self.mask)
        return latent, -scale.sum(dim=-1)


class NormalisingFlow(nn.Module):
    '''Compose affine coupling layers with alternating masks.
    Args
        dim (int): Number of data features.
        n_layers (int): Number of coupling layers.
        hidden (int): Width of each scale and translation network.
    '''

    def __init__(self, dim: int = 2, n_layers: int = 6, hidden: int = 64) -> None:
        super().__init__()
        _positive_integer(dim, 'dim')
        _positive_integer(n_layers, 'n_layers')
        _positive_integer(hidden, 'hidden')
        self.dim = dim
        self.n_layers = n_layers
        self.hidden = hidden
        layers = []
        for index in range(n_layers):
            mask = torch.zeros(dim)
            if index % 2 == 0:
                mask[:dim // 2] = 1
            else:
                mask[dim // 2:] = 1
            layers.append(CouplingLayer(mask, dim, hidden))
        self.layers = nn.ModuleList(layers)

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        '''Map standard-normal latent points to data space.
        Args
            latent (Tensor): Latent points with shape (..., dim).
        Returns
            tuple: Data points and forward log-determinants of shape (...,).
        '''

        _check_shape(latent, self.dim)
        points = latent
        log_det = latent.new_zeros(latent.shape[:-1])
        for layer in self.layers:
            points, contribution = layer(points)
            log_det = log_det + contribution
        return points, log_det

    def inverse(self, points: Tensor) -> tuple[Tensor, Tensor]:
        '''Map data points to the standard-normal latent space.
        Args
            points (Tensor): Data points with shape (..., dim).
        Returns
            tuple: Latent points and inverse log-determinants of shape (...,).
        '''

        _check_shape(points, self.dim)
        latent = points
        log_det = points.new_zeros(points.shape[:-1])
        for layer in reversed(self.layers):
            latent, contribution = layer.inverse(latent)
            log_det = log_det + contribution
        return latent, log_det

    def log_prob(self, points: Tensor) -> Tensor:
        '''Evaluate the model log density using the inverse Jacobian.
        Args
            points (Tensor): Data points with shape (..., dim).
        Returns
            Tensor: Log probability densities in nats, with shape (...,).
        '''

        latent, log_det = self.inverse(points)
        log_base = -0.5 * (latent.square().sum(dim=-1) + self.dim * math.log(2 * math.pi))
        return log_base + log_det

    def sample(self, n: int, *, generator: torch.Generator | None = None) -> Tensor:
        '''Draw samples using the model's dtype and device.
        Args
            n (int): Number of samples; must be nonnegative.
            generator (Generator or None): Random generator on the model's device.
        Returns
            Tensor: Generated points with shape (n, dim).
        '''

        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError('n must be a nonnegative integer.')
        weight = next(self.parameters())
        latent = torch.randn(
            n, self.dim, dtype=weight.dtype, device=weight.device, generator=generator,
        )
        return self(latent)[0]
