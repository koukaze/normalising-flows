'''Plot flow densities, optimization histories and shear transformations.'''

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch

from .data import MoonSplit
from .models import NormalisingFlow
from .training import FitResult
from .transforms import g_alpha


def training_figure(result: FitResult) -> Figure:
    '''Plot post-update training and validation NLL with the selected epoch.

    Args
        result (FitResult): Fitted model and recorded learning curves.

    Returns
        Figure: Negative-log-likelihood curves in nats per point.
    '''
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(result.history['epoch'], result.history['train_nll'], label='Training', color='#176b87')
    if result.history['val_nll']:
        ax.plot(result.history['epoch'], result.history['val_nll'], label='Validation', color='#e07a5f')
    ax.axvline(result.best_epoch, linestyle='--', linewidth=1, color='#444444', label='Selected epoch')
    ax.set(xlabel='Epoch', ylabel='Mean NLL [nats / point]', title='Maximum-likelihood training')
    ax.legend()
    return fig


def density_figure(flow: NormalisingFlow, data: MoonSplit, samples: torch.Tensor) -> Figure:
    '''Compare training coordinates, the learned density and generated samples.

    Args
        flow (NormalisingFlow): Trained two-dimensional CPU flow.
        data (MoonSplit): Training coordinates and optional display labels.
        samples (Tensor): Generated points with shape (n_samples, 2).

    Returns
        Figure: Three panels sharing coordinate limits and equal aspect ratios.
    '''
    generated = samples.detach().cpu().numpy()
    all_points = np.vstack((data.points, generated))
    lower, upper = all_points.min(axis=0) - 0.15, all_points.max(axis=0) + 0.15
    gx, gy = np.meshgrid(np.linspace(lower[0], upper[0], 180), np.linspace(lower[1], upper[1], 180))
    points = torch.tensor(np.column_stack((gx.ravel(), gy.ravel())), dtype=next(flow.parameters()).dtype)
    with torch.no_grad():
        density = flow.log_prob(points).exp().reshape(gx.shape).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].scatter(*data.points.T, c=data.labels if data.labels is not None else '#176b87',
                    cmap='coolwarm' if data.labels is not None else None, s=9, alpha=0.65)
    image = axes[1].pcolormesh(gx, gy, density, shading='auto', cmap='magma', rasterized=True)
    fig.colorbar(image, ax=axes[1], label='Probability density', shrink=0.8)
    axes[2].scatter(*generated.T, s=5, alpha=0.35, color='#176b87')
    for axis, title in zip(axes, ('Training data', 'Learned density', 'Generated samples')):
        axis.set(xlabel='$x_1$', ylabel='$x_2$', title=title, xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        axis.set_aspect('equal', adjustable='box')
    return fig


def shear_figure(samples: torch.Tensor, alphas: tuple[float, ...] = (-1.0, 0.0, 1.0)) -> Figure:
    '''Apply several shears to the same generated samples for a paired comparison.

    Args
        samples (Tensor): Base flow samples with shape (n_samples, 2).
        alphas (tuple): Finite shear coefficients, one per panel.

    Returns
        Figure: Sheared sample clouds with shared bounds and equal aspect ratios.
    '''
    if not alphas:
        raise ValueError('Provide at least one shear coefficient.')
    transformed = [g_alpha(samples, alpha)[0].detach().cpu().numpy() for alpha in alphas]
    stacked = np.concatenate(transformed)
    lower, upper = stacked.min(axis=0) - 0.15, stacked.max(axis=0) + 0.15
    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 3.6), squeeze=False, constrained_layout=True)
    for axis, points, alpha in zip(axes.ravel(), transformed, alphas):
        axis.scatter(*points.T, s=5, alpha=0.35, color='#176b87')
        axis.set(xlabel='$y_1$', ylabel='$y_2$', title=rf'$\alpha={alpha:g}$',
                 xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        axis.set_aspect('equal', adjustable='box')
    return fig
