'''Train a density model and save reproducible samples, metrics and figures.'''

from collections.abc import Callable
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import torch

from .checks import count_flops, invertibility_errors, log_det_check
from .data import load_moons_splits, make_moons_splits, save_moons_splits
from .plotting import density_figure, shear_figure, training_figure
from .training import TrainConfig, evaluate_nll, fit, save_checkpoint
from .transforms import g_alpha


def run_experiment(
    output_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    config: TrainConfig = TrainConfig(),
    data_seed: int = 42,
    noise: float = 0.08,
    sample_seed: int = 2026,
    n_samples: int = 2000,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    '''Fit on training data, select by validation NLL and evaluate the test split.

    Args
        output_dir (str or Path): Destination for metrics, weights, samples and figures.
        data_dir (str or Path or None): External CSV directory; None generates a synthetic example.
        config (TrainConfig): Architecture, optimizer and stopping settings.
        data_seed (int): Seed for the generated data, independent of model initialization.
        noise (float): Coordinate noise standard deviation for generated data.
        sample_seed (int): Seed for model samples, independent of training.
        n_samples (int): Number of generated points used for figures and sample arrays.
        progress (Callable or None): Optional training-epoch callback.

    Returns
        dict: Data provenance, selected-model metrics and numerical checks.
    '''
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError('n_samples must be a positive integer.')
    if config.dim != 2:
        raise ValueError('This experiment requires two-dimensional features.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if data_dir is None:
        splits = make_moons_splits(noise=noise, seed=data_seed)
        save_moons_splits(splits, output_dir / 'data')
        provenance = {'kind': 'generated_two_moons', 'seed': data_seed, 'noise': noise,
                      'geometry': 'two semicircles with independent Gaussian coordinate noise'}
        input_dir = output_dir / 'data'
    else:
        input_dir = Path(data_dir)
        splits = load_moons_splits(input_dir)
        provenance = {'kind': 'external_csv'}
    provenance['split_sizes'] = {name: len(split.points) for name, split in splits.items()}
    provenance['sha256'] = {
        name: hashlib.sha256((input_dir / f'moons_{name}.csv').read_bytes()).hexdigest()
        for name in ('train', 'val', 'test')
    }
    result = fit(splits['train'].tensor(), splits['val'].tensor(), config, progress=progress)
    # Evaluate the test NLL only after model selection is complete.
    test_nll = evaluate_nll(result.model, splits['test'].tensor())
    generator = torch.Generator(device='cpu').manual_seed(sample_seed)
    with torch.no_grad():
        samples = result.model.sample(n_samples, generator=generator)
    points = splits['train'].tensor()
    check_flow = copy.deepcopy(result.model).double()
    check_points = points[:32].double()
    logdets = log_det_check(check_flow, check_points)
    numerical = {
        'float32_roundtrip': invertibility_errors(result.model, points),
        'jacobian_points': len(check_points), 'jacobian_dtype': 'float64', 'finite_difference_step': 1e-5,
        'autograd_logdet_max_abs_error': float((logdets['analytic'] - logdets['autograd']).abs().max()),
        'finite_difference_logdet_max_abs_error': float((logdets['analytic'] - logdets['finite_difference']).abs().max()),
    }
    summary = {
        'data': provenance, 'config': asdict(config), 'sample_seed': sample_seed,
        'n_generated_samples': n_samples, 'epochs_run': len(result.history['epoch']),
        'best_epoch': result.best_epoch, 'parameter_count': sum(p.numel() for p in result.model.parameters()),
        'nll_nats_per_point': {'train': result.train_nll, 'validation': result.val_nll, 'test': test_nll},
        'initial_nll_nats_per_point': {'train': result.initial_train_nll, 'validation': result.initial_val_nll},
        'numerical_checks': numerical,
        'idealised_density_flops': count_flops(dim=config.dim, n_layers=config.n_layers, hidden=config.hidden,
                                              batch_size=len(points)),
        'software': {'python': platform.python_version(), 'torch': str(torch.__version__),
                     'numpy': np.__version__, 'torch_cpu_threads': torch.get_num_threads()},
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (output_dir / 'training_curves.json').write_text(json.dumps(result.history, indent=2, allow_nan=False) + '\n')
    save_checkpoint(output_dir / 'flow.pt', result, metadata={'data': provenance, 'sample_seed': sample_seed})
    np.savez_compressed(output_dir / 'samples.npz', base=samples.numpy(),
                        shear_minus_one=g_alpha(samples, -1.0)[0].numpy(),
                        shear_plus_one=g_alpha(samples, 1.0)[0].numpy())
    np.savez_compressed(output_dir / 'jacobian_checks.npz', **{name: value.numpy() for name, value in logdets.items()})
    for name, figure in (
        ('training', training_figure(result)),
        ('density', density_figure(result.model, splits['train'], samples)),
        ('shear_family', shear_figure(samples)),
    ):
        figure.savefig(output_dir / f'{name}.png', dpi=180, bbox_inches='tight')
        figure.savefig(output_dir / f'{name}.svg', bbox_inches='tight')
        plt.close(figure)
    return summary
