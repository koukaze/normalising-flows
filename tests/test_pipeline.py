'''Verify agreement between experiment outputs, selected weights and input data.'''

import hashlib
import json

import numpy as np
import pytest
import torch

from moonflow.data import load_moons_splits
from moonflow.pipeline import run_experiment
from moonflow.training import TrainConfig, evaluate_nll, load_checkpoint


def test_experiment_outputs_match_selected_model_and_external_inputs(tmp_path):
    '''Saved metrics and samples describe the restored model and recorded inputs.'''
    generated_dir = tmp_path / 'generated'
    config = TrainConfig(n_layers=2, hidden=8, max_epochs=4, patience=2)
    summary = run_experiment(generated_dir, config=config, n_samples=40)
    model, loaded_config, _ = load_checkpoint(generated_dir / 'flow.pt')
    assert loaded_config == config
    splits = load_moons_splits(generated_dir / 'data')
    curves = json.loads((generated_dir / 'training_curves.json').read_text())
    selected_index = int(np.argmin(curves['val_nll']))
    assert summary['best_epoch'] == curves['epoch'][selected_index]
    assert summary['nll_nats_per_point']['validation'] == curves['val_nll'][selected_index]
    for split_name, metric_name in [('train', 'train'), ('val', 'validation'), ('test', 'test')]:
        assert evaluate_nll(model, splits[split_name].tensor()) == pytest.approx(
            summary['nll_nats_per_point'][metric_name], abs=1e-7
        )
        data_path = generated_dir / 'data' / f'moons_{split_name}.csv'
        assert hashlib.sha256(data_path.read_bytes()).hexdigest() == summary['data']['sha256'][split_name]
    samples = np.load(generated_dir / 'samples.npz')
    for key, alpha in [('shear_minus_one', -1), ('shear_plus_one', 1)]:
        np.testing.assert_allclose(samples[key][:, 0], samples['base'][:, 0] + alpha * samples['base'][:, 1])
        np.testing.assert_array_equal(samples[key][:, 1], samples['base'][:, 1])
    expected = model.sample(40, generator=torch.Generator().manual_seed(summary['sample_seed']))
    np.testing.assert_array_equal(samples['base'], expected.detach().numpy())
    assert summary['numerical_checks']['autograd_logdet_max_abs_error'] < 1e-10
    for name in ('density', 'training', 'shear_family'):
        assert (generated_dir / f'{name}.png').stat().st_size > 0
        assert (generated_dir / f'{name}.svg').stat().st_size > 0

    external_dir = tmp_path / 'external_run'
    external = run_experiment(external_dir, data_dir=generated_dir / 'data', config=config, n_samples=40)
    assert external['data']['kind'] == 'external_csv'
    assert external['data']['sha256'] == summary['data']['sha256']
    assert external['nll_nats_per_point'] == summary['nll_nats_per_point']
    assert not (external_dir / 'data').exists()
