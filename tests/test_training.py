'''Test optimisation, validation selection and checkpoint integrity.'''

from dataclasses import replace

import pytest
import torch

from moonflow.models import NormalisingFlow
from moonflow.training import (
    TrainConfig, evaluate_nll, fit, load_checkpoint, save_checkpoint,
)


@pytest.fixture
def features():
    generator = torch.Generator().manual_seed(17)
    return (
        torch.randn(32, 2, generator=generator),
        torch.randn(16, 2, generator=generator),
    )


@pytest.fixture
def small_config():
    return TrainConfig(n_layers=2, hidden=8, max_epochs=8, patience=3)


def test_fit_is_reproducible_and_preserves_inputs_and_rng(features, small_config):
    train, val = features
    saved_train, saved_val = train.clone(), val.clone()
    rng_state = torch.random.get_rng_state().clone()
    first = fit(train, val, small_config)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    second = fit(train, val, small_config)
    assert first.history == second.history
    for key, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(train, saved_train, rtol=0, atol=0)
    torch.testing.assert_close(val, saved_val, rtol=0, atol=0)
    assert not first.model.training


def test_learning_curves_describe_the_same_post_update_model(features, small_config):
    train, val = features
    config = replace(small_config, max_epochs=1)
    result = fit(train, val, config)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        expected = NormalisingFlow(config.dim, config.n_layers, config.hidden)
    optimiser = torch.optim.Adam(expected.parameters(), lr=config.learning_rate)
    loss = -expected.log_prob(train).mean()
    assert result.initial_train_nll == loss.item()
    loss.backward()
    optimiser.step()
    assert result.history['train_nll'] == [evaluate_nll(expected, train)]
    assert result.history['val_nll'] == [evaluate_nll(expected, val)]
    assert result.history['learning_rate'] == [config.learning_rate]
    assert result.train_nll == result.history['train_nll'][0]
    assert result.val_nll == result.history['val_nll'][0]


def test_early_stopping_restores_best_validation_weights():
    generator = torch.Generator().manual_seed(1)
    train = torch.randn(32, 2, generator=generator) * 0.2
    val = torch.randn(16, 2, generator=generator) * 0.2 + 3
    config = TrainConfig(
        n_layers=2, hidden=8, learning_rate=0.02, max_epochs=30, patience=3
    )
    result = fit(train, val, config)
    assert result.best_epoch == 1
    assert len(result.history['epoch']) == result.best_epoch + config.patience
    assert result.val_nll == min(result.history['val_nll'])
    assert result.val_nll < result.history['val_nll'][-1]
    assert evaluate_nll(result.model, train) == result.train_nll
    assert result.train_nll == result.history['train_nll'][result.best_epoch - 1]


def test_tiny_fit_without_validation_reduces_nll(features, small_config):
    config = replace(small_config, max_epochs=50, patience=None, learning_rate=0.01)
    result = fit(features[0], None, config)
    assert result.best_epoch == config.max_epochs
    assert result.train_nll < result.initial_train_nll
    assert result.history['val_nll'] == []
    assert result.val_nll is None
    assert result.initial_val_nll is None
    assert result.train_nll == result.history['train_nll'][-1]


def test_checkpoint_round_trip_preserves_density_and_rng(tmp_path, features, small_config):
    result = fit(*features, small_config)
    path = tmp_path / 'flow.pt'
    save_checkpoint(path, result, metadata={'dataset': 'unit-test', 'noise': 0.1})
    rng_state = torch.random.get_rng_state().clone()
    loaded, config, metadata = load_checkpoint(path)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    torch.testing.assert_close(
        loaded.log_prob(features[1]), result.model.log_prob(features[1]), rtol=0, atol=0
    )
    assert config == small_config
    assert not loaded.training
    assert metadata['training']['best_epoch'] == result.best_epoch
    assert metadata['experiment']['dataset'] == 'unit-test'


def test_training_and_loading_use_float32_independent_of_default(
    tmp_path, features, small_config
):
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        result = fit(*features, small_config)
        path = tmp_path / 'flow.pt'
        save_checkpoint(path, result)
        loaded, _, _ = load_checkpoint(path)
        assert next(result.model.parameters()).dtype == torch.float32
        assert next(loaded.parameters()).dtype == torch.float32
        assert evaluate_nll(loaded, features[1]) == result.val_nll
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize('corruption', ['mask', 'nonfinite', 'shape', 'config', 'epoch'])
def test_checkpoint_rejects_inconsistent_contents(
    tmp_path, features, small_config, corruption
):
    path = tmp_path / 'flow.pt'
    save_checkpoint(path, fit(*features, small_config))
    payload = torch.load(path, weights_only=True)
    if corruption == 'mask':
        payload['state_dict']['layers.0.mask'][:] = 0
    elif corruption == 'nonfinite':
        payload['state_dict']['layers.0.net.0.weight'][0, 0] = float('nan')
    elif corruption == 'shape':
        payload['state_dict']['layers.0.net.0.weight'] = torch.zeros(1, 1)
    elif corruption == 'config':
        payload['config']['extra'] = 1
    elif corruption == 'epoch':
        payload['metadata']['training']['best_epoch'] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


@pytest.mark.parametrize('data', [torch.empty(0, 2), torch.zeros(3, 3), torch.full((2, 2), float('nan'))])
def test_fit_rejects_invalid_features(features, small_config, data):
    with pytest.raises(ValueError):
        fit(data, features[1], small_config)


def test_validation_is_required_for_early_stopping(features, small_config):
    with pytest.raises(ValueError, match='Validation'):
        fit(features[0], None, small_config)


def test_evaluation_preserves_model_mode(features):
    model = NormalisingFlow(n_layers=2, hidden=8)
    model.train()
    value = evaluate_nll(model, features[0])
    assert model.training
    assert isinstance(value, float)


@pytest.mark.parametrize('kwargs', [
    {'learning_rate': float('nan')}, {'max_epochs': 0},
    {'patience': 0}, {'dim': 3}, {'seed': True},
])
def test_configuration_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
