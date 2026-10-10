'''Train and evaluate affine coupling flows with validation-based model selection.'''

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .models import NormalisingFlow


@dataclass(frozen=True)
class TrainConfig:
    '''Configure full-batch Adam training and cosine learning-rate decay.'''

    dim: int = 2
    n_layers: int = 6
    hidden: int = 64
    learning_rate: float = 0.003
    max_epochs: int = 4000
    patience: int | None = 300
    seed: int = 42

    def __post_init__(self) -> None:
        '''Validate dimensions, optimisation settings and the random seed.'''
        for name in ('dim', 'n_layers', 'hidden', 'max_epochs'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.dim != 2:
            raise ValueError('The two-moon experiment requires dim=2.')
        if self.patience is not None and (
            type(self.patience) is not int or self.patience < 1
        ):
            raise ValueError('patience must be a positive integer or None.')
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError('learning_rate must be finite and positive.')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63).')


@dataclass
class FitResult:
    '''Store the selected model, post-update learning curves and selected NLLs.'''

    model: NormalisingFlow
    config: TrainConfig
    history: dict[str, list[int] | list[float]]
    best_epoch: int
    train_nll: float
    val_nll: float | None
    initial_train_nll: float
    initial_val_nll: float | None


def _features(data: Tensor, dim: int, dtype: torch.dtype) -> Tensor:
    '''Validate finite CPU features and detach them from any autograd graph.'''
    if not isinstance(data, Tensor):
        raise TypeError('Features must be a torch.Tensor.')
    if data.device.type != 'cpu':
        raise ValueError('This training pipeline uses CPU tensors.')
    if not data.is_floating_point():
        raise TypeError('Features must have a floating-point dtype.')
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != dim:
        raise ValueError(f'Features must have non-empty shape (N, {dim}).')
    if not bool(torch.isfinite(data).all()):
        raise ValueError('Features must contain only finite values.')
    return data.detach().to(dtype=dtype)


def evaluate_nll(model: NormalisingFlow, data: Tensor) -> float:
    '''Evaluate mean negative log likelihood in nats per observation.

    Args
        model (NormalisingFlow): CPU flow to evaluate.
        data (Tensor): Finite input features with shape (N, 2).

    Returns
        float: Mean negative log likelihood, which may be negative.
    '''
    parameter = next(model.parameters())
    if parameter.device.type != 'cpu':
        raise ValueError('This evaluation pipeline uses CPU models.')
    features = _features(data, model.dim, parameter.dtype)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            value = -model.log_prob(features).mean().item()
    finally:
        model.train(was_training)
    if not math.isfinite(value):
        raise FloatingPointError('The model produced a non-finite NLL.')
    return value


def fit(
    train: Tensor,
    val: Tensor | None,
    config: TrainConfig = TrainConfig(),
    progress: Callable[[dict[str, int | float]], None] | None = None,
) -> FitResult:
    '''Fit a flow and restore the epoch with the lowest validation NLL.

    Both learning curves describe the model after each optimiser update. The
    recorded learning rate is the value used for that update. Without validation,
    patience must be None and the final epoch is retained. Test data are not used.

    Args
        train (Tensor): Training features with shape (N_train, 2).
        val (Tensor | None): Validation features with shape (N_val, 2).
        config (TrainConfig): Architecture, optimisation and stopping settings.
        progress (Callable | None): Callback receiving each completed epoch.

    Returns
        FitResult: Selected model, learning curves and re-evaluated NLLs.
    '''
    if not isinstance(config, TrainConfig):
        raise TypeError('config must be a TrainConfig.')
    x_train = _features(train, config.dim, torch.float32)
    x_val = None if val is None else _features(val, config.dim, torch.float32)
    if x_val is None and config.patience is not None:
        raise ValueError('Validation data are required when patience is set.')

    history: dict[str, list[int] | list[float]] = {
        'epoch': [], 'train_nll': [], 'val_nll': [], 'learning_rate': []
    }
    best_state = None
    best_val_nll = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    # Keep seeded initialisation and training local to this call.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        model = NormalisingFlow(config.dim, config.n_layers, config.hidden).to(
            device='cpu', dtype=torch.float32
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=config.max_epochs
        )
        initial_train_nll = evaluate_nll(model, x_train)
        initial_val_nll = None if x_val is None else evaluate_nll(model, x_val)

        for epoch in range(1, config.max_epochs + 1):
            model.train()
            optimiser.zero_grad(set_to_none=True)
            loss = -model.log_prob(x_train).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch}.')
            loss.backward()
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f'Non-finite gradient at epoch {epoch}.')
            learning_rate = optimiser.param_groups[0]['lr']
            optimiser.step()
            scheduler.step()

            train_nll = evaluate_nll(model, x_train)
            val_nll = None if x_val is None else evaluate_nll(model, x_val)
            history['epoch'].append(epoch)
            history['train_nll'].append(train_nll)
            history['learning_rate'].append(learning_rate)

            if val_nll is not None:
                history['val_nll'].append(val_nll)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().clone()
                        for name, value in model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            else:
                best_epoch = epoch

            if progress is not None:
                record = {
                    'epoch': epoch, 'train_nll': train_nll,
                    'learning_rate': learning_rate,
                }
                if val_nll is not None:
                    record['val_nll'] = val_nll
                progress(record)
            if (
                config.patience is not None
                and epochs_without_improvement >= config.patience
            ):
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        train_nll = evaluate_nll(model, x_train)
        val_nll = None if x_val is None else evaluate_nll(model, x_val)

    return FitResult(
        model, config, history, best_epoch, train_nll, val_nll,
        initial_train_nll, initial_val_nll,
    )


def _json_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    '''Copy finite JSON-compatible metadata into plain Python containers.'''
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TypeError('Metadata must be a mapping with string keys.')
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError('Metadata must contain only finite JSON-compatible values.') from error


def save_checkpoint(
    path: str | Path, result: FitResult, metadata: Mapping[str, Any] | None = None
) -> None:
    '''Save model weights, configuration and JSON-compatible experiment metadata.

    Args
        path (str | Path): Checkpoint output path.
        result (FitResult): Fitted model and its selected-epoch metrics.
        metadata (Mapping | None): Optional experiment information.
    '''
    selection = {
        'best_epoch': result.best_epoch,
        'epochs_run': len(result.history['epoch']),
        'train_nll': result.train_nll,
        'val_nll': result.val_nll,
    }
    payload = {
        'format_version': 1,
        'config': asdict(result.config),
        'state_dict': {
            name: value.detach().cpu().clone()
            for name, value in result.model.state_dict().items()
        },
        'metadata': _json_metadata({
            'training': selection,
            'experiment': {} if metadata is None else _json_metadata(metadata),
        }),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
) -> tuple[NormalisingFlow, TrainConfig, dict[str, Any]]:
    '''Load a CPU model using restricted weight loading and strict validation.

    Args
        path (str | Path): Checkpoint written by save_checkpoint.

    Returns
        tuple: Evaluation-mode model, configuration and metadata containing
            training and experiment dictionaries.
    '''
    payload = torch.load(path, map_location='cpu', weights_only=True)
    required = {'format_version', 'config', 'state_dict', 'metadata'}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError('Checkpoint fields do not match the expected format.')
    if type(payload['format_version']) is not int or payload['format_version'] != 1:
        raise ValueError('Unsupported checkpoint format version.')
    config_fields = {field.name for field in fields(TrainConfig)}
    if not isinstance(payload['config'], dict) or set(payload['config']) != config_fields:
        raise ValueError('Checkpoint configuration fields are invalid.')
    config = TrainConfig(**payload['config'])
    metadata = _json_metadata(payload['metadata'])
    if set(metadata) != {'training', 'experiment'}:
        raise ValueError('Checkpoint metadata fields are invalid.')
    selection = metadata['training']
    if not isinstance(selection, dict) or set(selection) != {
        'best_epoch', 'epochs_run', 'train_nll', 'val_nll'
    }:
        raise ValueError('Checkpoint training metadata are invalid.')
    if (
        type(selection['best_epoch']) is not int
        or type(selection['epochs_run']) is not int
        or not 1 <= selection['best_epoch'] <= selection['epochs_run'] <= config.max_epochs
    ):
        raise ValueError('Checkpoint epoch metadata are invalid.')
    for key in ('train_nll', 'val_nll'):
        value = selection[key]
        if key == 'val_nll' and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Checkpoint NLL metadata must be finite.')
    if not isinstance(metadata['experiment'], dict):
        raise ValueError('Checkpoint experiment metadata must be a dictionary.')

    with torch.random.fork_rng(devices=[]):
        model = NormalisingFlow(config.dim, config.n_layers, config.hidden).to(
            device='cpu', dtype=torch.float32
        )
    expected = model.state_dict()
    state = payload['state_dict']
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Checkpoint parameter names do not match the architecture.')
    for name, value in state.items():
        if (
            not isinstance(value, Tensor)
            or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f'Invalid checkpoint tensor: {name}.')
        if name.endswith('.mask') and not torch.equal(value, expected[name]):
            raise ValueError(f'Checkpoint coupling mask is invalid: {name}.')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, metadata
