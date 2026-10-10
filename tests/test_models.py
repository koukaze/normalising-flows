'''Check change-of-variables identities and model execution semantics.'''

import math

import pytest
import torch

from moonflow.checks import count_flops, invertibility_errors, log_det_check
from moonflow.models import CouplingLayer, NormalisingFlow


@pytest.mark.parametrize('dim', [2, 3])
def test_coupling_roundtrip_and_jacobian(dim: int) -> None:
    torch.manual_seed(8)
    mask = (torch.arange(dim) % 2).double()
    layer = CouplingLayer(mask, dim=dim, hidden=8).double()
    latent = torch.randn(4, dim, dtype=torch.float64)
    points, forward_log_det = layer(latent)
    recovered, inverse_log_det = layer.inverse(points)
    torch.testing.assert_close(recovered, latent, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(forward_log_det, -inverse_log_det, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(points[:, mask.bool()], latent[:, mask.bool()], rtol=0, atol=0)
    for index, point in enumerate(latent):
        jacobian = torch.autograd.functional.jacobian(lambda value: layer(value)[0], point)
        torch.testing.assert_close(
            forward_log_det[index], torch.linalg.slogdet(jacobian).logabsdet,
            rtol=1e-12, atol=1e-12,
        )


@pytest.mark.parametrize('shape', [(2,), (7, 2), (3, 4, 2), (0, 2)])
def test_flow_handles_leading_batch_dimensions(shape: tuple[int, ...]) -> None:
    torch.manual_seed(13)
    flow = NormalisingFlow(n_layers=4, hidden=8).double()
    latent = torch.randn(shape, dtype=torch.float64)
    points, forward_log_det = flow(latent)
    recovered, inverse_log_det = flow.inverse(points)
    assert points.shape == latent.shape
    assert forward_log_det.shape == latent.shape[:-1]
    assert flow.log_prob(points).shape == latent.shape[:-1]
    torch.testing.assert_close(latent, recovered, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(forward_log_det, -inverse_log_det, rtol=1e-12, atol=1e-12)
    log_base = -0.5 * (latent.square().sum(dim=-1) + 2 * math.log(2 * math.pi))
    torch.testing.assert_close(flow.log_prob(points), log_base - forward_log_det, rtol=1e-12, atol=1e-12)


def test_flow_both_roundtrips_and_independent_jacobians() -> None:
    torch.manual_seed(5)
    flow = NormalisingFlow(n_layers=4, hidden=8).double()
    points = torch.randn(6, 2, dtype=torch.float64)
    errors = invertibility_errors(flow, points)
    assert max(errors.values()) < 1e-12
    estimates = log_det_check(flow, points, eps=1e-5)
    torch.testing.assert_close(estimates['analytic'], estimates['autograd'], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(estimates['analytic'], estimates['finite_difference'], rtol=1e-7, atol=1e-8)


def test_zero_network_is_standard_normal() -> None:
    flow = NormalisingFlow().double()
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.zero_()
    points = torch.tensor([[1.0, -2.0], [0.0, 0.0]], dtype=torch.float64)
    reference = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)),
        1,
    )
    torch.testing.assert_close(flow.log_prob(points), reference.log_prob(points), rtol=1e-14, atol=1e-14)


def test_sampling_respects_dtype_and_local_generator() -> None:
    torch.manual_seed(19)
    flow = NormalisingFlow(n_layers=2, hidden=8).double()
    global_state = torch.get_rng_state().clone()
    first = flow.sample(7, generator=torch.Generator().manual_seed(32))
    second = flow.sample(7, generator=torch.Generator().manual_seed(32))
    assert first.dtype == torch.float64
    assert first.device == next(flow.parameters()).device
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(global_state, torch.get_rng_state())
    assert flow.sample(0).shape == (0, 2)


def test_constructor_respects_default_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        flow = NormalisingFlow(n_layers=2, hidden=8)
        assert all(layer.mask.dtype == torch.float64 for layer in flow.layers)
        assert flow.sample(3).dtype == torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_likelihood_has_finite_parameter_gradients() -> None:
    torch.manual_seed(21)
    flow = NormalisingFlow(n_layers=3, hidden=8).double()
    points = torch.randn(12, 2, dtype=torch.float64)
    (-flow.log_prob(points).mean()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in flow.parameters())


@pytest.mark.parametrize('kwargs', [{'dim': 0}, {'n_layers': -1}, {'hidden': 0}, {'hidden': True}])
def test_invalid_architecture_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        NormalisingFlow(**kwargs)


def test_invalid_masks_and_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match='binary'):
        CouplingLayer(torch.tensor([1.0, 0.5]), dim=2, hidden=8)
    flow = NormalisingFlow(n_layers=2, hidden=8)
    with pytest.raises(ValueError, match='dimension'):
        flow.log_prob(torch.randn(4, 3))
    with pytest.raises(TypeError, match='floating-point'):
        flow.log_prob(torch.ones(4, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match='nonnegative'):
        flow.sample(-1)


def test_diagnostic_inputs_and_operation_count() -> None:
    flow = NormalisingFlow(n_layers=2, hidden=8)
    with pytest.raises(ValueError, match='At least one'):
        invertibility_errors(flow, torch.empty(0, 2))
    with pytest.raises(ValueError, match='eps'):
        log_det_check(flow, torch.randn(2, 2), eps=0)
    assert count_flops(dim=2, n_layers=6, hidden=64, batch_size=800) == 4_303_200
    with pytest.raises(ValueError, match='batch_size'):
        count_flops(dim=2, n_layers=6, hidden=64, batch_size=0)
