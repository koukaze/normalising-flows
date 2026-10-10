'''Check shear inversion, volume preservation and exact induced densities.'''

import pytest
import torch

from moonflow.models import NormalisingFlow
from moonflow.transforms import family_log_prob, g_alpha, g_alpha_inverse, sample_family


@pytest.mark.parametrize('alpha', [-2.0, 0.0, 2.5])
def test_shear_both_roundtrips_and_jacobian(alpha: float) -> None:
    points = torch.tensor([[1.0, 2.0], [-0.25, 0.8], [4.0, -3.0]], dtype=torch.float64)
    transformed, forward_log_det = g_alpha(points, alpha)
    recovered, inverse_log_det = g_alpha_inverse(transformed, alpha)
    torch.testing.assert_close(recovered, points, rtol=1e-14, atol=1e-14)
    reverse = g_alpha(g_alpha_inverse(points, alpha)[0], alpha)[0]
    torch.testing.assert_close(reverse, points, rtol=1e-14, atol=1e-14)
    assert torch.count_nonzero(forward_log_det) == 0
    assert torch.count_nonzero(inverse_log_det) == 0
    jacobian = torch.autograd.functional.jacobian(lambda value: g_alpha(value, alpha)[0], points[0])
    reference = torch.tensor([[1.0, alpha], [0.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(jacobian, reference, rtol=0, atol=0)
    torch.testing.assert_close(torch.linalg.det(jacobian), torch.tensor(1.0, dtype=torch.float64))


@pytest.mark.parametrize('alpha', [-2.0, 0.0, 1.25])
def test_sheared_gaussian_matches_multivariate_density(alpha: float) -> None:
    flow = NormalisingFlow(n_layers=2, hidden=4).double()
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.zero_()
    covariance = torch.tensor([[1 + alpha ** 2, alpha], [alpha, 1]], dtype=torch.float64)
    reference = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), covariance)
    points = torch.tensor([[1.0, -0.5], [-0.6, 1.3], [0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(family_log_prob(flow, points, alpha), reference.log_prob(points), rtol=1e-13, atol=1e-13)


def test_composed_flow_density_and_samples() -> None:
    torch.manual_seed(11)
    flow = NormalisingFlow(n_layers=4, hidden=8).double()
    alpha = 1.7
    latent = torch.tensor([0.3, -0.4], dtype=torch.float64)
    composed = lambda value: g_alpha(flow(value)[0], alpha)[0]
    points = composed(latent)
    jacobian = torch.autograd.functional.jacobian(composed, latent)
    reference = torch.distributions.Normal(0.0, 1.0).log_prob(latent).sum() - torch.linalg.slogdet(jacobian).logabsdet
    torch.testing.assert_close(family_log_prob(flow, points, alpha), reference, rtol=1e-13, atol=1e-13)
    base = flow.sample(12, generator=torch.Generator().manual_seed(26))
    sheared = sample_family(flow, alpha, 12, generator=torch.Generator().manual_seed(26))
    torch.testing.assert_close(sheared, g_alpha(base, alpha)[0], rtol=0, atol=0)


def test_shear_supports_batch_axes_and_rejects_invalid_inputs() -> None:
    points = torch.ones(3, 4, 2, dtype=torch.float64)
    transformed, log_det = g_alpha(points, 1.0)
    assert transformed.shape == (3, 4, 2)
    assert log_det.shape == (3, 4)
    with pytest.raises(ValueError, match='finite'):
        g_alpha(points, float('inf'))
    with pytest.raises(ValueError, match='dimension'):
        g_alpha(torch.ones(3), 1.0)
    with pytest.raises(ValueError, match='two-dimensional'):
        family_log_prob(NormalisingFlow(dim=3), points, 1.0)
