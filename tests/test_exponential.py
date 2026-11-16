import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

OUTPLACE_SHAPES = [
    (64, 64),
    (100, 1, 100),
    (10000, 1),
    (100, 256, 100),
    (10000, 256),
    (20, 320, 15),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
]

OUTPLACE_MOMENT_SHAPES = [(256, 1024), (1024, 1024), (4096, 4096), (64, 512, 512)]


@pytest.mark.exponential
@pytest.mark.parametrize("shape", OUTPLACE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exponential(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    x_ref = x.clone()
    y = flag_gems.exponential(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.min() > 0

    torch.testing.assert_close(x, x_ref)


@pytest.mark.exponential
@pytest.mark.parametrize("shape", OUTPLACE_MOMENT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exponential_moments(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    lambd = 1.0
    mean_tol = 0.05
    var_tol = 0.05
    y = flag_gems.exponential(x, lambd=lambd)

    y_res = utils.to_reference(y)
    mean_res = torch.mean(y_res.to(torch.float32)).to(dtype)
    var_res = torch.var(y_res.to(torch.float32)).to(dtype)
    mean_ref = 1.0 / lambd
    var_ref = 1.0 / (lambd**2)

    assert torch.abs(mean_res - mean_ref) < mean_tol
    assert torch.abs(var_res - var_ref) < var_tol


@pytest.mark.exponential
@pytest.mark.parametrize("seed", [260828, 2026, 7])
def test_exponential_far_left_tail(seed):
    """The far left tail must not carry an inflated atom (Issue #7040).

    The fast transform pinned every sample with ``u >= 1 - 1e-4`` to
    ``y == 0.5 * eps(float32) = 5.96e-8``, so ~1e-4 of the mass sat just above
    zero: ``P(X < 1e-5)`` measured ~1e-4 instead of ~1e-5.
    """
    x = torch.empty(1 << 22, dtype=torch.float32, device=flag_gems.device)
    generator = torch.Generator(device=flag_gems.device).manual_seed(seed)
    y = flag_gems.exponential(x, generator=generator)

    assert y.min().item() > 0, "exponential must not produce non-positive values"
    # Exp(1): P(X < t) = 1 - exp(-t) ~= t for small t. Bound each point by a
    # few times the true mass; the buggy atom overshoots by ~90x.
    for t, bound in ((1e-6, 5e-6), (1e-5, 3e-5)):
        observed = (y < t).float().mean().item()
        assert (
            observed < bound
        ), f"seed={seed}: P(X < {t:.0e}) = {observed:.3e}, expected ~{t:.0e}"
