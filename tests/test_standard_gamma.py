import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.standard_gamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma(shape, dtype):
    # Generate positive alpha values (shape parameter must be positive)
    res_inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5.0 + 0.5
    # Upcast the reference input: torch._standard_gamma has no bfloat16 CPU
    # kernel ("gamma_cpu" not implemented for 'BFloat16'), so in quick-cpu mode
    # the reference must run in a higher-precision float that CPU supports.
    ref_inp = utils.to_reference(res_inp, upcast=True)

    # For random number generators, we can't compare exact values
    # Instead, we verify:
    # 1. Output shape and dtype match
    # 2. All values are non-negative (Gamma distribution property)
    # 3. Statistical properties are reasonable

    ref_out = torch._standard_gamma(ref_inp)
    res_out = flag_gems.standard_gamma(res_inp)

    # Check shape and dtype
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype

    # Check all values are non-negative
    assert torch.all(res_out >= 0), "Gamma samples should be non-negative"

    # For larger samples, check basic statistical properties
    if res_inp.numel() >= 100:
        # Mean of Gamma(alpha, 1) should be approximately alpha
        # We check this with a large tolerance since it's stochastic
        mean_inp = res_inp.mean().item()
        mean_out = res_out.mean().item()
        # Allow 50% tolerance for statistical variation
        assert (
            abs(mean_out - mean_inp) < mean_inp * 0.5 + 1.0
        ), f"Mean mismatch: expected ~{mean_inp}, got {mean_out}"


@pytest.mark.standard_gamma
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_standard_gamma_statistics(alpha):
    """Test statistical properties with fixed alpha values."""
    # Use a large sample size for better statistical properties
    shape = (10000,)
    dtype = torch.float32

    inp = torch.full(shape, alpha, dtype=dtype, device=flag_gems.device)
    out = flag_gems.standard_gamma(inp)

    # Check basic properties
    assert out.shape == shape
    assert out.dtype == dtype
    assert torch.all(out >= 0)

    # Statistical checks (with relaxed tolerance due to randomness)
    # For Gamma(alpha, 1): mean = alpha, variance = alpha
    mean = out.mean().item()
    var = out.var().item()

    # Allow 20% error in mean and variance estimation
    assert (
        abs(mean - alpha) < alpha * 0.2 + 0.5
    ), f"Mean error too large: expected {alpha}, got {mean}"
    assert (
        abs(var - alpha) < alpha * 0.3 + 1.0
    ), f"Variance error too large: expected {alpha}, got {var}"


@pytest.mark.standard_gamma
def test_standard_gamma_randomness():
    """Test that the function generates different values on each call."""
    shape = (1000,)
    dtype = torch.float32

    inp = torch.ones(shape, dtype=dtype, device=flag_gems.device) * 2.0

    out1 = flag_gems.standard_gamma(inp)
    out2 = flag_gems.standard_gamma(inp)

    # The two outputs should be different (with very high probability)
    assert not torch.allclose(
        out1, out2
    ), "Multiple calls should generate different random values"


@pytest.mark.standard_gamma
@pytest.mark.parametrize("alpha", [1e-4, 1e-3, 1e-2, 0.05])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_standard_gamma_tiny_alpha(alpha, dtype):
    """Tiny alpha must not underflow to exactly zero.

    Like ATen, the sample is clamped to the smallest positive value of the
    output dtype, so every draw stays strictly positive.
    """
    shape = (4096,)
    inp = torch.full(shape, alpha, dtype=dtype, device=flag_gems.device)
    out = flag_gems.standard_gamma(inp)

    assert out.dtype == dtype
    assert torch.all(out > 0), "Tiny-alpha samples must be strictly positive"


@pytest.mark.standard_gamma
def test_standard_gamma_generator_reproducible():
    """Passing the same seeded generator must reproduce the same samples."""
    shape = (2048,)
    dtype = torch.float32
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.5

    gen = torch.Generator(device=flag_gems.device)
    gen.manual_seed(12345)
    out1 = flag_gems.standard_gamma(inp, generator=gen)

    gen.manual_seed(12345)
    out2 = flag_gems.standard_gamma(inp, generator=gen)

    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


@pytest.mark.standard_gamma
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.int16, torch.uint8, torch.bool]
)
def test_standard_gamma_rejects_non_floating(dtype):
    """Non-floating inputs must raise RuntimeError, matching ATen's
    'gamma_cuda not implemented for <int type>' behaviour, rather than
    leaking a TypeError from torch.finfo."""
    inp = torch.ones((16,), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.standard_gamma(inp)


@pytest.mark.standard_gamma
def test_standard_gamma_rejects_float64():
    """The fp32 sampling kernel cannot preserve the float64 tiny value, so
    float64 inputs are rejected explicitly instead of silently downcasting."""
    inp = torch.rand((16,), dtype=torch.float64, device=flag_gems.device) + 0.5
    with pytest.raises(RuntimeError):
        flag_gems.standard_gamma(inp)


@pytest.mark.standard_gamma
def test_standard_gamma_generator_advances_state():
    """Consecutive draws from one generator advance its state (differ)."""
    shape = (2048,)
    dtype = torch.float32
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.5

    gen = torch.Generator(device=flag_gems.device)
    gen.manual_seed(777)
    out1 = flag_gems.standard_gamma(inp, generator=gen)
    out2 = flag_gems.standard_gamma(inp, generator=gen)

    assert not torch.allclose(
        out1, out2
    ), "Generator state should advance between consecutive calls"
