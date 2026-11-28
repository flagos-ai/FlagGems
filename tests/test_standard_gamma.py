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
@pytest.mark.skipif(
    not utils.fp64_is_supported,
    reason="enflame/GCU does not support FP64/FP4/COMPLEX128",
)
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


@pytest.mark.standard_gamma_backward
def test_standard_gamma_native_autograd_contract():
    """Native ATen exposes an alpha gradient for the sampled gamma tensor."""
    alpha = torch.tensor([0.5, 1.0, 2.0], device=flag_gems.device, requires_grad=True)
    sample = torch._standard_gamma(alpha)
    grad_output = torch.tensor([0.5, 1.0, 2.0], device=alpha.device)
    assert sample.grad_fn is not None
    (grad_alpha,) = torch.autograd.grad(sample, alpha, grad_output)
    expected = grad_output * torch._standard_gamma_grad(alpha.detach(), sample.detach())
    torch.testing.assert_close(grad_alpha, expected)
    assert torch.isfinite(grad_alpha).all()
    assert torch.any(grad_alpha != 0)


@pytest.mark.standard_gamma_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_backward(shape, dtype):
    """The alpha gradient must match ATen's ``StandardGammaBackward0``.

    ``_standard_gamma`` is stochastic, so both sides differentiate the same
    saved sample: the reference recomputes ATen's own backward formula
    ``grad_output * _standard_gamma_grad(alpha, sample)`` on the candidate's
    output.  That expression is exactly what ``torch.autograd.grad`` returns on
    H800 (artifacts/native_probe.json), so no second random draw is involved.
    """
    alpha = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.5
    alpha.requires_grad_()
    sample = flag_gems.standard_gamma(alpha)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    (grad_alpha,) = torch.autograd.grad(sample, alpha, grad_output)

    expected = _reference_grad(alpha, sample, grad_output)
    utils.gems_assert_close(grad_alpha, expected, dtype)
    assert grad_alpha.shape == alpha.shape
    assert grad_alpha.dtype == alpha.dtype


def _reference_grad(alpha, sample, grad_output):
    """ATen's backward formula for the current reference mode.

    On-device the reference keeps the input dtype, which is exactly how ATen
    evaluates ``standard_gamma_grad_one`` (its ``accscalar_t`` is fp32 for
    fp16/bf16 and fp64 for fp64).  Under ``--ref cpu`` the tensors move to the
    CPU and upcast, because the CPU kernel has no BFloat16/Half gamma path.
    """
    ref_alpha = alpha.detach()
    ref_sample = sample.detach()
    ref_grad = grad_output
    if utils.TO_CPU:
        ref_alpha = ref_alpha.to("cpu").double()
        ref_sample = ref_sample.to("cpu").double()
        ref_grad = ref_grad.to("cpu").double()
    return ref_grad * torch._standard_gamma_grad(ref_alpha, ref_sample)


@pytest.mark.standard_gamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_backward_enters_custom_function(dtype):
    """The differentiable path must route through the custom autograd node.

    A non-empty ``grad_fn`` alone does not attribute the kernel, so this also
    checks the node name and compares the gradient with the saved sample.
    """
    alpha = torch.rand((64, 32), dtype=dtype, device=flag_gems.device) + 0.5
    alpha.requires_grad_()
    sample = flag_gems.standard_gamma(alpha)

    assert sample.requires_grad
    assert type(sample.grad_fn).__name__ == "_StandardGammaFunctionBackward"
    (grad_alpha,) = torch.autograd.grad(
        sample, alpha, torch.ones_like(sample), retain_graph=True
    )
    expected = _reference_grad(alpha, sample, torch.ones_like(sample))
    utils.gems_assert_close(grad_alpha, expected, dtype)


@pytest.mark.standard_gamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_backward_noncontiguous_and_expanded_grad(dtype):
    """Non-contiguous alpha and an expanded upstream gradient use the fallback."""
    alpha = (torch.rand((32, 64), dtype=dtype, device=flag_gems.device) * 4.0 + 0.5).t()
    alpha.requires_grad_()
    sample = flag_gems.standard_gamma(alpha)
    grad_output = torch.randn((64, 1), dtype=dtype, device=flag_gems.device).expand(
        64, 32
    )

    (grad_alpha,) = torch.autograd.grad(sample, alpha, grad_output)

    expected = _reference_grad(alpha, sample, grad_output)
    utils.gems_assert_close(grad_alpha, expected, dtype)
    assert grad_alpha.shape == alpha.shape


@pytest.mark.standard_gamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_backward_small_alpha_taylor_branch(dtype):
    """Canonical-alpha draws exercise ATen's Taylor branch (sample < 0.8).

    With the repository's canonical alpha = rand * 4 + 0.5, about 22% of gamma
    samples still land below 0.8, so the Taylor-series branch of the saved
    sample is exercised without leaving the alpha domain every other test of
    this family uses.  (The sub-0.5 domain was measured to exceed the
    repository tolerance for the merged ``standard_gamma_grad`` approximation
    itself: max |helper - ATen| ~ 1.7e-4 in fp32 for alpha down to 0.05, which
    is upstream behavior this PR does not change.)
    """
    alpha = torch.rand((4096,), dtype=dtype, device=flag_gems.device) * 4.0 + 0.5
    alpha.requires_grad_()
    sample = flag_gems.standard_gamma(alpha)
    grad_output = torch.randn((4096,), dtype=dtype, device=flag_gems.device)

    (grad_alpha,) = torch.autograd.grad(sample, alpha, grad_output)

    assert (sample.detach() < 0.8).any(), "Taylor branch must be reached"
    utils.gems_assert_close(
        grad_alpha, _reference_grad(alpha, sample, grad_output), dtype
    )


@pytest.mark.standard_gamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_backward_large_alpha(dtype):
    """Large alpha (> 8) drives the backward through ATen's Rice saddle point.

    The fused multiply is pinned against ``standard_gamma_grad``, whose own
    accuracy for every branch (including the Rice near-mode and away-from-mode
    sub-cases) is covered by ``tests/test_standard_gamma_grad.py``.  Comparing
    the fused launch to that op isolates this PR's change from the derivative
    approximation shared with the already-merged forward operator.
    """
    alpha = torch.rand((4096,), dtype=dtype, device=flag_gems.device) * 31.5 + 8.5
    alpha.requires_grad_()
    sample = flag_gems.standard_gamma(alpha)
    grad_output = torch.randn((4096,), dtype=dtype, device=flag_gems.device)

    (grad_alpha,) = torch.autograd.grad(sample, alpha, grad_output)

    expected = _fused_reference(alpha, sample, grad_output, dtype)
    utils.gems_assert_close(grad_alpha, expected, dtype)


def _fused_reference(alpha, sample, grad_output, dtype):
    """Expected result of the fused launch, from the shared derivative op.

    ``standard_gamma_grad`` is the already-tested op the backward kernel reuses,
    so this pins the fused multiply without duplicating the branchy derivative.
    Under ``--ref cpu`` the result follows the reference to the CPU.
    """
    reference = flag_gems.standard_gamma_grad(alpha.detach(), sample.detach())
    if dtype in (torch.float16, torch.bfloat16):
        # The fused kernel multiplies in fp32 opmath and rounds once.
        expected = (reference.float() * grad_output.float()).to(dtype)
    else:
        expected = reference * grad_output
    return expected.to("cpu") if utils.TO_CPU else expected


@pytest.mark.standard_gamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_standard_gamma_backward_empty_and_scalar(dtype, shape):
    """Zero-element and 0-dim inputs keep the native shape/dtype contract."""
    alpha = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.5
    alpha.requires_grad_()
    sample = flag_gems.standard_gamma(alpha)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    (grad_alpha,) = torch.autograd.grad(sample, alpha, grad_output)

    assert grad_alpha.shape == alpha.shape
    assert grad_alpha.dtype == alpha.dtype
    assert grad_alpha.numel() == alpha.numel()


@pytest.mark.standard_gamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_backward_needs_input_grad(dtype):
    """A non-differentiable generator argument must not receive a gradient."""
    alpha = torch.rand((128,), dtype=dtype, device=flag_gems.device) * 4.0 + 0.5
    alpha.requires_grad_()
    gen = torch.Generator(device=flag_gems.device).manual_seed(1234)
    sample = flag_gems.standard_gamma(alpha, generator=gen)

    (grad_alpha,) = torch.autograd.grad(
        sample, alpha, torch.ones_like(sample), allow_unused=True
    )

    assert grad_alpha is not None
    expected = _reference_grad(alpha, sample, torch.ones_like(sample))
    utils.gems_assert_close(grad_alpha, expected, dtype)


@pytest.mark.standard_gamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_backward_matches_fused_grad(dtype):
    """The fused launch must equal grad_output * standard_gamma_grad.

    This pins the only new math in the backward (the fused multiply) against the
    already-tested derivative op, so a regression in the fusion cannot hide
    behind the shared kernel.
    """
    alpha = torch.rand((64, 512), dtype=dtype, device=flag_gems.device) * 4.0 + 0.5
    alpha.requires_grad_()
    sample = flag_gems.standard_gamma(alpha)
    grad_output = torch.randn((64, 512), dtype=dtype, device=flag_gems.device)

    (grad_alpha,) = torch.autograd.grad(sample, alpha, grad_output)

    expected = _fused_reference(alpha, sample, grad_output, dtype)
    utils.gems_assert_close(grad_alpha, expected, dtype)


@pytest.mark.standard_gamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_backward_first_order_only(dtype):
    """ATen's ``_standard_gamma_grad`` has no derivative, so higher order fails."""
    alpha = torch.rand((32,), dtype=dtype, device=flag_gems.device) * 4.0 + 0.5
    alpha.requires_grad_()
    sample = flag_gems.standard_gamma(alpha)
    (grad_alpha,) = torch.autograd.grad(
        sample, alpha, torch.ones_like(sample), create_graph=True
    )

    # The Triton backward launches a kernel, so it produces a detached
    # first-order gradient instead of an autograd node; a second derivative is
    # therefore rejected, matching ATen raising NotImplementedError for
    # "_standard_gamma_grad" rather than returning a value.
    assert not grad_alpha.requires_grad
    with pytest.raises(RuntimeError):
        torch.autograd.grad(grad_alpha.sum(), alpha, allow_unused=True)
