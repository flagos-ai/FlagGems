import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.trapz
@pytest.mark.parametrize("shape", [(128,), (64, 128), (16, 32, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dx", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("dim", [-1, 0])
def test_trapz(shape, dtype, dx, dim):
    # Skip invalid dim for shape
    if dim >= len(shape) or dim < -len(shape):
        pytest.skip("Invalid dim for shape")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.trapezoid(ref_inp, dx=dx, dim=dim)

    res_out = flag_gems.trapz(inp, dx=dx, dim=dim)

    # Trapezoid is a reduction; scale atol by the reduced dimension length
    reduce_dim = shape[dim]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.trapz
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_edge_cases(dtype):
    """Test edge cases: single element, two elements."""
    # Single element - should return zero
    inp = torch.randn(1, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.trapezoid(ref_inp, dx=1.0)

    res_out = flag_gems.trapz(inp, dx=1.0)
    utils.gems_assert_close(res_out, ref_out, dtype)

    # Two elements
    inp = torch.randn(2, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.trapezoid(ref_inp, dx=1.0)

    res_out = flag_gems.trapz(inp, dx=1.0)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=2)


@pytest.mark.trapz
@pytest.mark.parametrize("shape", [(1024, 1024), (512, 2048)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_large_reduction(shape, dtype):
    """Test trapezoid with large reduction dimension."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.trapezoid(ref_inp, dx=0.01, dim=-1)

    res_out = flag_gems.trapz(inp, dx=0.01, dim=-1)

    # Large reduction accumulates more error
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[-1])


@pytest.mark.trapz
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_empty_input(dtype):
    """Reduction over a zero-length dim yields a zero-filled reduced shape."""
    inp = torch.empty((2, 0, 3), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.trapezoid(ref_inp, dx=1.0, dim=1)
    res_out = flag_gems.trapz(inp, dx=1.0, dim=1)

    assert res_out.shape == (2, 3)
    assert ref_out.shape == (2, 3)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.trapz
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_single_length_dim(dtype):
    """Reduction over a length-1 dim yields zeros with the dim removed."""
    inp = torch.randn((3, 1, 4), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.trapezoid(ref_inp, dx=1.0, dim=1)
    res_out = flag_gems.trapz(inp, dx=1.0, dim=1)

    assert res_out.shape == (3, 4)
    assert ref_out.shape == (3, 4)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.trapz
def test_trapz_invalid_dim():
    """Scalar input and out-of-range dims must raise IndexError."""
    inp = torch.randn((2, 3), dtype=torch.float32, device=flag_gems.device)
    scalar = torch.tensor(1.0, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(IndexError):
        flag_gems.trapz(scalar, dx=1.0, dim=0)
    with pytest.raises(IndexError):
        flag_gems.trapz(inp, dx=1.0, dim=2)
    with pytest.raises(IndexError):
        flag_gems.trapz(inp, dx=1.0, dim=-3)


@pytest.mark.trapz
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_non_contiguous(dtype):
    """Non-contiguous input must match the reference along every dim."""
    inp = torch.randn((8, 6, 4), dtype=dtype, device=flag_gems.device)
    inp = inp.transpose(0, 2)  # non-contiguous view
    ref_inp = utils.to_reference(inp, True)

    for dim in (0, 1, 2, -1):
        ref_out = torch.trapezoid(ref_inp, dx=1.0, dim=dim)
        res_out = flag_gems.trapz(inp, dx=1.0, dim=dim)
        utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.shape[dim])


@pytest.mark.trapz
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_trapz_int_promotion(dtype):
    """Integer inputs are promoted to float32, matching PyTorch."""
    inp = torch.randint(-8, 8, (3, 4), device="cpu", dtype=dtype).to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.trapezoid(ref_inp, dx=0.5, dim=-1)
    res_out = flag_gems.trapz(inp, dx=0.5, dim=-1)

    assert res_out.dtype == torch.float32
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=4)


@pytest.mark.trapz
@pytest.mark.skipif(not utils.fp64_is_supported, reason="fp64 not supported")
@pytest.mark.parametrize("shape", [(64, 128), (16, 32, 64)])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("dx", [0.5, 0.1])
def test_trapz_fp64(shape, dim, dx):
    """float64 inputs accumulate in float64 and round-trip losslessly.

    ``dx=0.1`` is not exactly representable in fp32, so this exercises the
    ``dx: tl.float64`` kernel annotation: an fp32 ``dx`` would round 0.1 and
    introduce a ~1.5e-8 relative error on the final ``acc * dx``.
    """
    if dim >= len(shape) or dim < -len(shape):
        pytest.skip("Invalid dim for shape")

    dtype = torch.float64
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.trapezoid(ref_inp, dx=dx, dim=dim)
    res_out = flag_gems.trapz(inp, dx=dx, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.trapz
@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0])
def test_trapz_complex(shape, dtype, dim):
    """Complex inputs are unsupported and must raise a RuntimeError."""
    if dim >= len(shape) or dim < -len(shape):
        pytest.skip("Invalid dim for shape")

    real = torch.randn(shape, device=flag_gems.device)
    imag = torch.randn(shape, device=flag_gems.device)
    inp = torch.complex(real, imag).to(dtype)

    with pytest.raises(RuntimeError):
        flag_gems.trapz(inp, dx=1.0, dim=dim)


@pytest.mark.trapz
def test_trapz_bool():
    """bool inputs must be rejected with a RuntimeError."""
    inp = torch.randint(0, 2, (3, 4), device=flag_gems.device, dtype=torch.bool)
    with pytest.raises(RuntimeError):
        flag_gems.trapz(inp, dx=1.0, dim=-1)


@pytest.mark.trapz
def test_trapz_invalid_dx():
    """bool and complex dx (scalar or 0-d tensor) are rejected, matching ATen.

    A plain float() coercion would silently turn True into 1.0 and drop the
    imaginary part of a complex dx, so validation must happen before conversion.
    """
    inp = torch.randn(4, device=flag_gems.device)
    for dx in [True, False, 1 + 2j, torch.tensor(True), torch.tensor(1 + 2j)]:
        with pytest.raises(RuntimeError, match="only support dx as a real number"):
            flag_gems.trapz(inp, dx=dx)

        with pytest.raises(RuntimeError, match="only support dx as a real number"):
            torch.trapezoid(inp.cpu(), dx=dx)

    # Real 0-d int/float tensors are accepted and match the scalar dx path.
    for dx in [2, 2.0, torch.tensor(2), torch.tensor(2.0)]:
        ref_inp = utils.to_reference(inp, True)
        ref_out = torch.trapezoid(ref_inp, dx=dx)
        res_out = flag_gems.trapz(inp, dx=dx)
        utils.gems_assert_close(res_out, ref_out, inp.dtype)


@pytest.mark.trapz
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + [torch.int32])
def test_trapz_empty_dim_ordering(dtype):
    """An empty reduction dim returns before the bool/complex/dx checks run.

    ATen short-circuits an empty reduction dimension to zeros of the input dtype
    *before* validating dx, so an invalid dx must not raise there. A degenerate
    N == 1 dim still validates dx, which the non-empty cases below cover.
    """
    for shape, dim in [((2, 0), -1), ((0, 3), 0)]:
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)

        for dx in [1.0, True, 1 + 2j]:
            ref_out = torch.trapezoid(ref_inp, dx=dx, dim=dim)
            res_out = flag_gems.trapz(inp, dx=dx, dim=dim)
            utils.gems_assert_close(res_out, ref_out, ref_out.dtype)


@pytest.mark.trapz
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_single_length_dim_ordering(dtype):
    """N == 1 still validates dx, matching ATen's ordering."""
    inp = torch.zeros((3, 1), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.trapezoid(ref_inp, dx=1.0, dim=-1)
    res_out = flag_gems.trapz(inp, dx=1.0, dim=-1)
    utils.gems_assert_close(res_out, ref_out, dtype)

    for dx in [True, 1 + 2j]:
        with pytest.raises(RuntimeError, match="only support dx as a real number"):
            flag_gems.trapz(inp, dx=dx, dim=-1)


@pytest.mark.trapz
def test_trapz_bool_complex_ordering():
    """A non-degenerate dim still rejects bool/complex y, and dx is checked too."""
    bool_inp = torch.zeros(2, 3, dtype=torch.bool, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="bool input"):
        flag_gems.trapz(bool_inp, dx=1.0, dim=-1)

    cplx_inp = torch.randn(2, 3, dtype=torch.complex64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="complex inputs are not supported"):
        flag_gems.trapz(cplx_inp, dx=1.0, dim=-1)

    # Invalid dim is reported before any dtype/dx rejection.
    with pytest.raises(IndexError):
        flag_gems.trapz(bool_inp, dx=True, dim=5)


@pytest.mark.trapz
@pytest.mark.parametrize("shape", [(16, 32), (4, 8, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0])
def test_trapz_backward(shape, dtype, dim):
    """Gradient is dx * [0.5, 1, ..., 1, 0.5] broadcast along the dim."""
    if dim >= len(shape) or dim < -len(shape):
        pytest.skip("Invalid dim for shape")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.trapezoid(ref_inp, dx=2.0, dim=dim)
    res_out = flag_gems.trapz(inp, dx=2.0, dim=dim)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad)

    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, ref_grad)[0]
    res_in_grad = torch.autograd.grad(res_out, inp, out_grad)[0]

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
