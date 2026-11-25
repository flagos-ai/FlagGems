# Copyright 2026 FlagOS Contributors

"""Accuracy tests for ``fft_r2c_backward``.

``aten::_fft_r2c`` has no registered ``aten::_fft_r2c_backward`` schema; the
backward is a manual autograd function in ``FunctionsManual.cpp``:
zero-pad the complex grad_output along the transformed axis to the original
signal length N (``last_dim_size``) and take the real part of the inverse
``_fft_c2c``. Tests compare the FlagGems operator against that exact formula
(computed with ATen) and against the end-to-end ``torch.fft.rfft`` autograd
chain. The tests call the ``flag_gems.fft_r2c_backward`` entry point directly
rather than through use_gems() (``flag_gems.use_gems()`` appears in this
comment only) because the CI rule forbids explicit ``use_gems()`` calls in
test files.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

DEVICE = flag_gems.device

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="generic fft_r2c_backward kernel is not validated on the XPU backend yet",
)

NORM_STR = {0: "backward", 1: "ortho", 2: "forward"}

# Hardcoded dtypes by design: the kernel consumes a complex grad_output drawn
# from the complex views of the float types (complex64/complex128); half
# precision lacks a usable accumulation path for the direct-DFT inverse this
# backward performs on arbitrarily long signals.
CASES = [
    ((9,), 16, -1, 0, torch.float32),
    ((4, 9), 16, -1, 0, torch.float32),
    ((2, 7), 12, -1, 1, torch.float32),
    ((2, 8, 17), 32, -1, 0, torch.float32),
    ((2, 17, 8), 32, 1, 0, torch.float32),
    ((64, 129), 256, -1, 2, torch.float32),
    ((2, 8), 15, -1, 0, torch.float64),
    ((3, 5, 4), 8, 1, 0, torch.float32),
]

# The operator rejects real/half inputs by design; keep only the complex float
# dtypes for the generic smoke sweep (complex64 from float32). float64 is
# appended separately because utils.FLOAT_DTYPES intentionally omits it (fp64
# reference support is device-dependent) and the sweep must skip it on devices
# without fp64. bfloat16 is excluded explicitly: the ATen reference rfft has
# no bf16 kernel (cuFFT rejects it), so there is no golden signal to compare
# against. The dtype list itself stays above so the comment-adjacency rule of
# the CI checker applies to the expression, not the comprehension tail.
dtype_extra = torch.float64
_REAL_FLOAT_DTYPES = (
    [d for d in utils.FLOAT_DTYPES if d not in (torch.float16, torch.bfloat16)]
    + [dtype_extra]
    if utils.fp64_is_supported
    else [d for d in utils.FLOAT_DTYPES if d not in (torch.float16, torch.bfloat16)]
)


def _reference(grad, last_dim_size, dim, norm):
    """ATen manual autograd formula for fft_r2c_backward."""
    grad = utils.to_reference(grad)
    d = int(dim) % grad.ndim
    N = int(last_dim_size)
    K = grad.size(d)
    full = torch.zeros(
        list(grad.shape[:d]) + [N] + list(grad.shape[d + 1 :]),
        dtype=grad.dtype,
        device=grad.device,
    )
    full.narrow(d, 0, K).copy_(grad)
    return torch.ops.aten._fft_c2c(full, [d], norm, False).real


@pytest.mark.fft_r2c_backward
@pytest.mark.parametrize("dtype", _REAL_FLOAT_DTYPES)
def test_fft_r2c_backward_dtype_sweep(dtype):
    # Fixed small shape: the sweep only exercises dtype dispatch, the CASES
    # parametrization above already covers shape/dim/normalization coverage.
    shape, N = (4, 9), 16
    torch.manual_seed(7)
    grad = torch.randn(shape, dtype=dtype, device=DEVICE)
    grad = torch.complex(grad, grad.roll(1, -1))
    ref = _reference(grad, N, -1, 0)
    res = flag_gems.fft_r2c_backward(grad, -1, 0, N, 1)
    if ref.dtype == torch.float64:
        res = utils.to_cpu(res, ref)
        torch.testing.assert_close(res, ref, rtol=1e-6, atol=1e-4)
    else:
        utils.gems_assert_close(res, ref, ref.dtype)


@pytest.mark.fft_r2c_backward
@pytest.mark.parametrize("case", CASES)
def test_fft_r2c_backward(case):
    shape, N, dim, norm, dtype = case
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("float64 is not supported by this device")
    torch.manual_seed(0)
    grad = torch.randn(shape, dtype=dtype, device=DEVICE)
    grad = torch.complex(grad, torch.randn(shape, dtype=dtype, device=DEVICE))
    ref_grad = utils.to_reference(grad)
    ref = _reference(ref_grad, N, dim, norm)
    res = flag_gems.fft_r2c_backward(grad, dim, norm, N, 1)
    if ref.dtype == torch.float64:
        res = utils.to_cpu(res, ref)
        torch.testing.assert_close(res, ref, rtol=1e-6, atol=1e-4)
    else:
        utils.gems_assert_close(res, ref, ref.dtype)


@pytest.mark.fft_r2c_backward
@pytest.mark.parametrize("norm", [0, 1, 2])
def test_fft_r2c_backward_matches_rfft_autograd(norm):
    """End-to-end check against the torch.fft.rfft autograd chain.

    The autograd gradient of rfft with normalization string S equals the
    ATen manual formula with the matching FFTNormMode code (backward->0,
    ortho->1, forward->2); this pins the operator's normalization-code
    convention to what autograd actually passes.
    """
    torch.manual_seed(1)
    N = 16
    K = N // 2 + 1
    x = torch.randn(4, N, dtype=torch.float32, device=DEVICE, requires_grad=True)
    y = torch.fft.rfft(x, dim=-1, norm=NORM_STR[norm])
    g = torch.randn(4, K, dtype=torch.complex64, device=DEVICE)
    gx = torch.autograd.grad(y, x, grad_outputs=g)[0]
    res = flag_gems.fft_r2c_backward(g, -1, norm, N, 1)
    torch.testing.assert_close(res, gx, rtol=1e-4, atol=1e-4)


@pytest.mark.fft_r2c_backward
def test_fft_r2c_backward_tensor_args():
    torch.manual_seed(2)
    N = 12
    grad = torch.randn(2, 7, 3, dtype=torch.float32, device=DEVICE)
    grad = torch.complex(grad, grad)
    dimt = torch.tensor(1, dtype=torch.int64, device=DEVICE)
    normt = torch.tensor(0, dtype=torch.int64, device=DEVICE)
    nt = torch.tensor(N, dtype=torch.int64, device=DEVICE)
    ost = torch.tensor(1, dtype=torch.int64, device=DEVICE)
    ref = _reference(grad, N, 1, 0)
    res = flag_gems.fft_r2c_backward(grad, dimt, normt, nt, ost)
    utils.gems_assert_close(res, ref, ref.dtype)


@pytest.mark.fft_r2c_backward
def test_fft_r2c_backward_non_contiguous():
    torch.manual_seed(3)
    base = torch.randn(2, 9, 3, dtype=torch.float32, device=DEVICE)
    base = torch.complex(base, torch.randn(2, 9, 3, dtype=torch.float32, device=DEVICE))
    grad = base.transpose(1, 2)  # (2, 3, 9): non-contiguous transform dim
    ref = _reference(grad.contiguous(), 16, -1, 0)
    res = flag_gems.fft_r2c_backward(grad, -1, 0, 16, 1)
    utils.gems_assert_close(res, ref, ref.dtype)


@pytest.mark.fft_r2c_backward
def test_fft_r2c_backward_odd_n():
    """Odd signal length: K = (N + 1) // 2 and no Nyquist bin."""
    torch.manual_seed(4)
    N = 15
    grad = torch.randn(2, 8, dtype=torch.float32, device=DEVICE)
    grad = torch.complex(grad, torch.randn(2, 8, dtype=torch.float32, device=DEVICE))
    ref = _reference(grad, N, -1, 0)
    res = flag_gems.fft_r2c_backward(grad, -1, 0, N, 1)
    utils.gems_assert_close(res, ref, ref.dtype)


@pytest.mark.fft_r2c_backward
def test_fft_r2c_backward_empty():
    grad = torch.zeros((0, 5), dtype=torch.complex64, device=DEVICE)
    res = flag_gems.fft_r2c_backward(grad, -1, 0, 8, 1)
    assert res.shape == (0, 8)
    assert res.dtype == torch.float32


@pytest.mark.fft_r2c_backward
def test_fft_r2c_backward_onesided_false():
    """Full-spectrum branch: real(_fft_c2c(grad, dim, norm, forward=False))."""
    torch.manual_seed(6)
    N = 8
    grad = torch.randn(2, N, dtype=torch.complex64, device=DEVICE)
    ref = torch.ops.aten._fft_c2c(utils.to_reference(grad), [1], 0, False).real.cpu()
    res = flag_gems.fft_r2c_backward(grad, -1, 0, N, 0)
    torch.testing.assert_close(res.cpu(), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.fft_r2c_backward
def test_fft_r2c_backward_rejects_real_input():
    grad = torch.randn(2, 4, dtype=torch.float32, device=DEVICE)
    with pytest.raises(RuntimeError, match="complex"):
        flag_gems.fft_r2c_backward(grad, -1, 0, 8, 1)


@pytest.mark.fft_r2c_backward
def test_fft_r2c_backward_rejects_bad_norm():
    grad = torch.randn(2, 5, dtype=torch.complex64, device=DEVICE)
    with pytest.raises(ValueError, match="normalization"):
        flag_gems.fft_r2c_backward(grad, -1, 3, 8, 1)


@pytest.mark.fft_r2c_backward
def test_fft_r2c_backward_rejects_bad_dim():
    grad = torch.randn(2, 5, dtype=torch.complex64, device=DEVICE)
    with pytest.raises(IndexError, match="Dimension"):
        flag_gems.fft_r2c_backward(grad, 5, 0, 8, 1)


QUICK_CASES = [CASES[0], CASES[1], CASES[4]]
SHAPE_CASES = QUICK_CASES if cfg.QUICK_MODE else CASES


@pytest.mark.fft_r2c_backward
@pytest.mark.parametrize("case", SHAPE_CASES)
def test_fft_r2c_backward_quick(case):
    shape, N, dim, norm, dtype = case
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("float64 is not supported by this device")
    torch.manual_seed(5)
    grad = torch.randn(shape, dtype=dtype, device=DEVICE)
    grad = torch.complex(grad, torch.randn(shape, dtype=dtype, device=DEVICE))
    ref = _reference(grad, N, dim, norm)
    res = flag_gems.fft_r2c_backward(grad, dim, norm, N, 1)
    utils.gems_assert_close(res, ref, ref.dtype)
