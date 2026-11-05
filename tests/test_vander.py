import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    INT_DTYPES = [torch.int16, torch.int32, torch.int64]

BOOL_DTYPES = [torch.bool]


def _make_input(M, dtype):
    if dtype.is_floating_point:
        return torch.randn(M, dtype=dtype, device=flag_gems.device)
    if dtype is torch.bool:
        return torch.randint(0, 2, (M,), device=flag_gems.device).to(torch.bool)
    return torch.randint(-4, 5, (M,), dtype=dtype, device=flag_gems.device)


@pytest.mark.vander
@pytest.mark.parametrize("M", [1, 4, 16, 64])
@pytest.mark.parametrize("N", [None, 1, 3, 8])
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_vander(M, N, increasing, dtype):
    res_inp = torch.randn(M, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=True)

    ref_out = torch.vander(ref_inp, N=N, increasing=increasing)
    res_out = flag_gems.vander(res_inp, N=N, increasing=increasing)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.vander
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_vander_non_contiguous(increasing, dtype):
    # x[::2] has stride 2; the kernel must honor x.stride(0) instead of assuming
    # a contiguous 1-D layout.
    base = torch.randn(2 * 64, dtype=dtype, device=flag_gems.device)
    res_inp = base[::2]
    ref_inp = utils.to_reference(res_inp, upcast=True)

    ref_out = torch.vander(ref_inp, N=8, increasing=increasing)
    res_out = flag_gems.vander(res_inp, N=8, increasing=increasing)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.vander
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_DTYPES)
def test_vander_int(increasing, dtype):
    M = 8
    res_inp = _make_input(M, dtype)
    ref_inp = utils.to_reference(res_inp, upcast=False)

    ref_out = torch.vander(ref_inp, N=5, increasing=increasing)
    res_out = flag_gems.vander(res_inp, N=5, increasing=increasing)

    # bool and narrow integer inputs promote to int64, matching ATen.
    assert res_out.dtype == torch.int64
    utils.gems_assert_close(res_out, ref_out, torch.int64)


@pytest.mark.vander
@pytest.mark.parametrize("increasing", [False, True])
def test_vander_int64_large(increasing):
    # Values beyond 2**24 need int64 arithmetic: an fp32 pow would drop the low
    # bits, so the integer kernel must compute the powers exactly.
    res_inp = torch.tensor(
        [16777216, 16777217, 33554433, -16777216, -2, 3],
        dtype=torch.int64,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(res_inp, upcast=False)

    ref_out = torch.vander(ref_inp, N=5, increasing=increasing)
    res_out = flag_gems.vander(res_inp, N=5, increasing=increasing)

    assert res_out.dtype == torch.int64
    utils.gems_assert_close(res_out, ref_out, torch.int64)


@pytest.mark.vander
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_vander_negative_values(dtype):
    # Negative bases with higher powers must produce the correct alternating sign.
    res_inp = torch.tensor([-2, -3, 2, -1], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=dtype.is_floating_point)

    ref_out = torch.vander(ref_inp, N=6, increasing=True)
    res_out = flag_gems.vander(res_inp, N=6, increasing=True)

    check_dtype = dtype if dtype.is_floating_point else torch.int64
    utils.gems_assert_close(res_out, ref_out, check_dtype)


@pytest.mark.vander
def test_vander_fp64():
    # FP64 precision must be preserved end-to-end (no fp32 downcast in the kernel).
    res_inp = torch.randn(32, dtype=torch.float64, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=False)

    ref_out = torch.vander(ref_inp, N=16)
    res_out = flag_gems.vander(res_inp, N=16)

    utils.gems_assert_close(res_out, ref_out, torch.float64)


@pytest.mark.vander
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_vander_complex(dtype, increasing):
    # Complex inputs are computed via a dedicated real/imag power kernel and must
    # match ATen's complex-capable implementation.
    x = torch.randn(6, dtype=dtype, device=flag_gems.device)
    ref_inp = x.cpu()

    ref_out = torch.vander(ref_inp, N=5, increasing=increasing).to(x.device)
    res_out = flag_gems.vander(x, N=5, increasing=increasing)

    assert res_out.dtype == dtype
    torch.testing.assert_close(res_out, ref_out)


@pytest.mark.vander
def test_vander_zero_n():
    res_inp = torch.randn(5, dtype=torch.float32, device=flag_gems.device)
    res_out = flag_gems.vander(res_inp, N=0)
    assert res_out.shape == (5, 0)


@pytest.mark.vander
def test_vander_negative_n():
    res_inp = torch.randn(5, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="N must be non-negative."):
        flag_gems.vander(res_inp, N=-1)


@pytest.mark.vander
def test_vander_invalid_rank():
    x2d = torch.randn(2, 3, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="x must be a one-dimensional tensor."):
        flag_gems.vander(x2d)
    x0d = torch.randn((), device=flag_gems.device)
    with pytest.raises(RuntimeError, match="x must be a one-dimensional tensor."):
        flag_gems.vander(x0d)


@pytest.mark.vander
def test_vander_output_dtype():
    # Explicit promotion contract: bool/integer -> int64, floating keeps its type.
    cases = [
        (torch.bool, torch.int64),
        (torch.int16, torch.int64),
        (torch.int32, torch.int64),
        (torch.int64, torch.int64),
        (torch.float16, torch.float16),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ]
    for input_dtype, expected in cases:
        res_inp = _make_input(4, input_dtype)
        res_out = flag_gems.vander(res_inp, N=3)
        assert (
            res_out.dtype == expected
        ), f"vander({input_dtype}) -> {res_out.dtype}, expected {expected}"
