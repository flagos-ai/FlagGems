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


def _vander_copy(value, device):
    if flag_gems.vendor_name == "kunlunxin" and value.dtype in (
        torch.float64,
        torch.complex128,
    ):
        return (
            value.contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .to(device)
            .view(value.dtype)
            .reshape(value.shape)
        )
    return value.to(device)


def _vander_random(shape, dtype):
    if flag_gems.vendor_name == "enflame" and dtype in (torch.float64, torch.complex128):
        pytest.skip("Enflame target transfer does not retain genuine wide-component precision")
    if flag_gems.vendor_name == "enflame" and dtype == torch.complex64:
        # The target complex randn constructor aborts; preserve its distribution
        # with a CPU-generator recipe, then transfer a genuine Complex64 tensor.
        return torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    if flag_gems.vendor_name == "kunlunxin" and dtype in (
        torch.float64,
        torch.complex128,
    ):
        return _vander_copy(
            torch.randn(shape, dtype=dtype, device="cpu"), flag_gems.device
        )
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _vander_reference_input(value, upcast=False):
    if (
        utils.TO_CPU
        and flag_gems.vendor_name == "kunlunxin"
        and value.dtype == torch.float64
    ):
        return _vander_copy(value, "cpu")
    return utils.to_reference(value, upcast)


def _vander_reference(value, N=None, increasing=False):
    if flag_gems.vendor_name == "enflame" and value.device.type != "cpu":
        if value.dtype in (torch.int16, torch.int32, torch.int64, torch.bool):
            pytest.skip(
                "Enflame native vander promotes integer/Bool to unsupported Long cumprod; CPU reference remains covered"
            )
        if value.dtype == torch.float32 and not value.is_contiguous():
            pytest.skip("Enflame native vander reference misreads strided Float32 input")
    if flag_gems.vendor_name == "kunlunxin" and value.device.type != "cpu":
        if value.dtype in (torch.float64, torch.int16, torch.int32, torch.bool):
            pytest.skip(
                "Kunlunxin native vander computes incorrect double or narrow-integer values; CPU reference remains covered"
            )
        if (
            value.numel() == 1 and N is not None and N > 2
        ) or not value.is_contiguous():
            pytest.skip(
                "Kunlunxin native vander computes incorrect singleton-rectangular or strided values; CPU reference remains covered"
            )
    return torch.vander(value, N=N, increasing=increasing)


def _vander_assert_close(result, reference, dtype, device):
    if flag_gems.vendor_name == "kunlunxin" and dtype in (
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        assert result.dtype == dtype
        assert result.device == device
        assert result.shape == reference.shape
        result, reference = _vander_copy(result, "cpu"), _vander_copy(reference, "cpu")
    utils.gems_assert_close(result, reference, dtype)


def _make_input(M, dtype):
    if dtype.is_floating_point:
        return _vander_random(M, dtype)
    if dtype is torch.bool:
        raw = torch.randint(0, 2, (M,), device=flag_gems.device)
        return (
            raw.cpu().to(torch.bool).to(raw.device)
            if flag_gems.vendor_name == "enflame"
            else raw.to(torch.bool)
        )
    return torch.randint(-4, 5, (M,), dtype=dtype, device=flag_gems.device)


@pytest.mark.vander
@pytest.mark.parametrize("M", [1, 4, 16, 64])
@pytest.mark.parametrize("N", [None, 1, 3, 8])
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_vander(M, N, increasing, dtype):
    res_inp = _vander_random(M, dtype)
    ref_inp = _vander_reference_input(res_inp, upcast=True)

    ref_out = _vander_reference(ref_inp, N=N, increasing=increasing)
    res_out = flag_gems.vander(res_inp, N=N, increasing=increasing)

    _vander_assert_close(res_out, ref_out, dtype, res_inp.device)


@pytest.mark.vander
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_vander_non_contiguous(increasing, dtype):
    # x[::2] has stride 2; the kernel must honor x.stride(0) instead of assuming
    # a contiguous 1-D layout.
    base = torch.randn(2 * 64, dtype=dtype, device=flag_gems.device)
    res_inp = base[::2]
    ref_inp = _vander_reference_input(res_inp, upcast=True)

    ref_out = _vander_reference(ref_inp, N=8, increasing=increasing)
    res_out = flag_gems.vander(res_inp, N=8, increasing=increasing)

    _vander_assert_close(res_out, ref_out, dtype, res_inp.device)


@pytest.mark.vander
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_DTYPES)
def test_vander_int(increasing, dtype):
    M = 8
    res_inp = _make_input(M, dtype)
    ref_inp = _vander_reference_input(res_inp, upcast=False)

    ref_out = _vander_reference(ref_inp, N=5, increasing=increasing)
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
    ref_inp = _vander_reference_input(res_inp, upcast=False)

    ref_out = _vander_reference(ref_inp, N=5, increasing=increasing)
    res_out = flag_gems.vander(res_inp, N=5, increasing=increasing)

    assert res_out.dtype == torch.int64
    utils.gems_assert_close(res_out, ref_out, torch.int64)


@pytest.mark.vander
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_vander_negative_values(dtype):
    # Negative bases with higher powers must produce the correct alternating sign.
    res_inp = torch.tensor([-2, -3, 2, -1], dtype=dtype, device=flag_gems.device)
    ref_inp = _vander_reference_input(res_inp, upcast=dtype.is_floating_point)

    ref_out = _vander_reference(ref_inp, N=6, increasing=True)
    res_out = flag_gems.vander(res_inp, N=6, increasing=True)

    check_dtype = dtype if dtype.is_floating_point else torch.int64
    utils.gems_assert_close(res_out, ref_out, check_dtype)


@pytest.mark.vander
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads" and not utils.TO_CPU,
    reason="MUSA torch.vander reference does not support float64",
)
def test_vander_fp64():
    # FP64 precision must be preserved end-to-end (no fp32 downcast in the kernel).
    res_inp = _vander_random(32, torch.float64)
    ref_inp = _vander_reference_input(res_inp, upcast=False)

    ref_out = _vander_reference(ref_inp, N=16)
    res_out = flag_gems.vander(res_inp, N=16)

    _vander_assert_close(res_out, ref_out, torch.float64, res_inp.device)


@pytest.mark.vander
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend torch.randn input construction does not support complex dtypes",
)
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_vander_complex(dtype, increasing):
    # Complex inputs are computed via a dedicated real/imag power kernel and must
    # match ATen's complex-capable implementation.
    x = _vander_random(6, dtype)
    ref_inp = _vander_copy(x, "cpu")

    ref_out = _vander_reference(ref_inp, N=5, increasing=increasing)
    res_out = flag_gems.vander(x, N=5, increasing=increasing)

    assert res_out.dtype == dtype
    assert res_out.device == x.device
    assert res_out.shape == ref_out.shape
    torch.testing.assert_close(_vander_copy(res_out, "cpu"), ref_out)


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
    if flag_gems.vendor_name == "enflame":
        pytest.skip("Original compound promotion fixture requires genuine target Float64 storage")
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
