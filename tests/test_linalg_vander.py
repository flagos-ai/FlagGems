import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

COMPLEX_DTYPES = (
    [torch.complex64, torch.complex128] if utils.complex_is_supported else []
)

# linalg_vander tests
VANDER_SHAPES = [
    (4,),  # 1D input
    (8,),  # 1D input
    (16,),  # 1D input
    (2, 4),  # 2D input (batch=2)
    (4, 8),  # 2D input (batch=4)
    (2, 3, 4),  # 3D input (batch=6)
]

# Use float32 only on metax since torch.linalg.vander doesn't support float16/bfloat16 there
if flag_gems.vendor_name == "metax":
    # torch.linalg.vander is not implemented for float16/bfloat16 on metax GPUs
    VANDER_DTYPES = [torch.float32]
else:
    VANDER_DTYPES = utils.ALL_FLOAT_DTYPES


def _cpu_values(value):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and value.dtype == torch.complex128
        and value.device.type != "cpu"
    ):
        return (
            value.contiguous()
            .view(torch.uint8)
            .cpu()
            .view(value.dtype)
            .reshape(value.shape)
        )
    return value.cpu()


def _make_input(shape, dtype):
    if dtype == torch.complex128 and flag_gems.vendor_name in ("kunlunxin", "iluvatar"):
        value = torch.randn(shape, dtype=dtype, device="cpu")
        if flag_gems.vendor_name == "kunlunxin":
            return (
                value.view(torch.uint8).to(flag_gems.device).view(dtype).reshape(shape)
            )
        return value.to(flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _reference_input(value, upcast=False):
    if (
        utils.TO_CPU
        and flag_gems.vendor_name == "kunlunxin"
        and value.dtype == torch.complex128
    ):
        return _cpu_values(value)
    return utils.to_reference(value, upcast)


def _reference_vander(x, **kwargs):
    if (
        flag_gems.vendor_name == "iluvatar"
        and x.device.type != "cpu"
        and x.dtype == torch.complex128
    ):
        pytest.skip(
            "Iluvatar native complex128 linalg.vander computes incorrect values; CPU reference remains covered"
        )
    try:
        return torch.linalg.vander(x, **kwargs)
    except (RuntimeError, ValueError) as error:
        if (
            flag_gems.vendor_name == "kunlunxin"
            and x.device.type == "cuda"
            and x.dtype in (torch.complex64, torch.complex128)
            and "[NOT IMPLEMENTED]: error code= 4(" in str(error)
        ):
            pytest.skip(
                "Kunlunxin native linalg.vander does not support complex inputs"
            )
        raise


def _assert_vander_close(res_out, ref_out, dtype, device):
    # flag_gems.testing.RESOLUTION has no entry for complex128 (only complex32/64),
    # so bypass gems_assert_close for complex128 and compare via torch.testing
    # directly, mirroring the approach used by test_linalg_ldl_solve.
    if dtype == torch.complex128:
        assert res_out.dtype == dtype
        assert res_out.device == device
        assert res_out.shape == ref_out.shape
        torch.testing.assert_close(
            _cpu_values(res_out), _cpu_values(ref_out), atol=1e-7, rtol=1e-7
        )
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_vander
@pytest.mark.parametrize("shape", VANDER_SHAPES)
@pytest.mark.parametrize("dtype", VANDER_DTYPES)
def test_linalg_vander(shape, dtype):
    # linalg_vander: input (*, n) -> output (*, n, N)
    # if N is not specified, N = n
    x = _make_input(shape, dtype)
    assert x.dtype == dtype

    # torch.linalg.vander does not support float16/bfloat16 (CPU nor GPU),
    # so upcast the reference input via to_reference(..., True).
    ref_x = (
        _reference_input(x, True)
        if dtype in [torch.float16, torch.bfloat16]
        else _reference_input(x, False)
    )
    ref_out = _reference_vander(ref_x)

    res_out = flag_gems.linalg_vander(x)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_vander
@pytest.mark.parametrize("shape", VANDER_SHAPES)
@pytest.mark.parametrize("N", [2, 3, 4, 8])
@pytest.mark.parametrize("dtype", VANDER_DTYPES)
def test_linalg_vander_with_N(shape, N, dtype):
    # Test with explicit N parameter
    # N is the number of columns in the Vandermonde matrix (shape (*, n, N)).
    # torch.linalg.vander supports any N >= 1, independent of input dimension n,
    # and the Triton kernel also handles N > n correctly via flat indexing.
    # N values that exceed shape[-1] produce rectangular matrices with more
    # columns than input elements, which is valid and tested here.

    x = _make_input(shape, dtype)
    assert x.dtype == dtype

    # torch.linalg.vander does not support float16/bfloat16 (CPU nor GPU),
    # so upcast the reference input via to_reference(..., True).
    ref_x = (
        _reference_input(x, True)
        if dtype in [torch.float16, torch.bfloat16]
        else _reference_input(x, False)
    )
    ref_out = _reference_vander(ref_x, N=N)

    res_out = flag_gems.linalg_vander(x, N=N)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_vander
@pytest.mark.parametrize("shape", VANDER_SHAPES)
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend torch.randn does not support complex input dtypes",
)
def test_linalg_vander_complex(shape, dtype):
    # Complex dtype coverage: torch.linalg.vander supports complex inputs.
    # No upcast is needed because the reference supports complex natively.
    x = _make_input(shape, dtype)
    assert x.dtype == dtype
    ref_x = _reference_input(x, False)

    ref_out = _reference_vander(ref_x)

    res_out = flag_gems.linalg_vander(x)

    _assert_vander_close(res_out, ref_out, dtype, x.device)


@pytest.mark.linalg_vander
@pytest.mark.parametrize("shape", VANDER_SHAPES)
@pytest.mark.parametrize("N", [2, 3, 4, 8])
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend torch.randn does not support complex input dtypes",
)
def test_linalg_vander_with_N_complex(shape, N, dtype):
    # Test complex dtype with explicit N parameter.
    x = _make_input(shape, dtype)
    assert x.dtype == dtype
    ref_x = _reference_input(x, False)

    ref_out = _reference_vander(ref_x, N=N)

    res_out = flag_gems.linalg_vander(x, N=N)

    _assert_vander_close(res_out, ref_out, dtype, x.device)
