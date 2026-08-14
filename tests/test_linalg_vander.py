import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLOAT_DTYPES = utils.FLOAT_DTYPES
COMPLEX_DTYPES = [torch.complex64, torch.complex128]

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
    VANDER_DTYPES = FLOAT_DTYPES


def _assert_vander_close(res_out, ref_out, dtype):
    # flag_gems.testing.RESOLUTION has no entry for complex128 (only complex32/64),
    # so bypass gems_assert_close for complex128 and compare via torch.testing
    # directly, mirroring the approach used by test_linalg_ldl_solve.
    if dtype == torch.complex128:
        res_out = utils.to_cpu(res_out, ref_out)
        torch.testing.assert_close(res_out, ref_out, atol=1e-7, rtol=1e-7)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_vander
@pytest.mark.parametrize("shape", VANDER_SHAPES)
@pytest.mark.parametrize("dtype", VANDER_DTYPES)
def test_linalg_vander(shape, dtype):
    # linalg_vander: input (*, n) -> output (*, n, N)
    # if N is not specified, N = n
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # torch.linalg.vander does not support float16/bfloat16 (CPU nor GPU),
    # so upcast the reference input via to_reference(..., True).
    ref_x = (
        utils.to_reference(x, True)
        if dtype in [torch.float16, torch.bfloat16]
        else utils.to_reference(x, False)
    )
    ref_out = torch.linalg.vander(ref_x)

    with flag_gems.use_gems():
        res_out = torch.linalg.vander(x)

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

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # torch.linalg.vander does not support float16/bfloat16 (CPU nor GPU),
    # so upcast the reference input via to_reference(..., True).
    ref_x = (
        utils.to_reference(x, True)
        if dtype in [torch.float16, torch.bfloat16]
        else utils.to_reference(x, False)
    )
    ref_out = torch.linalg.vander(ref_x, N=N)

    with flag_gems.use_gems():
        res_out = torch.linalg.vander(x, N=N)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_vander
@pytest.mark.parametrize("shape", VANDER_SHAPES)
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_linalg_vander_complex(shape, dtype):
    # Complex dtype coverage: torch.linalg.vander supports complex inputs.
    # No upcast is needed because the reference supports complex natively.
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x, False)

    ref_out = torch.linalg.vander(ref_x)

    with flag_gems.use_gems():
        res_out = torch.linalg.vander(x)

    _assert_vander_close(res_out, ref_out, dtype)


@pytest.mark.linalg_vander
@pytest.mark.parametrize("shape", VANDER_SHAPES)
@pytest.mark.parametrize("N", [2, 3, 4, 8])
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_linalg_vander_with_N_complex(shape, N, dtype):
    # Test complex dtype with explicit N parameter.
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x, False)

    ref_out = torch.linalg.vander(ref_x, N=N)

    with flag_gems.use_gems():
        res_out = torch.linalg.vander(x, N=N)

    _assert_vander_close(res_out, ref_out, dtype)
