import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


MK_SHAPES = (
    [(1, 32)]
    if QUICK_MODE
    else [
        (1, 32),
        (7, 33),
        (31, 65),
        (160, 1024),
        (257, 96),
        (1023, 255),
        (5333, 71),
    ]
)


# Issue #2833: fails at (1, 1, 2)
@pytest.mark.mm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_mm(M, N, K, dtype, b_column_major):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_broadcast_stride_zero(dtype):
    """Regression test: broadcast tensors (stride=0) must not crash TMA path."""
    torch.manual_seed(0)
    M, K, N = 128, 256, 256

    # Simulate the stride=(0,0) tensor that autograd produces from sum().backward():
    # scalar expand -> all strides are 0
    a = torch.randn((), dtype=dtype, device=flag_gems.device).expand(M, K)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    assert a.stride() == (0, 0)

    ref_a = utils.to_reference(a.contiguous(), True)
    ref_b = utils.to_reference(b, True)

    ref_out = torch.mm(ref_a, ref_b)
    with flag_gems.use_gems():
        res_out = torch.mm(a, b)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("M, K", MK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_self_transpose(M, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip(
            "Issue #2834: Skipping fp32 mm self-transpose test on tsingmicro platform"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    ref_mat = utils.to_reference(mat, True)

    ref_out = torch.mm(ref_mat, ref_mat.t())
    with flag_gems.use_gems():
        res_out = torch.mm(mat, mat.t())

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm_out
@pytest.mark.parametrize("M, K", MK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_out_self_transpose(M, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip(
            "Issue #2834: Skipping fp32 mm.out self-transpose test on tsingmicro platform"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    out = torch.empty((M, M), dtype=dtype, device=flag_gems.device)
    ref_mat = utils.to_reference(mat, True)
    ref_out = utils.to_reference(out, True)

    torch.mm(ref_mat, ref_mat.t(), out=ref_out)
    with flag_gems.use_gems():
        torch.mm(mat, mat.t(), out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)
