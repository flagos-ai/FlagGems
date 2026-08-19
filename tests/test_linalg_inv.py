import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _well_conditioned(shape, dtype):
    # Build a well-conditioned invertible matrix: A = B + n * I so the
    # diagonal dominates and the inverse is numerically stable.
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device)
    return B + eye * n


@pytest.mark.linalg_inv
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)])
# linalg_inv only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_inv(shape, dtype):
    A = _well_conditioned(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.inv(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.inv(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_inv
@pytest.mark.parametrize("shape", [(2, 4, 4), (3, 8, 8), (2, 3, 6, 6)])
# linalg_inv only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_inv_batch(shape, dtype):
    A = _well_conditioned(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.inv(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.inv(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_inv
@pytest.mark.linalg_inv_out
@pytest.mark.parametrize("shape", [(4, 4), (8, 8), (2, 6, 6)])
# linalg_inv only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_inv_out(shape, dtype):
    A = _well_conditioned(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.inv(ref_A)

    out = torch.empty_like(A)
    with flag_gems.use_gems():
        res_out = torch.linalg.inv(A, out=out)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(out, ref_out, dtype)
