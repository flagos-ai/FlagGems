import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# kthvalue tests
KTHVALUE_K_VALUES = [1, 2, 4]


@pytest.mark.kthvalue
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("k", KTHVALUE_K_VALUES)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
# kthvalue implementation relies on topk gemm path which requires float32 precision
@pytest.mark.parametrize("dtype", [torch.float32])
def test_kthvalue(shape, k, dim, keepdim, dtype):
    """Test kthvalue accuracy with float32"""
    rank = len(shape)
    if dim is not None:
        if dim >= rank or dim < -rank:
            return

    if any(d == 0 for d in shape):
        return

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    dim_size = shape[dim]
    if k < 1 or k > dim_size:
        with pytest.raises(RuntimeError, match="selected number k out of range"):
            torch.kthvalue(ref_inp, k, dim=dim, keepdim=keepdim)
        with pytest.raises(RuntimeError, match="selected number k out of range"):
            with flag_gems.use_gems():
                torch.kthvalue(inp, k, dim=dim, keepdim=keepdim)
        return

    ref_values, ref_indices = torch.kthvalue(ref_inp, k, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.kthvalue(inp, k, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(res_values, ref_values, dtype)
    utils.gems_assert_equal(res_indices, ref_indices)


@pytest.mark.kthvalue
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("k", [1, 2])
# kthvalue implementation relies on topk gemm path which requires float32 precision
@pytest.mark.parametrize("dtype", [torch.float32])
def test_kthvalue_default_dim(shape, k, dtype):
    """Test kthvalue with default dim (last dimension)"""
    if any(d == 0 for d in shape):
        return

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    dim_size = shape[-1]
    if k < 1 or k > dim_size:
        with pytest.raises(RuntimeError, match="selected number k out of range"):
            torch.kthvalue(ref_inp, k)
        with pytest.raises(RuntimeError, match="selected number k out of range"):
            with flag_gems.use_gems():
                torch.kthvalue(inp, k)
        return

    ref_values, ref_indices = torch.kthvalue(ref_inp, k)
    with flag_gems.use_gems():
        res_values, res_indices = torch.kthvalue(inp, k)

    utils.gems_assert_close(res_values, ref_values, dtype)
    utils.gems_assert_equal(res_indices, ref_indices)
