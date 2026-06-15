import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    KEEPDIM_DIMS = [(True, 1)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    KEEPDIM_DIMS = list(zip([True, False] * 2, DIMS_LIST))


@pytest.mark.mean
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.mean(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Shapes where K (product of dims after the reduction axis) exceeds the CUDA
# grid-Y limit of 65535, which used to trigger "Triton Error [CUDA]: invalid
# argument" before the mean_heur_tile_k grid-overflow fix.
MEAN_LARGE_K_SHAPES = [
    (1, 8, 256, 256),  # dim=1 → M=1, N=8, K=65536 (just over limit)
    (1, 4, 512, 512),  # dim=1 → M=1, N=4, K=262144 (well over limit)
]


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape", MEAN_LARGE_K_SHAPES)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_dim_large_k(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)


MEAN_LARGE_INNERDIM_SHAPES = [
    (1024, 1024, 1024),  # dim=1 → M=1, N=8, K=65536 (just over limit)
    (1024, 2048, 1024),  # dim=1 → M=1, N=4, K=262144 (well over limit)
]


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape", MEAN_LARGE_INNERDIM_SHAPES)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_dim_large_innerdim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)
