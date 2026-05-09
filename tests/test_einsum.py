import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

if QUICK_MODE:
    EINSUM_SHAPES = {
        "matmul": [(16, 32, 64)],
        "bmm": [(2, 16, 32, 64)],
        "dot": [64],
        "outer": [(16, 32)],
        "trace": [32],
        "transpose": [(16, 32)],
        "sum": [(16, 32, 64)],
    }
else:
    EINSUM_SHAPES = {
        "matmul": [(32, 64, 128), (16, 256, 32)],
        "bmm": [(4, 32, 64, 128), (8, 16, 256, 32)],
        "dot": [64, 256, 1024],
        "outer": [(32, 64), (128, 256)],
        "trace": [32, 64, 128],
        "transpose": [(32, 64), (64, 128, 256)],
        "sum": [(32, 64, 128)],
    }


@pytest.mark.einsum
@pytest.mark.parametrize("M, K, N", EINSUM_SHAPES["matmul"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_matmul(M, K, N, dtype):
    inp1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.einsum("ij,jk->ik", ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.einsum("ij,jk->ik", inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.einsum
@pytest.mark.parametrize("B, M, K, N", EINSUM_SHAPES["bmm"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_bmm(B, M, K, N, dtype):
    inp1 = torch.randn((B, M, K), dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn((B, K, N), dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.einsum("bij,bjk->bik", ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.einsum("bij,bjk->bik", inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.einsum
@pytest.mark.parametrize("size", EINSUM_SHAPES["dot"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_dot(size, dtype):
    inp1 = torch.randn(size, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(size, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.einsum("i,i->", ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.einsum("i,i->", inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=size)


@pytest.mark.einsum
@pytest.mark.parametrize("M, N", EINSUM_SHAPES["outer"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_outer(M, N, dtype):
    inp1 = torch.randn(M, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(N, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.einsum("i,j->ij", ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.einsum("i,j->ij", inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.einsum
@pytest.mark.parametrize("size", EINSUM_SHAPES["trace"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_trace(size, dtype):
    inp = torch.randn((size, size), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_out = torch.einsum("ii->", ref_inp)
    with flag_gems.use_gems():
        res_out = torch.einsum("ii->", inp)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=size)


@pytest.mark.einsum
@pytest.mark.parametrize("size", EINSUM_SHAPES["trace"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_diagonal(size, dtype):
    inp = torch.randn((size, size), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_out = torch.einsum("ii->i", ref_inp)
    with flag_gems.use_gems():
        res_out = torch.einsum("ii->i", inp)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.einsum
@pytest.mark.parametrize("shape", EINSUM_SHAPES["transpose"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_transpose(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    if len(shape) == 2:
        ref_out = torch.einsum("ij->ji", ref_inp)
        with flag_gems.use_gems():
            res_out = torch.einsum("ij->ji", inp)
    else:
        ref_out = torch.einsum("ijk->kji", ref_inp)
        with flag_gems.use_gems():
            res_out = torch.einsum("ijk->kji", inp)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.einsum
@pytest.mark.parametrize("shape", EINSUM_SHAPES["sum"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_sum_all(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_out = torch.einsum("ijk->", ref_inp)
    with flag_gems.use_gems():
        res_out = torch.einsum("ijk->", inp)
    reduce_dim = shape[0] * shape[1] * shape[2]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.einsum
@pytest.mark.parametrize("shape", EINSUM_SHAPES["sum"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_sum_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_out = torch.einsum("ijk->j", ref_inp)
    with flag_gems.use_gems():
        res_out = torch.einsum("ijk->j", inp)
    reduce_dim = shape[0] * shape[2]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.einsum
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_ellipsis(dtype):
    shape1 = (2, 3, 32, 64)
    shape2 = (2, 3, 64, 128)
    inp1 = torch.randn(shape1, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape2, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.einsum("...ij,...jk->...ik", ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.einsum("...ij,...jk->...ik", inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=64)
