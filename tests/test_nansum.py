import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]

INT_DTYPES = [torch.int8, torch.uint8]
DTYPES = FLOAT_DTYPES + INT_DTYPES


def _nan_input(shape, dtype, device, nan_ratio=0.3):
    x = torch.randn(shape, dtype=dtype, device=device) * 10
    mask = torch.rand(shape, device=device) < nan_ratio
    x[mask] = float("nan")
    return x


def _make_input(shape, dtype, device=flag_gems.device):
    # Integers hold no NaN; NPU has no int8/uint8 randint, so draw on CPU.
    if dtype is torch.int8:
        return torch.randint(-8, 9, shape, dtype=dtype, device="cpu").to(device)
    if dtype is torch.uint8:
        return torch.randint(0, 9, shape, dtype=dtype, device="cpu").to(device)
    return _nan_input(shape, dtype, device)


def _reference_input(inp):
    # Never upcast integers: their reference has to stay exact.
    return utils.to_reference(inp, inp.is_floating_point())


@pytest.mark.nansum
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_nansum(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = _reference_input(inp)

    ref_out = torch.nansum(ref_inp)

    res_out = flag_gems.nansum(inp)

    check_dtype = ref_out.dtype if dtype in INT_DTYPES else dtype
    utils.gems_assert_close(res_out, ref_out, check_dtype, reduce_dim=inp.numel())


INCLUDE_0_SHAPES = [(1, 0, 128), (4096, 1, 0), (200, 0, 3)]


@pytest.mark.nansum
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + INCLUDE_0_SHAPES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", DTYPES)
def test_nansum_dim(shape, dim, keepdim, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = _reference_input(inp)

    ref_out = torch.nansum(ref_inp, dim=dim, keepdim=keepdim)

    res_out = flag_gems.nansum(inp, dim=dim, keepdim=keepdim)

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    if dim == []:
        _dim = inp.numel()
    check_dtype = ref_out.dtype if dtype in INT_DTYPES else dtype
    utils.gems_assert_close(res_out, ref_out, check_dtype, reduce_dim=_dim)


@pytest.mark.nansum_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", DTYPES)
def test_nansum_dim_out(shape, dim, keepdim, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = _reference_input(inp)

    ref_out = torch.nansum(ref_inp, dim=dim, keepdim=keepdim)
    out_dtype = ref_out.dtype if dtype in INT_DTYPES else dtype
    ref_result = torch.empty(ref_out.shape, dtype=out_dtype, device=ref_inp.device)
    torch.nansum(ref_inp, dim=dim, keepdim=keepdim, out=ref_result)

    res_result = torch.empty(ref_out.shape, dtype=out_dtype, device=flag_gems.device)

    flag_gems.nansum_out(inp, dim, keepdim, out=res_result)

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    if dim == []:
        _dim = inp.numel()
    utils.gems_assert_close(res_result, ref_result, out_dtype, reduce_dim=_dim)


@pytest.mark.nansum_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_nansum_out(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = _reference_input(inp)

    ref_out = torch.nansum(ref_inp)
    out_dtype = ref_out.dtype if dtype in INT_DTYPES else dtype
    ref_result = torch.empty(ref_out.shape, dtype=out_dtype, device=ref_inp.device)
    torch.nansum(ref_inp, out=ref_result)

    res_result = torch.empty(ref_out.shape, dtype=out_dtype, device=flag_gems.device)

    flag_gems.nansum_out(inp, out=res_result)

    utils.gems_assert_close(res_result, ref_result, out_dtype, reduce_dim=inp.numel())


@pytest.mark.nansum
def test_nansum_edge():
    device = flag_gems.device

    # all NaN
    x = torch.full((5, 5), float("nan"), device=device)
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nansum(ref_inp, dim=1)

    res = flag_gems.nansum(x, dim=1)
    utils.gems_assert_close(res, ref_out, torch.float32, reduce_dim=5)

    # integer input
    x = torch.tensor([1, 2, 3, 4], device=device)
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nansum(ref_inp)

    res = flag_gems.nansum(x)
    utils.gems_assert_close(res, ref_out, torch.int64, reduce_dim=4)

    # empty tensor
    x = torch.empty(0, device=device)
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nansum(ref_inp)

    res = flag_gems.nansum(x)
    utils.gems_assert_close(res, ref_out, torch.float32, reduce_dim=0)
