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


def _nan_input(shape, dtype, device, nan_ratio=0.3):
    x = torch.randn(shape, dtype=dtype, device=device) * 10
    mask = torch.rand(shape, device=device) < nan_ratio
    x[mask] = float("nan")
    return x


@pytest.mark.nanmean
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nanmean(shape, dtype):
    inp = _nan_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nanmean(ref_inp)
    res_out = flag_gems.nanmean(inp)

    utils.gems_assert_close(
        res_out, ref_out, dtype, equal_nan=True, reduce_dim=inp.numel()
    )


@pytest.mark.nanmean
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nanmean_dim(shape, dim, keepdim, dtype):
    inp = _nan_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nanmean(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.nanmean(inp, dim=dim, keepdim=keepdim)

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    if dim == []:
        _dim = inp.numel()
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, reduce_dim=_dim)


@pytest.mark.nanmean
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", [[0, 1]])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nanmean_multi_dim(shape, dim, keepdim, dtype):
    inp = _nan_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nanmean(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.nanmean(inp, dim=dim, keepdim=keepdim)

    _dim = 1
    for d in dim:
        _dim *= shape[d]
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, reduce_dim=_dim)


@pytest.mark.nanmean_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nanmean_dim_out(shape, dim, keepdim, dtype):
    inp = _nan_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_shape = torch.nanmean(ref_inp, dim=dim, keepdim=keepdim).shape
    ref_result = torch.empty(ref_shape, dtype=dtype, device=ref_inp.device)
    torch.nanmean(ref_inp, dim=dim, keepdim=keepdim, out=ref_result)

    res_result = torch.empty(ref_shape, dtype=dtype, device=flag_gems.device)
    returned = flag_gems.nanmean_out(inp, dim=dim, keepdim=keepdim, out=res_result)

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    if dim == []:
        _dim = inp.numel()
    utils.gems_assert_close(
        res_result, ref_result, dtype, equal_nan=True, reduce_dim=_dim
    )
    assert returned is res_result


@pytest.mark.nanmean
def test_nanmean_edge():
    device = flag_gems.device

    # all NaN in a row -> result is NaN
    x = torch.full((5, 5), float("nan"), device=device)
    ref_out = torch.nanmean(x, dim=1)
    res = flag_gems.nanmean(x, dim=1)
    torch.testing.assert_close(res, ref_out, equal_nan=True)

    # no NaN present -> equals mean
    x = torch.randn((4, 8), device=device)
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nanmean(ref_inp)
    res = flag_gems.nanmean(x)
    utils.gems_assert_close(res, ref_out, torch.float32, reduce_dim=x.numel())


@pytest.mark.nanmean
def test_nanmean_empty_and_dtype():
    inp = torch.empty((2, 0, 3), device=flag_gems.device)
    result = flag_gems.nanmean(inp, dim=1, keepdim=True, dtype=torch.float64)
    reference = torch.nanmean(inp, dim=1, keepdim=True, dtype=torch.float64)
    torch.testing.assert_close(result, reference, equal_nan=True)


@pytest.mark.nanmean
def test_nanmean_invalid_dims():
    inp = torch.randn((2, 3), device=flag_gems.device)
    with pytest.raises(RuntimeError, match="appears multiple times"):
        flag_gems.nanmean(inp, dim=(0, -2))
    with pytest.raises(IndexError, match="Dimension out of range"):
        flag_gems.nanmean(inp, dim=2)


@pytest.mark.nanmean_out
def test_nanmean_out_resizes_and_returns_out():
    inp = _nan_input((4, 8), torch.float32, flag_gems.device)
    out = torch.empty((0,), dtype=torch.float16, device=flag_gems.device)
    returned = flag_gems.nanmean_out(inp, dim=1, out=out)
    reference = torch.nanmean(inp, dim=1).to(torch.float16)
    assert returned is out
    assert out.shape == (4,)
    torch.testing.assert_close(out, reference, equal_nan=True)


@pytest.mark.nanmean
def test_nanmean_complex_and_autograd():
    inp = torch.tensor(
        [complex(float("nan"), 1), complex(2, float("nan")), 3 + 4j],
        dtype=torch.complex64,
        device=flag_gems.device,
    )
    torch.testing.assert_close(
        flag_gems.nanmean(inp), torch.nanmean(inp), equal_nan=True
    )

    grad_inp = torch.tensor(
        [1.0, float("nan"), 3.0], device=flag_gems.device, requires_grad=True
    )
    result = flag_gems.nanmean(grad_inp)
    result.backward()
    torch.testing.assert_close(
        grad_inp.grad, torch.tensor([0.5, 0.0, 0.5], device=flag_gems.device)
    )
