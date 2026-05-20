import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
    SHAPES = [(64,), (4, 32), (3, 4, 17)]
    DIM_KEEPDIM = [(0, False), (-1, True)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    INT_DTYPES = utils.ALL_INT_DTYPES + [torch.int8, torch.uint8]
    SHAPES = [
        (1,),
        (37,),
        (256,),
        (4, 17),
        (16, 256),
        (1024, 1024),
        (3, 7, 11),
        (8, 16, 32),
        (2, 4, 5, 9),
    ]
    DIM_KEEPDIM = [(0, False), (0, True), (-1, False), (-1, True)]


def _make_tensor(shape, dtype):
    if dtype in (torch.uint8,):
        t = torch.randint(0, 200, shape, dtype=dtype, device=flag_gems.device)
    elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        t = torch.randint(-128, 128, shape, dtype=dtype, device=flag_gems.device)
    else:
        t = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return t


def _check_indices(inp, dim, values, indices, *, keepdim):
    if inp.numel() == 0 or inp.ndim == 0:
        return
    dim = dim % inp.ndim
    gather_idx = indices if keepdim else indices.unsqueeze(dim)
    expected = values if keepdim else values.unsqueeze(dim)
    gathered = torch.gather(inp, dim, gather_idx)
    if inp.dtype.is_floating_point:
        ok = torch.equal(gathered, expected) or (
            torch.isnan(gathered).all() and torch.isnan(expected).all()
        )
    else:
        ok = torch.equal(gathered, expected)
    assert ok, "indices do not select median values"


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_median(shape, dtype):
    inp = _make_tensor(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_uncontiguous(shape, dtype):
    if len(shape) < 2:
        pytest.skip("needs >=2 dims")
    inp = _make_tensor(shape, dtype)
    inp_uc = inp.transpose(-1, -2)
    ref_inp = utils.to_reference(inp_uc)
    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp_uc)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim,keepdim", DIM_KEEPDIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_median_dim(shape, dim, keepdim, dtype):
    if dim >= len(shape) or -dim > len(shape):
        pytest.skip("dim out of range for shape")
    inp = _make_tensor(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref_v, _ = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_v, res_i = torch.median(inp, dim=dim, keepdim=keepdim)
    utils.gems_assert_equal(res_v, ref_v)
    _check_indices(inp, dim, res_v, res_i, keepdim=keepdim)


@pytest.mark.median
@pytest.mark.parametrize("shape", [(16, 64), (4, 8, 32)])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_out(shape, dim, keepdim, dtype):
    inp = _make_tensor(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref_v, ref_i = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    out_v = torch.empty_like(ref_v, device=flag_gems.device)
    out_i = torch.empty_like(ref_i, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.median(inp, dim=dim, keepdim=keepdim, out=(out_v, out_i))
    utils.gems_assert_equal(out_v, ref_v)
    _check_indices(inp, dim, out_v, out_i, keepdim=keepdim)


@pytest.mark.median
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_with_nan(dtype):
    inp = torch.tensor(
        [[1.0, float("nan"), 3.0, 5.0], [2.0, 4.0, 6.0, 8.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)
    with flag_gems.use_gems():
        res_v, _ = torch.median(inp, dim=-1)
    ref_v, _ = torch.median(ref_inp, dim=-1)
    utils.gems_assert_equal(res_v, ref_v, equal_nan=True)
    with flag_gems.use_gems():
        res_all = torch.median(inp)
    ref_all = torch.median(ref_inp)
    utils.gems_assert_equal(res_all, ref_all, equal_nan=True)


@pytest.mark.median
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_even_count(dtype):
    inp = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype, device=flag_gems.device
    )
    with flag_gems.use_gems():
        out = torch.median(inp).item()
    assert out == 3.0, f"expected lower-of-two median 3.0, got {out}"


@pytest.mark.median
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_zero_dim(dtype):
    inp = torch.tensor(2.75, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    with flag_gems.use_gems():
        res = torch.median(inp)
        res_v, res_i = torch.median(inp, dim=0)
    utils.gems_assert_equal(res, ref_inp)
    utils.gems_assert_equal(res_v, ref_inp)
    assert res_i.item() == 0


@pytest.mark.median
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_empty(dtype):
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        out = torch.median(inp)
    assert torch.isnan(out).item()

    inp2 = torch.empty(3, 0, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        v, i = torch.median(inp2, dim=1)
    assert v.shape == (3,)
    assert i.shape == (3,)
    assert i.dtype == torch.int64


@pytest.mark.median
@pytest.mark.parametrize("width", [3, 8, 31, 256, 1024, 4097])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_widths(width, dtype):
    inp = torch.randn(7, width, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_v, _ = torch.median(ref_inp, dim=-1)
    with flag_gems.use_gems():
        res_v, res_i = torch.median(inp, dim=-1)
    utils.gems_assert_equal(res_v, ref_v)
    _check_indices(inp, -1, res_v, res_i, keepdim=False)


@pytest.mark.median
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_median_int_widths(dtype):
    if dtype == torch.uint8:
        inp = torch.randint(0, 200, (5, 4099), dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-50, 50, (5, 4099), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_v, _ = torch.median(ref_inp, dim=-1)
    with flag_gems.use_gems():
        res_v, _ = torch.median(inp, dim=-1)
    utils.gems_assert_equal(res_v, ref_v)
