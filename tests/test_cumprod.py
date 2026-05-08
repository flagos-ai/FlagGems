import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    CUMPROD_SHAPE_DIMS = [((2, 32), 1), ((2, 5, 3), 1)]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    CUMPROD_SHAPE_DIMS = [
        ((1, 2), -1),
        ((4096, 256), 1),
        ((200, 2560, 3), 1),
        ((2637,), 0),
        ((16, 1025, 255), 1),
    ]

INT_DTYPES = list(dict.fromkeys([torch.int8, torch.uint8] + utils.ALL_INT_DTYPES))
DTYPES = FLOAT_DTYPES + INT_DTYPES


def _make_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            0.99, 1.01
        )
    if dtype is torch.uint8:
        return torch.randint(0, 4, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    return torch.randint(-3, 4, shape, dtype=dtype, device="cpu").to(flag_gems.device)


def _reference_input(inp):
    return utils.to_reference(inp, inp.is_floating_point())


@pytest.mark.cumprod
@pytest.mark.parametrize("shape_dim", CUMPROD_SHAPE_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cumprod(shape_dim, dtype):
    shape, dim = shape_dim
    inp = _make_input(shape, dtype)
    ref_inp = _reference_input(inp)

    ref_out = torch.cumprod(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cumprod(inp, dim=dim)

    check_dtype = ref_out.dtype if dtype in INT_DTYPES else dtype
    utils.gems_assert_close(res_out, ref_out, check_dtype, reduce_dim=shape[dim])


@pytest.mark.cumprod
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [
        (torch.int8, torch.int32),
        (torch.uint8, torch.int64),
        (torch.float16, torch.float32),
    ],
)
def test_cumprod_dtype(input_dtype, output_dtype):
    inp = _make_input((8, 17), input_dtype)
    ref_inp = _reference_input(inp)

    ref_out = torch.cumprod(ref_inp, dim=1, dtype=output_dtype)
    with flag_gems.use_gems():
        res_out = torch.cumprod(inp, dim=1, dtype=output_dtype)

    utils.gems_assert_close(res_out, ref_out, output_dtype, reduce_dim=17)


@pytest.mark.cumprod_
@pytest.mark.parametrize("shape_dim", CUMPROD_SHAPE_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cumprod_inplace(shape_dim, dtype):
    shape, dim = shape_dim
    inp = _make_input(shape, dtype)
    ref_inp = _reference_input(inp)
    ref_out = torch.cumprod(ref_inp, dim=dim).to(dtype)

    with flag_gems.use_gems():
        res_out = inp.cumprod_(dim)

    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.cumprod_
def test_cumprod_inplace_non_contiguous():
    base = _make_input((4, 9), torch.float32)
    inp = base.t()
    ref_inp = _reference_input(inp)
    ref_out = torch.cumprod(ref_inp, dim=1).to(inp.dtype)

    with flag_gems.use_gems():
        res_out = inp.cumprod_(1)

    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_close(res_out, ref_out, inp.dtype, reduce_dim=inp.shape[1])


@pytest.mark.cumprod_
def test_cumprod_inplace_dtype_mismatch():
    inp = _make_input((4, 9), torch.int16)

    with flag_gems.use_gems():
        with pytest.raises(RuntimeError, match="Bad in-place call"):
            inp.cumprod_(1, dtype=torch.int64)
