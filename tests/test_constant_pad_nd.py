import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_equal,
    to_reference,
)


@pytest.mark.constant_pad_nd
@pytest.mark.parametrize(
    "shape",
    [s for s in POINTWISE_SHAPES if len(s) >= 1],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_constant_pad_nd(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    rank = len(shape)
    num_pad = rank * 2
    pad = [torch.randint(0, 10, (1,)).item() for _ in range(num_pad)]
    value = 1.5

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.constant_pad_nd
@pytest.mark.parametrize(
    "shape",
    [s for s in POINTWISE_SHAPES if len(s) >= 2],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_constant_pad_nd_non_contiguous(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = inp[::2, ::2]
    rank = inp.ndim
    num_pad = rank * 2
    pad = [torch.randint(0, 5, (1,)).item() for _ in range(num_pad)]
    value = -2.0

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.constant_pad_nd
@pytest.mark.parametrize(
    "shape",
    [s for s in POINTWISE_SHAPES if len(s) >= 1],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_constant_pad_nd_zero_value(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    rank = len(shape)
    num_pad = rank * 2
    pad = [torch.randint(0, 10, (1,)).item() for _ in range(num_pad)]
    value = 0.0

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.constant_pad_nd
@pytest.mark.parametrize(
    "shape",
    [s for s in POINTWISE_SHAPES if len(s) >= 2],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_constant_pad_nd_partial_dims(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pad = [2, 3]
    value = 7.0

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, value)

    gems_assert_equal(res_out, ref_out)
