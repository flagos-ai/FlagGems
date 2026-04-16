import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.__rshift__ import (
    rshift_scalar,
    rshift_scalar_,
    rshift_tensor,
    rshift_tensor_,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close, to_reference
except ImportError:

    def gems_assert_close(res, ref, dtype, **kwargs):
        torch.testing.assert_close(res, ref, **kwargs)

    def to_reference(x, upcast=False):
        return x.to("cpu")


ALL_INT_DTYPES = [torch.int16, torch.int32, torch.int64]

POINTWISE_SHAPES = [
    (1024,),
    (16, 1024),
    (16, 512, 256),
]

BITWISE_SHAPES = [
    ((1024,), (1024,)),
    ((16, 1024), (16, 1024)),
    ((16, 512, 256), (16, 512, 256)),
]


@pytest.mark.rshift
@pytest.mark.parametrize("shapes", BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + [torch.uint8])
def test_accuracy_rshift_tensor(shapes, dtype):
    shape_a, shape_b = shapes
    inp_a = torch.randint(0, 100, shape_a, dtype=dtype, device=flag_gems.device)
    inp_b = torch.randint(0, 8, shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = to_reference(inp_a)
    ref_b = to_reference(inp_b)

    ref_out = ref_a >> ref_b
    res_out = rshift_tensor(inp_a, inp_b)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rshift
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + [torch.uint8])
def test_accuracy_rshift_scalar(shape, dtype):
    inp_a = torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_a = to_reference(inp_a)
    shift_amount = 2

    ref_out = ref_a >> shift_amount
    res_out = rshift_scalar(inp_a, shift_amount)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rshift
@pytest.mark.parametrize("shapes", BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + [torch.uint8])
def test_accuracy_rshift_tensor_(shapes, dtype):
    shape_a, shape_b = shapes
    inp_a = torch.randint(0, 100, shape_a, dtype=dtype, device=flag_gems.device)
    inp_b = torch.randint(0, 8, shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = to_reference(inp_a.clone())
    ref_b = to_reference(inp_b)

    ref_a >>= ref_b
    rshift_tensor_(inp_a, inp_b)

    gems_assert_close(inp_a, ref_a, dtype)


@pytest.mark.rshift
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + [torch.uint8])
def test_accuracy_rshift_scalar_(shape, dtype):
    inp_a = torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_a = to_reference(inp_a.clone())
    shift_amount = 2

    ref_a >>= shift_amount
    rshift_scalar_(inp_a, shift_amount)

    gems_assert_close(inp_a, ref_a, dtype)
