import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes that work with __irshift__ (no broadcasting required)
IRSHIFT_SHAPES = [
    ((512, 1024), (512, 1024)),
    ((1024,), ()),
]

# Shapes for inplace bitwise ops (support broadcasting)
INPLACE_BITWISE_SHAPES = [
    ((512, 1024), (512, 1024)),
    ((256, 512), (1, 512)),
    ((256, 512), (256, 1)),
    ((1024,), ()),
]


@pytest.mark.irshift__
@pytest.mark.parametrize("shapes", IRSHIFT_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + [torch.uint8])
def test_irshift__(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_a = utils.to_reference(res_a)
    ref_b = utils.to_reference(res_b)

    # Use bitwise_right_shift as reference since __irshift__ has issues with broadcasting
    ref_out = torch.bitwise_right_shift(ref_a, ref_b)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.__irshift__(res_a, res_b)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.irshift__
@pytest.mark.parametrize("shapes", INPLACE_BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + [torch.uint8])
def test_irshift__inplace(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_a = utils.to_reference(res_a.clone())
    ref_b = utils.to_reference(res_b)

    # Use bitwise_right_shift as reference since __irshift__ has issues with broadcasting
    ref_a = ref_a.bitwise_right_shift_(ref_b)
    with flag_gems.use_gems():
        res_a = res_a.__irshift__(res_b)
    utils.gems_assert_close(res_a, ref_a, dtype)
