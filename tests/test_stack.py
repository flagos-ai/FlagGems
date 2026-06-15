import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

if flag_gems.vendor_name == "cambricon":
    CAMBRICON_STACK_SHAPES = [
        [
            (8, 8, 128),
            (8, 8, 128),
            (8, 8, 128),
        ],
        [
            (32, 64, 128, 8),
            (32, 64, 128, 8),
            (32, 64, 128, 8),
            (32, 64, 128, 8),
        ],
    ]

    STACK_SHAPES_TEST = utils.STACK_SHAPES + CAMBRICON_STACK_SHAPES
else:
    STACK_SHAPES_TEST = utils.STACK_SHAPES


@pytest.mark.stack
@pytest.mark.parametrize("shape", STACK_SHAPES_TEST)
@pytest.mark.parametrize("dim", utils.STACK_DIM_LIST)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_stack(shape, dim, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]

    ref_inp = [utils.to_reference(_) for _ in inp]
    ref_out = torch.stack(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.stack(inp, dim)

    utils.gems_assert_equal(res_out, ref_out)
