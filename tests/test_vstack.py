import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

VSTACK_SHAPES = [
    [(3,), (3,)],
    [(3, 33), (7, 33)],
    [(13, 3, 333), (17, 3, 333), (7, 3, 333)],
    [
        (13, 3, 64, 5, 2),
        (16, 3, 64, 5, 2),
        (7, 3, 64, 5, 2),
        (4, 3, 64, 5, 2),
        (1, 3, 64, 5, 2),
    ],
]

CAMBRICON_VSTACK_SHAPES = [
    [(16, 128, 64, 64), (16, 128, 64, 64), (16, 128, 64, 64), (16, 128, 64, 64)],
    [
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
    ],
]

if flag_gems.vendor_name == "cambricon":
    VSTACK_SHAPES_TEST = VSTACK_SHAPES + CAMBRICON_VSTACK_SHAPES
else:
    VSTACK_SHAPES_TEST = VSTACK_SHAPES


@pytest.mark.vstack
@pytest.mark.parametrize("shape", VSTACK_SHAPES_TEST)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_accuracy_vstack(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]
    ref_inp = [utils.to_reference(e) for e in inp]
    ref_out = torch.vstack(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.vstack(inp)

    utils.gems_assert_equal(res_out, ref_out)
