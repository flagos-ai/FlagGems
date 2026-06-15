import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

HSTACK_SHAPES = [
    [(8,), (16,)],
    [(16, 256), (16, 128)],
    [(20, 320, 15), (20, 160, 15), (20, 80, 15)],
]

HSTACK_EXCEPTION_SHAPES = [
    [(16, 256), (16,)],
    [(16, 256), (8, 128)],
]


@pytest.mark.hstack
@pytest.mark.parametrize("shape", HSTACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_accuracy_hstack(shape, dtype):
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
    ref_out = torch.hstack(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.hstack(inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.hstack
@pytest.mark.parametrize("shape", HSTACK_EXCEPTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_exception_hstack(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            _ = torch.hstack(inp)
