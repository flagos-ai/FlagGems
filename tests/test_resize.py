import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# (src_shape, dst_shape): resize flattens or reshapes without changing total elements
RESIZE_SHAPES = [
    ((1024, 1024), [1048576]),
    ((20, 320, 15), [96000]),
    ((16, 128, 64, 60), [16, 128, 64, 60]),
]


@pytest.mark.resize
@pytest.mark.parametrize("src_shape, dst_shape", RESIZE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_resize(src_shape, dst_shape, dtype):
    inp = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.resize(ref_inp, dst_shape)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.resize(inp, dst_shape)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.resize_
@pytest.mark.parametrize("src_shape, dst_shape", RESIZE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_resize_(src_shape, dst_shape, dtype):
    inp = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    torch.ops.aten.resize_(ref_inp, dst_shape)
    with flag_gems.use_gems():
        torch.ops.aten.resize_(inp, dst_shape)

    utils.gems_assert_close(inp, ref_inp, dtype)
