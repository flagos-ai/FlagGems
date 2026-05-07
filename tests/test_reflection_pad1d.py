import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reflection_pad1d
@pytest.mark.parametrize("shape", [(3, 33), (2, 4, 64), (8, 16, 256), (32, 64, 2048)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("padding", [(1, 1), (3, 5), (8, 8)])
def test_reflection_pad1d(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)

    ref_out = torch.ops.aten.reflection_pad1d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.reflection_pad1d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.reflection_pad1d_out
@pytest.mark.parametrize("shape", [(3, 33), (2, 4, 64), (32, 64, 2048)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("padding", [(1, 1), (3, 5), (8, 8)])
def test_reflection_pad1d_out(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)

    out_shape = list(shape)
    out_shape[-1] = out_shape[-1] + padding[0] + padding[1]
    out_shape = tuple(out_shape)

    ref_out_buf = torch.empty(out_shape, dtype=ref_x.dtype, device=ref_x.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.reflection_pad1d.out(ref_x, padding, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.reflection_pad1d.out(x, padding, out=act_out_buf)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)
