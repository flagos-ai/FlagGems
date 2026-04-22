import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.replication_pad1d
@pytest.mark.parametrize("shape", [(2, 3, 7), (4, 16, 64), (8, 32, 256), (32, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("padding", [(0, 0), (1, 2), (3, 1)])
def test_replication_pad1d(shape, dtype, padding):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.replication_pad1d(ref_inp, padding)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.replication_pad1d(inp, padding)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.replication_pad1d_out
@pytest.mark.parametrize("shape", [(2, 3, 7), (4, 16, 64), (8, 32, 256), (32, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("padding", [(0, 0), (1, 2), (3, 1)])
def test_replication_pad1d_out(shape, dtype, padding):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    pl, pr = padding
    w_out = shape[-1] + pl + pr
    if len(shape) == 3:
        N, C, _ = shape
        out_shape = (N, C, w_out)
    else:
        C, _ = shape
        out_shape = (C, w_out)

    ref_out_buf = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_out = torch.ops.aten.replication_pad1d.out(ref_inp, padding, out=ref_out_buf)

    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.replication_pad1d.out(inp, padding, out=act_out_buf)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)
