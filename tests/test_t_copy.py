import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.t_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_t_copy(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.t_copy(ref_x)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.t_copy(x)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.t_copy_out
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_t_copy_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    out_shape = (shape[1], shape[0])
    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=ref_x.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.ops.aten.t_copy(ref_x, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.t_copy(x, out=act_out_buf)

    utils.gems_assert_close(act_out, ref_out, dtype)
