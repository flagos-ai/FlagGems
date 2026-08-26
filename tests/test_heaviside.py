import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.heaviside
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_heaviside_tensor(shape, dtype):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < 0.1
    self_tensor[mask] = 0.0

    ref_self = utils.to_reference(self_tensor)
    ref_values = utils.to_reference(values_tensor)

    ref_out = torch.ops.aten.heaviside(ref_self, ref_values)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.heaviside(self_tensor, values_tensor)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.heaviside
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_heaviside_out(shape, dtype):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < 0.1
    self_tensor[mask] = 0.0

    ref_self = utils.to_reference(self_tensor)
    ref_values = utils.to_reference(values_tensor)
    ref_out_buf = torch.empty_like(ref_self)

    ref_out = torch.ops.aten.heaviside.out(ref_self, ref_values, out=ref_out_buf)

    act_out_buf = torch.empty_like(self_tensor)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.heaviside.out(
            self_tensor, values_tensor, out=act_out_buf
        )

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)
