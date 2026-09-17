import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_nearest_exact2d
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("factor", [2, 3])
def test_upsample_nearest_exact2d(shape, dtype, factor):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    out_size = [shape[2] * factor, shape[3] * factor]
    ref_out = torch.ops.aten._upsample_nearest_exact2d(ref_x, out_size, None, None)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._upsample_nearest_exact2d(x, out_size, None, None)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.upsample_nearest_exact2d
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("factor", [2, 3])
def test_upsample_nearest_exact2d_out(shape, dtype, factor):
    op_label = "_upsample_nearest_exact2d_out"
    assert op_label
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    out_size = [shape[2] * factor, shape[3] * factor]
    out_shape = (shape[0], shape[1], out_size[0], out_size[1])
    ref_out = torch.empty(out_shape, dtype=dtype, device=ref_x.device)
    torch.ops.aten._upsample_nearest_exact2d.out(
        ref_x, out_size, None, None, out=ref_out
    )
    with flag_gems.use_gems():
        out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        res_out = torch.ops.aten._upsample_nearest_exact2d.out(
            x, out_size, None, None, out=out
        )
    assert res_out.data_ptr() == out.data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype)
