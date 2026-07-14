import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES


ADAPTIVE_AVG_POOL3D_CONFIGS = [
    # (shape, output_size)
    # Basic cases
    ((1, 1, 4, 4, 4), (1, 1, 1)),
    ((1, 1, 4, 4, 4), (2, 2, 2)),
    ((1, 1, 4, 4, 4), (4, 4, 4)),
    # Non-cubic input/output
    ((2, 3, 8, 7, 6), (1, 1, 1)),
    ((2, 3, 8, 7, 6), (2, 3, 3)),
    ((2, 3, 8, 7, 6), (4, 5, 6)),
    # Larger tensors
    ((4, 16, 16, 16, 16), (1, 1, 1)),
    ((4, 16, 16, 16, 16), (4, 4, 4)),
    ((4, 16, 16, 16, 16), (8, 8, 8)),
]


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVG_POOL3D_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_adaptive_avg_pool3d(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten._adaptive_avg_pool3d(ref_inp, output_size)

    with flag_gems.use_gems():
        res_out = torch.ops.aten._adaptive_avg_pool3d(inp, output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVG_POOL3D_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_adaptive_avg_pool3d_out(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    out_shape = (shape[0], shape[1], output_size[0], output_size[1], output_size[2])
    ref_out_buf = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_out = torch.ops.aten._adaptive_avg_pool3d.out(
        ref_inp, output_size, out=ref_out_buf
    )

    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._adaptive_avg_pool3d.out(
            inp, output_size, out=act_out_buf
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
