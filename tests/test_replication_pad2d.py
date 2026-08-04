import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.replication_pad2d
@pytest.mark.parametrize("padding", [(1, 1, 1, 1), (2, 2, 0, 0), (0, 0, 3, 3), (3, 1, 2, 4)])
@pytest.mark.parametrize("shape", [(16, 3, 16, 16), (32, 16, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad2d(shape, padding, dtype):
    """Test replication_pad2d with various shapes and padding."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)

    ref_output = torch.ops.aten.replication_pad2d(ref_input, padding)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.replication_pad2d(input, padding)

    utils.gems_assert_close(res_output, ref_output, dtype)


@pytest.mark.replication_pad2d
@pytest.mark.parametrize("padding", [(1, 1, 1, 1), (2, 2, 0, 0), (0, 0, 3, 3)])
@pytest.mark.parametrize("shape", [(2, 8, 16, 16), (4, 16, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad2d_out(shape, padding, dtype):
    """Test replication_pad2d_out with pre-allocated output."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pad_h, pad_w = padding
    output_shape = (
        shape[0],
        shape[1],
        shape[2] + padding[0] + padding[1],
        shape[3] + padding[2] + padding[3],
    )
    out = torch.empty(output_shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_out = utils.to_reference(out)

    ref_output = torch.ops.aten.replication_pad2d(ref_input, padding, ref_out)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.replication_pad2d(input, padding, out)

    utils.gems_assert_close(res_output, ref_output, dtype)
