import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

REPLICATION_PAD1D_SHAPES = [(2, 3), (1, 5), (4, 10), (1, 8, 16)]
REPLICATION_PAD1D_PADDING = [(1, 1), (0, 2), (2, 1), (1, 2)]


@pytest.mark.replication_pad1d_backward
@pytest.mark.parametrize("shape", REPLICATION_PAD1D_SHAPES)
@pytest.mark.parametrize("padding", REPLICATION_PAD1D_PADDING)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad1d_backward(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    padded_out = torch.ops.aten.replication_pad1d(inp, padding)
    grad_output = torch.ones_like(padded_out)
    ref_grad = utils.to_reference(grad_output)

    ref_out = torch.ops.aten.replication_pad1d_backward(ref_grad, ref_inp, padding)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad1d_backward(grad_output, inp, padding)

    utils.gems_assert_close(res_out, ref_out, dtype)
