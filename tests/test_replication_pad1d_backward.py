import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes covering basic cases, edge cases (W_in=1, W_in=2, no padding, max padding)
REPLICATION_PAD1D_SHAPES = [
    (3, (2, 3)),  # Basic case
    (5, (1, 1)),  # Small padding
    (10, (2, 2)),  # Larger input, small padding
    (4, (0, 0)),  # No padding
    (4, (3, 3)),  # Large padding
    (1, (5, 5)),  # W_in=1 edge case
    (8, (7, 7)),  # Max padding case
    (2, (3, 4)),  # W_in=2 edge case
]


@pytest.mark.replication_pad1d_backward
@pytest.mark.parametrize("shape,padding", REPLICATION_PAD1D_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad1d_backward(shape, padding, dtype):
    W_in = shape if isinstance(shape, int) else shape[-1]
    W_out = W_in + padding[0] + padding[1]

    # Create grad_output with known values
    grad_output = torch.arange(1, W_out + 1, dtype=dtype, device=flag_gems.device)
    grad_output = grad_output.reshape(1, 1, W_out)

    # Create self tensor (not used in computation, just for shape)
    self_tensor = torch.randn(1, 1, W_in, dtype=dtype, device=flag_gems.device)

    ref_grad_output = utils.to_reference(grad_output)
    ref_self = utils.to_reference(self_tensor)

    ref_out = torch.ops.aten.replication_pad1d_backward(
        ref_grad_output, ref_self, padding
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad1d_backward(
            grad_output, self_tensor, padding
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
