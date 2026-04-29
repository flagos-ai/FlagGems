import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unfold_backward
@pytest.mark.parametrize(
    "input_sizes, dim, size, step",
    [
        ((32, 64), 1, 16, 16),
        ((16, 33), 0, 5, 2),
        ((4, 8, 12), -1, 6, 4),
        ((7, 13), 1, 13, 3),
        ((6, 20), 1, 7, 4),
        ((2, 3, 17), -1, 9, 1),
        ((2, 17), 1, 4, 6),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_unfold_backward(input_sizes, dim, size, step, dtype):
    d = dim % len(input_sizes)
    num_windows = (input_sizes[d] - size) // step + 1
    grad_shape = (
        list(input_sizes[:d]) + [num_windows] + list(input_sizes[d + 1 :]) + [size]
    )

    grad_in = torch.randn(grad_shape, dtype=dtype, device=flag_gems.device)

    ref_grad = utils.to_reference(grad_in, True)
    ref_out = torch.ops.aten.unfold_backward(ref_grad, input_sizes, dim, size, step)

    with flag_gems.use_gems():
        res_out = flag_gems.unfold_backward(grad_in, input_sizes, dim, size, step)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=size)
