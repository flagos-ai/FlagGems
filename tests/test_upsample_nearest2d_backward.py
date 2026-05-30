import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize(
    "output_size,scales",
    [((32, 32), None), ((64, 64), None), ((32, 32), (2.0, 2.0)), ((48, 48), (1.5, 1.5))],
)
@pytest.mark.parametrize("input_size", [(1, 3, 16, 16), (2, 16, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_backward(output_size, scales, input_size, dtype):
    """Test upsample_nearest2d_backward with various configurations."""
    input = torch.randn(input_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten.upsample_nearest2d(ref_input, output_size, scales[0] if scales else None, scales[1] if scales else None)
    ref_grad = torch.randn_like(ref_output)
    ref_output.backward(ref_grad)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.upsample_nearest2d(input, output_size, scales[0] if scales else None, scales[1] if scales else None)
        res_grad = torch.randn_like(res_output)
        res_output.backward(res_grad)

    utils.gems_assert_close(input.grad, ref_input.grad, dtype)
