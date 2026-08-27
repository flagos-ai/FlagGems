import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_trilinear3d_backward
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "scale", [(2, 2, 2), (1.5, 2.1, 3.7), (0.5, 0.5, 0.5), (0.3, 1.3, 0.7)]
)
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES_3D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_trilinear3d_backward(dtype, shape, scale, align_corners):
    input_size = list(shape)
    output_size = [max(1, int(shape[i + 2] * scale[i])) for i in range(3)]

    grad_output = torch.randn(
        [shape[0], shape[1], *output_size], dtype=dtype, device=flag_gems.device
    )
    ref_grad = utils.to_reference(grad_output).to(torch.float32)

    ref_out = torch.ops.aten.upsample_trilinear3d_backward.default(
        ref_grad, output_size, input_size, align_corners, None, None, None
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.upsample_trilinear3d_backward.default(
            grad_output, output_size, input_size, align_corners, None, None, None
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
