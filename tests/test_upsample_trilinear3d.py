import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_trilinear3d
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "scale", [(2, 2, 2), (1.5, 2.1, 3.7), (0.5, 0.5, 0.5), (0.3, 1.3, 0.7)]
)
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES_3D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_trilinear3d(dtype, shape, scale, align_corners):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(3)]
    ref_out = torch.ops.aten.upsample_trilinear3d.default(
        ref_i, output_size, align_corners, None, None, None
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.upsample_trilinear3d.default(
            input, output_size, align_corners, None, None, None
        )
    utils.gems_assert_close(res_out, ref_out, dtype)
