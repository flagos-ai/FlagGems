import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_bicubic2d
@pytest.mark.parametrize(
    "N, C, H, W, outH, outW, align_corners, use_scale",
    [
        (1, 1, 8, 8, 16, 16, False, False),
        (2, 3, 15, 20, 30, 35, True, False),
        (4, 3, 7, 5, 14, 10, False, True),
        (1, 16, 32, 24, 48, 36, True, True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_upsample_bicubic2d(N, C, H, W, outH, outW, align_corners, use_scale, dtype):
    x = torch.randn((N, C, H, W), dtype=dtype, device=flag_gems.device)

    if use_scale:
        output_size = None
        scale_factors = (outH / float(H), outW / float(W))
    else:
        output_size = (outH, outW)
        scale_factors = None

    ref_x = utils.to_reference(x, True)
    ref_out = torch._C._nn.upsample_bicubic2d(
        ref_x, output_size, align_corners, scale_factors
    ).to(dtype=dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_bicubic2d(
            ref_x, output_size, align_corners, scale_factors
        )

    utils.gems_assert_close(res_out.to(dtype=dtype), ref_out, dtype, reduce_dim=16)
