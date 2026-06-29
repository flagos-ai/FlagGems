import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_bilinear2d_aa
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1)])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 16, 128, 128),
        (15, 37, 256, 256),
        (3, 5, 127, 127),
        (128, 192, 42, 51),
        (3, 7, 1023, 1025),
    ],
)
# bfloat16 excluded: insufficient precision for bilinear AA on large input sizes
@pytest.mark.parametrize("dtype", utils.PRIMARY_FLOAT_DTYPES)
def test_upsample_bilinear2d_aa(dtype, shape, scale, align_corners):
    input = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input, True)
    output_size = tuple([int(input.shape[i + 2] * scale[i]) for i in range(2)])
    ref_out = torch.ops.aten._upsample_bilinear2d_aa(
        ref_i, output_size=output_size, align_corners=align_corners
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._upsample_bilinear2d_aa(
            input, output_size=output_size, align_corners=align_corners
        )

    def span(scale):
        support = 2 if (scale >= 1.0) else 2.0 / scale
        interpolate_range = int(support + 0.5) * 2 + 1
        return interpolate_range

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    # Bilinear uses 2x2 support window
    reduce_dim = span(scale[0]) * span(scale[1])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)
