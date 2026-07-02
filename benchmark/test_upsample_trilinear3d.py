import pytest
import torch

import flag_gems

from . import base, consts


class UpsampleBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return None


@pytest.mark.upsample_trilinear3d
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_upsample_trilinear3d(align_corners):
    def upsample_trilinear3d_input_fn(shape, dtype, device):
        batch, channel, height, width = shape
        depth = 4
        width = width // 4
        new_height = height // depth
        real_shape = (batch, channel, depth, new_height, width)

        input = torch.randn(size=real_shape, device=device, dtype=dtype)
        scale_factors = (2.0, 2.0, 2.0)
        output_size = (
            int(depth * scale_factors[0]),
            int(new_height * scale_factors[1]),
            int(width * scale_factors[2]),
        )

        yield input, output_size, align_corners, None, None, None

    bench = UpsampleBenchmark(
        input_fn=upsample_trilinear3d_input_fn,
        op_name="upsample_trilinear3d",
        torch_op=torch.ops.aten.upsample_trilinear3d.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
