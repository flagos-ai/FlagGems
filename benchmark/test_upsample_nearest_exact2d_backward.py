import pytest
import torch

from . import base, consts


class UpsampleNearestExact2dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Typical feature map sizes: small, medium, large spatial dims
        self.shapes = [(2, 3, 8, 8), (4, 8, 16, 16), (8, 16, 32, 32)]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # Create grad_output by doing a forward pass first
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            out_h = shape[2] * 2
            out_w = shape[3] * 2
            output_size = (out_h, out_w)

            # Forward pass to get output
            out = torch.ops.aten._upsample_nearest_exact2d(
                x, [out_h, out_w], None, None
            )
            grad_output = torch.ones_like(out)

            input_size = tuple(x.shape)
            yield grad_output, output_size, input_size


@pytest.mark.upsample_nearest_exact2d_backward
def test_upsample_nearest_exact2d_backward():
    bench = UpsampleNearestExact2dBackwardBenchmark(
        op_name="upsample_nearest_exact2d_backward",
        torch_op=torch.ops.aten._upsample_nearest_exact2d_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
