import pytest
import torch

from . import base, consts


class UpsampleNearest3dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Typical volumetric feature map sizes: small, medium, large spatial dims
        self.shapes = [(2, 3, 8, 8, 8), (4, 8, 16, 16, 16), (4, 16, 24, 24, 24)]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            out_d = shape[2] * 2
            out_h = shape[3] * 2
            out_w = shape[4] * 2
            output_size = (out_d, out_h, out_w)

            out = torch.ops.aten.upsample_nearest3d(
                x, [out_d, out_h, out_w], None, None, None
            )
            grad_output = torch.ones_like(out)

            input_size = tuple(x.shape)
            yield grad_output, output_size, input_size


@pytest.mark.upsample_nearest3d_backward
def test_upsample_nearest3d_backward():
    bench = UpsampleNearest3dBackwardBenchmark(
        op_name="upsample_nearest3d_backward",
        torch_op=torch.ops.aten.upsample_nearest3d_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@getattr(pytest.mark, "upsample_nearest3d_backward.grad_input")
def test_upsample_nearest3d_backward_grad_input():
    bench = UpsampleNearest3dBackwardBenchmark(
        op_name="upsample_nearest3d_backward_grad_input",
        torch_op=torch.ops.aten.upsample_nearest3d_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
