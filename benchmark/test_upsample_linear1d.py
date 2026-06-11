import pytest
import torch

from . import base, consts


# TODO(Qiming): Kill this class
class UpsampleBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return None


@pytest.mark.upsample_linear1d
@pytest.mark.parametrize("align_corners", [False, True])
def test_upsample_linear1d(align_corners):
    def upsample_linear1d_input_fn(shape, dtype, device):
        batch, channel, height, width = shape
        length = height * width
        input = torch.randn((batch, channel, length), device=device, dtype=dtype)
        scale_factors = 2
        output_size = int(length * scale_factors)
        yield {
            "input": input,
            "output_size": (output_size,),
            "align_corners": align_corners,
        },

    bench = UpsampleBenchmark(
        input_fn=upsample_linear1d_input_fn,
        op_name="upsample_linear1d",
        torch_op=torch._C._nn.upsample_linear1d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class UpsampleLinear1dBackwardBenchmark(base.Benchmark):
    def set_more_shapes(self):
        shapes = [
            (512 * 1024 * 1024,),
            (512, 1024, 1024),
        ]
        shapes_3d = [(4, 16, 2**i) for i in range(4, 14, 2)]
        shapes_2d = [(16, 2**i) for i in range(6, 16, 2)]
        return shapes + shapes_3d + shapes_2d

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            if len(shape) == 1:
                shape_3d = (1, 1, shape[0])
            elif len(shape) == 2:
                shape_3d = (1, shape[0], shape[1])
            else:
                shape_3d = shape

            for scale_factor in [0.5, 2.0]:
                for align_corners in [False, True]:
                    n, c, w_in = shape_3d
                    w_out = max(1, int(w_in * scale_factor))
                    if n * c * max(w_in, w_out) >= 2**30:
                        continue

                    grad = torch.randn(
                        [n, c, w_out],
                        device=self.device,
                        dtype=cur_dtype,
                    )
                    yield grad, [w_out], [n, c, w_in], align_corners, scale_factor

    def get_tflops(self, op, *args, **kwargs):
        grad = args[0]
        return grad.numel() * 2


@pytest.mark.upsample_linear1d_backward
def test_upsample_linear1d_backward():
    bench = UpsampleLinear1dBackwardBenchmark(
        op_name="upsample_linear1d_backward",
        torch_op=torch.ops.aten.upsample_linear1d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
