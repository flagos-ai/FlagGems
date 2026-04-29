import pytest
import torch

from . import base, consts


class UpsampleNearest2dBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [
        (4, 16, 64, 64, 128, 128),
        (8, 32, 32, 48, 64, 96),
        (8, 64, 32, 32, 64, 64),
        (16, 128, 16, 16, 32, 32),
        (8, 64, 64, 64, 128, 128),
    ]
    DEFAULT_SHAPE_DESC = "N, C, H_in, W_in, H_out, W_out"

    def get_input_iter(self, dtype):
        for n, c, h_in, w_in, h_out, w_out in self.shapes:
            inp = torch.randn((n, c, h_in, w_in), device=self.device, dtype=dtype)
            yield inp, [h_out, w_out], None, None


@pytest.mark.upsample_nearest2d
def test_upsample_nearest2d():
    bench = UpsampleNearest2dBenchmark(
        op_name="upsample_nearest2d",
        torch_op=torch._C._nn.upsample_nearest2d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


class UpsampleNearest2dBackwardBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [
        (4, 16, 64, 64, 128, 128),
        (8, 32, 32, 48, 64, 96),
        (8, 64, 32, 32, 64, 64),
        (16, 128, 16, 16, 32, 32),
        (8, 64, 64, 64, 128, 128),
    ]
    DEFAULT_SHAPE_DESC = "N, C, H_in, W_in, H_out, W_out"

    def get_input_iter(self, dtype):
        for n, c, h_in, w_in, h_out, w_out in self.shapes:
            grad = torch.randn((n, c, h_out, w_out), device=self.device, dtype=dtype)
            yield grad, [h_out, w_out], [n, c, h_in, w_in], None, None


@pytest.mark.upsample_nearest2d_backward
def test_upsample_nearest2d_backward():
    bench = UpsampleNearest2dBackwardBenchmark(
        op_name="upsample_nearest2d_backward",
        torch_op=torch.ops.aten.upsample_nearest2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
