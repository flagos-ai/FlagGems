import pytest
import torch

from flag_gems.ops.upsample_nearest1d_backward import (
    upsample_nearest1d_backward,
    upsample_nearest1d_backward_grad_input,
)

from . import base, consts


class UpsampleNearest1dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # (N, C, input width, output width): bounded core shapes spanning
        # launch-bound through throughput-bound workloads and both resize directions.
        self.shapes = [
            (1, 1, 8, 16),
            (2, 3, 64, 96),
            (8, 16, 512, 256),
            (16, 64, 2048, 4096),
            (8, 128, 8192, 12288),
        ]
        self.shape_desc = ["N", "C", "IW", "OW"]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for batch, channels, input_w, output_w in self.shapes:
            grad_output = torch.randn(
                (batch, channels, output_w),
                dtype=cur_dtype,
                device=self.device,
            )
            yield grad_output, (output_w,), (batch, channels, input_w)


class UpsampleNearest1dBackwardOutBenchmark(UpsampleNearest1dBackwardBenchmark):
    def get_input_iter(self, cur_dtype):
        for args in super().get_input_iter(cur_dtype):
            grad_output, output_size, input_size = args
            grad_input = torch.empty(input_size, dtype=cur_dtype, device=self.device)
            yield grad_output, output_size, input_size, {"grad_input": grad_input}


@pytest.mark.upsample_nearest1d_backward
def test_upsample_nearest1d_backward():
    bench = UpsampleNearest1dBackwardBenchmark(
        op_name="upsample_nearest1d_backward",
        torch_op=torch.ops.aten.upsample_nearest1d_backward.default,
        gems_op=upsample_nearest1d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@getattr(pytest.mark, "upsample_nearest1d_backward.grad_input")
def test_upsample_nearest1d_backward_grad_input():
    bench = UpsampleNearest1dBackwardOutBenchmark(
        op_name="upsample_nearest1d_backward_grad_input",
        torch_op=torch.ops.aten.upsample_nearest1d_backward.grad_input,
        gems_op=upsample_nearest1d_backward_grad_input,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
