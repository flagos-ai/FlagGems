import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark


class Col2ImBenchmark(Benchmark):
    """Benchmark for col2im operation."""

    def set_shapes(self, shape_file_path=None):
        # (batch, channels, kernel_size, output_size, stride, padding, dilation)
        self.shapes = [
            (2, 3, (2, 2), (4, 5), (1, 1), (0, 0), (1, 1)),
            (2, 16, (3, 3), (16, 16), (1, 1), (0, 0), (1, 1)),
            (4, 32, (3, 3), (32, 32), (1, 1), (1, 1), (1, 1)),
            (2, 64, (3, 3), (64, 64), (2, 2), (1, 1), (1, 1)),
            (1, 128, (3, 3), (128, 128), (1, 1), (1, 1), (1, 1)),
            (2, 64, (5, 5), (64, 64), (2, 2), (2, 2), (1, 1)),
            (4, 32, (3, 3), (128, 128), (2, 2), (1, 1), (2, 2)),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self.col2im_input_fn(config, cur_dtype, self.device)

    def col2im_input_fn(self, config, dtype, device):
        batch, channels, kernel_size, output_size, stride, padding, dilation = config
        kernel_h, kernel_w = kernel_size
        output_h, output_w = output_size
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        L_h = (
            output_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        L_w = (
            output_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1
        L = L_h * L_w

        inp = torch.randn(
            batch, channels * kernel_h * kernel_w, L, device=device, dtype=dtype
        )
        yield (inp, output_size, kernel_size, dilation, padding, stride)


@pytest.mark.col2im
def test_col2im():
    bench = Col2ImBenchmark(
        op_name="col2im",
        torch_op=torch.ops.aten.col2im,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
