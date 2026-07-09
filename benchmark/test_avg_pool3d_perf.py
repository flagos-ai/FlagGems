import pytest
import torch

from .attri_util import FLOAT_DTYPES, BenchLevel
from .performance_utils import Config, GenericBenchmark, generate_tensor_input


class AvgPool3dBenchmark(GenericBenchmark):
    DEFAULT_SHAPES = [
        (4, 3, 16, 8, 4),
        (8, 64, 8, 8, 4),
        (16, 128, 4, 8, 4),
        (32, 256, 4, 7, 4),
    ]
    DEFAULT_SHAPE_DESC = "N, C, D, H, W"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [tuple(shape) for shape in self.DEFAULT_SHAPES]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_shapes(self):
        return None


def avg_pool3d_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
    }
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": False,
            "count_include_pad": False,
            "divisor_override": None,
        }
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": True,
            "count_include_pad": True,
            "divisor_override": None,
        }
        yield inp, {
            "kernel_size": 2,
            "stride": 1,
            "padding": 0,
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": 3,
        }


@pytest.mark.avg_pool3d
def test_perf_avg_pool3d():
    bench = AvgPool3dBenchmark(
        input_fn=avg_pool3d_input_fn,
        op_name="avg_pool3d",
        torch_op=torch.ops.aten.avg_pool3d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.avg_pool3d_backward
def test_perf_avg_pool3d_backward():
    bench = AvgPool3dBenchmark(
        input_fn=avg_pool3d_input_fn,
        op_name="avg_pool3d",
        torch_op=torch.ops.aten.avg_pool3d,
        dtypes=FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
