from typing import Generator

import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, BenchLevel
from benchmark.performance_utils import Config, GenericBenchmark, generate_tensor_input


def adaptive_avg_pool2d_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    # Common cases - output_size must be (H, W)
    yield inp, (7, 7)
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield inp, (1, 1)
        yield inp, (16, 16)
        yield inp, (14, 14)


class AdaptiveAvgPool2dBenchmark(GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes_4d = [
            (4, 3, 32, 32),  # Small input
            (8, 64, 56, 56),  # Medium input
            (16, 128, 28, 28),  # Medium input
            (1, 64, 224, 224),  # Typical image size
            (4, 128, 112, 112),  # Typical intermediate feature map
        ]

        for shape in shapes_4d:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.adaptive_avg_pool2d
def test_perf_adaptive_avg_pool2d():
    bench = AdaptiveAvgPool2dBenchmark(
        input_fn=adaptive_avg_pool2d_input_fn,
        op_name="adaptive_avg_pool2d",
        torch_op=torch.ops.aten._adaptive_avg_pool2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
