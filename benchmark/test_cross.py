from typing import Generator

import pytest
import torch

from . import base, consts

# Mirror the layouts covered by tests/test_cross.py at the smallest performance
# scale that clears fixed kernel-launch overhead on each backend.
CROSS_COMMON_CASES = [
    ((262144, 3, 4), (1, 3, 4), 1),
    ((1, 3), (131072, 3), -1),
    ((65536, 4, 3), (65536, 4, 3), -1),
    ((131072, 3), (131072, 3), -1),
    ((262144, 3), (262144, 3), -1),
    ((524288, 3), (524288, 3), -1),
    ((1048576, 3), (1048576, 3), -1),
]


def _randn(shape, dtype, device):
    return torch.randn(shape, dtype=dtype, device=device)


class CrossBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CROSS_COMMON_CASES)

    def get_input_iter(self, cur_dtype) -> Generator:
        for input_shape, other_shape, dim in self.shapes:
            input = _randn(input_shape, cur_dtype, self.device)
            other = _randn(other_shape, cur_dtype, self.device)
            yield input, other, {"dim": dim}


class CrossOutBenchmark(CrossBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for input_shape, other_shape, dim in self.shapes:
            input = _randn(input_shape, cur_dtype, self.device)
            other = _randn(other_shape, cur_dtype, self.device)
            out = torch.empty(
                torch.broadcast_shapes(input_shape, other_shape),
                dtype=cur_dtype,
                device=self.device,
            )
            yield input, other, {"dim": dim, "out": out}


@pytest.mark.cross
def test_cross():
    bench = CrossBenchmark(
        op_name="cross",
        torch_op=torch.cross,
        # Final performance is the arithmetic mean of the per-shape averages
        # for FP16, FP32, and BF16.
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.cross_out
def test_cross_out():
    bench = CrossOutBenchmark(
        op_name="cross_out",
        torch_op=torch.ops.aten.cross.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
