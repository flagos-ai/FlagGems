from typing import Generator

import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, BenchLevel
from benchmark.performance_utils import Benchmark, Config, generate_tensor_input


class StackBenchmark(Benchmark):
    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)

    def set_more_shapes(self):
        more_shapes_2d = [(1024, 2**i) for i in range(1, 11, 4)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 8, 4)]
        return more_shapes_2d + more_shapes_3d


def _input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    inp3 = generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3], {"dim": 0},

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield [inp1, inp2, inp3], {"dim": -1},


@pytest.mark.skip(reason="CUDA error - illegal memory access: issue #2675")
@pytest.mark.stack
def test_stack():
    bench = StackBenchmark(
        op_name="stack", input_fn=_input_fn, torch_op=torch.stack, dtypes=FLOAT_DTYPES
    )
    bench.run()
