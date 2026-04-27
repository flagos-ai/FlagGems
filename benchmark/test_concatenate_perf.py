from typing import Generator

import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, INT_DTYPES, BenchLevel
from benchmark.performance_utils import Benchmark, Config, generate_tensor_input


class ConcatenateBenchmark(Benchmark):
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


def concatenate_input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    inp3 = generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3], {"dim": 0},
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield [inp1, inp2, inp3], {"dim": -1},


@pytest.mark.concatenate
def test_concatenate():
    bench = ConcatenateBenchmark(
        input_fn=concatenate_input_fn,
        op_name="concatenate",
        torch_op=torch.concatenate,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
    )
    bench.run()
