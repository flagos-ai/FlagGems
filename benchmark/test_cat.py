from typing import Generator

import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, INT_DTYPES, BenchLevel
from benchmark.performance_utils import Benchmark, Config, generate_tensor_input


def _input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    inp3 = generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3], {"dim": 0},

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield [inp1, inp2, inp3], {"dim": -1},


class CatBenchmark(Benchmark):
    def __init__(self, *args, **kwargs):
        self.input_fn = kwargs.pop("input_fn", _input_fn)
        super().__init__(*args, **kwargs)

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)

    def set_more_shapes(self):
        more_shapes_2d = [[1024, 2**i] for i in range(1, 11, 4)]
        more_shapes_3d = [[64, 64, 2**i] for i in range(0, 8, 4)]

        return more_shapes_2d + more_shapes_3d


@pytest.mark.skip("Benchmark test fails: issue #2673")
@pytest.mark.cat
def test_cat():
    bench = CatBenchmark(
        op_name="cat",
        input_fn=_input_fn,
        torch_op=torch.cat,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
    )
    bench.run()
