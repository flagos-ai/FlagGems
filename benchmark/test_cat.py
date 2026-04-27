from typing import Generator

import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3], {"dim": 0},

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield [inp1, inp2, inp3], {"dim": -1},


class CatBenchmark(base.Benchmark):
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
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
