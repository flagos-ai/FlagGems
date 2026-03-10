from typing import Generator

import pytest
import torch

from . import base, consts, utils


class MedianDimBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "(B), M, N"
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            dim = inp.ndim - 1
            yield inp, dim, {"keepdim": False}


@pytest.mark.median
def test_median_dim():
    bench = MedianDimBenchmark(
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
