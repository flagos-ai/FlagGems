from typing import Generator

import pytest
import torch

from . import base, consts, utils


class MedianNoDimBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield (utils.generate_tensor_input(shape, cur_dtype, self.device),)


class MedianReductionBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape; benchmark alternates dim/keepdim"

    def get_input_iter(self, cur_dtype) -> Generator:
        for case_id, shape in enumerate(self.shapes):
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            if inp.ndim == 1:
                dim = 0
            elif case_id % 2 == 0:
                dim = inp.ndim - 1
            else:
                dim = 0
            yield inp, dim, {"keepdim": case_id % 3 == 0}


@pytest.mark.median
def test_median():
    bench = MedianNoDimBenchmark(
        op_name="median",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.median
def test_median_dim():
    bench = MedianReductionBenchmark(
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
