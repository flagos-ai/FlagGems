from typing import Generator

import pytest
import torch

from . import base, consts


def _gen_input(shape, dtype, device):
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return torch.randint(-1000, 1000, shape, dtype=dtype, device=device)
    if dtype == torch.uint8:
        return torch.randint(0, 200, shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


class MedianBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (256,),
            (4096,),
            (65536,),
            (1024 * 1024,),
            (16, 1024),
            (1024, 1024),
            (4096, 4096),
            (8, 16384),
            (256, 4096),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield (_gen_input(shape, cur_dtype, self.device),)


class MedianDimBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (16, 256),
            (16, 1024),
            (16, 4096),
            (16, 16384),
            (256, 1024),
            (1024, 1024),
            (4096, 4096),
            (64, 32, 256),
            (4, 64, 4096),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = _gen_input(shape, cur_dtype, self.device)
            yield inp, -1, {"keepdim": False}


@pytest.mark.median
def test_perf_median():
    bench = MedianBenchmark(
        op_name="median",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.median
def test_perf_median_dim():
    bench = MedianDimBenchmark(
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
