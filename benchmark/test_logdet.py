import math

import pytest
import torch

import flag_gems

from . import base, consts
from .conftest import Config

LOGDET_SHAPES = [
    (1, 1),
    (4, 4),
    (8, 8),
    (16, 16),
    (4096, 4, 4),
    (1024, 8, 8),
    (128, 16, 16),
]
LOGDET_MORE_SHAPES = [(16384, 4, 4), (4096, 8, 8), (512, 16, 16)]
LOGDET_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


class LogdetBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(LOGDET_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(self.set_more_shapes())

    def set_more_shapes(self):
        return LOGDET_MORE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from self._inputs_for_shape(shape, cur_dtype)

    def _inputs_for_shape(self, shape, cur_dtype):
        n = shape[-1]
        matrix = torch.randn(shape, dtype=cur_dtype, device=self.device)
        matrix /= math.sqrt(n)
        inp = matrix @ matrix.transpose(-2, -1) + torch.eye(
            n, dtype=cur_dtype, device=self.device
        )
        yield (inp,)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        return next(self._inputs_for_shape(shape, case.dtype))


@pytest.mark.logdet
def test_logdet():
    bench = LogdetBenchmark(
        op_name="logdet",
        torch_op=torch.logdet,
        dtypes=LOGDET_DTYPES,
    )
    bench.run()
