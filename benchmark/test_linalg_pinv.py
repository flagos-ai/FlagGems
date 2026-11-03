import pytest
import torch

from . import base
from .test_linalg_cond import _MusaLinalgStaticBaseline

PINV_SHAPES = [
    (1, 4, 4),
    (1, 8, 8),
    (4, 4, 4),
    (4, 8, 8),
    (8, 4, 4),
    (8, 8, 8),
    (16, 4, 4),
    (16, 8, 8),
    (32, 4, 4),
    (32, 8, 8),
]
PINV_DTYPES = [torch.float32]


class LinalgPinvBenchmark(_MusaLinalgStaticBaseline, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PINV_SHAPES

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            dims = tuple(shape)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(dims)}, params={}, builder_args=(dims,)
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        return (torch.randn(shape, dtype=case.dtype, device=self.device),)


@pytest.mark.linalg_pinv
def test_linalg_pinv():
    LinalgPinvBenchmark(
        op_name="linalg_pinv",
        torch_op=torch.linalg.pinv,
        dtypes=PINV_DTYPES,
    ).run()
