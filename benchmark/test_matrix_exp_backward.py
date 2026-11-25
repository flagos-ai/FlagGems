from typing import Generator

import pytest
import torch

from . import base, consts

# Core benchmark shapes for matrix exponential backward: small to large square matrices
_SQUARE_SHAPES = [
    (64, 64),
    (256, 256),
    (512, 512),
    (16, 64, 64),
    (16, 128, 128),
]


class MatrixExpBackwardBenchmark(base.GenericBenchmark):
    """matrix_exp_backward operates on batches of square matrices, so we
    force square (n, n) / (batch, n, n) shapes. We override ``set_shapes``
    directly because the base ``Benchmark`` key in ``core_shapes.yaml`` would
    otherwise inject 1-D shapes via the MRO lookup and break ``tensor.mH``."""

    DEFAULT_SHAPES = _SQUARE_SHAPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = _SQUARE_SHAPES

    def set_more_shapes(self):
        return _SQUARE_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)


def _input_fn(shape, cur_dtype, device):
    self_inp = torch.randn(shape, dtype=cur_dtype, device=device)
    grad_inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield self_inp, grad_inp


@pytest.mark.matrix_exp_backward
def test_matrix_exp_backward():
    bench = MatrixExpBackwardBenchmark(
        op_name="matrix_exp_backward",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.matrix_exp_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
