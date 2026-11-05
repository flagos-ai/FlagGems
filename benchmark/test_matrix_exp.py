import pytest
import torch

import flag_gems

from . import base, consts


class MatrixExpBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Matrix exponential requires square matrices
        self.shapes = [
            (16, 16),
            (32, 32),
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
        ]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # Scale inputs to avoid numerical overflow
            x = torch.randn(shape, dtype=cur_dtype, device=self.device) * 0.1
            yield x,


@pytest.mark.matrix_exp
def test_matrix_exp():
    bench = MatrixExpBenchmark(
        op_name="matrix_exp",
        torch_op=torch.matrix_exp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.matrix_exp)
    bench.run()
