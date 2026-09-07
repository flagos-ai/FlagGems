import pytest
import torch

from . import base, consts


class LinalgMatrixExpBenchmark(base.Benchmark):
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
def test_linalg_matrix_exp():
    bench = LinalgMatrixExpBenchmark(
        op_name="matrix_exp",
        torch_op=torch.linalg.matrix_exp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
