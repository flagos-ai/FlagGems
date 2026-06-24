import pytest
import torch

from . import base


class LinalgEigvalsBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Eigenvalues require square matrices
        self.shapes = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield x,


@pytest.mark.linalg_eigvals
def test_linalg_eigvals():
    bench = LinalgEigvalsBenchmark(
        op_name="linalg_eigvals",
        torch_op=torch.linalg.eigvals,
        # _linalg_eigvals requires float32 for cuSOLVER eigenvalue computation
        dtypes=[torch.float32],
    )
    bench.run()
