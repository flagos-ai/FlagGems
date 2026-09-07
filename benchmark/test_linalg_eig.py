import pytest
import torch

from . import base


class LinalgEigBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # linalg.eig keeps the whole matrix in a register tile, so we cover
        # small-to-medium square matrices.
        self.shapes = [(8, 8), (16, 16), (32, 32), (64, 64)]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield x,


@pytest.mark.linalg_eig
def test_linalg_eig():
    bench = LinalgEigBenchmark(
        op_name="linalg_eig",
        torch_op=torch.linalg.eig,
        # The Francis QR kernel runs in float32 -> complex64 output.
        dtypes=[torch.float32],
    )
    bench.run()
