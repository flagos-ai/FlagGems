import pytest
import torch

from . import base

# Symmetric-eigenvalue benchmark shapes.
# Square matrices from 2x2 to 64x64. The single-program Jacobi kernel is serial
# per matrix; larger sizes (128x128+) take many seconds to minutes per solve
# under benchmark warmup/iteration and are excluded to keep the run tractable.
EIGVALSH_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
]


class EigvalshBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EIGVALSH_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # Create a symmetric matrix A = (B + B^T) / 2.
            B = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = (B + B.transpose(-2, -1)) / 2
            yield (A,)


@pytest.mark.linalg_eigvalsh
def test_linalg_eigvalsh():
    bench = EigvalshBenchmark(
        op_name="linalg_eigvalsh",
        torch_op=torch.ops.aten.linalg_eigvalsh,
        # eigvalsh only supports float32/float64; fp16/bf16 not supported by PyTorch
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
