import pytest
import torch

from . import base

# Square matrices for the linear solve; each shape is (N, N).
LINALG_SOLVE_EX_SHAPES = [
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]


def linalg_solve_ex_input_fn(shape, cur_dtype, device):
    n = shape[-1]
    a = torch.randn(shape, dtype=cur_dtype, device=device)
    # Strong diagonal keeps the system well-conditioned and non-singular.
    a = a + n * torch.eye(n, dtype=cur_dtype, device=device)
    b = torch.randn((n, n), dtype=cur_dtype, device=device)
    yield (a, b)


class LinalgSolveExBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LINALG_SOLVE_EX_SHAPES
        self.shape_desc = "N, N"


@pytest.mark.linalg_solve_ex
def test_linalg_solve_ex():
    bench = LinalgSolveExBenchmark(
        op_name="linalg_solve_ex",
        torch_op=torch.ops.aten._linalg_solve_ex,
        input_fn=linalg_solve_ex_input_fn,
        # PyTorch _linalg_solve_ex only reliably supports float32 on NPU.
        dtypes=[torch.float32],
    )
    bench.run()
