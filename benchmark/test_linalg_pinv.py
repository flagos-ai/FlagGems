import pytest
import torch

from flag_gems.ops.linalg_pinv import linalg_pinv

from . import base

# Benchmark shapes: (batch, m, n) - small matrices where Triton kernel excels
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
# Jacobi SVD kernel operates in float32 registers; float16/bfloat16 lack precision for convergence
PINV_DTYPES = [torch.float32]


class LinalgPinvBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PINV_SHAPES

    def get_input_iter(self, cur_dtype):
        for batch, m, n in self.shapes:
            A = torch.randn(batch, m, n, dtype=cur_dtype, device=self.device)
            yield (A,)


@pytest.mark.linalg_pinv
def test_linalg_pinv():
    bench = LinalgPinvBenchmark(
        op_name="linalg_pinv",
        torch_op=torch.linalg.pinv,
        gems_op=linalg_pinv,
        dtypes=PINV_DTYPES,
    )
    bench.run()
