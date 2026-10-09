import pytest
import torch

import flag_gems

from . import base

# Shapes for linalg_tensorinv benchmark - these are 2D square matrices for ind=1
TENSORINV_SHAPES = [
    (2, 2),  # 2x2 matrix
    (4, 4),  # 4x4 matrix
    (8, 8),  # 8x8 matrix
    (16, 16),  # 16x16 matrix
    (24, 24),  # 24x24 matrix
]


class LinalgTensorinvBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = TENSORINV_SHAPES

    def get_input_iter(self, cur_dtype):
        with_out = self.op_name == "linalg_tensorinv_out"
        for shape in self.shapes:
            # Create an invertible matrix
            # For 2D case, use ind=1 (matrix inverse)
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            # Make it invertible
            m = shape[0]
            A = A @ A.T + torch.eye(m, dtype=cur_dtype, device=self.device) * 0.1
            if with_out:
                # ind=1 on a 2D input yields a same-shape inverse
                out = torch.empty_like(A)
                yield A, 1, {"out": out}
            else:
                yield A, 1  # ind=1 for matrix inverse


@pytest.mark.linalg_tensorinv
def test_linalg_tensorinv():
    bench = LinalgTensorinvBenchmark(
        op_name="linalg_tensorinv",
        torch_op=torch.linalg.tensorinv,
        gems_op=flag_gems.linalg_tensorinv,
        # torch.linalg.tensorinv benchmark uses float32 because PyTorch does not support half here.
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.linalg_tensorinv_out
def test_linalg_tensorinv_out():
    bench = LinalgTensorinvBenchmark(
        op_name="linalg_tensorinv_out",
        torch_op=torch.linalg.tensorinv,
        gems_op=flag_gems.linalg_tensorinv_out,
        # torch.linalg.tensorinv benchmark uses float32 because PyTorch does not support half here.
        dtypes=[torch.float32],
    )
    bench.run()
