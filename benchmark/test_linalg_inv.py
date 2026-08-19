import pytest
import torch

from . import base

# Matrix inverse benchmark shapes.
# Square matrices from 2x2 to 128x128 covering small to medium use cases.
# The Gauss-Jordan kernel runs one program per matrix and fully unrolls the
# nested loops, so very large N makes Triton compilation extremely slow;
# 128 is a practical upper bound for the benchmark.
INV_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
]


class InvBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = INV_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n = shape[-1]
            # Build a well-conditioned invertible matrix: A = B + n * I so the
            # diagonal dominates and the inverse is numerically stable.
            B = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = B + torch.eye(n, dtype=cur_dtype, device=self.device) * n
            yield (A,)


@pytest.mark.linalg_inv
def test_linalg_inv():
    bench = InvBenchmark(
        op_name="linalg_inv",
        torch_op=torch.ops.aten.linalg_inv,
        # linalg_inv only supports float32/float64; fp16/bf16 not supported by PyTorch
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()


@pytest.mark.linalg_inv_out
def test_linalg_inv_out():
    bench = InvBenchmark(
        op_name="linalg_inv_out",
        torch_op=torch.ops.aten.linalg_inv,
        # linalg_inv only supports float32/float64; fp16/bf16 not supported by PyTorch
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
