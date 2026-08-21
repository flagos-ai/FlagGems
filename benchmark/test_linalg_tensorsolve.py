import pytest
import torch

from . import base

# tensorsolve benchmark shapes: (A.shape, B.ndim).
# The flattened (m, m) matrix ranges from small to medium sizes.
TENSORSOLVE_SHAPES = [
    ((4, 4), 1),
    ((16, 4, 4), 1),
    ((36, 6, 6), 1),
    ((6, 4, 2, 3, 4), 2),
    ((64, 8, 8), 1),
    ((128, 8, 16), 1),
]


class TensorsolveBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = TENSORSOLVE_SHAPES

    def get_input_iter(self, cur_dtype):
        for a_shape, b_ndim in self.shapes:
            m = 1
            for d in a_shape[:b_ndim]:
                m *= d
            A = torch.randn(a_shape, dtype=cur_dtype, device=self.device)
            # Make the flattened matrix diagonally dominant to keep it invertible.
            A_flat = A.reshape(m, m)
            A_flat = A_flat + torch.eye(m, dtype=cur_dtype, device=self.device) * m
            A = A_flat.reshape(a_shape)
            B = torch.randn(a_shape[:b_ndim], dtype=cur_dtype, device=self.device)
            yield (A, B)


class TensorsolveOutBenchmark(TensorsolveBenchmark):
    def get_input_iter(self, cur_dtype):
        for A, B in super().get_input_iter(cur_dtype):
            out = torch.empty(A.shape[B.ndim :], dtype=cur_dtype, device=self.device)
            yield (A, B, {"out": out})


@pytest.mark.linalg_tensorsolve
def test_linalg_tensorsolve():
    bench = TensorsolveBenchmark(
        op_name="linalg_tensorsolve",
        torch_op=torch.linalg.tensorsolve,
        # tensorsolve only supports float32/float64; fp16/bf16 not supported.
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()


@pytest.mark.linalg_tensorsolve_out
def test_linalg_tensorsolve_out():
    bench = TensorsolveOutBenchmark(
        op_name="linalg_tensorsolve_out",
        torch_op=torch.linalg.tensorsolve,
        # tensorsolve only supports float32/float64; fp16/bf16 not supported.
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
