import pytest
import torch

from . import base


# LDL Factorization benchmark
class LdlFactorBenchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # LDL factorization shapes (square matrices only)
        self.shapes = [
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64),
        ]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n = shape[0]
            # Create symmetric positive definite matrix
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = (
                A @ A.transpose(-2, -1)
                + torch.eye(n, dtype=cur_dtype, device=self.device) * n
            )
            yield (A,)

    def get_gems_input_iter(self, cur_dtype):
        return self.get_input_iter(cur_dtype)


@pytest.mark.linalg_ldl_factor
def test_linalg_ldl_factor():
    bench = LdlFactorBenchmark(
        op_name="linalg_ldl_factor",
        torch_op=torch.linalg.ldl_factor,
        # torch.linalg.ldl_factor on CUDA supports float32/float64 for this path.
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
