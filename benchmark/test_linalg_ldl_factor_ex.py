import pytest
import torch

from . import base

# Match the worktree accuracy-test square matrices; LDL requires N x N inputs.
LDL_FACTOR_EX_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]


def linalg_ldl_factor_ex_input_fn(shape, cur_dtype, device):
    n = shape[-1]
    a = torch.randn(shape, dtype=cur_dtype, device=device)
    # Symmetric positive definite input avoids singular LDL cases.
    yield (a @ a.mT + torch.eye(n, dtype=cur_dtype, device=device) * 0.1,)


class LinalgLdlFactorExBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LDL_FACTOR_EX_SHAPES
        self.shape_desc = "N, N"


@pytest.mark.linalg_ldl_factor_ex
def test_linalg_ldl_factor_ex():
    bench = LinalgLdlFactorExBenchmark(
        op_name="linalg_ldl_factor_ex",
        torch_op=torch.linalg.ldl_factor_ex,
        input_fn=linalg_ldl_factor_ex_input_fn,
        # PyTorch linalg_ldl_factor_ex only supports float32 on CUDA.
        dtypes=[torch.float32],
    )
    bench.run()
