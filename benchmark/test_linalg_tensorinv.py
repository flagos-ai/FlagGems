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

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            plan = base.BenchmarkCasePlan(
                shape={"input": list(shape)},
                params={"ind": 1},
                builder_args=(shape,),
            )
            yield self._case_from_plan(dtype, ordinal, plan)

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        eye = torch.eye(shape[0], dtype=case.dtype, device=self.device)
        matrix = matrix @ matrix.T + eye * 0.1
        if self.op_name == "linalg_tensorinv_out":
            return matrix, plan.params["ind"], {"out": torch.empty_like(matrix)}
        return matrix, plan.params["ind"]


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
