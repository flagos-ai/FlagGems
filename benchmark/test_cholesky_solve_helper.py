import pytest
import torch

from . import base

CHOLESKY_SOLVE_SHAPES = [(4, 1), (8, 2), (16, 4), (32, 4), (64, 8)]


class CholeskySolveHelperBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SOLVE_SHAPES
        self.shape_desc = "N, NRHS"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            n, nrhs = shape
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"rhs": [n, nrhs], "factor": [n, n]},
                    params={"upper": False},
                    builder_args=(n, nrhs),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        n, nrhs = plan.builder_args
        torch.manual_seed(0)
        matrix = torch.randn((n, n), dtype=case.dtype, device=self.device)
        matrix = matrix @ matrix.mT + n * torch.eye(
            n, dtype=case.dtype, device=self.device
        )
        factor = torch.linalg.cholesky(matrix)
        rhs = torch.randn((n, nrhs), dtype=case.dtype, device=self.device)
        return rhs, factor, False


@pytest.mark.cholesky_solve_helper
@pytest.mark.underscore_cholesky_solve_helper
def test_cholesky_solve_helper():
    bench = CholeskySolveHelperBenchmark(
        op_name="_cholesky_solve_helper",
        torch_op=torch.ops.aten._cholesky_solve_helper,
        dtypes=[torch.float32],
    )
    bench.run()
