import pytest
import torch

from . import base

# (m, n) shapes where m >= n (required by householder_product)
LINALG_HOUSEHOLDER_SHAPES = [
    (4, 3),
    (8, 5),
    (16, 8),
    (32, 16),
    (64, 32),
    (128, 64),
]


class LinalgHouseholderProductBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LINALG_HOUSEHOLDER_SHAPES

    def get_case_iter(self, cur_dtype):
        for ordinal, (m, n) in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": [m, n]},
                    builder_args=((m, n),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        m, n = plan.builder_args[0]
        # Use geqrf to generate valid (h, tau) pair
        A = torch.randn(m, n, dtype=case.dtype, device=self.device)
        h, tau = torch.geqrf(A)
        return h, tau


@pytest.mark.linalg_householder_product
def test_linalg_householder_product():
    bench = LinalgHouseholderProductBenchmark(
        op_name="linalg_householder_product",
        torch_op=torch.linalg.householder_product,
        # Only float32 supported: geqrf/householder_product requires float32/float64 on CUDA
        dtypes=[torch.float32],
    )
    bench.run()
