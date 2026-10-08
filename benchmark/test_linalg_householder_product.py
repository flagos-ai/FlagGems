import pytest
import torch

import flag_gems

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

    def get_input_iter(self, cur_dtype):
        for m, n in self.shapes:
            # Use geqrf to generate valid (h, tau) pair
            A = torch.randn(m, n, dtype=cur_dtype, device=self.device)
            h, tau = torch.geqrf(A)
            yield h, tau


@pytest.mark.linalg_householder_product
def test_linalg_householder_product():
    bench = LinalgHouseholderProductBenchmark(
        op_name="linalg_householder_product",
        torch_op=torch.linalg.householder_product,
        # Only float32 supported: geqrf/householder_product requires float32/float64 on CUDA
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.linalg_householder_product)
    bench.run()
