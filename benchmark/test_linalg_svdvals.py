import pytest
import torch

import flag_gems

from . import base

# Shapes for linalg_svdvals benchmark
SVD_BENCHMARK_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
]


class SvdBenchmark(base.GenericBenchmark2DOnly):
    """
    Benchmark for linalg_svdvals
    """

    def set_more_shapes(self):
        return SVD_BENCHMARK_SHAPES


@pytest.mark.linalg_svdvals
def test_linalg_svdvals():
    def svd_input_fn(shape, cur_dtype, device):
        m, n = shape
        # Only float32 is supported for SVD on CUDA
        inp = torch.randn([m, n], dtype=torch.float32, device=device)
        yield inp,

    bench = SvdBenchmark(
        input_fn=svd_input_fn,
        op_name="linalg_svdvals",
        torch_op=torch.linalg.svdvals,
        # Only float32 for SVD on CUDA (PyTorch limitation)
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.linalg_svdvals)
    bench.run()
