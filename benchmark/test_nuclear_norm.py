import pytest
import torch

import flag_gems

from . import base

# Shapes for nuclear_norm benchmark
NUCLEAR_NORM_BENCHMARK_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
]


class NuclearNormBenchmark(base.GenericBenchmark2DOnly):
    """
    Benchmark for nuclear_norm
    """

    def set_more_shapes(self):
        return NUCLEAR_NORM_BENCHMARK_SHAPES


@pytest.mark.nuclear_norm
def test_nuclear_norm():
    def nuclear_norm_input_fn(shape, cur_dtype, device):
        m, n = shape
        # Only float32 is supported for SVD on CUDA
        inp = torch.randn([m, n], dtype=torch.float32, device=device)
        yield inp,

    bench = NuclearNormBenchmark(
        input_fn=nuclear_norm_input_fn,
        op_name="nuclear_norm",
        torch_op=torch.nuclear_norm,
        # Only float32 for SVD on CUDA (PyTorch limitation)
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.nuclear_norm)
    bench.run()
