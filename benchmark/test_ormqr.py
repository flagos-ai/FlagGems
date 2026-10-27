import pytest
import torch

from . import base


# ormqr benchmark
# ormqr applies Householder reflectors which have inherent sequential dependency.
# The fused Triton kernel excels on small-to-medium matrices where the entire
# active region fits in GPU SRAM (<=128 in both dimensions).
class OrmqrBenchmark(base.GenericBenchmark2DOnly):
    # Override default shapes to include representative sizes for Householder application:
    # small matrices (fused kernel path) and medium/large matrices (tiled path)
    DEFAULT_SHAPES = [
        (32, 32),
        (48, 48),
        (64, 64),
        (96, 96),
        (128, 128),
        (256, 256),
        (1024, 1024),
        (4096, 4096),
        (1024, 65536),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_tflops(self, op, *args, **kwargs):
        # ormqr: multiply Q (m x m or n x n) with C (m x n)
        # Flops: 2 * m * n * min(m, n) for the matrix multiplication
        m, n = args[2].shape
        k = args[0].shape[-1]  # k = min(m, n) for ormqr
        return 2 * m * n * k


@pytest.mark.ormqr
def test_ormqr():
    def ormqr_input_fn(shape, dtype, device):
        m, n = shape
        k = min(m, n)
        # Generate valid Householder reflectors via QR decomposition
        a = torch.randn(m, k, dtype=dtype, device=device)
        input_tensor, tau = torch.geqrf(a)
        other = torch.randn(m, n, dtype=dtype, device=device)
        yield input_tensor, tau, other

    bench = OrmqrBenchmark(
        input_fn=ormqr_input_fn,
        op_name="ormqr",
        torch_op=torch.ormqr,
        # ormqr only supports float32 and float64 (LAPACK limitation, no half/bfloat16)
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
