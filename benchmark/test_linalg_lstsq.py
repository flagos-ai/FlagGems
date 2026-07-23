import pytest
import torch

from . import base

pytestmark = pytest.mark.filterwarnings(
    "ignore:Warning only once for all operators.*:UserWarning"
)


class LstsqBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "(*B), M, N  (M>=N tall, or M<N wide)"
    DEFAULT_DTYPES = [torch.float32]
    # gels covers both over- and underdetermined systems. b is a vector RHS with
    # shape (*B, M), derived from A's leading dims. Wide shapes come in two
    # regimes: within the single-tile budget (next_pow2(n)*next_pow2(m) <=
    # 32768 for fp32, 8192 for fp64) the fast Q-apply kernel runs; beyond it the
    # blocked TSQR-of-A^T path (no Q) runs — both native. fp64 additionally
    # routes every tall shape to the blocked path (measured: the 8-byte
    # monolithic tile loses at every NC), so these shapes exercise different
    # kernels per dtype.
    DEFAULT_SHAPES = [
        # tall (M >= N)
        (256, 32),
        (1024, 16),
        (4096, 8),
        (8, 4096, 8),
        (64, 2048, 16),
        (16, 8192, 16),
        # wide (M < N), within the tile budget (single-tile kernel)
        (8, 16, 512),
        (8, 64, 256),
        (16, 16, 1024),
        (64, 32, 256),
        # wide beyond the budget (blocked TSQR path, tile = 65536)
        (64, 1024),
        (128, 512),
        (8, 64, 1024),
        (16, 32, 2048),
    ]

    def set_more_shapes(self):
        return []

    def set_shapes(self, *args, **kwargs):
        # Force our tall shapes; the file-based default injects a 1D shape that
        # lstsq (which needs a 2D+ matrix, M >= N) cannot accept.
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=dtype, device=self.device)
            b = torch.randn(shape[:-1], dtype=dtype, device=self.device)
            yield A, b, {"driver": "gels"}


@pytest.mark.linalg_lstsq
def test_linalg_lstsq():
    bench = LstsqBenchmark(
        op_name="linalg_lstsq",
        torch_op=torch.linalg.lstsq,
        # gels supports float32/float64 only; fp16/bf16 are not supported by
        # PyTorch's reference, and complex is outside the native path.
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
