import pytest
import torch

from . import base

pytestmark = pytest.mark.filterwarnings(
    "ignore:Warning only once for all operators.*:UserWarning"
)

# The Triton SVD kernels only cover float32 CUDA matrices; the full_matrices
# path requires max(M, N) <= 64, so keep the shapes small and square.
LINALG_SVD_SHAPES = [(8, 8), (16, 16), (32, 32), (64, 64)]


class LinalgSvdBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "(*B), M, N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = LINALG_SVD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (inp,)


@pytest.mark.linalg_svd
def test_linalg_svd():
    bench = LinalgSvdBenchmark(
        op_name="linalg_svd",
        torch_op=torch.linalg.svd,
        # The Triton SVD kernels only support float32 CUDA matrices.
        dtypes=[torch.float32],
    )
    bench.run()
