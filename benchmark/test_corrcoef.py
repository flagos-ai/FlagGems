import pytest
import torch

from . import base, consts

# corrcoef takes a 2D (variables x observations) matrix and produces an N x N
# correlation matrix. Shapes range from small to large in both dimensions so the
# Gram (covariance) GEMM and the normalization kernel are both exercised.
CORRCOEF_SHAPES = [
    (32, 256),
    (64, 1024),
    (128, 1024),
    (256, 1024),
    (256, 4096),
    (512, 2048),
    (512, 4096),
]


class CorrcoefBenchmark(base.Benchmark):
    """Benchmark for aten::corrcoef (Pearson correlation coefficient matrix)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = CORRCOEF_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            yield inp,


@pytest.mark.corrcoef
def test_corrcoef():
    bench = CorrcoefBenchmark(
        op_name="corrcoef",
        torch_op=torch.corrcoef,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
