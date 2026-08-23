import pytest
import torch

from . import base, consts

# Shapes covering 2D, 3D, and 4D for benchmarking transpose.
# transpose.int is a zero-copy view op, so the benchmark measures
# dispatch + as_strided overhead rather than data movement.
TRANSPOSE_SHAPES = [
    (64, 64),
    (256, 512),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (128, 256, 64),
    (8, 16, 32, 64),
]


class TransposeBenchmark(base.Benchmark):
    """Benchmark for aten::transpose.int (zero-copy view operation)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = TRANSPOSE_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            # Swap first and last dimensions for every shape.
            ndim = inp.dim()
            yield inp, 0, ndim - 1


@pytest.mark.transpose
def test_transpose():
    bench = TransposeBenchmark(
        op_name="transpose",
        torch_op=torch.ops.aten.transpose.int,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
