import pytest
import torch

from . import base, consts

# narrow slices along dim 0; enumerate shapes explicitly.
NARROW_SHAPES = [(10000, 256), (10000, 4096), (10000, 65536)]


class NarrowBenchmark(base.Benchmark):
    """Benchmark for narrow operation (zero-copy view)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = NARROW_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            dim = 0
            start = shape[dim] // 4
            length = shape[dim] // 2
            yield inp, dim, start, length


@pytest.mark.narrow
def test_narrow():
    bench = NarrowBenchmark(
        op_name="narrow",
        torch_op=torch.narrow,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
