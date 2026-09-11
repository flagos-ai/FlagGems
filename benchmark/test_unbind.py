import pytest
import torch

from . import base, consts

UNBIND_SHAPES = [
    (2, 3),
    (4, 8),
    (16, 32),
    (4, 8, 16),
    (32, 64, 128),
    (2, 4, 8, 16),
]


class UnbindBenchmark(base.Benchmark):
    """Benchmark for unbind operation (zero-copy view)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = UNBIND_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            dim = 0
            yield inp, dim


@pytest.mark.unbind
def test_unbind():
    bench = UnbindBenchmark(
        op_name="unbind",
        torch_op=torch.unbind,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
