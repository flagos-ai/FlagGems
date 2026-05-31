import pytest
import torch

from . import base  # noqa: F401
from .base import Benchmark

# Square matrix shapes: batch+square and various square sizes for slogdet
SLOGDET_SHAPES = [
    (2, 3, 3),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]


class SlogdetBenchmark(Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SLOGDET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)


@pytest.mark.linalg_slogdet
def test_linalg_slogdet():
    from flag_gems.ops.linalg_slogdet import linalg_slogdet

    bench = SlogdetBenchmark(
        op_name="linalg_slogdet",
        torch_op=torch.linalg.slogdet,
        # linalg.slogdet only supports float32/float64 on CUDA
        dtypes=[torch.float32],
    )
    bench.set_gems(linalg_slogdet)
    bench.run()
