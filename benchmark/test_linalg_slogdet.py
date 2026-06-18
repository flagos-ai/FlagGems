import pytest
import torch

from . import base, consts

# Representative square matrix shapes for slogdet benchmarking
SLOGDET_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
]


class LinalgSlogdetBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SLOGDET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            a = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (a,)


@pytest.mark.linalg_slogdet
def test_linalg_slogdet():
    bench = LinalgSlogdetBenchmark(
        op_name="linalg_slogdet",
        torch_op=torch.linalg.slogdet,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
