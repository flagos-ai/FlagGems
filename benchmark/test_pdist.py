import pytest
import torch

from . import base

PDIST_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]


class PdistBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PDIST_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (x, 2.0)


@pytest.mark.pdist
def test_pdist():
    bench = PdistBenchmark(
        op_name="pdist",
        torch_op=torch.pdist,
        # pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
        dtypes=[torch.float32],
    )
    bench.run()
