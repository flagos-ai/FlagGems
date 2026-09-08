import pytest
import torch

from . import base

# Covers representative (N, M) pairs exercising different grid sizes and BLOCK_M widths.
PDIST_FORWARD_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]


class PdistForwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PDIST_FORWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield x, 2.0


@pytest.mark.pdist_forward
def test_pdist_forward():
    bench = PdistForwardBenchmark(
        op_name="pdist_forward",
        torch_op=torch.ops.aten._pdist_forward,
        # _pdist_forward only supports float32 in the reference implementation
        dtypes=[torch.float32],
    )
    bench.run()
