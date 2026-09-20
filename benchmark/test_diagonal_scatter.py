import pytest
import torch

from . import base, consts

# Shapes for diagonal_scatter benchmark
DIAGONAL_SCATTER_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (64, 128),
    (128, 256),
    (256, 512),
    (32, 64, 64),
    (64, 128, 128),
    (16, 32, 32, 32),
]


class DiagonalScatterBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = DIAGONAL_SCATTER_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            # Get diagonal to size src tensor correctly
            diag = torch.diagonal(inp, 0, -2, -1)
            src = torch.randn(diag.shape, dtype=cur_dtype, device=self.device)
            yield inp, src, 0, -2, -1


@pytest.mark.diagonal_scatter
def test_diagonal_scatter():
    bench = DiagonalScatterBenchmark(
        op_name="diagonal_scatter",
        torch_op=torch.diagonal_scatter,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
