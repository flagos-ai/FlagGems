import pytest
import torch

from . import base, consts

# Cubic growth shapes for profiling diagonal copy bandwidth at varying sizes,
# plus small uniform shapes to cover edge-case performance
DIAGONAL_COPY_SHAPES = [
    (16, 32, 64),
    (32, 64, 128),
    (64, 128, 256),
    (128, 256, 512),
    (256, 512, 1024),
    (16, 16, 16),
    (32, 32, 32),
]


class DiagonalCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = DIAGONAL_COPY_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield x, 0, 1, 2


@pytest.mark.diagonal_copy
def test_diagonal_copy():
    bench = DiagonalCopyBenchmark(
        op_name="diagonal_copy",
        torch_op=torch.diagonal_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
