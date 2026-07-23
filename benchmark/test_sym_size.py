import pytest
import torch

from . import base, consts


# Benchmark for sym_size - returns tensor shape as a list
class SymSizeBenchmark(base.Benchmark):
    """
    Benchmark for sym_size operator.
    Note: sym_size returns a Python list (metadata), not a tensor.
    """

    def set_shapes(self, shape_file_path=None):
        # Various shapes to benchmark
        self.shapes = [
            (2, 3),
            (128, 256),
            (512, 512),
            (1, 2, 3),
            (4, 8, 16, 32),
            (1024, 1024),
            (2048, 2048),
        ]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # Create tensor - dtype doesn't matter for sym_size
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield x,


@pytest.mark.sym_size
def test_sym_size():
    bench = SymSizeBenchmark(
        op_name="sym_size",
        torch_op=torch.ops.aten.sym_size,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
