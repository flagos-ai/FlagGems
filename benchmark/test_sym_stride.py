import pytest
import torch

from . import base, consts

# Benchmark shapes for sym_stride - covering various tensor dimensionalities
SYM_STRIDE_SHAPES = [(2, 3), (10, 20, 30), (5, 10), (100,), (1, 2, 3, 4)]


class SymStrideBenchmark(base.Benchmark):
    """Custom benchmark for sym_stride - returns tensor metadata (stride), not a computed tensor."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = SYM_STRIDE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (x,)


@pytest.mark.sym_stride
def test_sym_stride():
    bench = SymStrideBenchmark(
        op_name="sym_stride",
        torch_op=torch.ops.aten.sym_stride,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
