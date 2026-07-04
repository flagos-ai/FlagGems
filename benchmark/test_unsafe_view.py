import pytest
import torch

from . import base, consts

# Square 2D shapes covering common sizes for view benchmark
UNSAFE_VIEW_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
]


class UnsafeViewBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = UNSAFE_VIEW_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            new_shape = (shape[0] * shape[1],)
            yield inp, new_shape


@pytest.mark.unsafe_view
def test_unsafe_view():
    bench = UnsafeViewBenchmark(
        op_name="unsafe_view",
        torch_op=torch.ops.aten._unsafe_view,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
