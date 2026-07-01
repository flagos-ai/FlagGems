import pytest
import torch

import flag_gems

from . import base, consts

# Shapes covering 2D and 3D tensors for renorm benchmarking
RENORM_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
    (16, 32, 64),
    (8, 16, 128),
]


class RenormBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = RENORM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            p = 2.0
            dim = 1 if len(shape) > 1 else 0
            maxnorm = 1.0
            yield x, p, dim, maxnorm


@pytest.mark.renorm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_renorm():
    bench = RenormBenchmark(
        op_name="renorm",
        torch_op=torch.renorm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
