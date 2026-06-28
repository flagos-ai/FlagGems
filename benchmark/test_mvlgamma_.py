import pytest
import torch

from . import base, consts

# Benchmark for mvlgamma_ (multivariate log-gamma function)
MVLGAMMA_SHAPES = [
    (1024,),
    (1024, 1024),
    (16, 32, 64),
]


class MvlgammaBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MVLGAMMA_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # Input must be > (p-1)/2 for mvlgamma_ to be defined, use p=2
            x = torch.rand(shape, dtype=cur_dtype, device=self.device) + 1.0
            yield x, 2


@pytest.mark.mvlgamma_
def test_mvlgamma_():
    bench = MvlgammaBenchmark(
        op_name="mvlgamma_",
        torch_op=torch.ops.aten.mvlgamma_.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
