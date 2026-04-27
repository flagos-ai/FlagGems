import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


def aminmax_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    # Test dim=None (whole tensor reduction)
    yield inp,
    # Test dim=-1 (last dimension)
    yield inp, {"dim": -1}
    # Test dim=0 (first dimension)
    if len(shape) > 1:
        yield inp, {"dim": 0}


class AminmaxBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from aminmax_input_fn(shape, dtype, self.device)


@pytest.mark.aminmax
def test_aminmax():
    bench = AminmaxBenchmark(
        op_name="aminmax",
        torch_op=torch.aminmax,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
