import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def aminmax_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    # Test dim=None (whole tensor reduction)
    yield inp,
    # Test dim=-1 (last dimension)
    yield inp, {"dim": -1}
    # Test dim=0 (first dimension)
    if len(shape) > 1:
        yield inp, {"dim": 0}


class AminmaxBenchmark(utils.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from aminmax_input_fn(shape, cur_dtype, self.device)


@pytest.mark.aminmax
def test_aminmax():
    bench = AminmaxBenchmark(
        op_name="aminmax",
        torch_op=torch.aminmax,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
