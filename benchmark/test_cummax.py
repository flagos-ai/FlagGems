import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, 1


@pytest.mark.cummax
def test_cummax():
    bench = utils.GenericBenchmark2DOnly(
        input_fn=input_fn,
        op_name="cummax",
        torch_op=torch.cummax,
        dtypes=attr_utils.FLOAT_DTYPES + attr_utils.INT_DTYPES,
    )

    bench.run()
