import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, 1


@pytest.mark.cummin
def test_cummin():
    bench = utils.GenericBenchmark2DOnly(
        op_name="cummin",
        input_fn=input_fn,
        torch_op=torch.cummin,
        dtypes=attr_utils.FLOAT_DTYPES + attr_utils.INT_DTYPES,
    )
    bench.run()
