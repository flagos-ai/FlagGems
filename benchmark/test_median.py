import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def median_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, {"dim": -1}


@pytest.mark.median
def test_median():
    bench = utils.GenericBenchmark2DOnly(
        op_name="median",
        input_fn=median_input_fn,
        torch_op=torch.median,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
