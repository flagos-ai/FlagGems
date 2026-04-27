import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


def input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 1


@pytest.mark.cummax
def test_cummax():
    bench = base.GenericBenchmark2DOnly(
        input_fn=input_fn,
        op_name="cummax",
        torch_op=torch.cummax,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )

    bench.run()
