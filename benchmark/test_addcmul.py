import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)

    yield inp1, inp2, inp3, {"value": 0.5}


@pytest.mark.addcmul
def test_addcmul():
    bench = base.GenericBenchmark(
        op_name="addcmul",
        input_fn=_input_fn,
        torch_op=torch.addcmul,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
