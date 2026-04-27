import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


def _input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)

    if len(shape) > 1:
        yield inp, {"shifts": (1, 2), "dims": (0, 1)}
    else:
        yield inp, {"shifts": 1, "dims": 0}


@pytest.mark.roll
def test_roll():
    bench = base.GenericBenchmark(
        op_name="roll",
        input_fn=_input_fn,
        torch_op=torch.roll,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
