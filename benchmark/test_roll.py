import pytest
import torch

import flag_gems

from . import base, consts, utils


def _input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)

    if len(shape) > 1:
        yield inp, {"shifts": (1, 2), "dims": (0, 1)}
    else:
        yield inp, {"shifts": 1, "dims": 0}


@pytest.mark.roll
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_roll():
    bench = base.GenericBenchmark(
        op_name="roll",
        input_fn=_input_fn,
        torch_op=torch.roll,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
