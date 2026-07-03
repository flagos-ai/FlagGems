import pytest
import torch

from . import base, consts, utils


def if_input_fn(shape, cur_dtype, device):
    condition = utils.generate_tensor_input(shape, torch.bool, device)
    then_val = utils.generate_tensor_input(shape, cur_dtype, device)
    else_val = utils.generate_tensor_input(shape, cur_dtype, device)
    yield condition, then_val, else_val


@pytest.mark.If
def test_If():
    bench = base.GenericBenchmark(
        op_name="If",
        input_fn=if_input_fn,
        torch_op=torch.where,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
