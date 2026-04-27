import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)

    yield inp1, inp2, inp3, {"value": 0.5}


@pytest.mark.addcdiv
def test_addcdiv():
    bench = base.GenericBenchmark(
        op_name="addcdiv",
        input_fn=_input_fn,
        torch_op=torch.addcdiv,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
