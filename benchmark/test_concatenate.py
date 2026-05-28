import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3], {"dim": 0}


@pytest.mark.concatenate
def test_concatenate():
    bench = base.GenericBenchmark(
        op_name="concatenate",
        torch_op=torch.concatenate,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
