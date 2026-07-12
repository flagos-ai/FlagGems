import pytest
import torch

from . import base, consts, utils


def input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, 1


@pytest.mark.cummin
def test_cummin():
    bench = base.GenericBenchmark2DOnly(
        op_name="cummin",
        input_fn=input_fn,
        torch_op=torch.cummin,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
