import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = utils.generate_tensor_input(shape, cur_dtype, device)
    condition = inp1 > 0

    yield condition, inp1, inp2


@pytest.mark.where
def test_where():
    bench = base.GenericBenchmark(
        op_name="where",
        input_fn=_input_fn,
        torch_op=torch.where,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
