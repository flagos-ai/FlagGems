import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = utils.generate_tensor_input(shape, cur_dtype, device)
    condition = inp1 > 0

    yield condition, inp1, inp2


@pytest.mark.where_self
def test_where_self():
    bench = base.GenericBenchmark(
        op_name="where_self",
        input_fn=_input_fn,
        torch_op=torch.where,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _input_fn_out(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = utils.generate_tensor_input(shape, cur_dtype, device)
    condition = inp1 > 0
    out = torch.empty(shape, dtype=cur_dtype, device=device)

    yield condition, inp1, inp2, {"out": out}


@pytest.mark.where_self_out
def test_where_self_out():
    bench = base.GenericBenchmark(
        op_name="where_self_out",
        input_fn=_input_fn_out,
        torch_op=torch.where,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
