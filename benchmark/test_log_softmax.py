import pytest
import torch

from . import base, consts, utils


@pytest.mark.log_softmax
def test_log_softmax():
    bench = base.GenericBenchmark2DOnly(
        op_name="log_softmax",
        input_fn=utils.unary_input_fn,
        torch_op=torch.nn.functional.log_softmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def log_softmax_backward_data_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    log_sm = torch.nn.functional.log_softmax(inp, dim=-1)
    grad_output = torch.randn_like(log_sm)
    out = torch.empty_like(grad_output)
    yield grad_output, log_sm, -1, dtype, {"out": out}


@pytest.mark.log_softmax_backward_data_out
def test_log_softmax_backward_data_out():
    bench = base.GenericBenchmark2DOnly(
        op_name="log_softmax_backward_data_out",
        input_fn=log_softmax_backward_data_out_input_fn,
        torch_op=torch._log_softmax_backward_data,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
