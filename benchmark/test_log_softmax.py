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


@pytest.mark.log_softmax_backward_data
def test_log_softmax_backward_data():
    def log_softmax_backward_data_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        output = torch.nn.functional.log_softmax(inp, dim=-1)
        grad_output = torch.randn_like(output)
        yield grad_output, output, -1, dtype

    bench = base.GenericBenchmark2DOnly(
        op_name="log_softmax_backward_data",
        input_fn=log_softmax_backward_data_input_fn,
        torch_op=torch.ops.aten._log_softmax_backward_data,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
