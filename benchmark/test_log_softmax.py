import pytest
import torch

from . import base, consts, utils


def log_softmax_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty_like(inp)
    if len(shape) > 1:
        yield inp, 1, False, {"out": out}
    else:
        yield inp, 0, False, {"out": out}


@pytest.mark.log_softmax
def test_log_softmax():
    bench = base.GenericBenchmark2DOnly(
        op_name="log_softmax",
        input_fn=utils.unary_input_fn,
        torch_op=torch.nn.functional.log_softmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.log_softmax_out
def test_log_softmax_out():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="log_softmax_out",
        input_fn=log_softmax_out_input_fn,
        torch_op=torch._log_softmax,
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
