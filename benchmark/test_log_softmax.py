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
