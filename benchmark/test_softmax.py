import pytest
import torch

from . import base, consts


@pytest.mark.softmax
def test_softmax():
    bench = base.UnaryReductionBenchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.softmax_backward
def test_softmax_backward():
    bench = base.UnaryReductionBenchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )

    bench.run()


def softmax_backward_out_input_fn(shape, dtype, device):
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    output = torch.randn(shape, dtype=dtype, device=device)
    grad_input = torch.empty(shape, dtype=dtype, device=device)
    dim = 1 if len(shape) > 1 else 0
    yield grad_output, output, dim, dtype, {"grad_input": grad_input}


@pytest.mark.softmax_backward_out
def test_softmax_backward_out():
    bench = base.GenericBenchmark(
        op_name="softmax_backward_out",
        torch_op=torch.ops.aten._softmax_backward_data.out,
        input_fn=softmax_backward_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def softmax_out_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=dtype, device=device)
    dim = 1 if inp.ndim > 1 else 0
    yield inp, dim, False, {"out": out}


@pytest.mark.softmax_out
def test_softmax_out():
    bench = base.GenericBenchmark(
        op_name="softmax_out",
        torch_op=torch.ops.aten._softmax.out,
        input_fn=softmax_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
