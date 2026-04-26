import pytest
import torch

from . import performance_utils as base


def fill_tensor_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    yield input, 3.14159,


@pytest.mark.fill_tensor
def test_fill_tensor():
    bench = base.GenericBenchmark(
        op_name="fill_tensor",
        input_fn=fill_tensor_input_fn,
        torch_op=torch.fill,
        is_inplace=True,
    )
    bench.run()


def fill_inplace_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    yield input, 3.14159,


@pytest.mark.fill_tensor_
def test_fill_tensor_inplace():
    bench = base.GenericBenchmark(
        op_name="fill_tensor_",
        input_fn=fill_inplace_input_fn,
        torch_op=torch.fill_,
        is_inplace=True,
    )
    bench.run()


def fill_tensor_out_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    value = torch.tensor(3.14159, dtype=dtype, device=device)
    out = torch.empty_like(input)
    yield input, value, {"out": out}


@pytest.mark.fill_tensor_out
def test_fill_tensor_out():
    bench = base.GenericBenchmark(
        op_name="fill_tensor_out",
        input_fn=fill_tensor_out_input_fn,
        torch_op=torch.ops.aten.fill.Tensor_out,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.fill_scalar_
def test_fill_scalar_inplace():
    bench = base.GenericBenchmark(
        op_name="fill_scalar_",
        input_fn=fill_inplace_input_fn,
        torch_op=torch.ops.aten.fill_.Scalar,
        is_inplace=True,
    )
    bench.run()


def fill_scalar_out_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    out = torch.empty_like(input)
    yield input, 3.14159, {"out": out}


@pytest.mark.fill_scalar_out
def test_fill_scalar_out():
    bench = base.GenericBenchmark(
        op_name="fill_scalar_out",
        input_fn=fill_scalar_out_input_fn,
        torch_op=torch.ops.aten.fill.Scalar_out,
        is_inplace=True,
    )
    bench.run()
