import pytest
import torch

from . import base, consts


def normal_input_fn(shape, cur_dtype, device):
    loc = torch.full(shape, fill_value=3.0, dtype=cur_dtype, device=device)
    scale = torch.full(shape, fill_value=10.0, dtype=cur_dtype, device=device)
    yield loc, scale


@pytest.mark.normal_tensor_tensor
def test_normal_tensor_tensor():
    bench = base.GenericBenchmark(
        input_fn=normal_input_fn,
        op_name="normal",
        torch_op=torch.normal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def normal_inplace_input_fn(shape, dtype, device):
    self = torch.randn(shape, dtype=dtype, device=device)
    loc = 3.0
    scale = 10.0
    yield self, loc, scale


@pytest.mark.normal_tensor_tensor_
def test_normal_inplace():
    bench = base.GenericBenchmark(
        input_fn=normal_inplace_input_fn,
        op_name="normal_",
        torch_op=torch.Tensor.normal_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
