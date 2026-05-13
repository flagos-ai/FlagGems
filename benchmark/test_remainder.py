import pytest
import torch

from . import base, consts


@pytest.mark.remainder_tensor
def test_remainder_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="remainder_tensor",
        torch_op=torch.remainder,
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.remainder_tensor_
def test_remainder_tensor_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="remainder_tensor_",
        torch_op=lambda a, b: a.remainder_(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def scalar_tensor_remainder_input_fn(shape, dtype, device):
    inp = torch.randint(1, 100, shape, dtype=dtype, device=device)
    scalar = 7
    yield scalar, inp


@pytest.mark.remainder_scalar_tensor
def test_remainder_scalar_tensor():
    bench = base.GenericBenchmark(
        op_name="remainder_scalar_tensor",
        torch_op=torch.remainder,
        input_fn=scalar_tensor_remainder_input_fn,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
