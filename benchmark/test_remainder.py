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


@pytest.mark.remainder_scalar_
def test_remainder_scalar_inplace():
    def input_fn(shape, dtype, device):
        inp = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device=device,
        )
        scalar = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            (1,),
            dtype=dtype,
            device=device,
        ).item()
        if scalar == 0:
            scalar = 1
        yield inp, scalar

    bench = base.GenericBenchmark(
        input_fn=input_fn,
        op_name="remainder_scalar_",
        torch_op=lambda a, b: a.remainder_(b),
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
