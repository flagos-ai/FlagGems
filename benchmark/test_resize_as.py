import pytest
import torch

from . import base, consts


@pytest.mark.resize_as
def test_resize_as():
    def resize_as_input_fn(shape, dtype, device):
        numel = 1
        for s in shape:
            numel *= s
        target_shape = (numel,)
        inp = torch.randn(shape, dtype=dtype, device=device)
        template = torch.randn(target_shape, dtype=dtype, device=device)
        yield inp, template

    bench = base.GenericBenchmark(
        input_fn=resize_as_input_fn,
        op_name="resize_as",
        torch_op=torch.Tensor.resize_as,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.resize_as_
def test_resize_as_():
    def resize_as_inplace_input_fn(shape, dtype, device):
        numel = 1
        for s in shape:
            numel *= s
        target_shape = (numel,)
        inp = torch.randn(shape, dtype=dtype, device=device)
        template = torch.randn(target_shape, dtype=dtype, device=device)
        yield inp, template

    bench = base.GenericBenchmark(
        input_fn=resize_as_inplace_input_fn,
        op_name="resize_as_",
        torch_op=torch.Tensor.resize_as_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
