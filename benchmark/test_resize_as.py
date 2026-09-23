import pytest
import torch

from . import base, consts


def _resize_as_input_fn(shape, dtype, device):
    # Create a tensor and a template with different shape but same numel
    numel = 1
    for s in shape:
        numel *= s
    # Different shape with same numel
    target_shape = (numel,)
    inp = torch.randn(shape, dtype=dtype, device=device)
    template = torch.randn(target_shape, dtype=dtype, device=device)
    yield inp, template


@pytest.mark.resize_as
def test_resize_as():
    bench = base.GenericBenchmark(
        input_fn=_resize_as_input_fn,
        op_name="resize_as",
        torch_op=torch.Tensor.resize_as,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


@pytest.mark.resize_as_
def test_resize_as_():
    bench = base.GenericBenchmark(
        input_fn=_resize_as_input_fn,
        op_name="resize_as_",
        torch_op=torch.Tensor.resize_as_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )

    bench.run()
