import pytest
import torch

from . import base, consts


@pytest.mark.as_strided
def test_as_strided():
    def as_strided_input_fn(shape, dtype, device):
        inp = base.generate_tensor_input(shape, dtype, device)
        if len(shape) >= 2:
            size = (shape[0], min(shape[1] // 2 + 1, 1))
            stride = (shape[1], 1)
        elif len(shape) == 1:
            size = (max(shape[0] // 2 + 1, 1),)
            stride = (1,)
        else:
            size = (1,)
            stride = (1,)
        yield inp, size, stride

    bench = base.GenericBenchmarkExcluse1D(
        input_fn=as_strided_input_fn,
        op_name="as_strided",
        torch_op=torch.as_strided,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.as_strided_
def test_as_strided_():
    def as_strided_input_fn(shape, dtype, device):
        inp = base.generate_tensor_input(shape, dtype, device)
        if len(shape) >= 2:
            size = (shape[0], min(shape[1] // 2 + 1, 1))
            stride = (shape[1], 1)
        elif len(shape) == 1:
            size = (max(shape[0] // 2 + 1, 1),)
            stride = (1,)
        else:
            size = (1,)
            stride = (1,)
        yield inp, size, stride

    bench = base.GenericBenchmarkExcluse1D(
        input_fn=as_strided_input_fn,
        op_name="as_strided_",
        torch_op=lambda t, size, stride: t.as_strided_(size, stride),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
