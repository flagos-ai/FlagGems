import pytest
import torch

from . import base, consts


def _input_fn(shape, cur_dtype, device):
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    # Use dim=0 for 1D, dim=1 for 2D+
    if len(shape) == 1:
        yield inp, 0
    elif len(shape) >= 2:
        yield inp, 1


@pytest.mark.unsqueeze
def test_unsqueeze():
    bench = base.GenericBenchmark(
        op_name="unsqueeze",
        input_fn=_input_fn,
        torch_op=torch.unsqueeze,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.unsqueeze_
def test_unsqueeze_():
    bench = base.GenericBenchmark(
        op_name="unsqueeze_",
        input_fn=_input_fn,
        torch_op=torch.Tensor.unsqueeze_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
