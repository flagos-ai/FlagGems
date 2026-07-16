import pytest
import torch

from . import base, consts


def new_ones_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield {"tensor": inp, "size": shape},


@pytest.mark.new_ones
def test_new_ones():
    bench = base.GenericBenchmark(
        op_name="new_ones",
        torch_op=lambda tensor, size: tensor.new_ones(size),
        input_fn=new_ones_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
