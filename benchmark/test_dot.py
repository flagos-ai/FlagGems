import pytest
import torch

from . import base, consts, utils


def dot_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype=dtype, device=device)
    if inp.dim() > 1:
        inp = inp.flatten()

    yield inp, inp


@pytest.mark.dot
def test_dot():
    bench = base.GenericBenchmark(
        input_fn=dot_input_fn,
        op_name="dot",
        torch_op=torch.dot,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
