import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def dot_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype=dtype, device=device)
    if inp.dim() > 1:
        inp = inp.flatten()

    yield inp, inp


@pytest.mark.dot
def test_dot():
    bench = utils.GenericBenchmark(
        input_fn=dot_input_fn,
        op_name="dot",
        torch_op=torch.dot,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
