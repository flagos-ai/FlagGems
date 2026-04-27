import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"sorted": True, "return_inverse": True, "return_counts": False},


@pytest.mark.unique
def test_unique():
    bench = base.GenericBenchmark2DOnly(
        input_fn=_input_fn,
        op_name="unique",
        torch_op=torch.unique,
        dtypes=consts.INT_DTYPES,
    )

    bench.run()
