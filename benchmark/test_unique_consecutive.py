import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"return_inverse": True, "return_counts": False},


@pytest.mark.unique_consecutive
def test_unique_consecutive():
    bench = utils.GenericBenchmark2DOnly(
        input_fn=input_fn,
        op_name="unique_consecutive",
        torch_op=torch.unique_consecutive,
        dtypes=attr_utils.INT_DTYPES,
    )
    bench.run()
