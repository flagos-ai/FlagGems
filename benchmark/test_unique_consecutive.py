import pytest
import torch

from . import base, consts, utils


def input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"return_inverse": True, "return_counts": False},


@pytest.mark.unique_consecutive
def test_unique_consecutive():
    bench = base.GenericBenchmark2DOnly(
        input_fn=input_fn,
        op_name="unique_consecutive",
        torch_op=torch.unique_consecutive,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
