import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"sorted": True, "return_inverse": True, "return_counts": False},


@pytest.mark.unique2
def test_unique2():
    bench = base.GenericBenchmark2DOnly(
        input_fn=_input_fn,
        op_name="unique2",
        torch_op=torch.unique,
        dtypes=consts.INT_DTYPES,
    )

    bench.run()
