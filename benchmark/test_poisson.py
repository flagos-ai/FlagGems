import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp


@pytest.mark.poisson
def test_poisson():
    bench = base.GenericBenchmark2DOnly(
        op_name="poisson",
        torch_op=torch.poisson,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
