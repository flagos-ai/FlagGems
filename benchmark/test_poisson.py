import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device)
    yield (inp,)


@pytest.mark.poisson
def test_poisson():
    bench = base.GenericBenchmark2DOnly(
        op_name="poisson",
        torch_op=torch.poisson,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
