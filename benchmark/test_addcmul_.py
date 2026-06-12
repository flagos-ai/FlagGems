import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    inp3 = torch.randn(shape, dtype=dtype, device=device)

    yield inp1, inp2, inp3, {"value": 0.5}


@pytest.mark.addcmul_
def test_addcmul_():
    bench = base.GenericBenchmark(
        op_name="addcmul_",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.addcmul_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
