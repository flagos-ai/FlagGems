import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    inp3 = torch.randn(shape, dtype=dtype, device=device)

    yield inp1, inp2, inp3, {"value": 0.5}


@pytest.mark.addcmul_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_addcmul_():
    bench = base.GenericBenchmark(
        op_name="addcmul_",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.addcmul_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
