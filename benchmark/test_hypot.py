import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.hypot
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_hypot():
    bench = base.BinaryPointwiseBenchmark(
        op_name="hypot",
        torch_op=torch.hypot,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def hypot_out_input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.hypot_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_hypot_out():
    bench = base.GenericBenchmark(
        op_name="hypot_out",
        torch_op=torch.hypot,
        input_fn=hypot_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
