import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.logaddexp
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logaddexp():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logaddexp",
        torch_op=torch.logaddexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def logaddexp_out_input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.logaddexp_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logaddexp_out():
    bench = base.GenericBenchmark(
        op_name="logaddexp_out",
        torch_op=torch.logaddexp,
        input_fn=logaddexp_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
