import pytest
import torch

import flag_gems

from . import base


def _input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device) * 10
    yield inp, {"bins": 100, "min": 0, "max": 10}


@pytest.mark.histc
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_histc():
    bench = base.GenericBenchmark2DOnly(
        input_fn=_input_fn,
        op_name="histc",
        torch_op=torch.histc,
        dtypes=[torch.float32],
    )
    bench.run()
