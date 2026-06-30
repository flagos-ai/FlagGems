import pytest
import torch

import flag_gems

from . import base


def _input_fn(shape, dtype, device):
    yield {"size": shape, "dtype": dtype, "device": device},


@pytest.mark.rand
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_rand():
    bench = base.GenericBenchmark(
        op_name="rand", input_fn=_input_fn, torch_op=torch.rand
    )
    bench.run()
