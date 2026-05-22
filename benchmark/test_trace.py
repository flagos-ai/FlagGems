import pytest
import torch

import flag_gems

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype=dtype, device=device)
    yield inp,


@pytest.mark.skip(reason="Test case failure: issue #2663")
@pytest.mark.trace
def test_trace():
    if flag_gems.vendor_name == "mthreads":
        dtypes = consts.FLOAT_DTYPES
    else:
        dtypes = consts.FLOAT_DTYPES + consts.INT_DTYPES

    bench = base.GenericBenchmark2DOnly(
        op_name="trace",
        input_fn=_input_fn,
        torch_op=torch.trace,
        dtypes=dtypes,
    )

    bench.run()
