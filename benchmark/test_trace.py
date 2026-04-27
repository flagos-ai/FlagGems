import pytest
import torch

import flag_gems

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype=dtype, device=device)
    yield inp,


@pytest.mark.skip(reason="Test case failure: issue #2663")
@pytest.mark.trace
def test_trace():
    if flag_gems.vendor_name == "mthreads":
        dtypes = attr_utils.FLOAT_DTYPES
    else:
        dtypes = attr_utils.FLOAT_DTYPES + attr_utils.INT_DTYPES

    bench = utils.GenericBenchmark2DOnly(
        op_name="trace",
        input_fn=_input_fn,
        torch_op=torch.trace,
        dtypes=dtypes,
    )

    bench.run()
