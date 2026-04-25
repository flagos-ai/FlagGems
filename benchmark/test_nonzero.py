import pytest
import torch

import flag_gems

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin" and utils.SkipVersion("torch", "<2.5"),
    reason="Not supported in XPytorch 2.0. Please upgrade your PyTorch version >= 2.5",
)
def test_nonzero():
    bench = utils.GenericBenchmark2DOnly(
        input_fn=utils.unary_input_fn,
        op_name="nonzero",
        torch_op=torch.nonzero,
        dtypes=attr_utils.FLOAT_DTYPES + attr_utils.INT_DTYPES + attr_utils.BOOL_DTYPES,
    )
    bench.run()
