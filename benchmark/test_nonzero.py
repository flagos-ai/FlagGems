import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin" and utils.SkipVersion("torch", "<2.5"),
    reason="Not supported in XPytorch 2.0. Please upgrade your PyTorch version >= 2.5",
)
def test_nonzero():
    bench = base.GenericBenchmark2DOnly(
        input_fn=utils.unary_input_fn,
        op_name="nonzero",
        torch_op=torch.nonzero,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
