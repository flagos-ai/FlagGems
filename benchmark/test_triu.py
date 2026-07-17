import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.triu
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_triu():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="triu",
        torch_op=torch.triu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.triu_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_triu_inplace():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="triu_",
        torch_op=torch.Tensor.triu_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
