import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.sub
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_sub():
    bench = base.BinaryPointwiseBenchmark(
        op_name="sub",
        torch_op=torch.sub,
        dtypes=consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES,
    )
    bench.run()


# TODO(Qiming): Check why we don't have complex type here
@pytest.mark.sub_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_sub_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="sub_",
        torch_op=lambda a, b: a.sub_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
