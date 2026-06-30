import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.tan
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tan():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tan", torch_op=torch.tan, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.tan_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tan_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tan_", torch_op=torch.tan_, dtypes=consts.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
