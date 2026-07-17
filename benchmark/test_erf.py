import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.erf
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_erf():
    bench = base.UnaryPointwiseBenchmark(
        op_name="erf", torch_op=torch.erf, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.erf_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_erf_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="erf_", torch_op=torch.erf_, dtypes=consts.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
