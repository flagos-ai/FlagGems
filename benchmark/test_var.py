import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.var
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_var():
    bench = base.UnaryReductionBenchmark(
        op_name="var", torch_op=torch.var, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.var_correction
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_var_correction():
    bench = base.UnaryReductionBenchmark(
        op_name="var_correction", torch_op=torch.var, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
