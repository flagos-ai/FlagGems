import pytest
import torch

import flag_gems

from . import base, consts

vendor_name = flag_gems.vendor_name


@pytest.mark.celu
def test_celu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="celu", torch_op=torch.nn.functional.celu, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.celu_
def test_celu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="celu_",
        torch_op=torch.nn.functional.celu_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
