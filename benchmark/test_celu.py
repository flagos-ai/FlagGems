import pytest
import torch

import flag_gems

from . import attri_util as attrs
from . import performance_utils as base
from . import utils

vendor_name = flag_gems.vendor_name


@pytest.mark.celu
@pytest.mark.skipif(
    vendor_name == "kunlunxin" and utils.SkipVersion("torch", "<2.5"),
    reason="There is an error in kunlunxin torch 2.0 aten, please use torch 2.5 instead",
)
def test_celu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="celu", torch_op=torch.nn.functional.celu, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.celu_
@pytest.mark.skipif(
    vendor_name == "kunlunxin" and utils.SkipVersion("torch", "<2.5"),
    reason="There is an error in kunlunxin torch 2.0 aten, please use torch 2.5 instead",
)
def test_celu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="celu_",
        torch_op=torch.nn.functional.celu_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
