import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.sigmoid
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_sigmoid():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sigmoid", torch_op=torch.sigmoid, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.sigmoid_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_sigmoid_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sigmoid_",
        torch_op=torch.sigmoid_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.sigmoid_out
def test_sigmoid_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="sigmoid_out",
        torch_op=torch.ops.aten.sigmoid.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.sigmoid_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_sigmoid_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sigmoid_backward",
        torch_op=torch.sigmoid,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
