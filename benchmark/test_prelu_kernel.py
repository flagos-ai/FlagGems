import pytest
import torch

from . import base, consts


@pytest.mark.prelu_kernel
def test_prelu_kernel():
    bench = base.BinaryPointwiseBenchmark(
        op_name="prelu_kernel",
        torch_op=torch._prelu_kernel,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
