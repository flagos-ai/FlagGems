import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.ge
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_ge():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ge",
        torch_op=torch.ge,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def ge_scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0


@pytest.mark.ge_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_ge_scalar():
    bench = base.GenericBenchmark(
        op_name="ge_scalar",
        input_fn=ge_scalar_input_fn,
        torch_op=torch.ge,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
