import pytest
import torch

import flag_gems

from . import base, consts, utils


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0


@pytest.mark.le
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_le():
    bench = base.BinaryPointwiseBenchmark(
        op_name="le",
        torch_op=torch.le,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.le_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_le_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="le_scalar",
        torch_op=torch.le,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
