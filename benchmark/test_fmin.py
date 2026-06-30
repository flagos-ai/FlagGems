import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.fmin
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fmin():
    bench = base.BinaryPointwiseBenchmark(
        op_name="fmin",
        torch_op=torch.fmin,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def fmin_out_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.fmin_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fmin_out():
    bench = base.GenericBenchmark(
        op_name="fmin_out",
        torch_op=torch.fmin,
        input_fn=fmin_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
