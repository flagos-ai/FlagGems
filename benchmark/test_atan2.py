import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.atan2
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_atan2():
    bench = base.BinaryPointwiseBenchmark(
        op_name="atan2",
        torch_op=torch.atan2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.atan2_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_atan2_out():
    def input_fn(shape, dtype, device):
        inp1 = utils.generate_tensor_input(shape, dtype, device)
        inp2 = utils.generate_tensor_input(shape, dtype, device)
        out = torch.empty(shape, dtype=dtype, device=device)
        yield inp1, inp2, {"out": out}

    bench = base.GenericBenchmark(
        input_fn=input_fn,
        op_name="atan2_out",
        torch_op=torch.atan2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
