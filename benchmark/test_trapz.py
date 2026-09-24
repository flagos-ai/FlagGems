import pytest
import torch

import flag_gems

from . import base, consts


def trapz_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    dim = len(shape) - 1
    yield inp, {"dx": 1.0, "dim": dim}


@pytest.mark.trapz
def test_trapz():
    bench = base.GenericBenchmark(
        op_name="trapz",
        torch_op=torch.trapezoid,
        input_fn=trapz_input_fn,
        gems_op=flag_gems.trapz,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
