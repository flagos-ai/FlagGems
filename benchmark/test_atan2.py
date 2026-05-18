import pytest
import torch

from . import base, consts, utils


@pytest.mark.atan2
def test_atan2():
    bench = base.BinaryPointwiseBenchmark(
        op_name="atan2",
        torch_op=torch.atan2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def atan2_out_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.atan2_out
def test_atan2_out():
    bench = base.GenericBenchmark(
        op_name="atan2_out",
        input_fn=atan2_out_input_fn,
        torch_op=torch.atan2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
