import pytest
import torch

from . import base, consts


@pytest.mark.logaddexp2
def test_logaddexp2():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logaddexp2",
        torch_op=torch.logaddexp2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def logaddexp2_out_input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.logaddexp2_out
def test_logaddexp2_out():
    bench = base.GenericBenchmark(
        op_name="logaddexp2_out",
        torch_op=torch.logaddexp2,
        input_fn=logaddexp2_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
