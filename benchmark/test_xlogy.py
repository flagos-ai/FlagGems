import pytest
import torch

from . import base, consts


@pytest.mark.xlogy
def test_xlogy():
    bench = base.BinaryPointwiseBenchmark(
        op_name="xlogy",
        torch_op=torch.xlogy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def xlogy_out_input_fn(shape, dtype, device):
    inp1 = torch.rand(shape, dtype=dtype, device=device) * 4.0 + 0.1
    inp2 = torch.rand(shape, dtype=dtype, device=device) * 4.0 + 0.1
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.xlogy_out
def test_xlogy_out():
    bench = base.GenericBenchmark(
        op_name="xlogy_out",
        torch_op=torch.xlogy,
        input_fn=xlogy_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
