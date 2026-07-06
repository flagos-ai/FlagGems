import pytest
import torch

from . import base, consts


def xlogy_input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    # keep ``other`` positive so ``log`` stays finite
    inp2 = torch.rand(shape, dtype=dtype, device=device) * 5.0 + 0.01
    yield inp1, inp2


@pytest.mark.xlogy
def test_xlogy():
    bench = base.GenericBenchmark(
        op_name="xlogy",
        torch_op=torch.xlogy,
        input_fn=xlogy_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def xlogy_out_input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.rand(shape, dtype=dtype, device=device) * 5.0 + 0.01
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
