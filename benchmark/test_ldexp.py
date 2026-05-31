import pytest
import torch

from . import base, consts


@pytest.mark.ldexp
def test_ldexp():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ldexp",
        torch_op=torch.ldexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def ldexp_out_input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randint(-4, 4, shape, device=device).to(dtype)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.ldexp_out
def test_ldexp_out():
    bench = base.GenericBenchmark(
        op_name="ldexp_out",
        torch_op=torch.ldexp,
        input_fn=ldexp_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
