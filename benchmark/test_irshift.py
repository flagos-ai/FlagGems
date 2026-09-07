import pytest
import torch

from . import base, consts


@pytest.mark.irshift__
def test_irshift__():
    bench = base.BinaryPointwiseBenchmark(
        op_name="irshift__",
        torch_op=torch.ops.aten.__irshift__,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


def _irshift_input_fn(shape, dtype, device):
    inp1 = torch.randint(-100, 100, shape, dtype=dtype, device=device)
    inp2 = torch.randint(0, 8, shape, dtype=dtype, device=device)
    yield inp1, inp2


@pytest.mark.irshift
def test_irshift():
    bench = base.GenericBenchmark(
        input_fn=_irshift_input_fn,
        op_name="irshift",
        torch_op=torch.ops.aten.__irshift__,
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
