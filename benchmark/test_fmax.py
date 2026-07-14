import pytest
import torch

from . import base, consts, utils


@pytest.mark.fmax
def test_fmax():
    bench = base.BinaryPointwiseBenchmark(
        op_name="fmax",
        torch_op=torch.fmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def fmax_out_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.fmax_out
def test_fmax_out():
    bench = base.GenericBenchmark(
        op_name="fmax_out",
        torch_op=torch.fmax,
        input_fn=fmax_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
