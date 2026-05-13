import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


def _atan2_out_op(inp1, inp2, out):
    return torch.atan2(inp1, inp2, out=out)


@pytest.mark.atan2_out
def test_atan2_out():
    bench = base.GenericBenchmark(
        op_name="atan2_out",
        input_fn=_input_fn,
        torch_op=_atan2_out_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
