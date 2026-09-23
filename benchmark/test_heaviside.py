import pytest
import torch

from . import base, consts


class HeavisideBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return None


def heaviside_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    values = torch.randn(shape, dtype=dtype, device=device)
    yield inp, values


def heaviside_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    values = torch.randn(shape, dtype=dtype, device=device)
    out_buf = torch.empty_like(inp)
    yield inp, values, out_buf


def _heaviside_out_torch_op(inp, values, out_buf):
    return torch.ops.aten.heaviside.out(inp, values, out=out_buf)


@pytest.mark.heaviside
def test_heaviside():
    bench = HeavisideBenchmark(
        input_fn=heaviside_input_fn,
        op_name="heaviside",
        torch_op=torch.ops.aten.heaviside,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.heaviside
def test_heaviside_out():
    bench = HeavisideBenchmark(
        input_fn=heaviside_out_input_fn,
        op_name="heaviside",
        torch_op=_heaviside_out_torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
