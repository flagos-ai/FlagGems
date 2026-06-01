import pytest
import torch

from . import base, consts


@pytest.mark.special_i0
def test_special_i0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_i0",
        torch_op=torch.ops.aten.special_i0,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_i0
def test_special_i0_out():
    def torch_op(x):
        out = torch.empty_like(x)
        torch.ops.aten.special_i0.out(x, out=out)
        return out

    bench = base.UnaryPointwiseBenchmark(
        op_name="special_i0",
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
