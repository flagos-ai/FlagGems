import pytest
import torch

from . import base, consts


@pytest.mark.resize
def test_resize():
    bench = base.UnaryPointwiseBenchmark(
        op_name="resize",
        torch_op=lambda x: torch.ops.aten.resize(x, [x.numel()]),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.resize_
def test_resize_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="resize_",
        torch_op=lambda x: torch.ops.aten.resize_(x, [x.numel()]),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
