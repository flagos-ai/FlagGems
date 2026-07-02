import pytest
import torch

from flag_gems.ops.shape import shape as shape_fn
from flag_gems.ops.shape import shape_ as shape__fn

from . import base, consts


def shape_input_fn(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    yield (x,)


@pytest.mark.shape
def test_shape():
    bench = base.GenericBenchmark(
        op_name="shape",
        torch_op=lambda x: torch.tensor(x.shape, dtype=torch.int64, device=x.device),
        gems_op=shape_fn,
        input_fn=shape_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.shape_
def test_shape_():
    bench = base.GenericBenchmark(
        op_name="shape_",
        torch_op=lambda x: x,
        gems_op=shape__fn,
        input_fn=shape_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
