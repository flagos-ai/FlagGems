import pytest
import torch

from . import base, consts


def empty_input_fn(shape, dtype, device):
    yield shape


@pytest.mark.empty
def test_empty():
    bench = base.GenericBenchmark(
        op_name="empty",
        torch_op=torch.empty,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=empty_input_fn,
    )
    bench.run()
