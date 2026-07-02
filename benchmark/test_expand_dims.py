import pytest
import torch

from . import base, consts

EXPAND_DIMS_DIMS = [0, 1, -1]


def expand_dims_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    for dim in EXPAND_DIMS_DIMS:
        yield inp, dim


@pytest.mark.expand_dims
def test_expand_dims():
    bench = base.GenericBenchmark(
        op_name="expand_dims",
        torch_op=torch.unsqueeze,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=expand_dims_input_fn,
    )
    bench.run()
