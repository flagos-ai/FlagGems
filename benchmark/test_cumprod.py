import pytest
import torch

import flag_gems
from flag_gems.ops import cumprod as flag_gems_cumprod

from . import base, consts

CUMPROD_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.BOOL_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
)
CUMPROD_INPLACE_DTYPES = (
    consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.EXTRA_INT_DTYPES
)

CUMPROD_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.BOOL_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
)
CUMPROD_INPLACE_DTYPES = (
    consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.EXTRA_INT_DTYPES
)


def _make_input(shape, dtype, device):
    if dtype in consts.FLOAT_DTYPES:
        return torch.empty(shape, dtype=dtype, device=device).uniform_(0.99, 1.01)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.int8, device="cpu").to(
            device, dtype=dtype
        )
    if dtype is torch.uint8:
        return torch.randint(0, 4, shape, dtype=dtype, device="cpu").to(device)
    return torch.randint(-3, 4, shape, dtype=dtype, device="cpu").to(device)


def input_fn(shape, dtype, device):
    inp = _make_input(shape, dtype, device)
    yield inp, 1


def torch_cumprod(inp, dim):
    if flag_gems.vendor_name == "ascend" and inp.dtype is torch.bool:
        return torch.cumprod(inp.to(torch.uint8), dim)
    return torch.cumprod(inp, dim)


@pytest.mark.cumprod
def test_cumprod_perf():
    bench = base.GenericBenchmark2DOnly(
        op_name="cumprod",
        input_fn=input_fn,
        torch_op=torch_cumprod,
        gems_op=flag_gems_cumprod,
        dtypes=CUMPROD_DTYPES,
    )
    bench.run()


@pytest.mark.cumprod_
def test_cumprod_inplace_perf():
    bench = base.GenericBenchmark2DOnly(
        op_name="cumprod_",
        input_fn=input_fn,
        torch_op=torch.Tensor.cumprod_,
        dtypes=CUMPROD_INPLACE_DTYPES,
        is_inplace=True,
    )
    bench.run()
