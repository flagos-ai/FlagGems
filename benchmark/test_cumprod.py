import pytest
import torch

import flag_gems

from . import base, consts

CUMPROD_DTYPES = consts.FLOAT_DTYPES + [
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
]
CUMPROD_INPLACE_DTYPES = consts.FLOAT_DTYPES + [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
]


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


class CumprodBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        if flag_gems.vendor_name == "kunlunxin":
            return []
        return super().set_more_shapes()


@pytest.mark.cumprod
def test_cumprod_perf():
    bench = CumprodBenchmark(
        op_name="cumprod",
        input_fn=input_fn,
        torch_op=torch.cumprod,
        dtypes=CUMPROD_DTYPES,
    )
    bench.run()


@pytest.mark.cumprod_
def test_cumprod_inplace_perf():
    bench = CumprodBenchmark(
        op_name="cumprod_",
        input_fn=input_fn,
        torch_op=torch.Tensor.cumprod_,
        dtypes=CUMPROD_INPLACE_DTYPES,
        is_inplace=True,
    )
    bench.run()
