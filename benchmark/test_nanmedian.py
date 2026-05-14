import pytest
import torch

import flag_gems

from . import base, consts

ASCEND_UNSUPPORTED_REFERENCE_DTYPES = (torch.bfloat16, torch.float64)


def _filter_reference_supported(dtypes):
    if flag_gems.vendor_name == "ascend":
        return [
            dtype for dtype in dtypes if dtype not in ASCEND_UNSUPPORTED_REFERENCE_DTYPES
        ]
    return dtypes


NANMEDIAN_DTYPES = _filter_reference_supported(
    consts.FLOAT_DTYPES
    + [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.uint8,
    ]
)


class NanMedianBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [(1024, 1024), (256, 4096), (16, 128 * 1024)]


def _make_input(shape, dtype, device):
    if dtype is torch.uint8:
        return torch.randint(0, 101, shape, dtype=dtype, device="cpu").to(device)
    if dtype in (torch.int8, torch.int16, torch.int32):
        return torch.randint(-100, 101, shape, dtype=dtype, device="cpu").to(device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    if inp.numel() > 0:
        inp.reshape(-1)[::17] = float("nan")
    return inp


def _input_fn(shape, dtype, device):
    inp = _make_input(shape, dtype, device)
    if len(shape) == 1:
        yield (inp,)
    else:
        yield inp, {"dim": -1}


@pytest.mark.nanmedian
def test_perf_nanmedian():
    bench = NanMedianBenchmark(
        input_fn=_input_fn,
        op_name="nanmedian",
        torch_op=torch.nanmedian,
        dtypes=NANMEDIAN_DTYPES,
    )

    bench.run()
