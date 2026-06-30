import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.gcd
def test_gcd():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gcd",
        torch_op=torch.gcd,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


def gcd_out_input_fn(shape, dtype, device):
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        inp1 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(device)
        inp2 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(device)
    else:
        inp1 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device=device,
        )
        inp2 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device=device,
        )
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.gcd_out
def test_gcd_out():
    bench = base.GenericBenchmark(
        op_name="gcd_out",
        torch_op=torch.gcd,
        dtypes=consts.INT_DTYPES,
        input_fn=gcd_out_input_fn,
    )
    bench.run()
