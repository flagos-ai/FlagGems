import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(),
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2,


@pytest.mark.addmm
def test_addmm(monkeypatch):
    if flag_gems.vendor_name == "mthreads":
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    bench = base.BlasBenchmark(
        op_name="addmm",
        input_fn=_input_fn,
        torch_op=torch.addmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
