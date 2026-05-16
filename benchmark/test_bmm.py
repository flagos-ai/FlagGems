import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device)
        yield inp1, inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device)
        yield inp1, inp2


@pytest.mark.bmm
def test_bmm(monkeypatch):
    if flag_gems.vendor_name == "mthreads":
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    bench = base.BlasBenchmark(
        op_name="bmm",
        input_fn=_input_fn,
        torch_op=torch.bmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device)
        inp2 = inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device)
    out = torch.empty([b, m, n], dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.bmm_out
def test_bmm_out(monkeypatch):
    if flag_gems.vendor_name == "mthreads":
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    bench = base.BlasBenchmark(
        op_name="bmm_out",
        input_fn=_input_fn_out,
        torch_op=torch.bmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
