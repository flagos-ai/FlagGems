import pytest
import torch

from . import base, consts


@pytest.mark.addmm_
def test_addmm_(monkeypatch):
    def _input_fn(b, m, n, k, dtype, device, b_column_major):
        inp1 = torch.randn([m, k], dtype=dtype, device=device)
        bias = torch.randn([m, n], dtype=dtype, device=device)
        if b_column_major:
            inp2 = torch.randn([n, k], dtype=dtype, device=device)
            yield bias, inp1, inp2.t(),
        else:
            inp2 = torch.randn([k, n], dtype=dtype, device=device)
            yield bias, inp1, inp2,

    bench = base.BlasBenchmark(
        op_name="addmm_",
        input_fn=_input_fn,
        torch_op=lambda bias, mat1, mat2: bias.addmm_(mat1, mat2),
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
