import pytest
import torch

from . import base, consts


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device, requires_grad=True)

    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device, requires_grad=True)
        inp2 = inp2.transpose(1, 2).contiguous()
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device, requires_grad=True)

    # addbmm_ is inplace, bias must not require grad (leaf tensor constraint)
    bias = torch.randn([m, n], dtype=dtype, device=device, requires_grad=False)

    yield bias, inp1, inp2


@pytest.mark.addbmm_
def test_addbmm_():
    bench = base.BlasBenchmark(
        op_name="addbmm_",
        input_fn=_input_fn,
        torch_op=lambda bias, inp1, inp2: bias.addbmm_(inp1, inp2),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
