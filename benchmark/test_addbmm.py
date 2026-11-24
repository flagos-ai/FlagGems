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

    # addbmm has bias shape [m, n] (not batched like baddbmm)
    bias = torch.randn([m, n], dtype=dtype, device=device, requires_grad=True)

    yield bias, inp1, inp2


@pytest.mark.addbmm
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "torch_gcu has no PrivateUse1 kernel for aten::addbmm; the CPU fallback "
        "is about four orders of magnitude slower than the device path, so the "
        "native baseline cannot be timed"
    ),
)
def test_addbmm():
    bench = base.BlasBenchmark(
        op_name="addbmm",
        input_fn=_input_fn,
        torch_op=torch.addbmm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
