import pytest
import torch

from . import base, consts


def _standard_gamma_grad_input_fn(shape, dtype, device):
    # Both alpha (self_grad) and output must be positive for a Gamma distribution.
    self_grad = torch.rand(shape, dtype=dtype, device=device) * 4.0 + 0.5
    output = torch.rand(shape, dtype=dtype, device=device) * 4.0 + 0.5
    yield self_grad, output


@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "torch_gcu has no PrivateUse1 kernel for aten::_standard_gamma_grad; its "
        "CPU fallback only implements float32/float64, so half and bfloat16 have "
        "no native baseline"
    ),
)
@pytest.mark.standard_gamma_grad
def test_standard_gamma_grad():
    bench = base.GenericBenchmark(
        input_fn=_standard_gamma_grad_input_fn,
        op_name="standard_gamma_grad",
        torch_op=torch._standard_gamma_grad,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
