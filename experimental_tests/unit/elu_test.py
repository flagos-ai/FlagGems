# ELU operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.elu import elu as gems_elu  # noqa: E402
from flag_gems.experimental_ops.elu import elu_out as gems_elu_out  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.elu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("params", [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.5, 2.0, 0.5)])
def test_elu_tensor(shape, dtype, params):
    alpha, scale, input_scale = params
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(base)
    act_input = base.clone()

    ref_out = torch.ops.aten.elu(ref_input, alpha, scale, input_scale)

    with flag_gems.use_gems():
        act_out = gems_elu(act_input, alpha, scale, input_scale)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.elu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("params", [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.5, 2.0, 0.5)])
def test_elu_out(shape, dtype, params):
    alpha, scale, input_scale = params
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(base)
    act_input = base.clone()

    ref_outbuf = torch.empty_like(ref_input)
    ref_res = torch.ops.aten.elu.out(
        ref_input, alpha, scale, input_scale, out=ref_outbuf
    )

    act_outbuf = torch.empty_like(act_input)
    with flag_gems.use_gems():
        act_res = gems_elu_out(act_input, alpha, scale, input_scale, act_outbuf)

    gems_assert_close(act_res, ref_res, dtype=dtype)
