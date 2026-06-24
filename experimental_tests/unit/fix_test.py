# FIX operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.fix import fix as gems_fix  # noqa: E402
from flag_gems.experimental_ops.fix import fix_out as gems_fix_out  # noqa: E402

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


@pytest.mark.fix
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fix_tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 3.5

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.fix(ref_input)

    with flag_gems.use_gems():
        act_out = gems_fix(input_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.fix
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fix_out_tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 3.5

    ref_input = to_reference(input_tensor)
    act_input = input_tensor.clone()

    ref_out_buf = torch.empty_like(ref_input, device=ref_input.device)
    act_out_buf = torch.empty_like(act_input, device=flag_gems.device)

    ref_ret = torch.ops.aten.fix.out(ref_input, out=ref_out_buf)

    with flag_gems.use_gems():
        act_ret = gems_fix_out(act_input, act_out_buf)

    gems_assert_close(act_ret, ref_ret, dtype=dtype)
