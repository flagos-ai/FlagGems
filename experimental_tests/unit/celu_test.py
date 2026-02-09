# CELU operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.celu import celu as gems_celu
from flag_gems.experimental_ops.celu import celu_out as gems_celu_out

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


@pytest.mark.celu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_celu_tensor_default_alpha(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = torch.ops.aten.celu(ref_x)
    with flag_gems.use_gems():
        act_out = gems_celu(x)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.celu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_celu_tensor_alpha(shape, dtype, alpha):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = torch.ops.aten.celu(ref_x, alpha)
    with flag_gems.use_gems():
        act_out = gems_celu(x, alpha)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.celu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_celu_out(shape, dtype, alpha):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    out_ref = torch.empty_like(ref_x)
    ref_out = torch.ops.aten.celu.out(ref_x, alpha, out=out_ref)
    out_act = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = gems_celu_out(x, alpha, out_act)
    gems_assert_close(act_out, ref_out, dtype=dtype)
