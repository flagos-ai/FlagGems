# EYE operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.eye import eye as gems_eye
from flag_gems.experimental_ops.eye import eye_m_out as gems_eye_m_out
from flag_gems.experimental_ops.eye import eye_out as gems_eye_out

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


@pytest.mark.eye
@pytest.mark.parametrize("n", [2, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_base(n, dtype):
    ref_device = "cpu" if TO_CPU else flag_gems.device
    ref_out = torch.ops.aten.eye(n, dtype=dtype, device=ref_device)
    with flag_gems.use_gems():
        act_out = gems_eye(n, dtype=dtype, device=flag_gems.device)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.eye
@pytest.mark.parametrize("nm", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_m(nm, dtype):
    n, m = nm
    ref_device = "cpu" if TO_CPU else flag_gems.device
    ref_out = torch.ops.aten.eye(n, m, dtype=dtype, device=ref_device)
    with flag_gems.use_gems():
        act_out = gems_eye(n, m, dtype=dtype, device=flag_gems.device)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.eye
@pytest.mark.parametrize("n", [2, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_out(n, dtype):
    ref_device = "cpu" if TO_CPU else flag_gems.device
    ref_out_tensor = torch.empty((n, n), dtype=dtype, device=ref_device)
    act_out_tensor = torch.empty((n, n), dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.eye(n, out=ref_out_tensor)
    with flag_gems.use_gems():
        act_out = gems_eye_out(n, act_out_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.eye
@pytest.mark.parametrize("nm", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_m_out(nm, dtype):
    n, m = nm
    ref_device = "cpu" if TO_CPU else flag_gems.device
    ref_out_tensor = torch.empty((n, m), dtype=dtype, device=ref_device)
    act_out_tensor = torch.empty((n, m), dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.eye(n, m, out=ref_out_tensor)
    with flag_gems.use_gems():
        act_out = gems_eye_m_out(n, m, act_out_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)
