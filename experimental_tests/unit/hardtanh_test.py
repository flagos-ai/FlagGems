# HARDTANH operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.hardtanh import hardtanh as gems_hardtanh  # noqa: E402
from flag_gems.experimental_ops.hardtanh import (  # noqa: E402
    hardtanh_out as gems_hardtanh_out,
)

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


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardtanh_tensor_default(shape, dtype):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = to_reference(x)
    ref_out = torch.ops.aten.hardtanh(ref_x)

    with flag_gems.use_gems():
        act_x = x.clone()
        act_out = gems_hardtanh(act_x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("min_max", [(-1.0, 1.0), (-0.5, 0.5), (0.0, 6.0), (-2.0, 0.5)])
def test_hardtanh_tensor_explicit(shape, dtype, min_max):
    min_val, max_val = min_max
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = to_reference(x)
    ref_out = torch.ops.aten.hardtanh(ref_x, min_val, max_val)

    with flag_gems.use_gems():
        act_x = x.clone()
        act_out = gems_hardtanh(act_x, min_val, max_val)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardtanh_out_default(shape, dtype):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = to_reference(x)
    ref_out_buf = torch.empty_like(ref_x)
    ref_res = torch.ops.aten.hardtanh.out(ref_x, out=ref_out_buf)

    with flag_gems.use_gems():
        act_x = x.clone()
        act_out_buf = torch.empty_like(act_x)
        act_res = gems_hardtanh_out(act_x, out=act_out_buf)

    gems_assert_close(act_out_buf, ref_out_buf, dtype=dtype)
    gems_assert_close(act_res, ref_res, dtype=dtype)


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("min_max", [(-1.0, 1.0), (-0.5, 0.5), (0.0, 6.0), (-2.0, 0.5)])
def test_hardtanh_out_explicit(shape, dtype, min_max):
    min_val, max_val = min_max
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = to_reference(x)
    ref_out_buf = torch.empty_like(ref_x)
    ref_res = torch.ops.aten.hardtanh.out(ref_x, min_val, max_val, out=ref_out_buf)

    with flag_gems.use_gems():
        act_x = x.clone()
        act_out_buf = torch.empty_like(act_x)
        act_res = gems_hardtanh_out(act_x, min_val, max_val, out=act_out_buf)

    gems_assert_close(act_out_buf, ref_out_buf, dtype=dtype)
    gems_assert_close(act_res, ref_res, dtype=dtype)
