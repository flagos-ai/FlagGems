# MV operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.mv import mv as gems_mv  # noqa: E402
from flag_gems.experimental_ops.mv import mv_out as gems_mv_out  # noqa: E402

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


@pytest.mark.mv
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mv_tensor(shape, dtype):
    m, n = shape
    mat = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((n,), dtype=dtype, device=flag_gems.device)

    ref_mat = to_reference(mat)
    ref_vec = to_reference(vec)
    ref_out = torch.ops.aten.mv(ref_mat, ref_vec)

    with flag_gems.use_gems():
        act_out = gems_mv(mat, vec)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.mv
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mv_out_tensor(shape, dtype):
    m, n = shape
    mat = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((n,), dtype=dtype, device=flag_gems.device)

    ref_mat = to_reference(mat)
    ref_vec = to_reference(vec)
    ref_out_buf = torch.empty((m,), dtype=dtype, device=ref_mat.device)
    ref_out = torch.ops.aten.mv.out(ref_mat, ref_vec, out=ref_out_buf)

    act_out_buf = torch.empty((m,), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems_mv_out(mat, vec, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)
