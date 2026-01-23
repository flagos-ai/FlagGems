# TRACE operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.trace import trace as gems_trace
from flag_gems.experimental_ops.trace import trace_out as gems_trace_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison aligned with flag_gems.testing.assert_close
        from flag_gems.testing import assert_close as fg_assert_close  # noqa: E402

        kwargs = dict(kwargs)
        reduce_dim = kwargs.pop("reduce_dim", 1)
        equal_nan = kwargs.pop("equal_nan", False)
        fg_assert_close(res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim)


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


@pytest.mark.trace
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_trace_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x, upcast=True)

    ref_out = torch.ops.aten.trace(ref_x)

    with flag_gems.use_gems():
        act_out = gems_trace(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.trace
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_trace_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x, upcast=True)

    ref_out_tensor = torch.empty([], dtype=ref_x.dtype, device=ref_x.device)
    act_out_tensor = torch.empty([], dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.trace.out(ref_x, out=ref_out_tensor)

    with flag_gems.use_gems():
        act_out = gems_trace_out(x, act_out_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)
