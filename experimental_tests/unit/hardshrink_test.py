# HARDSHRINK operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.hardshrink import (  # noqa: E402
    hardshrink as gems_hardshrink,
)
from flag_gems.experimental_ops.hardshrink import (  # noqa: E402
    hardshrink_out as gems_hardshrink_out,
)

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


@pytest.mark.hardshrink
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lambd", [0.0, 0.5, 1.0])
def test_hardshrink_tensor(shape, dtype, lambd):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    ref_out = torch.ops.aten.hardshrink(ref_x, lambd)

    with flag_gems.use_gems():
        act_out = gems_hardshrink(x, lambd)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hardshrink
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lambd", [0.0, 0.75, 1.5])
def test_hardshrink_out(shape, dtype, lambd):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    out_ref = torch.empty_like(ref_x)
    out_act = torch.empty_like(x)

    ref_out = torch.ops.aten.hardshrink.out(ref_x, lambd, out=out_ref)

    with flag_gems.use_gems():
        act_out = gems_hardshrink_out(x, lambd, out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)
