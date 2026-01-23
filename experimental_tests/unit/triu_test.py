# TRIU operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.triu import triu as gems_triu  # noqa: E402
from flag_gems.experimental_ops.triu import triu_out as gems_triu_out  # noqa: E402

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
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


@pytest.mark.triu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 16, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-1, 0, 1, 3])
def test_triu_tensor(shape, dtype, diagonal):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    ref_out = torch.ops.aten.triu(ref_x, diagonal)

    with flag_gems.use_gems():
        act_out = gems_triu(x, diagonal)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.triu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 16, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-1, 0, 1, 3])
def test_triu_out(shape, dtype, diagonal):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out_buf = torch.empty_like(ref_x)

    ref_out = torch.ops.aten.triu.out(ref_x, diagonal, out=ref_out_buf)

    act_out_buf = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = gems_triu_out(x, diagonal, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)
