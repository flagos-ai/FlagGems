# MAXIMUM operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.maximum import maximum as gems_maximum  # noqa: E402
from flag_gems.experimental_ops.maximum import (  # noqa: E402
    maximum_out as gems_maximum_out,
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


@pytest.mark.maximum
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_maximum_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_y = to_reference(y)

    ref_out = torch.ops.aten.maximum(ref_x, ref_y)

    with flag_gems.use_gems():
        act_out = gems_maximum(x, y)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.maximum
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_maximum_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_y = to_reference(y)

    ref_out_buf = torch.empty(shape, dtype=dtype, device=ref_x.device)
    act_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.maximum.out(ref_x, ref_y, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = gems_maximum_out(x, y, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.maximum
@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 3, 1), (1, 3, 4)),
        ((8, 1, 16), (1, 12, 16)),
        ((64, 1, 256), (1, 128, 1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_maximum_tensor_broadcast(shapes, dtype):
    shape_a, shape_b = shapes
    x = torch.randn(shape_a, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape_b, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_y = to_reference(y)

    ref_out = torch.ops.aten.maximum(ref_x, ref_y)

    with flag_gems.use_gems():
        act_out = gems_maximum(x, y)

    gems_assert_close(act_out, ref_out, dtype=dtype)
