# REFLECTION_PAD2D operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.reflection_pad2d import (
    reflection_pad2d as gems_reflection_pad2d,
)
from flag_gems.experimental_ops.reflection_pad2d import (
    reflection_pad2d_out as gems_reflection_pad2d_out,
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


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize(
    "shape", [(3, 33, 33), (2, 4, 32, 64), (8, 16, 64, 64), (32, 64, 128, 256)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1),
        (2, 3, 2, 3),
        (3, 5, 3, 5),
        (0, 4, 0, 4),
        (4, 0, 4, 0),
    ],
)
def test_reflection_pad2d_accuracy(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = gems_reflection_pad2d(x, padding)

    gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize(
    "shape", [(3, 33, 33), (2, 4, 32, 64), (8, 16, 64, 64), (32, 64, 128, 256)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1),
        (2, 3, 2, 3),
        (3, 5, 3, 5),
        (0, 4, 0, 4),
        (4, 0, 4, 0),
    ],
)
def test_reflection_pad2d_out_accuracy(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    out_shape = list(shape)
    out_shape[-2] = out_shape[-2] + padding[2] + padding[3]
    out_shape[-1] = out_shape[-1] + padding[0] + padding[1]
    out_shape = tuple(out_shape)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x.clone())
    ref_out_buf_ref = to_reference(ref_out_buf)

    with flag_gems.use_gems():
        act_out = gems_reflection_pad2d_out(x, padding, act_out_buf)

    ref_out = torch.ops.aten.reflection_pad2d.out(ref_x, padding, out=ref_out_buf_ref)

    gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize("padding", [[1, 1, 1, 1], [2, 3, 4, 5]])
def test_reflection_pad2d_list_padding(padding):
    # Test with list format: [pad_left, pad_right, pad_top, pad_bottom]
    shape = (2, 4, 32, 64)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = gems_reflection_pad2d(x, padding)

    gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d
def test_reflection_pad2d_empty_padding():
    shape = (2, 4, 32, 64)
    dtype = torch.float32
    padding = (0, 0, 0, 0)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = gems_reflection_pad2d(x, padding)

    gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize("padding", [(1, 1, 1, 1), (2, 3, 4, 5)])
def test_reflection_pad2d_3d_input(padding):
    # Test with 3D input (C, H, W) - no batch dimension
    shape = (3, 32, 64)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = gems_reflection_pad2d(x, padding)

    gems_assert_close(act_out, ref_out, dtype, equal_nan=True)
