# REPLICATION_PAD3D operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.replication_pad3d import (  # noqa: E402
    replication_pad3d as gems_replication_pad3d,
)
from flag_gems.experimental_ops.replication_pad3d import (  # noqa: E402
    replication_pad3d_out as gems_replication_pad3d_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


@pytest.mark.replication_pad3d
@pytest.mark.parametrize(
    "shape", [(1, 2, 4, 5, 6), (2, 4, 16, 32, 32), (2, 4, 32, 64, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding", [(0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1), (2, 0, 1, 2, 0, 1)]
)
def test_replication_pad3d_tensor(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    ref_out = torch.ops.aten.replication_pad3d(ref_x, padding)
    with flag_gems.use_gems():
        act_out = gems_replication_pad3d(x, padding)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.replication_pad3d
@pytest.mark.parametrize(
    "shape", [(1, 2, 4, 5, 6), (2, 4, 16, 32, 32), (2, 4, 32, 64, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding", [(0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1), (2, 0, 1, 2, 0, 1)]
)
def test_replication_pad3d_out(shape, dtype, padding):
    def test__out_shape(s, p):
        return (s[0], s[1], s[2] + p[4] + p[5], s[3] + p[2] + p[3], s[4] + p[0] + p[1])

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    out_shape = test__out_shape(shape, padding)
    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=ref_x.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.replication_pad3d.out(ref_x, padding, out=ref_out_buf)
    with flag_gems.use_gems():
        act_ret = gems_replication_pad3d_out(x, padding, act_out_buf)

    gems_assert_close(act_ret, ref_ret, dtype=dtype)
