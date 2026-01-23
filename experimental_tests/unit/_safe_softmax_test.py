# _SAFE_SOFTMAX operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops._safe_softmax import _safe_softmax as gems__safe_softmax

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


@pytest.mark.safe_softmax
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize(
    "dtype_arg_sel", ["none", "same", torch.float32, torch.float16, torch.bfloat16]
)
def test__safe_softmax_tensor(shape, in_dtype, dim, dtype_arg_sel):
    x = torch.randn(shape, dtype=in_dtype, device=flag_gems.device)

    if dtype_arg_sel == "none":
        dtype_arg = None
    elif dtype_arg_sel == "same":
        dtype_arg = in_dtype
    else:
        dtype_arg = dtype_arg_sel

    ref_x = to_reference(x)

    # Use higher-precision reference for low-precision outputs, then cast back
    if dtype_arg in (torch.float16, torch.bfloat16):
        ref_x = ref_x.float()
        ref_out = torch.ops.aten._safe_softmax(ref_x, dim, dtype=torch.float32)
        ref_out = ref_out.to(dtype_arg)
    else:
        ref_out = torch.ops.aten._safe_softmax(ref_x, dim, dtype=dtype_arg)

    with flag_gems.use_gems():
        act_out = gems__safe_softmax(x, dim, dtype=dtype_arg)

    expected_dtype = dtype_arg if dtype_arg is not None else in_dtype
    gems_assert_close(act_out, ref_out, dtype=expected_dtype)
