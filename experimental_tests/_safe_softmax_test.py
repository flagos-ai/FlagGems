# _SAFE_SOFTMAX operator test

import os
import sys

import pytest
import torch
import triton

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


def to_reference(inp):
    """Move to CPU when TO_CPU is set, keep dtype/device otherwise."""
    if inp is None:
        return None
    return inp.to("cpu") if TO_CPU else inp.clone()


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


@pytest.mark.safe_softmax
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize(
    "dtype_arg_sel", ["none", "same", torch.float32, torch.float16, torch.bfloat16]
)
def test__safe_softmax_benchmark_tensor(shape, in_dtype, dim, dtype_arg_sel):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=in_dtype, device=flag_gems.device)

    if dtype_arg_sel == "none":
        dtype_arg = None
    elif dtype_arg_sel == "same":
        dtype_arg = in_dtype
    else:
        dtype_arg = dtype_arg_sel

    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._safe_softmax(ref_x, dim, dtype=dtype_arg),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__safe_softmax(x, dim, dtype=dtype_arg),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_safe_softmax {shape} {dtype_arg_sel}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
