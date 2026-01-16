# SPECIAL_I0E operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.special_i0e import special_i0e as gems_special_i0e
from flag_gems.experimental_ops.special_i0e import (
    special_i0e_out as gems_special_i0e_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from benchmark.performance_utils import GenericBenchmark
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


@pytest.mark.special_i0e
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_special_i0e_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_i0e(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.special_i0e(ref_x)
    with flag_gems.use_gems():
        act_out = gems_special_i0e(x)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.special_i0e
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_special_i0e_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        out_ref = torch.empty_like(ref_x, dtype=torch.float32)
        ref_out = torch.ops.aten.special_i0e.out(ref_x.float(), out=out_ref)
        out_ref = out_ref.to(dtype)
        ref_out = out_ref
    else:
        out_ref = torch.empty_like(ref_x)
        ref_out = torch.ops.aten.special_i0e.out(ref_x, out=out_ref)
    out_act = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = gems_special_i0e_out(x, out_act)
    gems_assert_close(act_out, ref_out, dtype=dtype)
    gems_assert_close(out_act, out_ref, dtype=dtype)


@pytest.mark.special_i0e
def test_perf_aten_special_i0e():
    # Define input generation logic matching the operator arguments
    def special_i0e_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=special_i0e_input_fn,
        op_name="special_i0e",
        torch_op=torch.ops.aten.special_i0e,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
