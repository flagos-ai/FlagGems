# MV operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.mv import mv as gems_mv
from flag_gems.experimental_ops.mv import mv_out as gems_mv_out


@pytest.mark.mv
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mv_tensor(shape, dtype):
    m, n = shape
    mat = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((n,), dtype=dtype, device=flag_gems.device)

    ref_mat = mat.clone()
    ref_vec = vec.clone()
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

    ref_mat = mat.clone()
    ref_vec = vec.clone()
    ref_out_buf = torch.empty((m,), dtype=dtype, device=flag_gems.device)
    ref_out = torch.ops.aten.mv.out(ref_mat, ref_vec, out=ref_out_buf)

    act_out_buf = torch.empty((m,), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems_mv_out(mat, vec, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.mv
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mv_benchmark_tensor(shape, dtype):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    m, n = shape
    mat = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((n,), dtype=dtype, device=flag_gems.device)

    ref_mat = mat.clone()
    ref_vec = vec.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.mv(ref_mat, ref_vec), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_mv(mat, vec), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"mv {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.mv
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mv_out_benchmark_tensor(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    m, n = shape
    mat = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((n,), dtype=dtype, device=flag_gems.device)

    ref_mat = mat.clone()
    ref_vec = vec.clone()
    ref_out_buf = torch.empty((m,), dtype=dtype, device=flag_gems.device)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.mv.out(ref_mat, ref_vec, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )

    act_out_buf = torch.empty((m,), dtype=dtype, device=flag_gems.device)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_mv_out(mat, vec, act_out_buf), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"mv {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
