# EYE operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.eye import eye as gems_eye
from flag_gems.experimental_ops.eye import eye_m_out as gems_eye_m_out
from flag_gems.experimental_ops.eye import eye_out as gems_eye_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
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


@pytest.mark.eye
@pytest.mark.parametrize("n", [2, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_base(n, dtype):
    ref_out = torch.ops.aten.eye(n, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems_eye(n, dtype=dtype, device=flag_gems.device)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.eye
@pytest.mark.parametrize("nm", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_m(nm, dtype):
    n, m = nm
    ref_out = torch.ops.aten.eye(n, m, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems_eye(n, m, dtype=dtype, device=flag_gems.device)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.eye
@pytest.mark.parametrize("n", [2, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_out(n, dtype):
    base_out = torch.empty((n, n), dtype=dtype, device=flag_gems.device)
    ref_out_tensor = to_reference(base_out)
    act_out_tensor = base_out.clone()

    ref_out = torch.ops.aten.eye(n, out=ref_out_tensor)
    with flag_gems.use_gems():
        act_out = gems_eye_out(n, act_out_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.eye
@pytest.mark.parametrize("nm", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_m_out(nm, dtype):
    n, m = nm
    base_out = torch.empty((n, m), dtype=dtype, device=flag_gems.device)
    ref_out_tensor = to_reference(base_out)
    act_out_tensor = base_out.clone()

    ref_out = torch.ops.aten.eye(n, m, out=ref_out_tensor)
    with flag_gems.use_gems():
        act_out = gems_eye_m_out(n, m, act_out_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.eye
@pytest.mark.parametrize("n", [2, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_base_performance(n, dtype):
    quantiles = [0.5, 0.2, 0.8]

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.eye(n, dtype=dtype, device=flag_gems.device),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_eye(n, dtype=dtype, device=flag_gems.device),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"eye {n} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.eye
@pytest.mark.parametrize("nm", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_m_performance(nm, dtype):
    quantiles = [0.5, 0.2, 0.8]

    n, m = nm
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.eye(n, m, dtype=dtype, device=flag_gems.device),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_eye(n, m, dtype=dtype, device=flag_gems.device),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"eye {nm} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.eye
@pytest.mark.parametrize("n", [2, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_benchmark_out(n, dtype):
    quantiles = [0.5, 0.2, 0.8]

    base_out = torch.empty((n, n), dtype=dtype, device=flag_gems.device)
    ref_out_tensor = base_out.clone()
    act_out_tensor = base_out.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.eye(n, out=ref_out_tensor), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_eye_out(n, act_out_tensor), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"eye {n} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.eye
@pytest.mark.parametrize("nm", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eye_m_benchmark_out(nm, dtype):
    quantiles = [0.5, 0.2, 0.8]

    n, m = nm
    base_out = torch.empty((n, m), dtype=dtype, device=flag_gems.device)
    ref_out_tensor = base_out.clone()
    act_out_tensor = base_out.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.eye(n, m, out=ref_out_tensor),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_eye_m_out(n, m, act_out_tensor), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"eye {nm} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
