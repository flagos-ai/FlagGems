# SLICE_SCATTER operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.slice_scatter import slice_scatter as gems_slice_scatter
from flag_gems.experimental_ops.slice_scatter import (
    slice_scatter_out as gems_slice_scatter_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp):
    """Convert tensor to reference device (CPU if TO_CPU is True)."""
    if TO_CPU:
        return inp.to("cpu")
    return inp.clone()


@pytest.mark.slice_scatter
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("mode", ["none", "front", "mid", "step2", "tail"])
def test_slice_scatter_tensor_2d(shape, dtype, dim, mode):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    L = shape[dim]
    if mode == "none":
        s = None
        e = None
        step = 1
    elif mode == "front":
        s = 0
        e = max(1, L // 2)
        step = 1
    elif mode == "mid":
        s = max(0, L // 3)
        e = max(s + 1, min(L, (2 * L) // 3))
        step = 1
    elif mode == "step2":
        s = 0
        e = None
        step = 2
    elif mode == "tail":
        s = max(0, L - max(1, L // 2))
        e = L
        step = 1
    else:
        s = None
        e = None
        step = 1
    s_eff = 0 if s is None else s
    e_eff = L if e is None else e
    slice_len = (e_eff - s_eff + step - 1) // step
    src_shape = list(shape)
    src_shape[dim] = slice_len
    src = torch.randn(tuple(src_shape), dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.slice_scatter(
        to_reference(x), to_reference(src), dim, s, e, step
    )
    with flag_gems.use_gems():
        act_out = gems_slice_scatter(x.clone(), src.clone(), dim, s, e, step)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.slice_scatter
@pytest.mark.parametrize("shape", [(3, 4, 5), (64, 128, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("mode", ["none", "front", "mid", "step2", "tail"])
def test_slice_scatter_tensor_3d(shape, dtype, dim, mode):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    L = shape[dim]
    if mode == "none":
        s = None
        e = None
        step = 1
    elif mode == "front":
        s = 0
        e = max(1, L // 2)
        step = 1
    elif mode == "mid":
        s = max(0, L // 3)
        e = max(s + 1, min(L, (2 * L) // 3))
        step = 1
    elif mode == "step2":
        s = 0
        e = None
        step = 2
    elif mode == "tail":
        s = max(0, L - max(1, L // 2))
        e = L
        step = 1
    else:
        s = None
        e = None
        step = 1
    s_eff = 0 if s is None else s
    e_eff = L if e is None else e
    slice_len = (e_eff - s_eff + step - 1) // step
    src_shape = list(shape)
    src_shape[dim] = slice_len
    src = torch.randn(tuple(src_shape), dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.slice_scatter(
        to_reference(x), to_reference(src), dim, s, e, step
    )
    with flag_gems.use_gems():
        act_out = gems_slice_scatter(x.clone(), src.clone(), dim, s, e, step)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.slice_scatter
@pytest.mark.parametrize("shape", [(2, 3), (256, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("mode", ["none", "front", "step2"])
def test_slice_scatter_out_2d(shape, dtype, dim, mode):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    L = shape[dim]
    if mode == "none":
        s = None
        e = None
        step = 1
    elif mode == "front":
        s = 0
        e = max(1, L // 2)
        step = 1
    elif mode == "step2":
        s = 0
        e = None
        step = 2
    else:
        s = None
        e = None
        step = 1
    s_eff = 0 if s is None else s
    e_eff = L if e is None else e
    slice_len = (e_eff - s_eff + step - 1) // step
    src_shape = list(shape)
    src_shape[dim] = slice_len
    src = torch.randn(tuple(src_shape), dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_src = to_reference(src)
    out_ref = torch.empty_like(ref_x)
    out_act = torch.empty_like(x)
    ref_out = torch.ops.aten.slice_scatter.out(
        ref_x, ref_src, dim, s, e, step, out=out_ref
    )
    with flag_gems.use_gems():
        act_out = gems_slice_scatter_out(
            x.clone(), src.clone(), dim, s, e, step, out_act
        )
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.slice_scatter
@pytest.mark.parametrize("shape", [(3, 4, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("mode", ["mid", "tail"])
def test_slice_scatter_out_3d(shape, dtype, dim, mode):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    L = shape[dim]
    if mode == "mid":
        s = max(0, L // 3)
        e = max(s + 1, min(L, (2 * L) // 3))
        step = 1
    elif mode == "tail":
        s = max(0, L - max(1, L // 2))
        e = L
        step = 1
    else:
        s = None
        e = None
        step = 1
    s_eff = 0 if s is None else s
    e_eff = L if e is None else e
    slice_len = (e_eff - s_eff + step - 1) // step
    src_shape = list(shape)
    src_shape[dim] = slice_len
    src = torch.randn(tuple(src_shape), dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_src = to_reference(src)
    out_ref = torch.empty_like(ref_x)
    out_act = torch.empty_like(x)
    ref_out = torch.ops.aten.slice_scatter.out(
        ref_x, ref_src, dim, s, e, step, out=out_ref
    )
    with flag_gems.use_gems():
        act_out = gems_slice_scatter_out(
            x.clone(), src.clone(), dim, s, e, step, out_act
        )
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.slice_scatter
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("mode", ["none", "front", "mid", "step2", "tail"])
def test_slice_scatter_tensor_2d_performance(shape, dtype, dim, mode):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    L = shape[dim]
    if mode == "none":
        s = None
        e = None
        step = 1
    elif mode == "front":
        s = 0
        e = max(1, L // 2)
        step = 1
    elif mode == "mid":
        s = max(0, L // 3)
        e = max(s + 1, min(L, (2 * L) // 3))
        step = 1
    elif mode == "step2":
        s = 0
        e = None
        step = 2
    elif mode == "tail":
        s = max(0, L - max(1, L // 2))
        e = L
        step = 1
    else:
        s = None
        e = None
        step = 1
    s_eff = 0 if s is None else s
    e_eff = L if e is None else e
    slice_len = (e_eff - s_eff + step - 1) // step
    src_shape = list(shape)
    src_shape[dim] = slice_len
    src = torch.randn(tuple(src_shape), dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.slice_scatter(x.clone(), src.clone(), dim, s, e, step),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_slice_scatter(x.clone(), src.clone(), dim, s, e, step),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"slice_scatter {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.slice_scatter
@pytest.mark.parametrize("shape", [(3, 4, 5), (64, 128, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("mode", ["none", "front", "mid", "step2", "tail"])
def test_slice_scatter_tensor_3d_performance(shape, dtype, dim, mode):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    L = shape[dim]
    if mode == "none":
        s = None
        e = None
        step = 1
    elif mode == "front":
        s = 0
        e = max(1, L // 2)
        step = 1
    elif mode == "mid":
        s = max(0, L // 3)
        e = max(s + 1, min(L, (2 * L) // 3))
        step = 1
    elif mode == "step2":
        s = 0
        e = None
        step = 2
    elif mode == "tail":
        s = max(0, L - max(1, L // 2))
        e = L
        step = 1
    else:
        s = None
        e = None
        step = 1
    s_eff = 0 if s is None else s
    e_eff = L if e is None else e
    slice_len = (e_eff - s_eff + step - 1) // step
    src_shape = list(shape)
    src_shape[dim] = slice_len
    src = torch.randn(tuple(src_shape), dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.slice_scatter(x.clone(), src.clone(), dim, s, e, step),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_slice_scatter(x.clone(), src.clone(), dim, s, e, step),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"slice_scatter {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.slice_scatter
@pytest.mark.parametrize("shape", [(2, 3), (256, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("mode", ["none", "front", "step2"])
def test_slice_scatter_out_2d_performance(shape, dtype, dim, mode):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    L = shape[dim]
    if mode == "none":
        s = None
        e = None
        step = 1
    elif mode == "front":
        s = 0
        e = max(1, L // 2)
        step = 1
    elif mode == "step2":
        s = 0
        e = None
        step = 2
    else:
        s = None
        e = None
        step = 1
    s_eff = 0 if s is None else s
    e_eff = L if e is None else e
    slice_len = (e_eff - s_eff + step - 1) // step
    src_shape = list(shape)
    src_shape[dim] = slice_len
    src = torch.randn(tuple(src_shape), dtype=dtype, device=flag_gems.device)

    out_ref = torch.empty_like(x)
    out_act = torch.empty_like(x)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.slice_scatter.out(
            x.clone(), src.clone(), dim, s, e, step, out=out_ref
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_slice_scatter_out(
                x.clone(), src.clone(), dim, s, e, step, out_act
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"slice_scatter {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.slice_scatter
@pytest.mark.parametrize("shape", [(3, 4, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("mode", ["mid", "tail"])
def test_slice_scatter_out_3d_performance(shape, dtype, dim, mode):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    L = shape[dim]
    if mode == "mid":
        s = max(0, L // 3)
        e = max(s + 1, min(L, (2 * L) // 3))
        step = 1
    elif mode == "tail":
        s = max(0, L - max(1, L // 2))
        e = L
        step = 1
    else:
        s = None
        e = None
        step = 1
    s_eff = 0 if s is None else s
    e_eff = L if e is None else e
    slice_len = (e_eff - s_eff + step - 1) // step
    src_shape = list(shape)
    src_shape[dim] = slice_len
    src = torch.randn(tuple(src_shape), dtype=dtype, device=flag_gems.device)

    out_ref = torch.empty_like(x)
    out_act = torch.empty_like(x)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.slice_scatter.out(
            x.clone(), src.clone(), dim, s, e, step, out=out_ref
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_slice_scatter_out(
                x.clone(), src.clone(), dim, s, e, step, out_act
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"slice_scatter {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
