# EXPAND_COPY operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.expand_copy import expand_copy as gems_expand_copy
from flag_gems.experimental_ops.expand_copy import (
    expand_copy_out as gems_expand_copy_out,
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


@pytest.mark.expand_copy
@pytest.mark.parametrize(
    "self_shape,size,implicit",
    [
        ((), (4, 5), False),
        ((1, 3), (2, 3), False),
        ((2, 1, 3), (2, 4, 3), False),
        ((128, 256), (128, -1), True),
        ((3, 4), (2, 3, 4), False),
        ((256, 1, 256), (256, 64, 256), True),
        ((1, 512, 512), (2, 512, 512), False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_expand_copy_default(self_shape, size, implicit, dtype):
    x = (
        torch.randn(self_shape, device=flag_gems.device, dtype=dtype)
        if len(self_shape) > 0
        else torch.randn((), device=flag_gems.device, dtype=dtype)
    )
    x_ref = to_reference(x)
    ref_out = torch.ops.aten.expand_copy(x_ref, list(size), implicit=implicit)
    with flag_gems.use_gems():
        act_out = gems_expand_copy(x, list(size), implicit=implicit)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.expand_copy
@pytest.mark.parametrize(
    "self_shape,size,implicit",
    [
        ((), (4, 5), False),
        ((1, 3), (2, 3), False),
        ((2, 1, 3), (2, 4, 3), True),
        ((128, 256), (128, -1), False),
        ((3, 4), (2, 3, 4), False),
        ((256, 1, 256), (256, 64, 256), True),
        ((1, 512, 512), (2, 512, 512), False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_expand_copy_out(self_shape, size, implicit, dtype):
    def compute_final_shape(src_shape, tgt_size):
        n_self = len(src_shape)
        n_tgt = len(tgt_size)
        s_aligned = (1,) * (n_tgt - n_self) + tuple(src_shape)
        final = []
        for i in range(n_tgt):
            final.append(s_aligned[i] if tgt_size[i] == -1 else tgt_size[i])
        return tuple(final)

    x = (
        torch.randn(self_shape, device=flag_gems.device, dtype=dtype)
        if len(self_shape) > 0
        else torch.randn((), device=flag_gems.device, dtype=dtype)
    )
    final_shape = compute_final_shape(self_shape, list(size))
    x_ref = to_reference(x)
    out_ref = torch.empty(final_shape, device=x_ref.device, dtype=dtype)
    out_act = torch.empty(final_shape, device=flag_gems.device, dtype=dtype)

    ref_out = torch.ops.aten.expand_copy.out(
        x_ref, list(size), implicit=implicit, out=out_ref
    )
    with flag_gems.use_gems():
        act_out = gems_expand_copy_out(x, list(size), implicit=implicit, out=out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.expand_copy
@pytest.mark.parametrize(
    "self_shape,size,implicit",
    [
        ((), (4, 5), False),
        ((1, 3), (2, 3), False),
        ((2, 1, 3), (2, 4, 3), False),
        ((128, 256), (128, -1), True),
        ((3, 4), (2, 3, 4), False),
        ((256, 1, 256), (256, 64, 256), True),
        ((1, 512, 512), (2, 512, 512), False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_expand_copy_default_performance(self_shape, size, implicit, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = (
        torch.randn(self_shape, device=flag_gems.device, dtype=dtype)
        if len(self_shape) > 0
        else torch.randn((), device=flag_gems.device, dtype=dtype)
    )
    x_ref = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.expand_copy(x_ref, list(size), implicit=implicit),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_expand_copy(x, list(size), implicit=implicit),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"expand_copy {self_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.expand_copy
@pytest.mark.parametrize(
    "self_shape,size,implicit",
    [
        ((), (4, 5), False),
        ((1, 3), (2, 3), False),
        ((2, 1, 3), (2, 4, 3), True),
        ((128, 256), (128, -1), False),
        ((3, 4), (2, 3, 4), False),
        ((256, 1, 256), (256, 64, 256), True),
        ((1, 512, 512), (2, 512, 512), False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_expand_copy_benchmark_out(self_shape, size, implicit, dtype):
    quantiles = [0.5, 0.2, 0.8]

    def compute_final_shape(src_shape, tgt_size):
        n_self = len(src_shape)
        n_tgt = len(tgt_size)
        s_aligned = (1,) * (n_tgt - n_self) + tuple(src_shape)
        final = []
        for i in range(n_tgt):
            final.append(s_aligned[i] if tgt_size[i] == -1 else tgt_size[i])
        return tuple(final)

    x = (
        torch.randn(self_shape, device=flag_gems.device, dtype=dtype)
        if len(self_shape) > 0
        else torch.randn((), device=flag_gems.device, dtype=dtype)
    )
    final_shape = compute_final_shape(self_shape, list(size))
    out_ref = torch.empty(final_shape, device=flag_gems.device, dtype=dtype)
    out_act = torch.empty(final_shape, device=flag_gems.device, dtype=dtype)

    x_ref = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.expand_copy.out(
            x_ref, list(size), implicit=implicit, out=out_ref
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_expand_copy_out(x, list(size), implicit=implicit, out=out_act),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"expand_copy {self_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
