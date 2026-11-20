"""Tsingmicro (TX81) specialized softmax.

The generic inner softmax (``softmax_kernel_inner``) launches one CTA per row
(grid == M). On TX81 each CTA carries a fixed ~0.5us cost (3 barriers, two
layout-conversion gather/scatters, two hardware reductions over 128 lanes, DMA
setup), so when N == 128 a row is only 256B and the kernel is per-CTA overhead
bound: it sustains ~1 GB/s regardless of M (measured: time == M * ~0.5us).

This override groups ``TILE_M`` rows per CTA (a 2D ``[TILE_M, TILE_N]`` tile with
the reduce along axis=1) so the fixed per-CTA cost is amortized over many rows.
Sweep-verified sweet spot is ~64K elements/CTA: TILE_M = clamp(65536 // TILE_N,
1, 256) (TILE_N == 32768 exempted at 128K, see heuristics). (131072,128) improves
~8x over the generic path (0.9 -> 7.1 GB/s), and wide-row shapes (N >= 1024) also
gain ~1.2-1.4x.

Two hard boundaries (both measured on TX81, see
``docs/softmax_tsingmicro_vs_generic_analysis.md``):
- ``N > 32768`` (multi-chunk, ``TILE_N`` clamped to 4096): the 2D loop-carried
  ``[TILE_M, 4096]`` m/z running tile is pathological on TX81 codegen (2D tile
  padded to ``[16,4096]`` by the hardware layout + 256KB gatherscatter per chunk,
  164~203x slower than generic at TILE_M=1), so the kernel's
  ``ONE_TILE_PER_CTA == False`` branch uses the generic 1D per-row implementation
  verbatim (one row/CTA, pointer-promoted, unmasked block DMA) and the grid
  switches to ``M`` CTAs.
- ``TILE_M * TILE_N`` must stay <= ~131072 elements (SPM ~3MB limit); larger
  tiles are OutOfResources at compile time.

The non-inner (dim != last) kernel is copied verbatim from the generic
implementation (it is not slow on TX81); the backward pass is left to the generic
implementation.
"""

import logging
import os

import torch
import triton
import triton.language as tl
from triton.backends.tsingmicro.precision import (
    restore_precision_mode,
    set_precision_mode,
)

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from .zeros import zero_

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b == 0
    return tl.cdiv(a, b) * b - b


# ---------------------------------------------------------------------------
# inner softmax (over the last dim): TILE_M rows per CTA
# ---------------------------------------------------------------------------
@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner_tsm"))
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets[None, :]
        mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
        inp = tl.load(input_ptr + offsets, mask=mask, other=float("-inf")).to(
            tl.float32
        )
        m = tl.max(inp, 1)
        e = tl.exp(inp - m[:, None])
        z = tl.sum(e, 1)
        out = (e / z[:, None]).to(output_ptr.dtype.element_ty)
        tl.store(output_ptr + offsets, out, mask=mask)
    else:
        # Multi-chunk (N > 32768, TILE_N clamped to 4096). The 2D row-grouped
        # loop-carried [TILE_M, TILE_N] m/z running tiles are pathological on
        # TX81 codegen (2D tile padded to [16,TILE_N] by the hardware layout,
        # 256KB gatherscatter layout conversion per chunk, SPM-resident single
        # CTA), 14~203x slower than the generic path. Use the generic 1D
        # per-row implementation verbatim: one row per CTA, pointer-promoted,
        # unmasked block-DMA loads. The grid adapts via ONE_TILE_PER_CTA
        # (M CTAs here, cdiv(M, TILE_M) for the single-tile branch).
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            # it is possible that there are -inf's in the input
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        # specialize the last (possibly partial) chunk
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=float("-inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        # store pass, walk chunks in reverse with evict_first
        previous_multiple = prev_multiple_of(N, TILE_N)
        # specialize the first (last-chunk) iteration
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=float("-inf"),
                eviction_policy="evict_first",
            )
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)


# ---------------------------------------------------------------------------
# non-inner softmax (dim != last): copied verbatim from src/flag_gems/ops/softmax.py
# ---------------------------------------------------------------------------
@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_k = tle.program_id(1)
    pid_m = tle.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        # specialization does not improve performance inn this example, as tested
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
        m = m_reduced

        # specialization does not improve performance inn this example, as tested
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)


def softmax_impl(self, dim, half_to_float=False, out=None):
    # Must in [-self.ndim, self.ndim)
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        if out is None:
            dtype = torch.float32 if half_to_float else self.dtype
            out = torch.empty(list(self.shape), dtype=dtype, device=self.device)
        else:
            if tuple(out.shape) != tuple(self.shape):
                out.resize_(self.shape)
        zero_(out)
        return out

    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    self = self.contiguous()
    dtype = torch.float32 if half_to_float else self.dtype
    if out is None:
        out = torch.empty_like(self, dtype=dtype)
    else:
        if tuple(out.shape) != tuple(self.shape):
            out.resize_(self.shape)
        if out.dtype != dtype:
            raise RuntimeError(
                f"_softmax.out: expected out dtype {dtype}, got {out.dtype}"
            )
    K = self.numel() // M // N

    with torch_device_fn.device(self.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            softmax_kernel_non_inner[grid](
                out,
                self,
                M,
                N,
                K,
            )
        else:
            # inner softmax: grid adapts per branch. Multi-chunk
            # (ONE_TILE_PER_CTA=False, N>32768) runs one row per CTA (generic 1D
            # structure inside the kernel); single-tile groups TILE_M rows/CTA.
            grid = lambda meta: (
                M if not meta["ONE_TILE_PER_CTA"] else triton.cdiv(M, meta["TILE_M"]),
                1,
                1,
            )
            softmax_kernel_inner[grid](
                out,
                self,
                M,
                N,
            )
    return out


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_TSINGMICRO SOFTMAX")

    user_precision_priority = os.environ.get("TSM_SOFTMAX_PRECISION_MODE", None)
    # force to default mode unless the user explicitly opts into another mode
    precision_token = set_precision_mode(user_precision_priority or "0")

    try:
        return softmax_impl(self, dim, half_to_float)

    finally:
        restore_precision_mode(precision_token)


def softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_TSINGMICRO SOFTMAX_OUT")

    user_precision_priority = os.environ.get("TSM_SOFTMAX_PRECISION_MODE", None)
    # force to default mode unless the user explicitly opts into another mode
    precision_token = set_precision_mode(user_precision_priority or "0")

    try:
        return softmax_impl(self, dim, half_to_float, out=out)
    finally:
        restore_precision_mode(precision_token)
