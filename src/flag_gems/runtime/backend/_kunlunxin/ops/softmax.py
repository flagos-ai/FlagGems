# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cdiv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        # Pre-offset the base pointers so the inner `ptr + n_offsets` access is a
        # scalar-base + stride-1 arange that OffsetAnalysis proves contiguous
        # (block DMA). The old inline `pid_m * N + n_offsets` addressing blocked
        # the analysis -> discrete scalar gather (~1-3 GB/s, e.g. [4096,4096] took
        # ~37ms). Pre-offsetting drops it to ~1.1ms (~35x).
        input_ptr += pid_m * N
        output_ptr += pid_m * N
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            output_ptr.dtype.element_ty
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        tl.store(output_ptr + n_offsets, out, mask=mask)
    else:
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
        # specialize the last iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        # Normalize pass. Iterate ASCENDING so each `input_ptr + n_offsets` load
        # and `output_ptr + n_offsets` store is a scalar-base + stride-1 arange
        # (block DMA). The old code walked the tiles DESCENDING
        # (`previous_multiple - start_n`) as a cache-locality trick, but on this
        # XPU the backward walk defeats OffsetAnalysis/prefetch -> discrete access
        # (~1-3 GB/s: [1024,65536] took ~154ms). Ascending drops it to ~4ms (~35x).
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)


# ------------------------  backward -------------------------------


def softmax_backward_kernel_inner_heru_tile_n(args):
    N = args["N"]
    if N <= 32768:
        return triton.next_power_of_2(N)
    return 4096


def softmax_backward_kernel_inner_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


@libentry()
@triton.heuristics(
    values={
        "TILE_N": softmax_backward_kernel_inner_heru_tile_n,
        "ONE_TILE_PER_CTA": softmax_backward_kernel_inner_heur_one_tile_per_cta,
    },
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # One program per row (grid=(M,)), mirroring the forward. Pre-offset the base
    # pointers so the inner `ptr + n_offsets` access is a scalar-base + stride-1
    # arange that OffsetAnalysis proves contiguous (block DMA). The old impl used a
    # fixed grid=(12,) with a [TILE_M, TILE_N] tile whose `m_offsets[:,None]*N +
    # n_offsets` addressing blocked the analysis -> discrete scalar gather
    # (~1-3 GB/s: [4096,4096] took ~38ms). It also computed in float64 (2x traffic,
    # unnecessary). float32 accumulation matches the forward and the generic backend.
    pid_m = ext.program_id(0)
    out_ptr += pid_m * N
    out_grad_ptr += pid_m * N
    in_grad_ptr += pid_m * N
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.sum(out_tile * out_grad_tile, 0)
        in_grad_tile = out_tile * (out_grad_tile - scale)
        tl.store(in_grad_ptr + n_offsets, in_grad_tile, mask=mask)
    else:
        # Pass 1: accumulate scale = sum(out * out_grad) over the row. Iterate
        # ASCENDING so each load is a scalar-base + stride-1 arange (block DMA).
        scale = tl.zeros([TILE_N], dtype=tl.float32)
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            out_tile = tl.load(out_ptr + n_offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale += out_tile * out_grad_tile
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            scale += out_tile * out_grad_tile
        scale = tl.sum(scale, 0)  # scalar

        # Pass 2: write in_grad = out * (out_grad - scale), ASCENDING.
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            out_tile = tl.load(out_ptr + n_offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + n_offsets, in_grad_tile)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + n_offsets, in_grad_tile, mask=mask)


# --- Multi-row 2D-tile backward kernel (launch-bound huge-rows / small-N only) ----
# Same recipe as rms_norm_multirow_kernel: when the row count is large but N is
# tiny, the per-row kernel launches `rows` tiny programs and per-program launch
# latency (~0.6-0.9us) dominates (e.g. K>1 backward of [64,256,64] -> 4096 rows of
# 256 elements -> ~1.6ms). Each program here owns a [TILE_M, N] tile of TILE_M
# consecutive rows (the whole softmax dim as ONE contiguous column block) and
# reduces along axis=1, cutting the grid to cdiv(rows, TILE_M). N is a constexpr
# and the columns span exactly [0, N) with NO power-of-2 padding, so the tile is
# one stride-1 contiguous block (block DMA); padded/runtime-N addressing forces
# discrete access (~2x slower, HARNESS 1.3). Gated to power-of-2 N <= 256 and
# rows >= MULTIROW_M; otherwise keep the per-row kernel.
MULTIROW_N = 256
MULTIROW_M = 512
TILE_BUDGET = 4096  # rows*cols per 2D tile; two input tiles live in registers


@libentry()
@triton.jit
def softmax_backward_multirow_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = ext.program_id(0)
    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_off < M
    n_off = tl.arange(0, N)
    offs = m_off[:, None] * N + n_off[None, :]
    # Only rows are masked. Out-of-range rows may load garbage (XPU ignores
    # `other=`) but the axis=1 reduction is per-row and independent, and their
    # store is masked out, so they never affect valid rows or the output.
    out_tile = tl.load(out_ptr + offs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    out_grad_tile = tl.load(out_grad_ptr + offs, mask=m_mask[:, None], other=0.0).to(
        tl.float32
    )
    scale = tl.sum(out_tile * out_grad_tile, axis=1)
    in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
    tl.store(in_grad_ptr + offs, in_grad_tile, mask=m_mask[:, None])


def softmax_backward_kernel_inner_heur_multirow(args):
    N = args["N"]
    rows = args["M"]
    return N <= MULTIROW_N and (N & (N - 1)) == 0 and rows >= MULTIROW_M


def softmax_backward_multirow_tile_m(args):
    return max(1, TILE_BUDGET // args["N"])


# --- Split-N kernels (few rows x huge N) ------------------------------------------
# The per-row kernel gives ONE program to a whole row and walks it twice (scale
# pass, then write pass) in TILE_N=4096 chunks. With rows=1 and N in the
# hundreds of millions that is a single CTA doing hundreds of thousands of
# serial tile iterations (~20GB/s: 1D [1073741824] took ~270ms). Split the row
# across `split` CTAs: kernel 1 accumulates a partial sum(out*grad) per chunk,
# kernel 2 re-loads the row scale from the partials and writes its chunk. The
# scale accumulator stays within the tl.sum safety limit (TILE_N=4096 <= 8192,
# HARNESS 1.5).
#
# Tail handling: this backend CANNOT compile the per-row kernel's
# full-loop + masked-tail-loop structure for bf16 ("LLVM ERROR: SmallVector
# unable to grow"), clamped tail addresses defeat vectorized access (~10-18x
# slower), and a reduce inside a runtime branch is illegal. So coverage is
# partitioned at N_ALIGNED = N // TILE_N * TILE_N: every aligned part spans an
# exact multiple of TILE_N and its loops need NO masked lanes at all (block
# DMA). The < TILE_N-element remainder [N_ALIGNED, N) is handled by separate
# one-tile masked kernels (the per-row ONE_TILE_PER_CTA shape, which compiles
# for all dtypes), launched only when N is not TILE_N-aligned.
SPLIT_TILE_N = 4096
SPLIT_CHUNK_TARGET = 16384
SPLIT_MAX = 2048
SPLIT_MAX_PROGRAMS = 32768


@libentry()
@triton.jit
def softmax_backward_split_scale_kernel(
    out_ptr,
    out_grad_ptr,
    part_scale_ptr,
    N,
    N_ALIGNED,
    CHUNK: tl.constexpr,
    TILE_S: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # grid = (rows, TILE_S). The part dim is padded to TILE_S (a power of 2) so
    # the partial-sum vector the write kernel reduces over has NO masked lanes:
    # this XPU miscompiles non-trivially-masked reductions (HARNESS 1.5), and a
    # SPLIT < TILE_S mask there silently corrupted the scale. Padded parts
    # (part >= real split) have start >= N_ALIGNED, so their loop is empty and
    # they store a harmless 0.0. `N` is the true row stride (addressing);
    # `N_ALIGNED` only bounds the loop coverage.
    row = ext.program_id(0)
    part = ext.program_id(1)
    start = tl.minimum(part * CHUNK, N_ALIGNED)
    end = tl.minimum(start + CHUNK, N_ALIGNED)
    out_ptr += row * N
    out_grad_ptr += row * N
    acc = tl.zeros([TILE_N], dtype=tl.float32)
    for off in range(start, end, TILE_N):
        idx = off + tl.arange(0, TILE_N)
        o = tl.load(out_ptr + idx).to(tl.float32)
        g = tl.load(out_grad_ptr + idx).to(tl.float32)
        acc += o * g
    tl.store(part_scale_ptr + row * TILE_S + part, tl.sum(acc, 0))


@libentry()
@triton.jit
def softmax_backward_split_scale_remainder_kernel(
    out_ptr,
    out_grad_ptr,
    part_scale_ptr,
    N,
    N_ALIGNED,
    SPLIT,
    TILE_S: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # One program per row: the ragged remainder [N_ALIGNED, N), ONE masked tile
    # (r < TILE_N by construction; same shape as the per-row kernel's
    # ONE_TILE_PER_CTA path, which compiles for all dtypes). Writes slot SPLIT
    # of the padded partial buffer (slots > SPLIT stay zero from zero-init).
    row = ext.program_id(0)
    out_ptr += row * N + N_ALIGNED
    out_grad_ptr += row * N + N_ALIGNED
    r = N - N_ALIGNED
    idx = tl.arange(0, TILE_N)
    mask = idx < r
    o = tl.load(out_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(out_grad_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    tl.store(part_scale_ptr + row * TILE_S + SPLIT, tl.sum(o * g, 0))


@libentry()
@triton.jit
def softmax_backward_split_write_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    part_scale_ptr,
    N,
    N_ALIGNED,
    CHUNK: tl.constexpr,
    TILE_S: tl.constexpr,
    TILE_N: tl.constexpr,
):
    row = ext.program_id(0)
    part = ext.program_id(1)
    # All TILE_S lanes are valid: the scale kernels write every slot of the
    # padded (rows, TILE_S) buffer (aligned parts 0.0 when empty, remainder
    # slot SPLIT, slots > SPLIT zero-init), so this reduction needs NO mask
    # (a non-trivial mask here miscompiles, HARNESS 1.5).
    s = tl.load(part_scale_ptr + row * TILE_S + tl.arange(0, TILE_S))
    scale = tl.sum(s, 0)
    start = tl.minimum(part * CHUNK, N_ALIGNED)
    end = tl.minimum(start + CHUNK, N_ALIGNED)
    out_ptr += row * N
    out_grad_ptr += row * N
    in_grad_ptr += row * N
    for off in range(start, end, TILE_N):
        idx = off + tl.arange(0, TILE_N)
        o = tl.load(out_ptr + idx).to(tl.float32)
        g = tl.load(out_grad_ptr + idx).to(tl.float32)
        tl.store(in_grad_ptr + idx, (o * (g - scale)).to(in_grad_ptr.dtype.element_ty))


@libentry()
@triton.jit
def softmax_backward_split_write_remainder_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    part_scale_ptr,
    N,
    N_ALIGNED,
    SPLIT,
    TILE_S: tl.constexpr,
    TILE_N: tl.constexpr,
):
    row = ext.program_id(0)
    s = tl.load(part_scale_ptr + row * TILE_S + tl.arange(0, TILE_S))
    scale = tl.sum(s, 0)
    out_ptr += row * N + N_ALIGNED
    out_grad_ptr += row * N + N_ALIGNED
    in_grad_ptr += row * N + N_ALIGNED
    r = N - N_ALIGNED
    idx = tl.arange(0, TILE_N)
    mask = idx < r
    o = tl.load(out_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(out_grad_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        in_grad_ptr + idx, (o * (g - scale)).to(in_grad_ptr.dtype.element_ty), mask=mask
    )


def softmax_backward_split_plan(rows, N):
    """Return (split, chunk, tile_s, n_aligned, has_remainder) or None."""
    # Empirical crossover (micro-bench, fp32): the split's two-kernel structure
    # (kernel barrier + partial round-trip) only pays when the per-row kernel
    # leaves the machine idle, i.e. very few rows AND a long serial row walk.
    # rows=1/N=1048576: 2.8x faster; rows=4: 1.6x; rows=8: 1.14x SLOWER;
    # rows>=16: up to 1.45x slower. At N=65536 the split is ~2x slower even
    # with rows<=4 (overhead dominates), hence the N floor.
    if N < 1048576 or rows > 4:
        return None
    n_aligned = N // SPLIT_TILE_N * SPLIT_TILE_N
    split = min(
        triton.cdiv(n_aligned, SPLIT_CHUNK_TARGET),
        SPLIT_MAX,
        max(1, SPLIT_MAX_PROGRAMS // rows),
    )
    if split < 2:
        return None
    chunk = triton.cdiv(triton.cdiv(n_aligned, split), SPLIT_TILE_N) * SPLIT_TILE_N
    split = triton.cdiv(n_aligned, chunk)
    if split < 2:
        return None
    has_remainder = N > n_aligned
    return (
        split,
        chunk,
        triton.next_power_of_2(split + 1 if has_remainder else split),
        n_aligned,
        has_remainder,
    )


def launch_softmax_backward_inner(out_mat, out_grad_mat, in_grad_mat, rows, N):
    plan = softmax_backward_split_plan(rows, N)
    if plan is not None:
        split, chunk, tile_s, n_aligned, has_remainder = plan
        # zeros, NOT empty: slots beyond the written ones (padded parts / the
        # remainder slot when N is aligned) must reduce as 0.0.
        part_scale = torch.zeros(
            (rows * tile_s,), dtype=torch.float32, device=out_mat.device
        )
        grid = (rows, tile_s, 1)
        softmax_backward_split_scale_kernel[grid](
            out_mat,
            out_grad_mat,
            part_scale,
            N,
            n_aligned,
            CHUNK=chunk,
            TILE_S=tile_s,
            TILE_N=SPLIT_TILE_N,
            buffer_size_limit=2048,
        )
        if has_remainder:
            softmax_backward_split_scale_remainder_kernel[(rows, 1, 1)](
                out_mat,
                out_grad_mat,
                part_scale,
                N,
                n_aligned,
                split,
                TILE_S=tile_s,
                TILE_N=SPLIT_TILE_N,
                buffer_size_limit=2048,
            )
        softmax_backward_split_write_kernel[grid](
            out_mat,
            out_grad_mat,
            in_grad_mat,
            part_scale,
            N,
            n_aligned,
            CHUNK=chunk,
            TILE_S=tile_s,
            TILE_N=SPLIT_TILE_N,
            buffer_size_limit=2048,
        )
        if has_remainder:
            softmax_backward_split_write_remainder_kernel[(rows, 1, 1)](
                out_mat,
                out_grad_mat,
                in_grad_mat,
                part_scale,
                N,
                n_aligned,
                split,
                TILE_S=tile_s,
                TILE_N=SPLIT_TILE_N,
                buffer_size_limit=2048,
            )
    elif softmax_backward_kernel_inner_heur_multirow({"N": N, "M": rows}):
        tile_m = softmax_backward_multirow_tile_m({"N": N})
        grid = (triton.cdiv(rows, tile_m), 1, 1)
        softmax_backward_multirow_kernel[grid](
            out_mat,
            out_grad_mat,
            in_grad_mat,
            rows,
            N,
            TILE_M=tile_m,
            buffer_size_limit=2048,
        )
    else:
        grid = (rows, 1, 1)
        softmax_backward_kernel_inner[grid](
            out_mat,
            out_grad_mat,
            in_grad_mat,
            rows,
            N,
            buffer_size_limit=2048,
        )


def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    if tuple(grad_input.shape) != tuple(output.shape):
        grad_input.resize_(output.shape)
    if grad_input.dtype != input_dtype:
        raise RuntimeError(
            f"_softmax_backward_data.out: expected grad_input dtype {input_dtype}, got {grad_input.dtype}"
        )
    if output.numel() == 0:
        return grad_input
    # The generic `_softmax_backward_data.out` implementation lowers to
    # softmax_backward_kernel_non_inner, whose 2D-tile `tl.sum(axis=0)`
    # reduction fails to compile on this XPU ("axis must not be 0 for 2D+
    # shapes"). Reuse the tuned kunlunxin functional (which handles K > 1 by
    # transposing the reduced dim innermost) and write the result back into
    # `grad_input` via the native strided copy; gems never overrides
    # `_copy_from`, so this reaches the vendor engine directly.
    in_grad = softmax_backward(grad_output, output, dim, input_dtype)
    torch.ops.aten._copy_from(in_grad, grad_input, False)
    return grad_input


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    # special handling for dim = 0 and empty tensor
    if self.numel() == 0:
        out_shape = list(self.shape)
        out = torch.empty(out_shape, dtype=self.dtype, device=self.device)
        zero_(out)
        return out

    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]  # pre_dim
    self = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    K = self.numel() // M // N  # post_dim

    with torch_device_fn.device(self.device):
        if K > 1:
            origin_dim = self.ndim
            if origin_dim == 3:
                m, n, k = self.shape
            else:  # 2D, dim == 0 -> M == 1
                n, k = self.shape
                m = 1
            # Rearrange [M, N, K] -> [M, K, N] so the reduced dim N is innermost
            # (the only fast axis on this XPU). Allocate the output tile directly
            # instead of `empty_like(self).view(...).transpose(...).contiguous()`,
            # which used to copy an uninitialized [M,K,N] buffer (a wasted
            # transpose-copy on top of the input transpose).
            inp_reshaped = (
                self.view(M, N, K).transpose(1, 2).contiguous().view(M * K, N)
            )
            out_reshaped = torch.empty((M * K, N), dtype=dtype, device=self.device)

            grid = lambda meta: (M * K, 1, 1)  # noqa: E731

            softmax_kernel_inner[grid](
                out_reshaped,
                inp_reshaped,
                M * K,
                N,
                buffer_size_limit=2048,
                is_use_mask_zero=True,
            )

            # Restore original layout (returns a transposed view, no copy).
            if M == 1 and origin_dim == 2:
                out = out_reshaped.view(K, N).transpose(0, 1)
            elif M == 1 and origin_dim == 3:
                out = out_reshaped.transpose(0, 1).view(m, n, k)
            else:
                out = out_reshaped.view(m, k, n).transpose(1, 2)
        else:
            out = torch.empty_like(self, dtype=dtype)
            grid = (M, 1, 1)
            softmax_kernel_inner[grid](
                out,
                self,
                M,
                N,
                buffer_size_limit=2048,
                is_use_mask_zero=True,
            )
    return out


def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    K = output.numel() // M // N

    with torch_device_fn.device(output.device):
        if K > 1:
            # Transpose so the reduced dim N is innermost and contiguous, then
            # treat [M, N, K] as a [M*K, N] row-major matrix.
            out_mat = output.view(M, N, K).transpose(1, 2).contiguous().view(M * K, N)
            out_grad_mat = (
                grad_output.view(M, N, K).transpose(1, 2).contiguous().view(M * K, N)
            )
            rows = M * K
            in_grad_mat = torch.empty(
                (rows, N), dtype=torch.float32, device=output.device
            )
            # The kernel writes a fresh [rows, N] buffer. The old code produced
            # this buffer via `empty_like(...).view(M,N,K).transpose(1,2)
            # .contiguous()`, which materialized a copy of an UNINITIALIZED
            # tensor (a full read+write pass with no information), and then
            # recovered the layout from views + `.to(input_dtype)` (another
            # full pass). Layout restore + dtype cast now fold into a single
            # native strided copy via `_copy_from`.
            launch_softmax_backward_inner(out_mat, out_grad_mat, in_grad_mat, rows, N)
            # 将输入梯度恢复到原始布局（同一趟原生 strided copy 内完成 cast）
            in_grad = torch.empty_like(output, dtype=input_dtype)
            torch.ops.aten._copy_from(
                in_grad_mat.view(M, K, N),
                in_grad.view(M, N, K).transpose(1, 2),
                False,
            )
            return in_grad
        else:
            in_grad = torch.empty_like(output, dtype=torch.float32)
            launch_softmax_backward_inner(output, grad_output, in_grad, M, N)
    return in_grad.to(input_dtype)
