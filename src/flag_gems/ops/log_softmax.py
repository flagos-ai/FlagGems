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
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def log_softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
        mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(inp, 0)
        z = tl.sum(tl.exp(inp - m[None, :]), 0)
        out = inp - m[None, :] - tl.log(z)[None, :]
        tl.store(output_ptr + offsets, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(inp, m)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)
        m = m_reduced
        log_z = tl.log(z)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            out = inp - m[None, :] - log_z[None, :]
            tl.store(output_ptr + offsets, out, mask=mask)


@libentry()
@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr = 8,
    BLOCK_N: tl.constexpr = 256,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # TODO(chenfeiyu): consider float64 add add a utility function to get accumulator type
    m = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32)
    z = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        # '&' for element-wise tensor AND; Python 'and' is deprecated on tensors
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(inp, m)
        all_neg_inf = m_new == float("-inf")
        z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
        m = m_new

    m_reduced = tl.max(m, 1)
    z = tl.sum(z * tl.exp(m - m_reduced[:, None]), 1)
    m = m_reduced

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        # '&' for element-wise tensor AND; Python 'and' is deprecated on tensors
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        o = inp - m[:, None] - tl.log(z[:, None])
        tl.store(output_ptr + offset, o, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
@triton.jit
def log_softmax_backward_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    scale = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        # '&' for element-wise tensor AND; Python 'and' is deprecated on tensors
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask, other=0.0).to(tl.float32)
        scale += out_grad
    scale = tl.sum(scale, 1)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        # '&' for element-wise tensor AND; Python 'and' is deprecated on tensors
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_ptrs = out_ptr + offsets
        out = tl.load(out_ptrs, mask=mask).to(tl.float32)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        in_grad = out_grad - tl.exp(out) * scale[:, None]
        in_grad_ptrs = in_grad_ptr + offsets
        tl.store(in_grad_ptrs, in_grad, mask=mask)


def log_softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS LOG_SOFTMAX_OUT")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    if out.dtype != dtype:
        raise RuntimeError(
            f"_log_softmax.out: expected out dtype {dtype}, got {out.dtype}"
        )
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            log_softmax_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        else:
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                K,
            )
            log_softmax_kernel[grid](
                out,
                inp,
                M,
                N,
                K,
                num_warps=8,
            )
    return out


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS LOG_SOFTMAX")
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    dtype = torch.float32 if half_to_float else self.dtype
    out = torch.empty_like(self.contiguous(), dtype=dtype)
    return log_softmax_out(self, dim, half_to_float, out=out)


def log_softmax_backward_out(grad_output, output, dim, input_dtype, *, out):
    logger.debug("GEMS LOG_SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    if tuple(out.shape) != tuple(output.shape):
        out.resize_(output.shape)
    if out.dtype != input_dtype:
        raise RuntimeError(
            f"_log_softmax_backward_data.out: expected out dtype {input_dtype}, got {out.dtype}"
        )
    K = output.numel() // M // N

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(out.device):
        log_softmax_backward_kernel[grid](
            output,
            grad_output,
            out,
            M,
            N,
            K,
        )
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS LOG_SOFTMAX_BACKWARD")
    if output.is_cuda and grad_output.is_cuda:
        return log_softmax_backward_optimized grad_output, output, dim, input_dtype
    in_grad = torch.empty_like(output, dtype=input_dtype)
    return log_softmax_backward_out(grad_output, output, dim, input_dtype, out=in_grad)


# --------------------------------------------------------------------------- #
# Row-wise backward for ``_log_softmax``
#
# ``grad_input = grad_output - exp(output) * sum(grad_output, dim, keepdim=True)``
# needs exactly one reduction over the normalized dimension.  The kernels below
# keep that reduction inside the program that also writes the epilogue, so no
# per-row ``scale`` is staged through memory on the common short-row path, and a
# wide row is walked in chunks that stay resident in L2 between the reduction pass
# and the epilogue pass.
# --------------------------------------------------------------------------- #


SHORT_LIMIT = 4096

# Bounded search for the wide two-pass path.  The reuse of grad_output between the
# two passes only survives in L2 if the aggregate footprint of concurrently resident
# rows stays under 50 MB, so process several rows per program: with ROWS_PER_PROG>1
# the grid shrinks and far fewer 128 KB (fp16) / 256 KB (fp32) rows are in flight at
# once.  Wider CTAs reduce residency for the same reason.
#
# The two element sizes prefer opposite settings: 128 KB fp16/bf16 rows already fit L2
# with one row per program and a wide chunk, whereas 256 KB fp32 rows need the deep
# residency caps to get under it.  Removing the single-row/wide-chunk entries cost
# fp16 about 13% in an earlier round, so the list keeps both families and lets the
# per-key autotune choose: `ESIZE` is part of the key, so each element size is tuned
# independently instead of sharing one winner.
_WIDE_CONFIGS = [
    # fits fp16/bf16 rows at one row per program, widest chunk
    triton.Config({"BLOCK_C": 8192, "ROWS_PER_PROG": 1}, num_warps=32),
    triton.Config({"BLOCK_C": 4096, "ROWS_PER_PROG": 1}, num_warps=8),
    triton.Config({"BLOCK_C": 8192, "ROWS_PER_PROG": 8}, num_warps=32),
    triton.Config({"BLOCK_C": 4096, "ROWS_PER_PROG": 8}, num_warps=8),
    # deeper residency caps, which is what 256 KB fp32 rows need
    triton.Config({"BLOCK_C": 4096, "ROWS_PER_PROG": 16}, num_warps=8),
    triton.Config({"BLOCK_C": 4096, "ROWS_PER_PROG": 32}, num_warps=8),
    triton.Config({"BLOCK_C": 8192, "ROWS_PER_PROG": 32}, num_warps=32),
]


@libentry()
@triton.jit
def _rowwise(
    grad_output_ptr,
    output_ptr,
    grad_input_ptr,
    N: tl.constexpr,
    INNER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // INNER
    inner = pid % INNER
    base = row.to(tl.int64) * (N * INNER) + inner

    offsets = tl.arange(0, BLOCK_N)
    ptrs = base + offsets * INNER

    if EVEN_N:
        grad_output = tl.load(grad_output_ptr + ptrs).to(tl.float32)
        out_v = tl.load(output_ptr + ptrs).to(tl.float32)
    else:
        mask = offsets < N
        grad_output = tl.load(grad_output_ptr + ptrs, mask=mask, other=0.0).to(
            tl.float32
        )
        out_v = tl.load(output_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

    scale = tl.sum(grad_output, 0)
    grad_input = grad_output - tl.exp(out_v) * scale

    if EVEN_N:
        tl.store(grad_input_ptr + ptrs, grad_input)
    else:
        tl.store(grad_input_ptr + ptrs, grad_input, mask=mask)


@triton.autotune(configs=_WIDE_CONFIGS, key=["N", "INNER", "ESIZE"])
@triton.jit
def _chunked(
    grad_output_ptr,
    output_ptr,
    grad_input_ptr,
    N: tl.constexpr,
    INNER: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    ESIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_stride = N * INNER

    # Tile-derived fast path: the unmasked path is only valid when the chosen
    # BLOCK_C divides the row, so this is derived from the tile rather than from N
    # and can never outlive a change to the config list.
    EVEN_C: tl.constexpr = (N % BLOCK_C) == 0

    cols = tl.arange(0, BLOCK_C)

    for r in range(ROWS_PER_PROG):
        # Programs stride by ROWS_PER_PROG rows, so with ROWS_PER_PROG=1 this is the
        # original one-row-per-program mapping.
        rid = pid * ROWS_PER_PROG + r
        if rid < NUM_ROWS:
            row = rid // INNER
            inner = rid % INNER
            base = row.to(tl.int64) * row_stride + inner

            acc = tl.zeros([BLOCK_C], dtype=tl.float32)

            # Pass 1: row sum of grad_output, vector-accumulated in fp32.  `evict_last`
            # asks L2 to keep these lines alive until pass 2 re-reads them.
            if EVEN_C:
                for start in range(0, N, BLOCK_C):
                    ptrs = base + (start + cols) * INNER
                    acc += tl.load(
                        grad_output_ptr + ptrs, eviction_policy="evict_last"
                    ).to(tl.float32)
            else:
                for start in range(0, N, BLOCK_C):
                    ptrs = base + (start + cols) * INNER
                    mask = (start + cols) < N
                    acc += tl.load(
                        grad_output_ptr + ptrs,
                        mask=mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    ).to(tl.float32)

            scale = tl.sum(acc, 0)

            # Pass 2: epilogue in reverse chunk order to reuse the row tail in L2.  This
            # re-read of grad_output is its LAST use, so it is loaded `evict_first` to
            # release those lines as soon as it is done with them rather than pinning
            # them against the pass-1 lines that have not been re-read yet; `output` is
            # likewise streamed once, and the store is streaming so it does not
            # write-allocate over the row lines pass 1 left behind.
            n_full = (N // BLOCK_C) * BLOCK_C
            for start in range(n_full - BLOCK_C, -BLOCK_C, -BLOCK_C):
                ptrs = base + (start + cols) * INNER
                g = tl.load(grad_output_ptr + ptrs, eviction_policy="evict_first").to(
                    tl.float32
                )
                o = tl.load(output_ptr + ptrs, eviction_policy="evict_first").to(
                    tl.float32
                )
                tl.store(
                    grad_input_ptr + ptrs, g - tl.exp(o) * scale, cache_modifier=".cs"
                )

            if n_full < N:
                ptrs = base + (n_full + cols) * INNER
                mask = (n_full + cols) < N
                g = tl.load(grad_output_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
                o = tl.load(output_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
                tl.store(grad_input_ptr + ptrs, g - tl.exp(o) * scale, mask=mask)


def _launch(grad_output, output, grad_input, N, inner, rows):
    """Grid/config selection and the two JIT launches, out of the hot path body."""
    if N <= SHORT_LIMIT:
        block_n = triton.next_power_of_2(N)
        if block_n <= 256:
            num_warps = 2
        elif block_n <= 1024:
            num_warps = 4
        else:
            num_warps = 8
        _rowwise[(rows,)](
            grad_output,
            output,
            grad_input,
            N=N,
            INNER=inner,
            BLOCK_N=block_n,
            EVEN_N=block_n == N,
            num_warps=num_warps,
        )
    else:
        # The grid depends on ROWS_PER_PROG, which each autotune config chooses, so
        # the launch grid is a callable over the bound config arguments.
        def _grid(meta):
            n_rows = rows
            rpp = meta["ROWS_PER_PROG"]
            return (triton.cdiv(n_rows, rpp),)

        _chunked[_grid](
            grad_output,
            output,
            grad_input,
            N=N,
            INNER=inner,
            NUM_ROWS=rows,
            ESIZE=grad_output.element_size(),
        )


def log_softmax_backward_optimized(grad_output, output, dim, input_dtype):
    ndim = output.ndim
    if dim < 0:
        dim += ndim
    shape = output.shape
    N = shape[dim]
    if not output.is_contiguous():
        output = output.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(output, dtype=input_dtype)
    if not N:
        return grad_input

    inner = 1
    for d in range(dim + 1, ndim):
        inner *= shape[d]
    rows = (output.numel() // N) if inner else 0
    if not rows:
        return grad_input

    # Scope the launch through the backend handle the rest of FlagGems uses.  This
    # is unconditional rather than cached: a cached "already current" answer goes
    # stale when the ambient device changes and would launch against foreign
    # pointers.
    with torch_device_fn.device(output.device):
        _launch(grad_output, output, grad_input, N, inner, rows)
    return grad_input
