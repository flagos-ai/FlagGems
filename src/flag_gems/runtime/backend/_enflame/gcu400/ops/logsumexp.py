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
def logsumexp_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """Kernel for logsumexp when reduction dimension is not the innermost."""
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        input_ptrs = input_ptr + inp_offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, axis=0, keep_dims=True)
        # Handle case where entire column is -inf
        safe_m = tl.where(m == float("-inf"), tl.zeros_like(m), m)
        e = tl.exp(inp - safe_m)
        z = tl.sum(e, axis=0, keep_dims=True)
        out = safe_m + tl.log(z)
        # If all inputs were -inf, result should be -inf
        out = tl.where(m == float("-inf"), m, out)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, axis=0, keep_dims=True)
        z = tl.sum(z * tl.exp(m - m_reduced), axis=0, keep_dims=True)
        m = m_reduced
        # Handle case where all inputs were -inf
        out = tl.where(m == float("-inf"), m, m + tl.log(z))
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def logsumexp_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """Kernel for logsumexp when reduction dimension is the innermost."""
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, axis=0)
        # Handle case where all inputs are -inf
        safe_m = tl.where(m == float("-inf"), 0.0, m)
        e = tl.exp(inp - safe_m)
        z = tl.sum(e, axis=0)
        out = safe_m + tl.log(z)
        # If all inputs were -inf, result should be -inf
        out = tl.where(m == float("-inf"), m, out)
        output_ptrs = output_ptr + pid_m
        tl.store(output_ptrs, out)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, axis=0)
        z = tl.sum(z * tl.exp(m - m_reduced), axis=0)
        m = m_reduced
        # Handle case where all inputs were -inf
        out = tl.where(m == float("-inf"), m, m + tl.log(z))
        output_ptrs = output_ptr + pid_m
        tl.store(output_ptrs, out)


# --- TUNING PARAMS START ---
# Multi-row inner path: rows per program are chosen so one program owns
# INNER_ROW_ELEMENTS fp32 operands (the live tile), which keeps the reduction
# tile inside the vector register budget while amortising the per-program
# max/exp/log epilogue over several rows.
INNER_ROW_ELEMENTS = 32768
NUM_WARPS_INNER_ROWS = 4
# Above this tile width a stacked tile spills, so wide rows switch to the
# row-loop kernel with this many rows per program.
MAX_TILE_2D = 2048
WIDE_ROW_ROWS = 8
NUM_WARPS_WIDE = 4
# 2026-11-23 sweep (official do_bench, GEO over the 15 core cells: 1.734x at HEAD):
#   stacked-row budget 4096 / 8192 / 16384 / 32768 / 65536 -> 2.519x / 2.734x / 2.906x /
#   3.079x / 3.024x, so the budget is set where a program still fills the 48 cores;
#   wide rows 4 / 8 / 16 -> 1.517x / 1.656x / 1.626x at 4096^2 (8 rows shipped);
#   a split-N two-pass variant measured 0.265x at 4096^2 (many small programs) and was dropped;
#   exp2/log2 rewriting of the epilogue failed 6 accuracy cells and was dropped.
# --- TUNING PARAMS END ---


def _inner_rows(M, N):
    """Rows per program for the fused multi-row inner kernel (1 = use the per-row kernel).

    The innermost-dimension reduction ships one program per row, so the row count
    is the only lever on how much work a program amortises. A row that is already
    at the tile budget stays on the historical kernel.
    """
    tile_n = triton.next_power_of_2(N)
    if tile_n > MAX_TILE_2D:
        return 1
    # ROWS feeds ``tl.arange(0, ROWS)``, which only accepts a power of two, so a
    # non-power-of-two row count must round up: (2,3,4,5) reduced on dim 3 has
    # M=24 and used to fail to compile. Overshooting M is harmless, the tile
    # carries a row mask.
    return min(max(1, INNER_ROW_ELEMENTS // tile_n), triton.next_power_of_2(M))


@libentry()
@triton.jit
def logsumexp_kernel_inner_rows(
    output_ptr,
    input_ptr,
    M,
    N,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``logsumexp`` over the innermost dimension, ROWS rows per program.

    Numerics are those of ``logsumexp_kernel_inner``: an fp32 max pass, a shifted
    exponential sum, and the same all ``-inf`` row guard, so the trimmed tile can
    only change the launch geometry, not the result.
    """
    pid = ext.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M
    mask = row_mask[:, None] & (cols < N)[None, :]
    inp = tl.load(
        input_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=-float("inf")
    ).to(tl.float32)
    m = tl.max(inp, axis=1)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    z = tl.sum(tl.exp(inp - safe_m[:, None]), axis=1)
    out = safe_m + tl.log(z)
    out = tl.where(m == float("-inf"), m, out)
    tl.store(output_ptr + rows, out, mask=row_mask)


@libentry()
@triton.jit
def logsumexp_kernel_inner_rowloop(
    output_ptr,
    input_ptr,
    M,
    N,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Wide-row variant: ROWS rows per program, one row at a time.

    A row wider than the register tile cannot be stacked (a ``(ROWS, N)`` tile
    spills), but a program can still take several rows sequentially to cut the
    program count, which is what the flat geometry is short of at 4096 columns.
    """
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    for row_offset in tl.static_range(ROWS):
        row = pid * ROWS + row_offset
        if row < M:
            inp = tl.load(
                input_ptr + row * N + cols, mask=mask, other=-float("inf")
            ).to(tl.float32)
            m = tl.max(inp, axis=0)
            safe_m = tl.where(m == float("-inf"), 0.0, m)
            z = tl.sum(tl.exp(inp - safe_m), axis=0)
            out = safe_m + tl.log(z)
            out = tl.where(m == float("-inf"), m, out)
            tl.store(output_ptr + row, out)


def logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS_ENFLAME LOGSUMEXP")

    dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
    normalized = []
    for axis in dims:
        if not -inp.ndim <= axis < inp.ndim:
            raise IndexError(f"Dimension out of range: {axis}")
        axis %= inp.ndim
        if axis in normalized:
            raise RuntimeError(f"dim {axis} appears multiple times in the list of dims")
        normalized.append(axis)
    # Route differentiable inputs through this backend's autograd bridge: the shared
    # implementation is replaced here, so its backward kernels are unreachable otherwise.
    if torch.is_grad_enabled() and inp.requires_grad:
        return _LogSumExp.apply(inp, tuple(normalized), keepdim)

    if isinstance(dim, (list, tuple)):
        # Handle multi-dimensional reduction
        if len(dim) == 0:
            # Empty dim list means no reduction, just return the input
            return inp.clone()
        if len(dim) == 1:
            dim = dim[0]
        else:
            # For multiple dims, reduce sequentially
            # Sort dims in descending order to handle dimension shifts correctly
            sorted_dims = sorted([d % inp.ndim for d in dim], reverse=True)
            result = inp
            for d in sorted_dims:
                result = logsumexp(result, d, keepdim=True)
            if not keepdim:
                # Remove the reduced dimensions
                for d in sorted(sorted_dims, reverse=True):
                    result = result.squeeze(d)
            return result

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    M = 1
    N = inp.shape[dim]
    for i in range(dim):
        M *= inp.shape[i]
    inp = inp.contiguous()
    if M == 0 or N == 0:
        # Empty reduction axis: ATen returns -inf for every output element and
        # the historical code divided by N right here.
        empty_shape = list(inp.shape)
        if keepdim:
            empty_shape[dim] = 1
        else:
            empty_shape.pop(dim)
        return torch.full(
            empty_shape, float("-inf"), dtype=inp.dtype, device=inp.device
        )
    K = inp.numel() // M // N

    # Output shape with reduction dimension set to 1
    shape = list(inp.shape)
    shape[dim] = 1
    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            logsumexp_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        else:
            rows = _inner_rows(M, N)
            if rows > 1:
                # Multi-row inner path: ROWS consecutive rows per program.
                grid = (triton.cdiv(M, rows), 1, 1)
                logsumexp_kernel_inner_rows[grid](
                    out,
                    inp,
                    M,
                    N,
                    rows,
                    triton.next_power_of_2(N),
                    num_warps=NUM_WARPS_INNER_ROWS,
                    num_stages=1,
                )
            elif N >= MAX_TILE_2D:
                grid = (triton.cdiv(M, WIDE_ROW_ROWS), 1, 1)
                logsumexp_kernel_inner_rowloop[grid](
                    out,
                    inp,
                    M,
                    N,
                    WIDE_ROW_ROWS,
                    triton.next_power_of_2(N),
                    num_warps=NUM_WARPS_WIDE,
                    num_stages=1,
                )
            else:
                grid = (M, 1, 1)
                logsumexp_kernel_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                )

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


# --- TUNING PARAMS START ---
# Backward tile budget: elements (fp32 operands) a single program owns in the inner path. The
# adjoint is elementwise work over ``grad_out``/``out`` broadcast along the reduced axis, so the
# budget only trades the per-program epilogue against the program count.
BACKWARD_TILE_ELEMENTS = 16384
NUM_WARPS_BACKWARD = 4
BACKWARD_MIN_ROWS = 4
BACKWARD_MAX_ROWS = 64
# Flat single-axis path: lanes per program (byte budget / operand size).
BACKWARD_LANE_BYTES = 4096
BACKWARD_MAX_BLOCK = 4096
# --- TUNING PARAMS END ---

# ``out``/``grad_out`` metadata for the multi-axis adjoint, keyed by (shape, reduced axes). Both
# index vectors are constants of the layout, so caching them removes the per-call upload that
# dominates these (small) shapes.
_META_CACHE = {}
_META_CACHE_LIMIT = 64


@triton.jit
def _logsumexp_backward_inner_kernel(
    grad_out_ptr,
    input_ptr,
    out_ptr,
    grad_in_ptr,
    M,
    N,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Adjoint of an innermost reduction: ``grad_out[m] * exp(input[m, n] - out[m])``.

    ``out`` (keepdim) and ``grad_out`` are scalars per row, so one tile covers ``ROWS`` rows and
    the broadcast is implicit - the tile needs no index arithmetic at all.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * ROWS + tl.arange(0, ROWS)
    cols = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    mask = row_mask[:, None] & (cols < N)[None, :]
    grad_out = tl.load(grad_out_ptr + rows, mask=row_mask, other=0.0)
    out = tl.load(out_ptr + rows, mask=row_mask, other=0.0)
    offset = rows[:, None] * N + cols[None, :]
    inp = tl.load(input_ptr + offset, mask=mask, other=0.0)
    # Same rounding as ATen's composite expression: subtract and multiply in the operand dtype,
    # exponentiate in fp32.
    grad_in = grad_out[:, None] * tl.exp((inp - out[:, None]).to(tl.float32)).to(
        grad_out.dtype
    )
    tl.store(grad_in_ptr + offset, grad_in, mask=mask)


@triton.jit
def _logsumexp_backward_axis_kernel(
    grad_out_ptr,
    input_ptr,
    out_ptr,
    grad_in_ptr,
    N_ELEMENTS,
    NK: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Adjoint of a single non-innermost reduction, one flat block per program.

    The keepdim source of element ``(m, n, k)`` is ``(m, k)``: with ``N*K`` and ``K`` as
    compile-time constants both divisions fold into multiply-shift sequences.
    """
    pid = tl.program_id(0)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    mask = index < N_ELEMENTS
    source = (index // NK) * K + index % K
    grad_out = tl.load(grad_out_ptr + source, mask=mask, other=0.0)
    out = tl.load(out_ptr + source, mask=mask, other=0.0)
    inp = tl.load(input_ptr + index, mask=mask, other=0.0)
    grad_in = grad_out * tl.exp((inp - out).to(tl.float32)).to(grad_out.dtype)
    tl.store(grad_in_ptr + index, grad_in, mask=mask)


@triton.jit
def _logsumexp_backward_multi_axis_kernel(
    grad_out_ptr,
    input_ptr,
    out_ptr,
    grad_in_ptr,
    sizes_ptr,
    out_strides_ptr,
    n_elements,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Adjoint of an interleaved set of reduced axes."""
    pid = tl.program_id(0)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    mask = index < n_elements
    remaining = index
    source = tl.zeros([BLOCK], dtype=tl.int32)
    for step in tl.static_range(NDIM):
        axis = NDIM - 1 - step
        size = tl.load(sizes_ptr + axis)
        stride = tl.load(out_strides_ptr + axis)
        coord = remaining % size
        remaining = remaining // size
        source += coord * stride
    grad_out = tl.load(grad_out_ptr + source, mask=mask, other=0.0)
    out = tl.load(out_ptr + source, mask=mask, other=0.0)
    inp = tl.load(input_ptr + index, mask=mask, other=0.0)
    grad_in = grad_out * tl.exp((inp - out).to(tl.float32)).to(grad_out.dtype)
    tl.store(grad_in_ptr + index, grad_in, mask=mask)


def _contiguous_or_self(tensor):
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _broadcast_keepdim(grad_out, out, dims, keepdim):
    """Restore the reduced axes of ``grad_out`` so it broadcasts against the keepdim ``out``."""
    if not keepdim:
        for reduced in sorted(dims):
            grad_out = grad_out.unsqueeze(reduced)
    return _contiguous_or_self(grad_out), _contiguous_or_self(out)


def _backward_lane_block(element_size):
    return min(
        BACKWARD_MAX_BLOCK,
        triton.next_power_of_2(max(16, BACKWARD_LANE_BYTES // max(element_size, 1))),
    )


def _backward_inner(grad_out, inp, out, dim):
    """Adjoint when the reduced axis is the innermost one (``K == 1``)."""
    n_reduce = inp.shape[dim]
    n_rows = inp.numel() // n_reduce if n_reduce else 0
    grad_in = torch.empty_like(inp)
    if n_rows == 0 or n_reduce == 0:
        return grad_in
    block_n = min(2048, triton.next_power_of_2(n_reduce))
    rows = max(BACKWARD_MIN_ROWS, BACKWARD_TILE_ELEMENTS // block_n)
    rows = min(rows, BACKWARD_MAX_ROWS, triton.next_power_of_2(max(n_rows, 1)))
    grid = (triton.cdiv(n_rows, rows), triton.cdiv(n_reduce, block_n))
    with torch_device_fn.device(inp.device):
        _logsumexp_backward_inner_kernel[grid](
            grad_out,
            inp,
            out,
            grad_in,
            n_rows,
            n_reduce,
            rows,
            block_n,
            num_warps=NUM_WARPS_BACKWARD,
            num_stages=1,
        )
    return grad_in


def _backward_axis(grad_out, inp, out, dim):
    """Adjoint when a single non-innermost axis is reduced (``K > 1``)."""
    n_reduce = inp.shape[dim]
    inner = 1
    for size in inp.shape[dim + 1 :]:
        inner *= size
    n_elements = inp.numel()
    grad_in = torch.empty_like(inp)
    if n_elements == 0:
        return grad_in
    block = _backward_lane_block(inp.element_size())
    with torch_device_fn.device(inp.device):
        _logsumexp_backward_axis_kernel[(triton.cdiv(n_elements, block),)](
            grad_out,
            inp,
            out,
            grad_in,
            n_elements,
            NK=n_reduce * inner,
            K=inner,
            BLOCK=block,
            num_warps=NUM_WARPS_BACKWARD,
            num_stages=1,
        )
    return grad_in


def _axis_metadata(inp, out, dims):
    """Device-side layout description of the keepdim ``out`` for the multi-axis adjoint."""
    key = (inp.shape, dims)
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    out_strides = list(out.stride())
    for reduced in dims:
        out_strides[reduced] = 0
    sizes = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)
    strides = torch.tensor(out_strides, dtype=torch.int32, device=inp.device)
    if len(_META_CACHE) < _META_CACHE_LIMIT:
        _META_CACHE[key] = (sizes, strides)
    return sizes, strides


def _backward_multi_axis(grad_out, inp, out, dims):
    """Adjoint for an interleaved set of reduced axes."""
    grad_in = torch.empty_like(inp)
    if inp.numel() == 0:
        return grad_in
    sizes, out_strides = _axis_metadata(inp, out, dims)
    block = _backward_lane_block(inp.element_size())
    with torch_device_fn.device(inp.device):
        _logsumexp_backward_multi_axis_kernel[(triton.cdiv(inp.numel(), block),)](
            grad_out,
            inp,
            out,
            grad_in,
            sizes,
            out_strides,
            inp.numel(),
            NDIM=inp.ndim,
            BLOCK=block,
            num_warps=NUM_WARPS_BACKWARD,
            num_stages=1,
        )
    return grad_in


class _LogSumExp(torch.autograd.Function):
    """Autograd bridge for ``aten::logsumexp``.

    ``d/dx logsumexp(x) = exp(x - logsumexp(x))``, so the adjoint is one elementwise pass over the
    input multiplied by the broadcast saved result - no second reduction.
    """

    @staticmethod
    def forward(ctx, inp, dims, keepdim):
        contiguous = inp.contiguous()
        out = logsumexp(contiguous, dims, keepdim=True)
        ctx.dims = dims
        ctx.keepdim = keepdim
        ctx.save_for_backward(contiguous, out)
        if keepdim:
            return out
        for reduced in sorted(dims, reverse=True):
            out = out.squeeze(reduced)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not ctx.needs_input_grad[0]:
            return None, None, None
        contiguous, out = ctx.saved_tensors
        grad_out, out = _broadcast_keepdim(grad_out, out, ctx.dims, ctx.keepdim)
        if not ctx.dims:
            # Empty dim list: the forward is an identity, so is the adjoint.
            return grad_out, None, None
        if len(ctx.dims) == 1:
            dim = ctx.dims[0]
            if dim == contiguous.ndim - 1:
                grad_in = _backward_inner(grad_out, contiguous, out, dim)
            else:
                grad_in = _backward_axis(grad_out, contiguous, out, dim)
        else:
            grad_in = _backward_multi_axis(grad_out, contiguous, out, ctx.dims)
        return grad_in, None, None
