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


# Tile sizes for the fused first-order backward. The backward pass is a pure
# elementwise pass, so the tile only trades coalescing on the innermost ``K``
# axis against the number of launched programs. The splits below come from the
# KernelGen sweep over the five default shapes in FP16/FP32/BF16: narrow
# contiguous rows do best with an exact-fit ``BLOCK_N`` and four warps, while
# the wide rows (N > 1024) and the strided (K > 1) cases benefit from a swept
# larger tile, so only those two use ``@triton.autotune``.
_MAX_BACKWARD_INNER_BLOCK = 1024
_MIN_BACKWARD_INNER_BLOCK = 64
_STRIDED_LOOP_MIN_ROWS = 512
_STRIDED_LOOP_MAX_BLOCK_K = 256
_GENERIC_BLOCK = 1024

_WIDE_INNER_CONFIGS = [
    triton.Config({"BLOCK_N": 1024}, num_warps=4),
    triton.Config({"BLOCK_N": 2048}, num_warps=4),
    triton.Config({"BLOCK_N": 2048}, num_warps=8),
    triton.Config({"BLOCK_N": 4096}, num_warps=8),
]

_NON_INNER_CONFIGS = [
    triton.Config({"BLOCK_N": 16, "BLOCK_K": 128}, num_warps=4),
    triton.Config({"BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4),
    triton.Config({"BLOCK_N": 16, "BLOCK_K": 256}, num_warps=4),
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 256}, num_warps=4),
    triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=8),
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 512}, num_warps=4),
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8),
]


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


def logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS LOGSUMEXP")

    dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
    normalized = []
    for axis in dims:
        if not -inp.ndim <= axis < inp.ndim:
            raise IndexError(f"Dimension out of range: {axis}")
        axis %= inp.ndim
        if axis in normalized:
            raise RuntimeError(f"dim {axis} appears multiple times in the list of dims")
        normalized.append(axis)
    if normalized and torch.is_grad_enabled() and inp.requires_grad:
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
    if inp.numel() == 0:
        shape = list(inp.shape)
        shape[dim] = 1
        dtype = torch.result_type(inp, 0.0)
        out = inp.new_full(shape, float("-inf"), dtype=dtype)
        return out if keepdim else out.squeeze(dim)
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


# The backward kernels take explicit block sizes and need no heuristics, so
# they are launched through plain ``triton.jit``: LibEntry would add its
# per-launch key computation (~14us) to every backward, which is the same order
# as the kernel itself for the small shapes.
@triton.jit
def logsumexp_backward_kernel_inner(
    grad_out_ptr,
    input_ptr,
    out_ptr,
    grad_in_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Backward pass for an innermost reduction (``K == 1``).

    ``out`` (keepdim) and ``grad_out`` are scalars per row, so they are loaded
    once and broadcast over the reduction axis:

    ``grad_in[m, n] = grad_out[m] * exp(input[m, n] - out[m])``.
    """
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    out = tl.load(out_ptr + pid_m)
    grad_out = tl.load(grad_out_ptr + pid_m)
    row = pid_m.to(tl.int64) * N
    inp = tl.load(input_ptr + row + offsets, mask=mask, other=0.0)
    grad_in = grad_out * tl.exp((inp - out).to(tl.float32)).to(grad_out.dtype)
    tl.store(grad_in_ptr + row + offsets, grad_in, mask=mask)


@triton.autotune(configs=_WIDE_INNER_CONFIGS, key=["N"])
@triton.jit
def logsumexp_backward_kernel_inner_wide(
    grad_out_ptr,
    input_ptr,
    out_ptr,
    grad_in_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Same expression for wide contiguous rows, with a swept tile."""
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    out = tl.load(out_ptr + pid_m)
    grad_out = tl.load(grad_out_ptr + pid_m)
    row = pid_m.to(tl.int64) * N
    inp = tl.load(input_ptr + row + offsets, mask=mask, other=0.0)
    grad_in = grad_out * tl.exp((inp - out).to(tl.float32)).to(grad_out.dtype)
    tl.store(grad_in_ptr + row + offsets, grad_in, mask=mask)


@triton.autotune(configs=_NON_INNER_CONFIGS, key=["N", "K"])
@triton.jit
def logsumexp_backward_kernel_non_inner(
    grad_out_ptr,
    input_ptr,
    out_ptr,
    grad_in_ptr,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Backward pass for a non-innermost reduction (``K > 1``).

    Each program owns a ``BLOCK_N x BLOCK_K`` tile; the broadcast values are
    loaded once per tile, so ``input`` is streamed exactly once.
    """
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    pid_n = ext.program_id(2)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    out_offset = pid_m.to(tl.int64) * K + k_offsets
    out = tl.load(out_ptr + out_offset, mask=k_mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + out_offset, mask=k_mask, other=0.0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (n_offsets[:, None] < N) & k_mask[None, :]
    offsets = pid_m.to(tl.int64) * N * K + n_offsets[:, None] * K + k_offsets[None, :]
    inp = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    grad_in = grad_out[None, :] * tl.exp((inp - out[None, :]).to(tl.float32)).to(
        grad_out.dtype
    )
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


@triton.jit
def logsumexp_backward_kernel_strided_loop(
    grad_out_ptr,
    input_ptr,
    out_ptr,
    grad_in_ptr,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Strided reduction axis, one program per (row, k-block) with a loop.

    The broadcast slice for this row and k-block stays in registers for the
    whole reduced axis, so it is read once per program instead of once per tile
    and the launch drops from ``M * cdiv(K, BLOCK_K) * cdiv(N, BLOCK_N)``
    programs to ``M * cdiv(K, BLOCK_K)``. Used only when that still leaves the
    device well populated.
    """
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    out_offset = pid_m.to(tl.int64) * K + k_offsets
    out = tl.load(out_ptr + out_offset, mask=k_mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + out_offset, mask=k_mask, other=0.0)
    row_base = pid_m.to(tl.int64) * N * K
    for n0 in range(0, N, BLOCK_N):
        n_offsets = n0 + tl.arange(0, BLOCK_N)
        mask = (n_offsets[:, None] < N) & k_mask[None, :]
        offsets = row_base + n_offsets[:, None] * K + k_offsets[None, :]
        inp = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        grad_in = grad_out[None, :] * tl.exp((inp - out[None, :]).to(tl.float32)).to(
            grad_out.dtype
        )
        tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


@triton.jit
def logsumexp_backward_kernel_multi_axis(
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
    """Backward pass for several reduced axes at once.

    The keepdim ``out``/``grad_out`` layout is described by ``sizes_ptr`` (input
    sizes) and ``out_strides_ptr`` (contiguous strides of the keepdim output,
    zero on the reduced axes), so every input element maps itself to its
    broadcast source without any host-side index tensor.
    """
    pid = ext.program_id(0)
    offsets = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    remaining = offsets
    out_offset = tl.zeros([BLOCK], dtype=tl.int64)
    for step in tl.static_range(NDIM):
        axis = NDIM - 1 - step
        size = tl.load(sizes_ptr + axis)
        stride = tl.load(out_strides_ptr + axis)
        index = remaining % size
        remaining = remaining // size
        out_offset += index * stride
    grad_out = tl.load(grad_out_ptr + out_offset, mask=mask, other=0.0)
    out = tl.load(out_ptr + out_offset, mask=mask, other=0.0)
    inp = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    grad_in = grad_out * tl.exp((inp - out).to(tl.float32)).to(grad_out.dtype)
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


def _contiguous_or_self(tensor):
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _broadcast_keepdim(grad_out, out, dims, keepdim):
    """Restore the reduced axes of ``grad_out``/``out`` so both broadcast.

    ``out`` is stored in keepdim form. A non-keepdim ``grad_out`` is missing
    exactly the reduced axes, so they are inserted back before the elementwise
    backward launch; the result shares storage with the incoming gradient.
    """
    if not keepdim:
        for reduced in sorted(dims):
            grad_out = grad_out.unsqueeze(reduced)
    return _contiguous_or_self(grad_out), _contiguous_or_self(out)


def _backward_block(grad_out, inp, out, dim):
    """Gradient for one reduced axis; ``grad_out``/``out`` are keepdim-shaped."""
    N = inp.shape[dim]
    grad_in = torch.empty_like(inp)
    if inp.numel() == 0:
        return grad_in
    M = 1
    for i in range(dim):
        M *= inp.shape[i]
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            if M >= _STRIDED_LOOP_MIN_ROWS:
                # Enough rows to fill the device with one program per
                # (row, k-block): loop the reduced axis instead of tiling it.
                block_k = min(_STRIDED_LOOP_MAX_BLOCK_K, triton.next_power_of_2(K))
                grid = (M, triton.cdiv(K, block_k))
                logsumexp_backward_kernel_strided_loop[grid](
                    grad_out,
                    inp,
                    out,
                    grad_in,
                    N,
                    K,
                    BLOCK_N=8,
                    BLOCK_K=block_k,
                    num_warps=4,
                )
            else:
                grid = lambda meta: (
                    M,
                    triton.cdiv(K, meta["BLOCK_K"]),
                    triton.cdiv(N, meta["BLOCK_N"]),
                )
                logsumexp_backward_kernel_non_inner[grid](
                    grad_out,
                    inp,
                    out,
                    grad_in,
                    N,
                    K,
                )
        elif N > _MAX_BACKWARD_INNER_BLOCK:
            grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
            logsumexp_backward_kernel_inner_wide[grid](
                grad_out,
                inp,
                out,
                grad_in,
                N,
            )
        else:
            block_n = min(
                _MAX_BACKWARD_INNER_BLOCK,
                max(_MIN_BACKWARD_INNER_BLOCK, triton.next_power_of_2(N)),
            )
            grid = (M, triton.cdiv(N, block_n))
            logsumexp_backward_kernel_inner[grid](
                grad_out,
                inp,
                out,
                grad_in,
                N,
                BLOCK_N=block_n,
                num_warps=4,
            )
    return grad_in


def _backward_multi_axis(grad_out, inp, out, dims):
    """Gradient for an interleaved set of reduced axes.

    Both ``out`` and ``grad_out`` are keepdim-shaped here; the kernel maps each
    input element to its broadcast source through the keepdim strides, with the
    reduced axes pinned to stride zero.
    """
    grad_in = torch.empty_like(inp)
    if inp.numel() == 0:
        return grad_in
    ndim = inp.ndim
    out_strides = list(out.stride())
    for reduced in dims:
        out_strides[reduced] = 0
    sizes = torch.tensor(inp.shape, dtype=torch.int64, device=inp.device)
    out_strides = torch.tensor(out_strides, dtype=torch.int64, device=inp.device)
    n_elements = inp.numel()
    grid = (triton.cdiv(n_elements, _GENERIC_BLOCK),)
    with torch_device_fn.device(inp.device):
        logsumexp_backward_kernel_multi_axis[grid](
            grad_out,
            inp,
            out,
            grad_in,
            sizes,
            out_strides,
            n_elements,
            NDIM=ndim,
            BLOCK=_GENERIC_BLOCK,
        )
    return grad_in


class _LogSumExp(torch.autograd.Function):
    """Autograd bridge for ``aten::logsumexp``.

    The forward keeps using the existing FlagGems reduction kernels. The
    backward differentiates ``logsumexp`` without a second reduction:

    ``d/dx logsumexp(x) = exp(x - logsumexp(x))``

    so multiplying the saved keepdim output with the incoming gradient only
    requires a single elementwise Triton pass over the input.
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
        if len(ctx.dims) == 1:
            grad_in = _backward_block(grad_out, contiguous, out, ctx.dims[0])
        else:
            grad_in = _backward_multi_axis(grad_out, contiguous, out, ctx.dims)
        return grad_in, None, None
