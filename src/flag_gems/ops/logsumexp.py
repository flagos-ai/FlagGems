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

# Block sizes for the fused first-order backward kernel. The backward pass is
# purely elementwise, so the tile only balances coalescing on the innermost
# ``K`` axis against the number of launched programs.
_MAX_BACKWARD_BLOCK_K = 128
_MAX_BACKWARD_BLOCK_N = 32
_MAX_BACKWARD_TILE = 2048
_MAX_BACKWARD_INNER_BLOCK = 1024
_MIN_BACKWARD_INNER_BLOCK = 64
_GENERIC_MAX_NDIM = 8
_GENERIC_BLOCK = 1024


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
    """Kernel for logsumexp when the reduction dimension is the innermost."""
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, axis=0)
        # Handle case where all inputs were -inf
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


# The backward kernels take fixed constexpr block sizes and need neither
# autotuning nor heuristics, so they are launched through plain ``triton.jit``:
# LibEntry would add its per-launch key computation (~14us) to every backward.
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

    Each program owns a ``BLOCK_N x BLOCK_K`` tile. ``out`` and ``grad_out``
    are read once per tile and broadcast over the reduced axis, so the kernel
    streams the input exactly once.
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
def logsumexp_backward_kernel_generic(
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
    """Backward pass for an arbitrary set of reduced axes.

    The keepdim ``out``/``grad_out`` layout is described by ``sizes_ptr``
    (input sizes) and ``out_strides_ptr`` (contiguous strides of the keepdim
    output, zero on reduced axes), so every input element can map itself to its
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


def _normalize_dim(dim, ndim):
    assert -ndim <= dim < ndim, "Invalid dim"
    return dim % ndim


def _normalize_dims(dim, ndim):
    if isinstance(dim, (list, tuple)):
        if len(dim) == 0:
            return ()
        # Reducing the same axis twice is identical to reducing it once.
        return tuple(sorted({_normalize_dim(d, ndim) for d in dim}))
    return (_normalize_dim(dim, ndim),)


def _launch_forward(inp, dim):
    """Run the forward kernel over one normalized axis, keeping that axis.

    Returns the keepdim output and the contiguous input the kernel read.
    """
    ndim = inp.ndim
    assert 0 <= dim < ndim, "Invalid dim"
    N = inp.shape[dim]
    inp = inp.contiguous()
    shape = list(inp.shape)
    shape[dim] = 1
    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    if N == 0:
        # ATen defines logsumexp over an empty reduction axis as -inf.
        out.fill_(float("-inf"))
        return out, inp
    if inp.numel() == 0:
        return out, inp
    M = 1
    for i in range(dim):
        M *= inp.shape[i]
    K = inp.numel() // M // N

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
    return out, inp


def _forward_keepdim(inp, dims):
    """Forward over one or more normalized axes, always keeping every axis.

    Single-axis reductions keep the original kernel path. Multiple axes are
    reduced sequentially, exactly like the pre-existing forward implementation,
    so results are unchanged.
    """
    if len(dims) == 1:
        return _launch_forward(inp, dims[0])
    contiguous = inp.contiguous()
    result = contiguous
    for dim in sorted(dims, reverse=True):
        result, _ = _launch_forward(result, dim)
    return result, contiguous


def _contiguous_or_self(tensor):
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _broadcast_keepdim(grad_out, out, dims, keepdim):
    """Restore the reduced axes of ``grad_out``/``out`` so both broadcast.

    ``out`` is stored in keepdim form. A non-keepdim ``grad_out`` is missing
    exactly the reduced axes, so they are inserted back before the elementwise
    backward launch.
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
            block_k = min(_MAX_BACKWARD_BLOCK_K, triton.next_power_of_2(K))
            block_n = max(
                1,
                min(_MAX_BACKWARD_BLOCK_N, _MAX_BACKWARD_TILE // block_k),
            )
            grid = (M, triton.cdiv(K, block_k), triton.cdiv(N, block_n))
            logsumexp_backward_kernel_non_inner[grid](
                grad_out,
                inp,
                out,
                grad_in,
                N,
                K,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
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
            )
    return grad_in


def _backward_generic(grad_out, inp, out, dims):
    """Gradient for an interleaved set of reduced axes.

    Both ``out`` and ``grad_out`` are keepdim-shaped here; the kernel maps each
    input element to its broadcast source through the keepdim strides, with the
    reduced axes pinned to stride zero.
    """
    grad_in = torch.empty_like(inp)
    if inp.numel() == 0:
        return grad_in
    ndim = inp.ndim
    if ndim > _GENERIC_MAX_NDIM:
        raise NotImplementedError(
            "logsumexp backward supports at most "
            f"{_GENERIC_MAX_NDIM} dimensions for multi-axis reductions"
        )
    out_strides = list(out.stride())
    for reduced in dims:
        out_strides[reduced] = 0
    sizes = torch.tensor(inp.shape, dtype=torch.int64, device=inp.device)
    out_strides = torch.tensor(out_strides, dtype=torch.int64, device=inp.device)
    n_elements = inp.numel()
    grid = (triton.cdiv(n_elements, _GENERIC_BLOCK),)
    with torch_device_fn.device(inp.device):
        logsumexp_backward_kernel_generic[grid](
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
    def forward(ctx, inp, dim, keepdim):
        dims = _normalize_dims(dim, inp.ndim)
        out, contiguous = _forward_keepdim(inp, dims)
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
            grad_in = _backward_generic(grad_out, contiguous, out, ctx.dims)
        return grad_in, None, None


def logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS LOGSUMEXP")

    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        # Empty dim list means no reduction, just return the input.
        return inp.clone()
    return _LogSumExp.apply(inp, dim, keepdim)
