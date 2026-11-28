# Copyright 2026, The FlagOS Contributors.
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

"""First-order gradients of ``aten::_sparse_addmm``.

ATen declares no ``_sparse_addmm_backward`` schema: the autograd engine derives
the three input gradients from the forward formula in ``derivatives.yaml``
(``SparseAddmmBackward0``)::

    grad_self = beta  * grad_output          (reduced over the bias broadcast)
    grad_mat1 = alpha * (grad_output @ mat2.T) sampled at mat1's pattern
    grad_mat2 = alpha * mat1.T @ grad_output

``grad_mat1`` is a *sparse* gradient: native autograd returns it with mat1's
own layout and stored pattern.  Every kernel below walks stored entries only,
so mat1 is never densified.

ATen builds ``grad_mat1`` with ``sparse_sampled_addmm`` (CSR) or
``addmm_sparse_cuda`` + ``sparse_mask`` (COO); both are several times slower
than sampling the dense product at the stored coordinates, and both reject
fp16/bf16.  ``grad_mat2`` is the sparse-transpose product: the CSR pattern is
transposed into CSC with a histogram, a prefix sum and a scatter over stored
entries, and a Triton SpMM multiplies it by ``grad_output`` instead of calling
cuSPARSE.

The bias reduction covers every broadcast ``self`` accepts in ``addmm``:
scalar, ``(N,)``, ``(1, N)``, ``(M, 1)`` and the full ``(M, N)`` matrix.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


_BLOCK = 1024


@libentry()
@triton.jit
def _sparse_addmm_hist_kernel(
    crow_ptr,
    col_ptr,
    counts_ptr,
    BLOCK: tl.constexpr,
):
    """Count stored entries per column, so the CSR pattern can be transposed."""
    row = ext.program_id(0)
    start = tl.load(crow_ptr + row).to(tl.int64)
    end = tl.load(crow_ptr + row + 1).to(tl.int64)
    for e0 in range(start, end, BLOCK):
        e = e0 + tl.arange(0, BLOCK)
        mask = e < end
        col = tl.load(col_ptr + e, mask=mask, other=0).to(tl.int32)
        tl.atomic_add(counts_ptr + col, 1, mask=mask)


@libentry()
@triton.jit
def _sparse_addmm_scan_kernel(
    counts_ptr,
    crow_ptr,
    K,
    BLOCK_K: tl.constexpr,
):
    """Exclusive prefix sum: ``crow_ptr`` holds the counts on entry and the
    offsets on exit; ``counts_ptr`` receives the unmodified counts."""
    running = 0
    for k0 in range(0, K, BLOCK_K):
        off = k0 + tl.arange(0, BLOCK_K)
        mask = off < K
        counts = tl.load(counts_ptr + off, mask=mask, other=0).to(tl.int32)
        prefix = tl.cumsum(counts, 0) - counts
        tl.store(crow_ptr + off, running + prefix, mask=mask)
        running += tl.sum(counts, 0)
    tl.store(crow_ptr + K, running)


@libentry()
@triton.jit
def _sparse_addmm_scatter_kernel(
    crow_ptr,
    col_ptr,
    val_ptr,
    cursor_ptr,
    row_ptr,
    t_val_ptr,
    BLOCK: tl.constexpr,
):
    """Scatter ``(row, value)`` into CSC order over the stored entries."""
    row = ext.program_id(0)
    start = tl.load(crow_ptr + row).to(tl.int64)
    end = tl.load(crow_ptr + row + 1).to(tl.int64)
    for e0 in range(start, end, BLOCK):
        e = e0 + tl.arange(0, BLOCK)
        mask = e < end
        col = tl.load(col_ptr + e, mask=mask, other=0).to(tl.int32)
        pos = tl.atomic_add(cursor_ptr + col, 1, mask=mask)
        tl.store(row_ptr + pos, row, mask=mask)
        tl.store(t_val_ptr + pos, tl.load(val_ptr + e, mask=mask), mask=mask)


@libentry()
@triton.jit
def _sparse_addmm_sample_kernel(
    dense_ptr,
    crow_ptr,
    col_ptr,
    out_ptr,
    alpha,
    stride_dm,
    stride_dk,
):
    """Sample ``alpha * dense`` at mat1's stored pattern, in stored order.

    One program per row of mat1; only stored entries are read, so no dense
    pattern mask is ever materialised.
    """
    row = ext.program_id(0)
    row64 = row.to(tl.int64)
    start = tl.load(crow_ptr + row64).to(tl.int64)
    end = tl.load(crow_ptr + row64 + 1).to(tl.int64)
    for e in range(start, end):
        col = tl.load(col_ptr + e).to(tl.int64)
        v = tl.load(dense_ptr + row64 * stride_dm + col * stride_dk)
        tl.store(out_ptr + e, (v * alpha).to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit
def _sparse_addmm_sample_transposed_kernel(
    dense_ptr,
    ccol_ptr,
    row_ptr,
    out_ptr,
    alpha,
    stride_dm,
    stride_dk,
):
    """Sample ``alpha * dense`` at a CSC pattern (column-major storage)."""
    col = ext.program_id(0)
    col64 = col.to(tl.int64)
    start = tl.load(ccol_ptr + col64).to(tl.int64)
    end = tl.load(ccol_ptr + col64 + 1).to(tl.int64)
    for e in range(start, end):
        row = tl.load(row_ptr + e).to(tl.int64)
        v = tl.load(dense_ptr + row * stride_dm + col64 * stride_dk)
        tl.store(out_ptr + e, (v * alpha).to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit
def _sparse_addmm_spmm_kernel(
    crow_ptr,
    row_ptr,
    val_ptr,
    grad_ptr,
    out_ptr,
    alpha,
    N,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    """``out = alpha * mat1.T @ grad_output`` over the transposed pattern.

    One program per (column of mat1.T, column block of grad_output).  The
    accumulator is fp32 for every input dtype.
    """
    k = ext.program_id(0)
    rn = ext.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = rn < N
    start = tl.load(crow_ptr + k).to(tl.int64)
    end = tl.load(crow_ptr + k + 1).to(tl.int64)
    goff = rn.to(tl.int64) * stride_gn
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for e in range(start, end):
        row = tl.load(row_ptr + e).to(tl.int64)
        value = tl.load(val_ptr + e).to(tl.float32)
        acc += (
            tl.load(grad_ptr + row * stride_gm + goff, mask=mask, other=0.0).to(
                tl.float32
            )
            * value
        )
    tl.store(
        out_ptr + k.to(tl.int64) * stride_om + rn.to(tl.int64) * stride_on,
        (acc * alpha).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _sparse_addmm_bias_full_kernel(
    grad_ptr,
    out_ptr,
    beta,
    M,
    N,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``grad_self = beta * grad_output`` for a full ``(M, N)`` bias."""
    om = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    on = ext.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (om[:, None] < M) & (on[None, :] < N)
    acc = tl.load(
        grad_ptr + om[:, None].to(tl.int64) * stride_gm + on[None, :] * stride_gn,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + om[:, None].to(tl.int64) * stride_om + on[None, :] * stride_on,
        (acc * beta).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _sparse_addmm_bias_reduce_kernel(
    grad_ptr,
    out_ptr,
    beta,
    M,
    N,
    stride_gm,
    stride_gn,
    SUM_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum ``grad_output`` over the axis a row/column bias broadcast away.

    ``SUM_ROWS=True`` folds the M axis for a ``(1, N)`` bias, otherwise the N
    axis is folded for an ``(M, 1)`` bias.
    """
    pid = ext.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    if SUM_ROWS:
        mask = off < N
        for m0 in range(0, M, BLOCK):
            om = m0 + tl.arange(0, BLOCK)
            values = tl.load(
                grad_ptr
                + om[:, None].to(tl.int64) * stride_gm
                + off[None, :].to(tl.int64) * stride_gn,
                mask=(om[:, None] < M) & mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(values, 0)
    else:
        mask = off < M
        for n0 in range(0, N, BLOCK):
            on = n0 + tl.arange(0, BLOCK)
            values = tl.load(
                grad_ptr
                + off[:, None].to(tl.int64) * stride_gm
                + on[None, :].to(tl.int64) * stride_gn,
                mask=mask[:, None] & (on[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(values, 1)
    tl.store(out_ptr + off, (acc * beta).to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _sparse_addmm_bias_vector_kernel(
    grad_ptr,
    out_ptr,
    beta,
    M,
    N,
    stride_gm,
    stride_gn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``(N,)`` bias: ``beta * grad_output.sum(0)`` in a single pass."""
    pid = ext.program_id(0)
    on = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = on < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        om = m0 + tl.arange(0, BLOCK_M)
        values = tl.load(
            grad_ptr + om[:, None].to(tl.int64) * stride_gm + on[None, :] * stride_gn,
            mask=(om[:, None] < M) & mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(values, 0)
    tl.store(out_ptr + on, (acc * beta).to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _sparse_addmm_scalar_kernel(
    grad_ptr,
    out_ptr,
    beta,
    n_elements,
    BLOCK: tl.constexpr,
):
    """0-D ``self``: one fp32 accumulator over the whole output gradient."""
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n_elements, BLOCK):
        off = start + tl.arange(0, BLOCK)
        acc += tl.load(
            grad_ptr + off.to(tl.int64), mask=off < n_elements, other=0.0
        ).to(tl.float32)
    total = tl.sum(acc, 0) * beta
    tl.store(out_ptr, total.to(out_ptr.dtype.element_ty))


def _spmm_block_n(N):
    # A row of mat1.T is expanded over N, so the block keeps one program's
    # accumulator in registers while covering enough columns to amortise the
    # per-entry gather.
    if N <= 256:
        return 256
    if N <= 2048:
        return 512
    return 256


def _bias_reduction(bias, grad_output):
    """Classify the broadcast ``self`` performed in the forward."""
    if bias.shape == grad_output.shape:
        return "full"
    if bias.dim() == 0:
        return "scalar"
    if bias.dim() == 1 and bias.shape[0] == grad_output.shape[1]:
        return "vector"
    if bias.dim() == 2 and bias.shape[0] == 1 and bias.shape[1] == grad_output.shape[1]:
        return "row"
    if bias.dim() == 2 and bias.shape[1] == 1 and bias.shape[0] == grad_output.shape[0]:
        return "col"
    raise RuntimeError(
        f"_sparse_addmm_backward: unsupported self shape {tuple(bias.shape)} "
        f"for grad_output {tuple(grad_output.shape)}"
    )


def _bias_grad(bias, grad_output, beta):
    """``beta * grad_output`` reduced back to ``bias.shape``."""
    plan = _bias_reduction(bias, grad_output)
    M, N = grad_output.shape
    device = grad_output.device
    dtype = grad_output.dtype
    with torch_device_fn.device(device):
        if plan == "full":
            out = torch.empty((M, N), dtype=dtype, device=device)
            if out.numel():
                block_m = min(triton.next_power_of_2(max(M, 1)), 64)
                block_n = min(triton.next_power_of_2(max(N, 1)), 256)
                _sparse_addmm_bias_full_kernel[
                    (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
                ](
                    grad_output,
                    out,
                    beta,
                    M,
                    N,
                    grad_output.stride(0),
                    grad_output.stride(1),
                    out.stride(0),
                    out.stride(1),
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=4,
                )
            return out
        if plan == "vector":
            out = torch.empty((N,), dtype=dtype, device=device)
            if N:
                _sparse_addmm_bias_vector_kernel[(triton.cdiv(N, 256),)](
                    grad_output,
                    out,
                    beta,
                    M,
                    N,
                    grad_output.stride(0),
                    grad_output.stride(1),
                    BLOCK_M=32,
                    BLOCK_N=256,
                    num_warps=4,
                )
            return out
        if plan == "scalar":
            out = torch.empty((), dtype=dtype, device=device)
            n = grad_output.numel()
            if n:
                _sparse_addmm_scalar_kernel[(1,)](
                    grad_output, out, beta, n, BLOCK=1024, num_warps=8
                )
            else:
                out.zero_()
            return out
        axis_len = N if plan == "row" else M
        out = torch.empty(
            (1, axis_len) if plan == "row" else (axis_len, 1),
            dtype=dtype,
            device=device,
        )
        if axis_len:
            block = min(triton.next_power_of_2(max(axis_len, 1)), 256)
            _sparse_addmm_bias_reduce_kernel[(triton.cdiv(axis_len, block),)](
                grad_output,
                out,
                beta,
                M,
                N,
                grad_output.stride(0),
                grad_output.stride(1),
                SUM_ROWS=plan == "row",
                BLOCK=block,
                num_warps=4,
            )
        return out


def _transpose_pattern(csr):
    """Transpose a CSR pattern into CSC over the stored entries only.

    Returns ``(crow, row, value)`` describing ``csr.T`` in CSC form: entries
    ``crow[k]:crow[k + 1]`` hold row ``k`` of the transpose.  This stage stays
    linear in ``nnz`` and never touches the dense extents of ``csr``.
    """
    K = csr.shape[1]
    nnz = csr.values().numel()
    crow = torch.zeros(K + 1, dtype=torch.int32, device=csr.device)
    row = torch.empty(nnz, dtype=torch.int32, device=csr.device)
    value = torch.empty(nnz, dtype=csr.values().dtype, device=csr.device)
    if nnz == 0:
        return crow, row, value
    # ``crow`` holds the histogram, then the offsets; the scatter advances a
    # disposable copy so the immutable offsets survive for the caller.
    with torch_device_fn.device(csr.device):
        _sparse_addmm_hist_kernel[(csr.shape[0],)](
            csr.crow_indices(), csr.col_indices(), crow, BLOCK=_BLOCK
        )
        _sparse_addmm_scan_kernel[(1,)](crow, crow, K, BLOCK_K=_BLOCK)
        cursor = crow.clone()
        _sparse_addmm_scatter_kernel[(csr.shape[0],)](
            csr.crow_indices(),
            csr.col_indices(),
            csr.values(),
            cursor,
            row,
            value,
            BLOCK=_BLOCK,
        )
    return crow, row, value


def _csr_of(mat1):
    """Return ``mat1`` itself in CSR form without a dense conversion."""
    if mat1.layout is torch.sparse_csr:
        return mat1
    if mat1.layout is torch.sparse_csc:
        # csc.T is exactly the CSR view of the same stored entries, so the
        # transpose is free: only the index vectors swap roles.
        return torch.sparse_csr_tensor(  # allow-torch-compute: container ctor, no device math
            mat1.ccol_indices(),
            mat1.row_indices(),
            mat1.values(),
            size=(mat1.shape[1], mat1.shape[0]),
            dtype=mat1.dtype,
            device=mat1.device,
        )
    if mat1.layout is torch.sparse_coo:
        coalesced = mat1.coalesce() if not mat1.is_coalesced() else mat1
        return coalesced.to_sparse_csr()
    raise RuntimeError(
        f"_sparse_addmm_backward: mat1 must be CSR, CSC or COO, "
        f"but got {mat1.layout}"
    )


def _sample_csr(csr, dense, alpha):
    """Sample ``alpha * dense`` at a CSR pattern, in stored entry order."""
    values = torch.empty(
        csr.values().numel(), dtype=csr.values().dtype, device=csr.device
    )
    if values.numel():
        with torch_device_fn.device(csr.device):
            _sparse_addmm_sample_kernel[(csr.shape[0],)](
                dense,
                csr.crow_indices(),
                csr.col_indices(),
                values,
                alpha,
                dense.stride(0),
                dense.stride(1),
            )
    return values


def _sample_csc(csc, dense, alpha):
    """Sample ``alpha * dense`` at a CSC pattern, in stored entry order.

    CSC stores column ``j`` contiguously, so the dense product is walked with
    its column coordinate from ``ccol`` and its row coordinate from
    ``row_indices``: the transpose of the usual CSR walk.
    """
    values = torch.empty(
        csc.values().numel(), dtype=csc.values().dtype, device=csc.device
    )
    if values.numel():
        with torch_device_fn.device(csc.device):
            _sparse_addmm_sample_transposed_kernel[(csc.shape[1],)](
                dense,
                csc.ccol_indices(),
                csc.row_indices(),
                values,
                alpha,
                dense.stride(0),
                dense.stride(1),
            )
    return values


def _sparse_mat1_grad(mat1, dense, alpha):
    """Sample ``alpha * dense`` at mat1's pattern, keeping mat1's layout.

    ``dense`` is ``grad_output @ mat2.T`` in mat1's logical coordinates.  A
    CSC input is sampled through the CSR form of its transpose, because both
    describe the same column-major stored order.
    """
    layout = mat1.layout
    if layout is torch.sparse_coo:
        # Coalesced COO indices and its CSR conversion share row-major order.
        pattern = mat1.coalesce()
        values = _sample_csr(_csr_of(pattern), dense, alpha)
        return torch.sparse_coo_tensor(  # allow-torch-compute: container constructor
            pattern.indices(),
            values,
            size=mat1.shape,
            dtype=mat1.dtype,
            device=mat1.device,
            is_coalesced=True,
        )
    if layout is torch.sparse_csc:
        values = _sample_csc(mat1, dense, alpha)
        return torch.sparse_csc_tensor(  # allow-torch-compute: container ctor, no device math
            mat1.ccol_indices(),
            mat1.row_indices(),
            values,
            size=mat1.shape,
            dtype=mat1.dtype,
            device=mat1.device,
        )
    values = _sample_csr(mat1, dense, alpha)
    return (
        torch.sparse_csr_tensor(  # allow-torch-compute: container ctor, no device math
            mat1.crow_indices(),
            mat1.col_indices(),
            values,
            size=mat1.shape,
            dtype=mat1.dtype,
            device=mat1.device,
        )
    )


def _dense_grad_mat2(mat1, grad_output, alpha):
    """``alpha * mat1.T @ grad_output`` over the stored entries only."""
    K, N = mat1.shape[1], grad_output.shape[1]
    out = torch.empty((K, N), dtype=grad_output.dtype, device=grad_output.device)
    if K == 0 or N == 0:
        return out
    if mat1.values().numel() == 0:
        return out.zero_()
    # ``mat1.T @ grad_output`` needs the CSR form of mat1 itself.  For CSC
    # input the stored entries are already mat1.T in CSR order, which is the
    # transposed pattern the SpMM consumes, so no extra transpose is needed.
    if mat1.layout is torch.sparse_csc:
        crow, row, value = mat1.ccol_indices(), mat1.row_indices(), mat1.values()
    else:
        crow, row, value = _transpose_pattern(_csr_of(mat1))
    block_n = _spmm_block_n(N)
    with torch_device_fn.device(grad_output.device):
        _sparse_addmm_spmm_kernel[(K, triton.cdiv(N, block_n))](
            crow,
            row,
            value,
            grad_output,
            out,
            alpha,
            N,
            grad_output.stride(0),
            grad_output.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_N=block_n,
            num_warps=4,
        )
    return out


def _dense_matmul(a, b):
    """`a @ b` through the live FlagGems ``mm`` entry (no torch fallback)."""
    import flag_gems

    # Empty tensors may share a null data pointer and be misidentified as
    # a transpose pair by the SYRK dispatch (which has no empty launch).
    if a.shape[0] == 0 or b.shape[1] == 0 or a.shape[1] == 0:
        return torch.zeros((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    return flag_gems.mm(a, b)


def sparse_addmm_backward(
    grad_output, self, mat1, mat2, *, beta=1.0, alpha=1.0, needs_input_grad=None
):
    """First-order gradients of ``beta * self + alpha * (mat1 @ mat2)``.

    Returns ``(grad_self, grad_mat1, grad_mat2)`` in ATen argument order; an
    entry is ``None`` when the caller passes a false ``needs_input_grad`` (or
    never creates a gradient for that input).  ``mat1`` keeps its own sparse
    layout (CSR/CSC/COO), indices and values, so the sparse gradient is sampled
    at its stored pattern instead of being densified.
    """
    if needs_input_grad is None:
        needs_input_grad = (True, True, True)
    if mat1.layout not in (torch.sparse_csr, torch.sparse_csc, torch.sparse_coo):
        raise RuntimeError(
            f"_sparse_addmm_backward: mat1 must be CSR, CSC or COO, "
            f"but got {mat1.layout}"
        )

    grad_self = _bias_grad(self, grad_output, beta) if needs_input_grad[0] else None
    grad_mat1 = None
    grad_mat2 = None
    if needs_input_grad[1]:
        dense = _dense_matmul(grad_output, mat2.transpose(0, 1))
        grad_mat1 = _sparse_mat1_grad(mat1, dense, alpha)
    if needs_input_grad[2]:
        grad_mat2 = _dense_grad_mat2(mat1, grad_output, alpha)
    return grad_self, grad_mat1, grad_mat2


class _SparseAddmmFunction(torch.autograd.Function):
    """Autograd bridge for ``_sparse_addmm``'s engine-derived gradient.

    ATen owns the ``SparseAddmmBackward0`` formula and FlagGems only serves the
    device key, so the native sparse CUDA backward would run even when the
    forward kernel is FlagGems'.  Pairing this wrapper with an Autograd
    fallthrough for ``aten::_sparse_addmm`` (the ``clone`` /
    ``adaptive_avg_pool3d`` precedent) routes the differentiable path through
    the Triton kernels while inference keeps the forward-only entry.
    """

    @staticmethod
    def forward(ctx, self, mat1, mat2, beta, alpha):
        ctx.save_for_backward(self, mat1, mat2)
        ctx.beta = beta
        ctx.alpha = alpha
        from flag_gems.ops._sparse_addmm import _sparse_addmm_impl

        return _sparse_addmm_impl(self, mat1, mat2, beta=beta, alpha=alpha)

    @staticmethod
    def backward(ctx, grad_output):
        self, mat1, mat2 = ctx.saved_tensors
        grad_self, grad_mat1, grad_mat2 = sparse_addmm_backward(
            grad_output,
            self,
            mat1,
            mat2,
            beta=ctx.beta,
            alpha=ctx.alpha,
            needs_input_grad=ctx.needs_input_grad,
        )
        return grad_self, grad_mat1, grad_mat2, None, None
