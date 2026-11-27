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

"""GCU400 einsum specialization: dispatch common equations to Triton paths inside the use_gems() context.

aten's generic einsum is split into a chain of pointwise + generic reductions on the enflame backend;
`i,i->` on 65536 elements compiles a single-block reduction kernel needing 2.6MB DSM (hardware limit
917504B), which directly raises OutOfResources; `i,j->ij` lands on the broadcast-mul gather path.
This module takes over only within the FlagGems registration context; the native reference still uses the original aten
implementation.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_ORIGINAL_EINSUM = torch.einsum
_DOT_SINGLE_MAX = 4096
_DOT_BLOCK = 4096
_TRACE_SINGLE_MAX = 1024
_TRACE_BLOCK = 256
_GEMM_MM_MAX_DIM = 512
_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@triton.jit
def _einsum_dot_single(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    xv = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    yv = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, tl.sum(xv * yv))


@triton.jit
def _einsum_dot_partial(x_ptr, y_ptr, partial_ptr, N, BLOCK: tl.constexpr):
    # Each program reduces one BLOCK segment; the in-block state vector has fixed length, so DSM is decoupled from N
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    xv = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    yv = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partial_ptr + pid, tl.sum(xv * yv))


@triton.jit
def _einsum_reduce_final(partial_ptr, out_ptr, M, BLOCK_M: tl.constexpr):
    offsets = tl.arange(0, BLOCK_M)
    vals = tl.load(partial_ptr + offsets, mask=offsets < M, other=0.0)
    tl.store(out_ptr, tl.sum(vals))


@triton.jit
def _einsum_trace_single(a_ptr, out_ptr, N, STRIDE, BLOCK: tl.constexpr):
    # The diagonal element a[i, i] has stride stride(0)+stride(1); the upper bound of N is guarded on the host
    offsets = tl.arange(0, BLOCK)
    vals = tl.load(a_ptr + offsets * STRIDE, mask=offsets < N, other=0.0).to(tl.float32)
    tl.store(out_ptr, tl.sum(vals))


@triton.jit
def _einsum_trace_partial(a_ptr, partial_ptr, N, STRIDE, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    vals = tl.load(a_ptr + offsets * STRIDE, mask=offsets < N, other=0.0).to(tl.float32)
    tl.store(partial_ptr + pid, tl.sum(vals))


def _einsum_dot(x, y):
    # Deferred import avoids a circular dependency during flag_gems initialization; fp32 partial sums match einsum
    # reduction semantics
    from flag_gems.runtime import torch_device_fn

    num_elements = x.numel()
    out = torch.empty((), dtype=x.dtype, device=x.device)
    with torch_device_fn.device(x.device):
        if num_elements <= _DOT_SINGLE_MAX:
            _einsum_dot_single[(1,)](
                x,
                y,
                out,
                num_elements,
                BLOCK=triton.next_power_of_2(num_elements),
                num_warps=4,
            )
        else:
            num_partials = triton.cdiv(num_elements, _DOT_BLOCK)
            partial = torch.empty((num_partials,), dtype=torch.float32, device=x.device)
            _einsum_dot_partial[(num_partials,)](
                x, y, partial, num_elements, BLOCK=_DOT_BLOCK, num_warps=8
            )
            _einsum_reduce_final[(1,)](
                partial,
                out,
                num_partials,
                BLOCK_M=triton.next_power_of_2(num_partials),
                num_warps=4,
            )
    return out


def _einsum_trace(a):
    from flag_gems.runtime import torch_device_fn

    n = a.shape[0]
    stride = a.stride(0) + a.stride(1)
    out = torch.empty((), dtype=a.dtype, device=a.device)
    with torch_device_fn.device(a.device):
        if n <= _TRACE_SINGLE_MAX:
            _einsum_trace_single[(1,)](
                a, out, n, stride, BLOCK=triton.next_power_of_2(n), num_warps=4
            )
        else:
            num_partials = triton.cdiv(n, _TRACE_BLOCK)
            partial = torch.empty((num_partials,), dtype=torch.float32, device=a.device)
            _einsum_trace_partial[(num_partials,)](
                a, partial, n, stride, BLOCK=_TRACE_BLOCK, num_warps=4
            )
            _einsum_reduce_final[(1,)](
                partial,
                out,
                num_partials,
                BLOCK_M=triton.next_power_of_2(num_partials),
                num_warps=2,
            )
    return out


def _einsum_gemm_2d(a, b):
    # The FlagGems mm kernel is faster than bmm on small GEMMs and vice versa on large ones (L600 parameter sweep);
    # large shapes go through the bmm kernel via a [None] view, and the result takes a [0] view to restore 2D semantics
    if a.dim() == 2 and b.dim() == 2:
        if max(a.shape[0], a.shape[1], b.shape[1]) <= _GEMM_MM_MAX_DIM:
            return torch.matmul(a, b)
        return torch.matmul(a[None], b[None])[0]
    return torch.matmul(a, b)


def _einsum_gemm_batched(a, b):
    # Batched GEMM with b==1 folds into a 2D mm (small shapes); b>1 or non-square batches keep bmm
    if (
        a.dim() == 3
        and b.dim() == 3
        and a.shape[0] == 1
        and b.shape[0] == 1
        and max(a.shape[1], a.shape[2], b.shape[2]) <= _GEMM_MM_MAX_DIM
    ):
        return torch.matmul(a[0], b[0])[None]
    return torch.matmul(a, b)


def _gems_is_active():
    # use_gems().__enter__ sets this global via enable() and deletes it on exit
    try:
        import flag_gems

        return getattr(flag_gems, "current_work_registrar", None) is not None
    except Exception:
        return False


def _dispatch(equation, operands):
    eq = equation.replace(" ", "")
    if eq == "i,i->":
        x, y = operands[0], operands[1]
        if (
            x.dim() == 1
            and y.dim() == 1
            and x.numel() == y.numel()
            and 0 < x.numel() < 2**31
            and x.dtype == y.dtype
            and x.dtype in _FLOAT_DTYPES
        ):
            # A 1D view may carry a stride; the native aten path crashes on GCU with a DSM overflow for such inputs,
            # so here, following the same precedent as enflame fused outer, we make it contiguous before entering the
            # dedicated kernel
            if not x.is_contiguous():
                x = x.contiguous()
            if not y.is_contiguous():
                y = y.contiguous()
            return _einsum_dot(x, y)
        if x.dim() == 1 and y.dim() == 1 and x.numel() == y.numel():
            # int / mixed dtype: native einsum crashes at compile time on GCU for this shape,
            # so we use the FlagGems mul+sum combination (same promotion semantics as aten)
            return (x * y).sum()
        return _ORIGINAL_EINSUM(equation, *operands)
    if eq == "ii->":
        a = operands[0]
        n = a.shape[0] if a.dim() == 2 else 0
        if (
            a.dim() == 2
            and a.shape[0] == a.shape[1]
            and a.dtype in _FLOAT_DTYPES
            and 0 < n < 2**16
            and n * (n + 1) < 2**31
            and a.is_contiguous()
        ):
            return _einsum_trace(a)
        return torch.diagonal(operands[0]).sum()
    if eq == "i,j->ij":
        # Directly call enflame's row-streaming Triton kernel, avoiding the aten broadcasting mul path
        import flag_gems

        return flag_gems.outer(operands[0], operands[1])
    if eq == "ij,jk->ik":
        return _einsum_gemm_2d(operands[0], operands[1])
    if eq == "bij,bjk->bik":
        return _einsum_gemm_batched(operands[0], operands[1])
    if eq == "...ij,...jk->...ik":
        return torch.matmul(operands[0], operands[1])
    if eq == "ii->i":
        return torch.diagonal(operands[0])
    if eq == "ij->ji":
        return operands[0].transpose(0, 1)
    if eq == "ijk->":
        return operands[0].sum()
    if eq == "ijk->j":
        return operands[0].sum(dim=(0, 2))
    # Uncovered equations keep native aten behavior to avoid affecting other operators
    return _ORIGINAL_EINSUM(equation, *operands)


def einsum(equation, *operands):
    if _gems_is_active():
        try:
            return _dispatch(equation, operands)
        except Exception:
            logger.debug("GCU400 einsum fell back for equation %s", equation)
            return _ORIGINAL_EINSUM(equation, *operands)
    return _ORIGINAL_EINSUM(equation, *operands)


torch.einsum = einsum
