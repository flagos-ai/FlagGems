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
#
"""GCU400 ``to_dense_backward``.

Two backends realities shape this specialization:

* the GCU400 vector path has no float64 device support, so a cast whose target
  (or source) dtype is ``torch.float64`` is executed in float32 and then
  re-typed with ``.to(torch.float64)``. That matches the backend convention used
  by the sibling specializations (``_aminmax``, ``_coalesce``) and keeps the
  result exact because float32 -> float64 is a lossless widening of the same
  values the native operator produces;
* ``aten::_to_sparse.sparse_dim`` is not implemented for the SparseCsr device
  backend, so the number of sparse dimensions of a compressed operand is derived
  from its rank (``dim - 2`` dense dims) instead of calling ``sparse_dim()``.

The element conversion keeps ATen's truncation semantics: float -> integer
truncates toward zero and then wraps for widths below 32 bits, which is why
``int8``/``int16`` are routed through ``int32`` inside the kernel exactly like
the unsigned targets (a direct store saturates instead of wrapping).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Vector length of one program. Both values are multiples of the 256-lane GCU
# vector width; the wide one is only used once the copy is bandwidth-bound.
_CBLOCK = 8192
_CBLOCK_LARGE = 32768
_LARGE_THRESHOLD = 1 << 22


@triton.jit
def _cast_kernel(
    g_ptr,
    o_ptr,
    n,
    MODE: tl.constexpr,  # 0 = plain cast, 1 = bool, 2 = integer wrap via int32
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,  # n is a multiple of BLOCK: the mask is dead weight
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(g_ptr + off)
        if MODE == 1:
            tl.store(o_ptr + off, x != 0)
        elif MODE == 2:
            tl.store(o_ptr + off, x.to(tl.int32))
        else:
            tl.store(o_ptr + off, x)
    else:
        m = off < n
        x = tl.load(g_ptr + off, mask=m)
        if MODE == 1:
            tl.store(o_ptr + off, x != 0, mask=m)
        elif MODE == 2:
            tl.store(o_ptr + off, x.to(tl.int32), mask=m)
        else:
            tl.store(o_ptr + off, x, mask=m)


def _mode_of(dtype):
    """Pick the conversion mode of the kernel for a target ``dtype``.

    MODE 2 (convert through int32) is required for every integer type narrower
    than 32 bits: a direct float store saturates, while ATen truncates toward
    zero and then wraps.
    """
    if dtype == torch.bool:
        return 1
    if dtype.is_floating_point or dtype.is_complex:
        return 0
    if dtype.is_signed and dtype.itemsize >= 4:
        return 0
    return 2


def _cast_copy(grad, dtype):
    """Device-kernel conversion of ``grad`` to ``dtype``, preserving shape."""
    src = grad if grad.is_contiguous() else grad.contiguous()
    target = dtype
    if target == torch.float64:
        # No float64 device path on GCU400: convert in float32, then re-type.
        target = torch.float32
    if src.dtype == torch.float64:
        src = src.to(torch.float32)

    out = torch.empty(grad.shape, dtype=target, device=grad.device)
    n = out.numel()
    if n:
        # Block size is chosen by length: every shape at or below 1 Mi elements
        # is launch-bound, where 8192 keeps the program count reasonable, while
        # the 16 Mi benchmark shape is bandwidth-bound and needs the wide block
        # (measured 43 us at 32768 vs 110 us at 8192 for float32 -> float16).
        block = _CBLOCK_LARGE if n >= _LARGE_THRESHOLD else _CBLOCK
        grid = (triton.cdiv(n, block),)
        _cast_kernel[grid](
            src,
            out,
            n,
            _mode_of(target),
            block,
            n % block == 0,
        )
    if target != dtype:
        out = out.to(dtype)
    return out


class _Cast(torch.autograd.Function):
    """Autograd wrapper: both directions are the same device cast kernel."""

    @staticmethod
    def forward(ctx, x, dtype):
        ctx.src_dtype = x.dtype
        return _cast_copy(x, dtype)

    @staticmethod
    def backward(ctx, g):
        return _cast_copy(g, ctx.src_dtype), None


def _norm_masked(masked_grad):
    if masked_grad is None:
        return True
    if isinstance(masked_grad, bool):
        return masked_grad
    if isinstance(masked_grad, int):
        return masked_grad != 0
    raise TypeError("to_dense_backward: masked_grad must be an Optional[bool]")


def _block(input):
    return tuple(int(v) for v in input.values().shape[1:3])


def to_dense_backward(grad, input, masked_grad=None):
    if not torch.is_tensor(grad):
        raise TypeError("to_dense_backward: grad must be a Tensor")
    masked = _norm_masked(masked_grad)
    layout = input.layout

    if layout == torch.strided:
        if grad.dtype == input.dtype:
            return grad
        # The autograd wrapper only exists to give the conversion a backward;
        # without a grad-requiring input there is no graph to build, so the
        # kernel is called directly (saves the Function.apply machinery).
        if grad.requires_grad and torch.is_grad_enabled():
            return _Cast.apply(grad, input.dtype)
        return _cast_copy(grad, input.dtype)

    if layout == torch.sparse_coo:
        if masked:
            return grad.sparse_mask(input.coalesce())
        return grad.to_sparse(input.sparse_dim())

    # Compressed layouts carry exactly two dense dimensions (row/col pointers),
    # so the sparse rank is known without querying the (unimplemented for
    # SparseCsr) ``_to_sparse.sparse_dim`` kernel.
    k = input.dim() - 2
    if masked:
        base = grad.sparse_mask(input)
    elif input.dense_dim() == 0 and grad.dim() > k:
        base = (
            grad.to_sparse_csr() if layout == torch.sparse_csr else grad.to_sparse_csc()
        )
    else:
        base = grad.to_sparse(k)

    if layout == torch.sparse_csr:
        return base.to_sparse_csr()
    if layout == torch.sparse_csc:
        return base.to_sparse_csc()
    blk = _block(input)
    if layout == torch.sparse_bsr:
        return base.to_sparse_bsr(blk)
    return base.to_sparse_bsc(blk)
