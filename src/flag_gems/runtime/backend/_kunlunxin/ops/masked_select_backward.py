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

"""Kunlunxin XPU3 override for ``aten::masked_select_backward``.

Differences from the generic implementation, all forced by XPU3 behaviour:

* The generic tail blocks read the mask with ``tl.load(..., mask=..., other=0)``.
  On XPU3 a masked load returns *in-bounds* lanes instead of ``other``
  (``pattern-xpu3-masked-load-other-inbounds-leak``), so the per-block popcount
  of the tail was inflated and the generic ``total_count`` validation raised
  "Number of elements of source < number of ones in mask" for every shape whose
  numel is not a multiple of the tile (e.g. 4097, 32769, 65537, 262145).
  Here every mask read uses a clamped index plus an explicit ``off < n``
  predicate, so no masked load is ever issued and no all-false mask can abort a
  program (``pattern-xpu3-all-false-mask-load-aborts``).
* The generic large path launches only ``sm_count`` (=8) programs and walks all
  blocks in a serial loop.  Here the apply phase is one program per block (the
  measured peak on this backend); block prefixes come from a workspace scan.
  (4096,4096): 10.19 ms -> ~2.1 ms.
* Output stores are unmasked: the destination is allocated padded to a block
  multiple and the tail lanes land in the padding (the returned tensor is a
  view of the first ``n`` elements), avoiding the slow / unreliable masked and
  sub-128-lane stores.  The block is never wider than 16384 lanes; a bare
  launch at 32768 lanes is known to fault on XPU3
  (``language-xpu3-bare-launch-block32768-illegal-address``).
* ``aten::masked_select_backward`` returns a *contiguous* tensor (it is
  ``zeros_like + masked_scatter_``), while the generic override preserved the
  input layout.  ``test_*_preserves_layout`` asserts stride equality, so the
  result here is always contiguous.

The compaction stays output-centric (a monotone compressed gather from ``grad``)
because atomics are unreliable on XPU3 and a data-dependent scatter store is a
known ~0.03 GB/s wall.  ``tl.gather`` (SRAM gather) does not legalize on this
backend, so the monotone global gather is the best available primitive.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.contiguous import contiguous
from flag_gems.ops.copy import copy_
from flag_gems.ops.neg import neg
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MAX_BLOCK = 16384  # bare launch at 32768 lanes faults on XPU3
_SCAN_GROUP = 8192  # lanes a single scan program may own


@libentry()
@triton.jit
def _msb_single_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    total_ptr,
    n_elements,
    grad_numel,
    BLOCK: tl.constexpr,
    HAS_GRAD: tl.constexpr,
):
    off = tl.arange(0, BLOCK)
    valid = off < n_elements
    selected = (tl.load(mask_ptr + tl.minimum(off, n_elements - 1)) != 0) & valid
    selected_int = selected.to(tl.int32)
    tl.store(total_ptr, tl.sum(selected_int, axis=0))
    if HAS_GRAD:
        g = tl.cumsum(selected_int, axis=0) - 1
        g = tl.minimum(tl.maximum(g, 0), grad_numel - 1)
        values = tl.load(grad_ptr + g)
        zero = tl.zeros((BLOCK,), dtype=values.dtype)
        tl.store(out_ptr + off, tl.where(selected, values, zero))
    else:
        zero = tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptr + off, zero)


@libentry()
@triton.jit
def _msb_count_kernel(mask_ptr, parts_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    selected = (tl.load(mask_ptr + tl.minimum(off, n_elements - 1)) != 0) & (
        off < n_elements
    )
    tl.store(parts_ptr + pid, tl.sum(selected.to(tl.int32), axis=0))


@libentry()
@triton.jit
def _msb_scan_group_kernel(buf_ptr, group_ptr, n, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * GROUP + tl.arange(0, GROUP)
    valid = off < n
    v = tl.load(buf_ptr + tl.minimum(off, n - 1))
    v = tl.where(valid, v, 0)
    tl.store(buf_ptr + off, tl.cumsum(v, axis=0) - v, mask=off < n)
    tl.store(group_ptr + pid, tl.sum(v, axis=0))


@libentry()
@triton.jit
def _msb_scan_add_kernel(buf_ptr, group_ptr, n, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * GROUP + tl.arange(0, GROUP)
    base = tl.load(group_ptr + pid)
    v = tl.load(buf_ptr + tl.minimum(off, n - 1))
    tl.store(buf_ptr + off, v + base, mask=off < n)


@libentry()
@triton.jit
def _msb_scan_final_kernel(buf_ptr, total_ptr, n, NP: tl.constexpr):
    off = tl.arange(0, NP)
    valid = off < n
    v = tl.load(buf_ptr + tl.minimum(off, n - 1))
    v = tl.where(valid, v, 0)
    tl.store(total_ptr, tl.sum(v, axis=0))
    tl.store(buf_ptr + off, tl.cumsum(v, axis=0) - v, mask=valid)


@libentry()
@triton.jit
def _msb_apply_kernel(
    grad_ptr,
    mask_ptr,
    parts_ptr,
    out_ptr,
    n_elements,
    grad_numel,
    BLOCK: tl.constexpr,
    HAS_GRAD: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    base = tl.load(parts_ptr + pid)
    selected = (tl.load(mask_ptr + tl.minimum(off, n_elements - 1)) != 0) & (
        off < n_elements
    )
    if HAS_GRAD:
        g = base + tl.cumsum(selected.to(tl.int32), axis=0) - 1
        g = tl.minimum(tl.maximum(g, 0), grad_numel - 1)
        values = tl.load(grad_ptr + g)
        zero = tl.zeros((BLOCK,), dtype=values.dtype)
        tl.store(out_ptr + off, tl.where(selected, values, zero))
    else:
        zero = tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptr + off, zero)


def _tile_for(numel):
    """Block ladder measured on XPU3 (one program per block above 16384)."""
    if numel <= 16384:
        # A sub-64-lane store spills on XPU3; pad tiny tiles up to 64 lanes.
        return max(64, triton.next_power_of_2(numel)), 4
    if numel <= 65536:
        return 2048, 8
    return _MAX_BLOCK, 8


def _scan_prefix(parts, total, nblk, device, index_dtype):
    """Exclusive prefix sum of ``parts`` in place and write the total."""
    level = 0
    saved = []
    n = nblk
    cur = parts
    while n > _SCAN_GROUP:
        ng = triton.cdiv(n, _SCAN_GROUP)
        groups = torch.empty(ng, dtype=index_dtype, device=device)
        _msb_scan_group_kernel[(ng,)](
            cur, groups, n, GROUP=_SCAN_GROUP, num_warps=4
        )
        saved.append((cur, groups, n))
        cur = groups
        n = ng
        level += 1
    _msb_scan_final_kernel[(1,)](
        cur, total, n, NP=triton.next_power_of_2(n), num_warps=4
    )
    for buf, groups, nn in reversed(saved):
        _msb_scan_add_kernel[(triton.cdiv(nn, _SCAN_GROUP),)](
            buf, groups, nn, GROUP=_SCAN_GROUP, num_warps=4
        )


def _msb_flat(grad, mask_flat, numel, dtype, device, validate=True):
    """Return a contiguous 1-D tensor of ``numel`` elements (shape ``mask``)."""
    if numel == 0:
        return torch.empty((0,), dtype=dtype, device=device)

    grad = contiguous(grad)
    grad_numel = grad.numel()
    has_grad = grad_numel > 0
    block, warps = _tile_for(numel)

    if block >= numel:
        # One program owns the whole tensor; no workspace / scan kernels.
        out = torch.empty(triton.next_power_of_2(numel), dtype=dtype, device=device)
        total = torch.empty((), dtype=torch.int32, device=device)
        with torch_device_fn.device(device):
            _msb_single_kernel[(1,)](
                grad,
                mask_flat,
                out,
                total,
                numel,
                grad_numel,
                BLOCK=block,
                HAS_GRAD=has_grad,
                num_warps=warps,
            )
        result = out[:numel]
        if validate and grad_numel < numel and grad_numel < total.item():
            raise RuntimeError(
                "Number of elements of source < number of ones in mask"
            )
        return result

    nblk = triton.cdiv(numel, block)
    index_dtype = torch.int32 if numel < 2**31 else torch.int64
    out = torch.empty(nblk * block, dtype=dtype, device=device)
    parts = torch.empty(nblk, dtype=index_dtype, device=device)
    total = torch.empty((), dtype=index_dtype, device=device)
    with torch_device_fn.device(device):
        _msb_count_kernel[(nblk,)](
            mask_flat, parts, numel, BLOCK=block, num_warps=warps
        )
        _scan_prefix(parts, total, nblk, device, index_dtype)
        _msb_apply_kernel[(nblk,)](
            grad,
            mask_flat,
            parts,
            out,
            numel,
            grad_numel,
            BLOCK=block,
            HAS_GRAD=has_grad,
            num_warps=warps,
        )
    result = out[:numel]
    if validate and grad_numel < numel and grad_numel < total.item():
        raise RuntimeError("Number of elements of source < number of ones in mask")
    return result


def _broadcast_views(input, mask):
    ndim = max(input.ndim, mask.ndim)
    input_shape = (1,) * (ndim - input.ndim) + tuple(input.shape)
    mask_shape = (1,) * (ndim - mask.ndim) + tuple(mask.shape)
    input_strides = (0,) * (ndim - input.ndim) + tuple(input.stride())
    mask_strides = (0,) * (ndim - mask.ndim) + tuple(mask.stride())
    shape = []
    for left, right in zip(input_shape, mask_shape):
        if left == right or right == 1:
            shape.append(left)
        elif left == 1:
            shape.append(right)
        else:
            raise RuntimeError("input and mask shapes cannot be broadcast together")
    input_strides = tuple(
        0 if size == 1 and output != 1 else stride
        for size, output, stride in zip(input_shape, shape, input_strides)
    )
    mask_strides = tuple(
        0 if size == 1 and output != 1 else stride
        for size, output, stride in zip(mask_shape, shape, mask_strides)
    )
    return input.as_strided(shape, input_strides), mask.as_strided(shape, mask_strides)


def masked_select_backward(grad, input, mask):
    logger.debug("GEMS_KUNLUNXIN MASKED_SELECT_BACKWARD")

    if mask.dtype != torch.bool:
        raise RuntimeError(
            f"masked_scatter_ only supports boolean masks, but got {mask.dtype}"
        )
    if grad.dtype != input.dtype:
        raise RuntimeError(
            "masked_scatter: expected self and source to have same dtypes but got "
            f"{input.dtype} and {grad.dtype}"
        )
    if grad.device != input.device or mask.device != input.device:
        raise RuntimeError("grad, input, and mask must be on the same device")

    input_expanded, mask_expanded = _broadcast_views(input, mask)
    shape = tuple(input_expanded.shape)
    numel = input_expanded.numel()
    device = input.device

    # ATen returns a contiguous tensor regardless of the input layout.
    result = torch.empty(shape, dtype=input.dtype, device=device)
    if numel == 0:
        return result

    mask_flat = contiguous(mask_expanded).reshape(-1)

    if input.is_complex():
        physical_grad = grad.conj() if grad.is_conj() else grad
        grad_parts = torch.view_as_real(physical_grad)
        real = _msb_flat(grad_parts[..., 0], mask_flat, numel, grad_parts.dtype, device)
        imag_src = grad_parts[..., 1]
        if grad.is_conj():
            imag_src = neg(contiguous(imag_src))
        imag = _msb_flat(imag_src, mask_flat, numel, grad_parts.dtype, device, False)
        result_parts = torch.view_as_real(result)
        copy_(result_parts[..., 0], real.view(shape))
        copy_(result_parts[..., 1], imag.view(shape))
        return result

    flat = _msb_flat(grad, mask_flat, numel, input.dtype, device)
    return flat.view(shape)
