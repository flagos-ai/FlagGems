"""Kunlunxin vendor override for ``_foreach_addcmul.ScalarList``.

The shared ``foreach_ternary`` executor is a single meta-pointer launch with
``BLOCK=1024`` and no vectorization, which underutilizes the XPU.  This override
splits the list by total traffic volume:

* small/medium (<= ~128 MiB): one meta-pointer launch (single kernel, no
  per-tensor launch overhead),
* large: reuse the tuned ``addcmul`` pointwise_dynamic kernel per tensor; launch
  overhead is amortized by memory time and the direct-pointer form vectorizes.

Non-floating or non-uniform lists fall back to the shared executor, which owns
the exact dtype-promotion and non-contiguous semantics.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import foreach

from ._foreach_direct import _pick_block, _select_ptr
from .addcmul import addcmul_forward

logger = logging.getLogger(__name__)

_TL_DT = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

# Crossover measured 2026-10-14 on KL3: the meta launch runs at ~180 GB/s
# (scalar loads from runtime pointers are not vectorized) while the direct
# pointwise_dynamic kernel vectorizes to >1.5 TB/s.  Below this many bytes of
# total traffic the single launch wins, above it the per-tensor loop wins.
_META_BYTES_LIMIT = 128 * 1024 * 1024

_BLOCK = 8192

# Direct-pointer slots.  Matching the other foreach overrides, one pointer per
# slot per operand list: 4 lists (a, b, c, out) x 16 slots = 64 scalar pointer
# args.  Longer lists fall back to the meta kernel.
_MAX_DIRECT = 16


@triton.jit
def _foreach_addcmul_direct_kernel(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    b8,
    b9,
    b10,
    b11,
    b12,
    b13,
    b14,
    b15,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Direct-pointer addcmul: scalar pointer args keep block-DMA.

    Reading the pointers through a meta buffer forces a scalar load per
    program, which the XPU backend does not vectorize (~100 GB/s); passing the
    pointers as scalar kernel args lowers the accesses to block-DMA
    (~850 GB/s measured for this op).  The per-tensor scalar is passed the
    same way (selected with ``tl.where``), so no device scalar buffer is built
    and copied from the host on every call.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    a_ptr = _select_ptr(
        t, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
    ).to(tl.pointer_type(DT))
    b_ptr = _select_ptr(
        t, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15
    ).to(tl.pointer_type(DT))
    c_ptr = _select_ptr(
        t, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15
    ).to(tl.pointer_type(DT))
    o_ptr = _select_ptr(
        t, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    ).to(tl.pointer_type(DT))
    s = _select_ptr(
        t, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
    ).to(DT)
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + idx)
    y = tl.load(b_ptr + idx)
    z = tl.load(c_ptr + idx)
    tl.store(o_ptr + idx, x + s * y * z)


def _launch_direct(tensors, tensor1, tensor2, values, dtype, device):
    """Launch the direct-pointer kernel; returns the output list or ``None``."""
    nt = len(tensors)
    if nt > _MAX_DIRECT:
        return None
    n_elements = tensors[0].numel()
    for lst in (tensors, tensor1, tensor2):
        if any(t.numel() != n_elements for t in lst):
            return None
    block, num_warps = _pick_block(n_elements)
    if n_elements % block != 0:
        return None
    # One allocator call for every output, then view it into one tensor per
    # list slot: ``nt`` x ``empty_like`` measures ~36us at k=8, a single
    # ``empty`` + ``unbind`` ~8us, and the kernel's grid writes straight into
    # the contiguous rows.  Output pointers are derived from the single buffer
    # base address (one ``data_ptr`` call) instead of one per view.
    buf = torch.empty(nt, n_elements, dtype=dtype, device=device)
    out = list(buf.unbind(0))
    a_ptrs = [t.data_ptr() for t in tensors]
    a_ptrs += [a_ptrs[0]] * (_MAX_DIRECT - nt)
    b_ptrs = [t.data_ptr() for t in tensor1]
    b_ptrs += [b_ptrs[0]] * (_MAX_DIRECT - nt)
    c_ptrs = [t.data_ptr() for t in tensor2]
    c_ptrs += [c_ptrs[0]] * (_MAX_DIRECT - nt)
    base = buf.data_ptr()
    stride = n_elements * dtype.itemsize
    o_ptrs = [base + i * stride for i in range(nt)]
    o_ptrs += [o_ptrs[0]] * (_MAX_DIRECT - nt)
    vs = [float(v) for v in values]
    vs += [vs[0]] * (_MAX_DIRECT - nt)
    grid = (nt, n_elements // block)
    _foreach_addcmul_direct_kernel[grid](
        *a_ptrs,
        *b_ptrs,
        *c_ptrs,
        *o_ptrs,
        *vs,
        nt,
        _TL_DT[dtype],
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return out


@triton.jit
def _foreach_addcmul_meta_kernel(
    meta, scalar_buf, NT: tl.constexpr, DT: tl.constexpr, BLOCK: tl.constexpr
):
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK
    a_ptr = tl.load(meta + t).to(tl.pointer_type(DT))
    b_ptr = tl.load(meta + NT + t).to(tl.pointer_type(DT))
    c_ptr = tl.load(meta + 2 * NT + t).to(tl.pointer_type(DT))
    o_ptr = tl.load(meta + 3 * NT + t).to(tl.pointer_type(DT))
    n = tl.load(meta + 4 * NT + t)
    s = tl.load(scalar_buf + t).to(DT)
    idx = offset + tl.arange(0, BLOCK)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m)
    b = tl.load(b_ptr + idx, mask=m)
    c = tl.load(c_ptr + idx, mask=m)
    tl.store(o_ptr + idx, a + s * b * c, mask=m)


def _is_fast_path(self, tensor1, tensor2):
    if not self or not tensor1 or not tensor2:
        return False
    dtype = self[0].dtype
    if not dtype.is_floating_point or dtype not in _TL_DT:
        return False
    device = self[0].device
    for a, b, c in zip(self, tensor1, tensor2):
        if a.dtype != dtype or b.dtype != dtype or c.dtype != dtype:
            return False
        if a.device != device or b.device != device or c.device != device:
            return False
        if not (a.is_contiguous() and b.is_contiguous() and c.is_contiguous()):
            return False
    return True


def _fast_path(self, tensor1, tensor2, value):
    values, _, _ = foreach._resolve_scalars(self, value)
    dtype = self[0].dtype
    device = self[0].device
    n = len(self)

    out = _launch_direct(self, tensor1, tensor2, values, dtype, device)
    if out is not None:
        return out
    out = [torch.empty_like(a) for a in self]
    numels = [a.numel() for a in self]
    total_bytes = sum(numels) * dtype.itemsize * 4

    if total_bytes <= _META_BYTES_LIMIT:
        meta = torch.tensor(
            [a.data_ptr() for a in self]
            + [b.data_ptr() for b in tensor1]
            + [c.data_ptr() for c in tensor2]
            + [o.data_ptr() for o in out]
            + numels,
            dtype=torch.int64,
        ).to(device, non_blocking=True)
        scalar_buf = torch.tensor(values, dtype=dtype).to(device, non_blocking=True)
        maxn = max(numels)
        grid = (n, triton.cdiv(maxn, _BLOCK))
        _foreach_addcmul_meta_kernel[grid](meta, scalar_buf, n, _TL_DT[dtype], _BLOCK)
    else:
        for i, (a, b, c) in enumerate(zip(self, tensor1, tensor2)):
            addcmul_forward(a, b, c, values[i], out0=out[i])
    return out


def _foreach_addcmul_ScalarList(self, tensor1, tensor2, value=1):
    self = foreach.check_tensor_list(self)
    tensor1 = foreach.check_tensor_list(tensor1)
    tensor2 = foreach.check_tensor_list(tensor2)
    foreach._check_same_length(self, tensor1, "tensor list")
    foreach._check_same_length(self, tensor2, "tensor list")

    if _is_fast_path(self, tensor1, tensor2):
        return _fast_path(self, tensor1, tensor2, value)
    return foreach.foreach_ternary(self, tensor1, tensor2, value, _addcmul_fn)


@triton.jit
def _addcmul_fn(a, b, c, v):
    return a + v * b * c
