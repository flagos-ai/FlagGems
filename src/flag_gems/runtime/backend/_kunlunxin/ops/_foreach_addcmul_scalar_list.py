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
    values = foreach._resolve_scalars(self, value)
    dtype = self[0].dtype
    device = self[0].device
    n = len(self)

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
