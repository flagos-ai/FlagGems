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

"""In-place foreach maximum with a scalar (aten::_foreach_maximum_.Scalar).

Reference semantics (torch.maximum elementwise, in place):
    for t in self:  t[i] <- NaN-aware maximum(t[i], scalar)
with the scalar converted to t.dtype before the comparison, NaN in either
operand propagated, and ties resolved to the scalar operand.

Host side only reads metadata and launches Triton kernels; every element of the
mutated tensors is written by a kernel in this module.  Contiguous list entries
are served by a single fused launch, and the hot path reuses the compiled
kernel's own launcher so the per-call host cost stays small.

The tile is chosen so that one thread moves a full 128-bit access: 8 elements
per thread for 1-2 byte dtypes and 4 for wider ones, both with 512-thread
programs (warp_size is 64 on this target).

Elements that alias between list entries cannot corrupt the result: applying
``max(x, s)`` repeatedly to the same location, in any order, always converges to
``max(x_original, s)``, so overlapping (even partially) tensors need no
serialization.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

try:  # stream lookup for the prepared launcher
    from triton.runtime import driver as _triton_driver
except Exception:  # pragma: no cover
    _triton_driver = None

_MAX_ND = 8
_FUSE = 4
_INT32_SAFE_BLOCKS = 1 << 21  # keeps every n/e kernel argument inside int32
# dtype -> (elements per program, warps): 8 elements/thread for narrow types,
# 4 for 4-8 byte types, i.e. one 128-bit access per thread, in 1024-thread
# programs (warp_size is 64 on this target).
_TILE_BY_DTYPE = {
    torch.float16: (8192, 16),
    torch.bfloat16: (8192, 16),
    torch.int16: (8192, 16),
    torch.int8: (8192, 16),
    torch.uint8: (8192, 16),
    torch.bool: (8192, 16),
}
_DEFAULT_TILE = (4096, 16)

# (dtype, device, aligned16) -> prepared launcher for the fused kernel
_LAUNCHERS = {}


@triton.jit
def _mt_tile(x_ptr, n, bpid, pid, s, BLOCK: tl.constexpr):
    offs = (pid - bpid).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    s_d = s.to(x.dtype)
    r = tl.where(x != x, x, tl.where(x > s_d, x, s_d))
    tl.store(x_ptr + offs, r, mask=mask)


@triton.jit
def _max_scalar_fused4_kernel(
    p0,
    p1,
    p2,
    p3,
    n0,
    n1,
    n2,
    n3,
    e0,
    e1,
    e2,
    s,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < e0:
        _mt_tile(p0, n0, 0, pid, s, BLOCK)
    elif pid < e1:
        _mt_tile(p1, n1, e0, pid, s, BLOCK)
    elif pid < e2:
        _mt_tile(p2, n2, e1, pid, s, BLOCK)
    else:
        _mt_tile(p3, n3, e2, pid, s, BLOCK)


@triton.jit
def _max_scalar_index_kernel(
    x_ptr,
    n,
    s,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if ND > 0:
        off += (rem % s0) * t0
        rem = rem // s0
    if ND > 1:
        off += (rem % s1) * t1
        rem = rem // s1
    if ND > 2:
        off += (rem % s2) * t2
        rem = rem // s2
    if ND > 3:
        off += (rem % s3) * t3
        rem = rem // s3
    if ND > 4:
        off += (rem % s4) * t4
        rem = rem // s4
    if ND > 5:
        off += (rem % s5) * t5
        rem = rem // s5
    if ND > 6:
        off += (rem % s6) * t6
        rem = rem // s6
    if ND > 7:
        off += (rem % s7) * t7
        rem = rem // s7
    # data_ptr() already includes storage_offset, so offsets are 0-based.
    x = tl.load(x_ptr + off, mask=mask)
    s_d = s.to(x.dtype)
    r = tl.where(x != x, x, tl.where(x > s_d, x, s_d))
    tl.store(x_ptr + off, r, mask=mask)


def _collapse(t):
    """Collapse a strided view into (size, stride) pairs, dropping size-1 dims."""
    shape = t.shape
    strides = t.stride()
    items = []
    for i in range(len(shape)):
        sz = int(shape[i])
        if sz == 1:
            continue
        items.append((sz, int(strides[i])))
    merged = []
    for sz, st in items:
        if merged and merged[-1][1] == st * sz:
            merged[-1] = (merged[-1][0] * sz, st)
        else:
            merged.append((sz, st))
    return merged


def _launch_strided(t, s, block, warps):
    n = t.numel()
    merged = _collapse(t)
    if len(merged) > _MAX_ND:
        # Extremely rare: peel the leading dimension with host-side views.
        for i in range(merged[0][0]):
            _launch_strided(t[i], s, block, warps)
        return
    sizes = [m[0] for m in merged] + [1] * (_MAX_ND - len(merged))
    strides_ = [m[1] for m in merged] + [0] * (_MAX_ND - len(merged))
    _max_scalar_index_kernel[((n + block - 1) // block,)](
        t,
        n,
        s,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        strides_[0],
        strides_[1],
        strides_[2],
        strides_[3],
        strides_[4],
        strides_[5],
        strides_[6],
        strides_[7],
        ND=len(merged),
        BLOCK=block,
        num_warps=warps,
    )


def _make_launcher(sample, dev_idx, aligned, block, warps):
    """Prepare the compiled kernel's launcher for one (dtype, device, layout)."""
    # ``aligned`` reproduces the divisibility-by-16 attributes the ordinary JIT
    # launch would infer when every size and block boundary is a multiple of 16;
    # otherwise the general (unaligned) variant is built.
    v = 4096 if aligned else 4095
    ck = _max_scalar_fused4_kernel.warmup(
        sample,
        sample,
        sample,
        sample,
        v,
        v,
        v,
        v,
        v,
        v,
        v,
        1.5,
        grid=(1,),
        BLOCK=block,
        num_warps=warps,
    )
    ck._init_handles()
    launch = ck.run
    fn = ck.function
    meta = ck.packed_metadata
    get_stream = _triton_driver.active.get_current_stream

    # The trailing 1 fills the launcher slot that holds the BLOCK constexpr.
    def fast(total, p0, p1, p2, p3, n0, n1, n2, n3, e0, e1, e2, s):
        launch(
            total,
            1,
            1,
            get_stream(dev_idx),
            fn,
            meta,
            None,
            None,
            None,
            p0,
            p1,
            p2,
            p3,
            n0,
            n1,
            n2,
            n3,
            e0,
            e1,
            e2,
            s,
            1,
        )

    return fast


def _launch_group(t0, n0, t1, n1, t2, n2, t3, n3, k, s, dtype, device):
    """One launch for 1..4 contiguous same-dtype/device tensors (unused slots
    are already padded by the caller with n=0 and the first pointer)."""
    block, warps = _TILE_BY_DTYPE.get(dtype, _DEFAULT_TILE)
    e0 = (n0 + block - 1) // block
    if k == 1:
        total = e1 = e2 = e0
    elif k == 2:
        total = e1 = e2 = e0 + ((n1 + block - 1) // block)
    elif k == 3:
        e1 = e0 + ((n1 + block - 1) // block)
        total = e2 = e1 + ((n2 + block - 1) // block)
    else:
        e1 = e0 + ((n1 + block - 1) // block)
        e2 = e1 + ((n2 + block - 1) // block)
        total = e2 + ((n3 + block - 1) // block)
    if (
        _triton_driver is not None
        and total < _INT32_SAFE_BLOCKS
        and ((t0.data_ptr() | t1.data_ptr() | t2.data_ptr() | t3.data_ptr()) & 15) == 0
    ):
        aligned = ((e0 | e1 | e2 | n0 | n1 | n2 | n3) & 15) == 0
        key = (dtype, device, aligned)
        fast = _LAUNCHERS.get(key)
        if fast is None:
            if len(_LAUNCHERS) > 32:
                _LAUNCHERS.clear()
            dev_idx = t0.device.index
            if dev_idx is None:
                dev_idx = torch.cuda.current_device()
            try:
                fast = _make_launcher(t0, dev_idx, aligned, block, warps)
            except Exception:
                fast = False
            _LAUNCHERS[key] = fast
        if fast:
            fast(total, t0, t1, t2, t3, n0, n1, n2, n3, e0, e1, e2, s)
            return
    _max_scalar_fused4_kernel[(total,)](
        t0,
        t1,
        t2,
        t3,
        n0,
        n1,
        n2,
        n3,
        e0,
        e1,
        e2,
        s,
        BLOCK=block,
        num_warps=warps,
    )


def _foreach_maximum__Scalar(self, scalar):
    logger.debug("GEMS_HYGON _FOREACH_MAXIMUM__SCALAR")
    if isinstance(self, torch.Tensor):
        tensors = (self,)
    else:
        tensors = self
    if type(scalar) is not float:
        scalar = (
            float(scalar.item()) if isinstance(scalar, torch.Tensor) else float(scalar)
        )
    n_tensors = len(tensors)
    i = 0
    while i < n_tensors:
        t0 = tensors[i]
        i += 1
        n0 = t0.numel()
        if n0 == 0:
            continue
        dtype = t0.dtype
        if not t0.is_contiguous():
            block, warps = _TILE_BY_DTYPE.get(dtype, _DEFAULT_TILE)
            _launch_strided(t0, scalar, block, warps)
            continue
        device = t0.device
        t1 = t2 = t3 = t0
        n1 = n2 = n3 = 0
        k = 1
        while i < n_tensors and k < _FUSE:
            u = tensors[i]
            un = u.numel()
            if un == 0:
                i += 1
                continue
            if u.dtype is not dtype or u.device != device or not u.is_contiguous():
                break
            if k == 1:
                t1 = u
                n1 = un
            elif k == 2:
                t2 = u
                n2 = un
            else:
                t3 = u
                n3 = un
            k += 1
            i += 1
        _launch_group(t0, n0, t1, n1, t2, n2, t3, n3, k, scalar, dtype, device)
    return None
