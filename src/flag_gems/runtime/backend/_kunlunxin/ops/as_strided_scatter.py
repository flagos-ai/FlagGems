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
"""Kunlunxin(XPU) specialization of ``aten::as_strided_scatter``.

Two XPU facts drive this implementation (both measured on XPU 1, see
``harness/solution/performance/as_strided_scatter_xpu1_20260830.md``):

1. The generic implementation's ``tl.store(out + target, values,
   mask=mask & is_selected_writer)`` silently writes garbage whenever the
   store mask is a *compound* predicate derived from a discrete ``tl.load``
   with ``other=``.  ``values`` and the predicate are both computed
   correctly, but the payload that lands in memory is not the payload that
   was requested.  Dropping the compound mask makes the very same discrete
   store correct.
2. On this backend a *strided/discrete store* runs at roughly 1-4 GB/s while
   a *strided/discrete load* and a *contiguous store* both run at ~50 GB/s.

So instead of scattering ``src`` into the cloned storage, we walk the output
storage contiguously and *gather*: every storage slot decides for itself
whether it belongs to the as-strided view.  That removes the compound-mask
store (correctness) and replaces the discrete store with a contiguous one
(performance).  No mask is used anywhere; every address is either provably
in range or clamped into range, and ``tl.where`` does the gating.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Maximum view rank handled by the arithmetic-inverse (gather) kernel.
_MAX_RANK = 6
# Smallest tile we are willing to emit; a 64-element vector store is the
# native store granularity on this backend.
_MIN_BLOCK = 64
_MAX_BLOCK = 1024
# Pure contiguous copies keep scaling past 1024 elements per program; the
# gather-flavoured select pass does not (measured on XPU 1, 2026-08-30).
_COPY_BLOCK = 8192
_INT32_GUARD = (1 << 31) - (1 << 16)


@triton.jit
def _fill_i32_kernel(
    dst,
    value,
    NLAST: tl.constexpr,
    START: tl.constexpr,
    CLAMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    j = START + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CLAMP:
        j = tl.minimum(j, NLAST)
    tl.store(dst + j, tl.zeros((BLOCK,), tl.int32) + value)


@triton.jit
def _owner_scatter_kernel(
    owners,
    sizes,
    strides,
    n_src,
    storage_offset,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """First-writer-wins bookkeeping for views we cannot invert."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    i = tl.minimum(i, n_src - 1)
    remaining = i
    target = tl.zeros((BLOCK,), tl.int32) + storage_offset
    for reverse_dim in tl.static_range(NDIM):
        dim = NDIM - reverse_dim - 1
        dim_size = tl.load(sizes + dim)
        dim_stride = tl.load(strides + dim)
        target += (remaining % dim_size) * dim_stride
        remaining = remaining // dim_size
    tl.atomic_min(owners + target, i)


@triton.jit
def _owner_gather_kernel(
    base,
    src,
    owners,
    out,
    n_src,
    NLAST: tl.constexpr,
    START: tl.constexpr,
    CLAMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    j = START + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CLAMP:
        j = tl.minimum(j, NLAST)
    owner = tl.load(owners + j)
    inside = owner < n_src
    sidx = tl.minimum(tl.maximum(owner, 0), n_src - 1)
    tl.store(out + j, tl.where(inside, tl.load(src + sidx), tl.load(base + j)))


@triton.jit
def _view_select_kernel(
    base,
    src,
    out,
    S0: tl.constexpr,
    D0: tl.constexpr,
    X0: tl.constexpr,
    S1: tl.constexpr,
    D1: tl.constexpr,
    X1: tl.constexpr,
    S2: tl.constexpr,
    D2: tl.constexpr,
    X2: tl.constexpr,
    S3: tl.constexpr,
    D3: tl.constexpr,
    X3: tl.constexpr,
    S4: tl.constexpr,
    D4: tl.constexpr,
    X4: tl.constexpr,
    S5: tl.constexpr,
    D5: tl.constexpr,
    X5: tl.constexpr,
    OFFSET: tl.constexpr,
    SRC_MAX: tl.constexpr,
    NDIM: tl.constexpr,
    HAS_VIEW: tl.constexpr,
    FULL: tl.constexpr,
    DIRECT: tl.constexpr,
    NLAST: tl.constexpr,
    START: tl.constexpr,
    CLAMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One contiguous pass over the destination storage.

    ``j`` is a storage index.  The view membership test and the source index
    are recovered arithmetically (greedy divmod over dims sorted by
    descending destination stride, which is exact because the caller has
    verified the strides are properly nested).
    """
    j = START + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CLAMP:
        j = tl.minimum(j, NLAST)

    if FULL:
        # The view covers every storage slot, so no membership test and no
        # address clamping are needed.  Keeping the source index free of
        # ``tl.minimum`` is what lets the backend prove stride-1 and emit a
        # block DMA instead of a gather.
        if DIRECT:
            tl.store(out + j, tl.load(src + j))
        else:
            p = j
            sidx = tl.zeros((BLOCK,), tl.int32)
            if NDIM >= 1:
                c = p // D0
                sidx += c * X0
                p -= c * D0
            if NDIM >= 2:
                c = p // D1
                sidx += c * X1
                p -= c * D1
            if NDIM >= 3:
                c = p // D2
                sidx += c * X2
                p -= c * D2
            if NDIM >= 4:
                c = p // D3
                sidx += c * X3
                p -= c * D3
            if NDIM >= 5:
                c = p // D4
                sidx += c * X4
                p -= c * D4
            if NDIM >= 6:
                c = p // D5
                sidx += c * X5
            tl.store(out + j, tl.load(src + sidx))
    elif HAS_VIEW:
        p = j - OFFSET
        inside = p >= 0
        q = tl.maximum(p, 0)
        sidx = tl.zeros((BLOCK,), tl.int32)
        if NDIM >= 1:
            c = q // D0
            inside = inside & (c < S0)
            c = tl.minimum(c, S0 - 1)
            sidx += c * X0
            q -= c * D0
        if NDIM >= 2:
            c = q // D1
            inside = inside & (c < S1)
            c = tl.minimum(c, S1 - 1)
            sidx += c * X1
            q -= c * D1
        if NDIM >= 3:
            c = q // D2
            inside = inside & (c < S2)
            c = tl.minimum(c, S2 - 1)
            sidx += c * X2
            q -= c * D2
        if NDIM >= 4:
            c = q // D3
            inside = inside & (c < S3)
            c = tl.minimum(c, S3 - 1)
            sidx += c * X3
            q -= c * D3
        if NDIM >= 5:
            c = q // D4
            inside = inside & (c < S4)
            c = tl.minimum(c, S4 - 1)
            sidx += c * X4
            q -= c * D4
        if NDIM >= 6:
            c = q // D5
            inside = inside & (c < S5)
            c = tl.minimum(c, S5 - 1)
            sidx += c * X5
            q -= c * D5
        inside = inside & (q == 0)
        sidx = tl.minimum(sidx, SRC_MAX)
        tl.store(
            out + j,
            tl.where(inside, tl.load(src + sidx), tl.load(base + j)),
        )
    else:
        tl.store(out + j, tl.load(base + j))


# Neutral constexpr payload for ``_view_select_kernel``: a plain storage copy.
_NO_VIEW_META = {
    "S0": 1,
    "D0": 1,
    "X0": 0,
    "S1": 1,
    "D1": 1,
    "X1": 0,
    "S2": 1,
    "D2": 1,
    "X2": 0,
    "S3": 1,
    "D3": 1,
    "X3": 0,
    "S4": 1,
    "D4": 1,
    "X4": 0,
    "S5": 1,
    "D5": 1,
    "X5": 0,
    "OFFSET": 0,
    "SRC_MAX": 0,
    "NDIM": 0,
    "HAS_VIEW": False,
    "FULL": False,
    "DIRECT": False,
}


def _prev_pow2(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def _pick_block(n: int, cap: int = _MAX_BLOCK) -> int:
    if n < _MIN_BLOCK:
        return _MIN_BLOCK
    return min(cap, _prev_pow2(n))


def _launch_1d(kernel, n: int, *args, cap: int = _MAX_BLOCK, **kwargs):
    """Run ``kernel`` over ``[0, n)`` without ever using a store mask.

    Full tiles use a statically affine ``pid * BLOCK + arange`` index so the
    backend can prove stride-1 and emit a block DMA.  A leftover tail is
    covered by a single extra program whose *constexpr* start is
    ``n - BLOCK``; it re-writes already-written slots with identical values,
    which is safe because every value depends only on ``j``.  Storages
    smaller than one tile fall back to a clamped (still unmasked, still
    in-bounds) single program.
    """
    block = _pick_block(n, cap)
    if n < block:
        kernel[(1,)](*args, NLAST=n - 1, START=0, CLAMP=True, BLOCK=block, **kwargs)
        return
    n_full = n // block
    kernel[(n_full,)](*args, NLAST=n - 1, START=0, CLAMP=False, BLOCK=block, **kwargs)
    if n % block:
        kernel[(1,)](
            *args, NLAST=n - 1, START=n - block, CLAMP=False, BLOCK=block, **kwargs
        )


def _plan_view(size, dst_stride, src_stride):
    """Reduce the view to properly nested dims, or return ``None``.

    Dims of extent 1 and dims with destination stride 0 are dropped: their
    coordinate is pinned to 0, which is exactly ATen's "first logical source
    element wins" rule for the aliasing they cause.  Adjacent dims are
    collapsed when both the destination and the source strides collapse.
    The survivors are sorted by descending destination stride and checked
    for proper nesting, which makes greedy divmod an exact inverse.
    """
    dims = [
        [int(sz), int(ds), int(ss)]
        for sz, ds, ss in zip(size, dst_stride, src_stride)
        if int(sz) > 1 and int(ds) != 0
    ]
    if any(d[1] < 0 or d[2] < 0 for d in dims):
        return None

    merged = []
    for dim in dims:
        if merged:
            prev = merged[-1]
            if prev[1] == dim[1] * dim[0] and prev[2] == dim[2] * dim[0]:
                merged[-1] = [prev[0] * dim[0], dim[1], dim[2]]
                continue
        merged.append(list(dim))
    merged.sort(key=lambda d: -d[1])
    if len(merged) > _MAX_RANK:
        return None

    tail_span = 1
    for sz, ds, _ in reversed(merged):
        if ds < tail_span:
            return None
        tail_span += (sz - 1) * ds
    return merged


def _scatter_by_owner(
    base, src, out, size, stride, storage_offset, storage_numel, n_src
):
    """Generic fallback for views we cannot invert arithmetically.

    Still avoids the discrete store: ``owners`` records the winning source
    index per storage slot, then the value pass walks the storage
    contiguously and gathers.  Only reachable for aliasing (non-nested)
    strides, rank > 6 views or int32-overflowing storages.
    """
    device = out.device
    src = src.contiguous()
    sizes = torch.tensor(
        tuple(int(v) for v in size) or (1,), dtype=torch.int32, device=device
    )
    strides = torch.tensor(
        tuple(int(v) for v in stride) or (0,), dtype=torch.int32, device=device
    )
    owners = torch.empty(storage_numel, dtype=torch.int32, device=device)
    _launch_1d(_fill_i32_kernel, storage_numel, owners, n_src)

    block = _pick_block(n_src)
    _owner_scatter_kernel[(triton.cdiv(n_src, block),)](
        owners,
        sizes,
        strides,
        n_src,
        int(storage_offset),
        NDIM=max(len(size), 1),
        BLOCK=block,
    )
    _launch_1d(_owner_gather_kernel, storage_numel, base, src, owners, out, n_src)


def as_strided_scatter(
    self: torch.Tensor,
    src: torch.Tensor,
    size,
    stride,
    storage_offset=None,
) -> torch.Tensor:
    """Clone ``self`` storage and scatter ``src`` through an as-strided view."""
    logger.debug("GEMS_KUNLUNXIN AS_STRIDED_SCATTER")
    if self.device != src.device or self.dtype != src.dtype:
        raise RuntimeError("self and src must have the same device and dtype")
    if self.layout != torch.strided or src.layout != torch.strided:
        raise RuntimeError("as_strided_scatter supports strided tensors only")
    if len(size) != len(stride):
        raise RuntimeError("mismatch in length of strides and shape")

    size = tuple(int(value) for value in size)
    stride = tuple(int(value) for value in stride)
    target_offset = (
        self.storage_offset() if storage_offset is None else int(storage_offset)
    )
    # Let ATen validate negative strides and storage bounds without copying.
    torch.as_strided(self, size, stride, target_offset)
    expected_numel = math.prod(size)
    if src.numel() != expected_numel:
        raise RuntimeError("src size must match the requested as_strided view size")

    storage_numel = self.untyped_storage().nbytes() // self.element_size()
    out_storage = torch.empty(storage_numel, dtype=self.dtype, device=self.device)
    if storage_numel == 0:
        return torch.as_strided(
            out_storage, self.size(), self.stride(), self.storage_offset()
        )
    storage_view = torch.as_strided(self, (storage_numel,), (1,), 0)

    plan = None
    if expected_numel:
        plan = _plan_view(size, stride, src.stride())
        if plan is not None:
            src_max = sum((sz - 1) * ss for sz, _, ss in plan)
            widest = max(storage_numel, src_max + 1, target_offset + 1)
            if widest > _INT32_GUARD:
                plan = None

    with torch_device_fn.device(self.device):
        if not expected_numel:
            # Nothing to scatter: the result is a plain storage clone.
            _launch_1d(
                _view_select_kernel,
                storage_numel,
                storage_view,
                storage_view,
                out_storage,
                cap=_COPY_BLOCK,
                **_NO_VIEW_META,
            )
        elif plan is None:
            _scatter_by_owner(
                storage_view,
                src,
                out_storage,
                size,
                stride,
                target_offset,
                storage_numel,
                expected_numel,
            )
        else:
            src_max = sum((sz - 1) * ss for sz, _, ss in plan)
            meta = dict(_NO_VIEW_META)
            for axis, (sz, ds, ss) in enumerate(plan):
                meta[f"S{axis}"] = sz
                meta[f"D{axis}"] = ds
                meta[f"X{axis}"] = ss
            meta["OFFSET"] = target_offset
            meta["SRC_MAX"] = src_max
            meta["NDIM"] = len(plan)
            meta["HAS_VIEW"] = True
            meta["FULL"] = target_offset == 0 and (
                (len(plan) == 1 and plan[0][1] == 1 and plan[0][0] == storage_numel)
                or (len(plan) == 0 and storage_numel == 1)
            )
            meta["DIRECT"] = meta["FULL"] and (len(plan) == 0 or plan[0][2] == 1)
            _launch_1d(
                _view_select_kernel,
                storage_numel,
                storage_view,
                src,
                out_storage,
                cap=_COPY_BLOCK if meta["FULL"] else _MAX_BLOCK,
                **meta,
            )

    return torch.as_strided(
        out_storage, self.size(), self.stride(), self.storage_offset()
    )
