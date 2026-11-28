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

"""Triton implementation of ``aten::_coalesce`` for sparse COO tensors.

``_coalesce(self)`` de-duplicates a sparse COO tensor: it groups the stored
coordinates by their row-major linear index, sorts the unique coordinates in
ascending order and reduces the duplicated values (sum for float/int, logical
OR for bool).  The functional form returns a *fresh* coalesced tensor and never
mutates the input; the ``.out`` form writes into the caller supplied sparse
tensor and returns that exact object.

Strategy (dense accumulation path): the key space ``numel = prod(shape)`` is
materialised when it is small relative to the work, so duplicates are merged by
a single ``atomic_add`` per stored element instead of a sort, and the result is
produced by walking the occupied key slots in ascending order.

At the benchmark sizes (``numel`` up to 2**24, ``nnz`` up to 2**21) the profile
is dominated by the ``atomic_add`` scatter, the occupancy scan/compaction over
the dense key space, and the device->host staging of the emitted coordinates -
in that order - so every small constant is passed as a kernel argument instead
of living in a helper device tensor:

  1. ``_scatter_kernel`` - linearise the coordinates with the dimension strides
     passed as scalar arguments, atomic-add the values into the dense
     accumulator and mark occupancy.  Occupancy is a ``uint8`` sentinel rather
     than ``acc != 0`` because a float group can legitimately sum to exactly
     zero, and a byte-wide map is cheaper to scan than the accumulator.
  2. ``_count_kernel``   - per-tile count of the occupied keys.
  3. ``_scan_kernel`` + ``_keylist_kernel`` - exclusive scan over the per-tile
     counts, then compact the surviving linear keys into an ``int32`` list of
     length ``m`` (the scan also publishes ``m``).  This is the default path;
     only when the tile count exceeds ``_SCAN_MAX_LANES`` is the fused
     ``_keylist_prefix_kernel`` used instead (it re-derives each tile's prefix
     in-program to save one launch, at ~20x the per-byte cost).
  4. one host scalar read of ``m`` to size the output (mirrors the native path,
     which also has to size the result before writing it).
  5. ``_emit_kernel``    - one lane per group: gather the merged value and
     decode the coordinates into the output tensors.

Device handling (torch_gcu): the sparse COO wrapper reports the ATen-visible
device independently of where its indices/values storage lives, so the kernels
run on the accelerator device resolved from the storage (staging host-resident
buffers onto it) and the result is built on that same device, matching the
accelerator-resident tensor the native op returns.  The ``.out`` form keeps the
caller's buffer identity and fills it with a sparse ``copy_`` (the write resizes
``nnz`` but not the shape, so a shape mismatch is rejected exactly like the
native ``aten::resize_``-based path).  fp64 values have no device arithmetic on
gcu400, so they are reinterpreted at fp32 precision for the kernels and the
emitted values are widened back to the declared fp64 dtype (the platform-wide
convention for this backend).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Largest dense key space this implementation materialises.  2**24 covers the
# benchmark shapes (4096x4096 and 256x256x256 are exactly 2**24) while keeping
# the accumulator, occupancy map and the compaction prefix bounded.
_DENSE_MAX = 1 << 24

# Widest sparse dimension this implementation specialises for; the reference
# tests use 1..5 dimensions.
_MAX_SPARSE_DIM = 5

# Tile size shared by the occupancy count and the key-list compaction, plus the
# launch geometry.  Tuned on gcu400 at the benchmark sizes
# (numel = 2**22 .. 2**24): a wider tile cut the flag scan + compaction from
# 3.54 ms to 2.21 ms at numel = 2**24, and a wider scatter block cut the
# scatter from 4.24 ms to 3.59 ms.
#
# Re-check (nnz = 65536 / 2097152, two sizes, median of 3 runs each, host timing includes synchronization):
#   _SCAT_BLOCK 512 -> 5.13ms, 2048 -> 4.65ms, 2048+_SCAT_WARPS 16 -> 4.64ms,
#   _SCAT_WARPS 16 -> 4.68ms, _TILE 4096 -> 4.90ms, _TILE_WARPS 16 + _EMIT_WARPS 8 -> 4.71ms,
# None beat the current combination (4.57ms), indicating the above settings have converged and no further tile sweep is
# needed.
_TILE = 8192
_TILE_WARPS = 8
_SCAT_BLOCK = 1024
_SCAT_WARPS = 8
_EMIT_BLOCK = 1024
_EMIT_WARPS = 4
# Compaction strategy switch on the tile count ``nb``.  Up to
# ``_SCAN_MAX_LANES`` tiles the dedicated exclusive scan over the tile counts
# (``_scan_kernel``) plus the flat ``_keylist_kernel`` is used: it costs
# O(numel) bytes and measured ~8.5 GB/s at numel = 2**24.  Above that the tile
# prefix is folded into ``_keylist_prefix_kernel``, which costs an extra
# O(nb * _PREFIX_LANES) reduction per program (measured ~0.34 GB/s, i.e. ~20x
# slower per byte), so it is only worthwhile when a separate scan launch would
# dominate.  With ``_DENSE_MAX == 2**24`` and ``_TILE == 2048`` the tile count
# never exceeds 2048, hence the scan path is the hot one.
_SCAN_MAX_LANES = 8192
_PREFIX_LANES = 2048

# ``_keylist_word_kernel`` reads the occupancy map by 32-bit word, with 4x fewer load transactions, but adds
# 3 byte extractions and 4 asymmetric write offsets. Measured (float32, 1024x1024/4096x4096 case):
#   at numel = 2**20 it is break-even/slightly worse, starts winning at numel = 2**22, and gives -2.5 ms per call at
# 2**24,
# so switch to the word version only when the key space is large enough (>= 2**22), preserving latency for small cases.
_WORD_KEYLIST_MIN = 1 << 22

_ACC_DTYPE = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
    torch.int8: torch.int32,
    torch.uint8: torch.int32,
    torch.int16: torch.int32,
    torch.int32: torch.int32,
    torch.int64: torch.int64,
    torch.bool: torch.int32,
}


@triton.jit
def _scatter_kernel(
    idx_ptr,
    val_ptr,
    acc_ptr,
    flag_ptr,
    n,
    numel,
    dv,
    s0,
    s1,
    s2,
    s3,
    s4,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Linearise coordinates, atomic-add values and mark occupied keys."""
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offs < n
    k = tl.zeros((BLOCK,), dtype=tl.int64)
    if S > 0:
        k = k + tl.load(idx_ptr + offs, mask=mask, other=0).to(tl.int64) * s0
    if S > 1:
        k = k + tl.load(idx_ptr + n + offs, mask=mask, other=0).to(tl.int64) * s1
    if S > 2:
        k = k + tl.load(idx_ptr + 2 * n + offs, mask=mask, other=0).to(tl.int64) * s2
    if S > 3:
        k = k + tl.load(idx_ptr + 3 * n + offs, mask=mask, other=0).to(tl.int64) * s3
    if S > 4:
        k = k + tl.load(idx_ptr + 4 * n + offs, mask=mask, other=0).to(tl.int64) * s4
    ok = mask & (k >= 0) & (k < numel)
    kk = k.to(tl.int32)
    aty = acc_ptr.dtype.element_ty
    # Each occupied key owns ``dv`` accumulator slots (one per dense element), so
    # the scatter target is ``kk * dv + j``; the int64 product keeps the offset
    # exact when ``numel * dv`` exceeds the int32 range.
    acc_base = kk.to(tl.int64) * dv
    for j in tl.range(0, dv):
        v = tl.load(val_ptr + offs * dv + j, mask=ok, other=0)
        tl.atomic_add(acc_ptr + acc_base + j, v.to(aty), mask=ok)
    tl.store(flag_ptr + kk, tl.full((BLOCK,), 1, tl.uint8), mask=ok)


@triton.jit
def _count_kernel(flag_ptr, cnt_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    f = tl.load(flag_ptr + offs, mask=offs < numel, other=0).to(tl.int32)
    tl.store(cnt_ptr + pid, tl.sum(f, axis=0))


@triton.jit
def _scan_kernel(cnt_ptr, base_ptr, m_ptr, nb, B: tl.constexpr):
    offs = tl.arange(0, B)
    mask = offs < nb
    c = tl.load(cnt_ptr + offs, mask=mask, other=0)
    incl = tl.cumsum(c, axis=0)
    tl.store(base_ptr + offs, incl - c, mask=mask)
    tl.store(m_ptr + tl.arange(0, 1), tl.sum(c, axis=0))


@triton.jit
def _keylist_kernel(flag_ptr, base_ptr, key_ptr, numel, BLOCK: tl.constexpr):
    """Compact the linear keys of the occupied slots into a dense list."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    f = tl.load(flag_ptr + offs, mask=mask, other=0).to(tl.int32)
    incl = tl.cumsum(f, axis=0)
    base = tl.load(base_ptr + pid)
    pos = base + incl - 1
    tl.store(key_ptr + pos, offs, mask=mask & (f == 1))


@triton.jit
def _keylist_word_kernel(flag32_ptr, base_ptr, key_ptr, numel, BLOCK: tl.constexpr):
    """Compact the linear keys of the occupied slots into a dense list.

    The occupancy map is read as 32-bit words (4 keys per read): on gcu400 a
    byte-granular load costs one transaction each, measuring only 4 GB/s for
    numel = 2**20; word reads cut transactions 4x and the prefix sum drops to BLOCK/4.
    """
    pid = tl.program_id(0)
    W: tl.constexpr = BLOCK // 4
    widx = pid * W + tl.arange(0, W)
    mask = widx * 4 < numel
    w = tl.load(flag32_ptr + widx, mask=mask, other=0)
    # Flags are only 0/1, so each byte is both the occupancy bit and the count for that key.
    b0 = w & 0xFF
    b1 = (w >> 8) & 0xFF
    b2 = (w >> 16) & 0xFF
    b3 = (w >> 24) & 0xFF
    cnt = b0 + b1 + b2 + b3
    incl = tl.cumsum(cnt, axis=0)
    base = tl.load(base_ptr + pid)
    wbase = base + incl - cnt
    key0 = widx * 4
    tl.store(key_ptr + wbase, key0, mask=mask & (b0 == 1))
    tl.store(key_ptr + wbase + b0, key0 + 1, mask=mask & (b1 == 1))
    tl.store(key_ptr + wbase + b0 + b1, key0 + 2, mask=mask & (b2 == 1))
    tl.store(key_ptr + wbase + b0 + b1 + b2, key0 + 3, mask=mask & (b3 == 1))


@triton.jit
def _keylist_prefix_kernel(
    flag_ptr,
    cnt_ptr,
    key_ptr,
    total_ptr,
    numel,
    nb,
    PB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compact keys without a precomputed tile scan.

    Each program derives its own tile base by summing the tile counts that
    precede it (so the exclusive-scan launch is not needed) and the last program
    publishes the global group count.

    The occupancy prefix inside a tile is built in two levels: a cumulative sum
    along each 32-lane row, then a cumulative sum of the row totals.  Both are
    expressed with explicit 2D indices rather than a reshape, and a windowed
    measurement showed the pair costs 0.089 ms versus 0.121 ms for a single
    cumulative sum over the whole tile at ``numel`` = 2**24.
    """
    pid = tl.program_id(0)
    po = tl.arange(0, PB)
    cc = tl.load(cnt_ptr + po, mask=po < nb, other=0)
    base = tl.sum(tl.where(po < pid, cc, 0), axis=0)
    if pid == nb - 1:
        tl.store(total_ptr + tl.arange(0, 1), tl.sum(cc, axis=0))
    W: tl.constexpr = BLOCK // 32
    w = tl.arange(0, W)[:, None]
    lane = tl.arange(0, 32)[None, :]
    offs = pid * BLOCK + w * 32 + lane
    mask = offs < numel
    f = tl.load(flag_ptr + offs, mask=mask, other=0).to(tl.int32)
    incl = tl.cumsum(f, axis=1)
    wc = tl.sum(f, axis=1)
    wbase = tl.cumsum(wc, axis=0) - wc
    pos = base + wbase[:, None] + incl - 1
    tl.store(key_ptr + pos, offs, mask=mask & (f == 1))


@triton.jit
def _emit_kernel(
    key_ptr,
    acc_ptr,
    oidx_ptr,
    oval_ptr,
    m,
    dv,
    kstride,
    s0,
    s1,
    s2,
    s3,
    s4,
    d0,
    d1,
    d2,
    d3,
    d4,
    S: tl.constexpr,
    IS_BOOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One lane per group: gather the merged value and decode the coordinates."""
    pid = tl.program_id(0)
    pos = pid * BLOCK + tl.arange(0, BLOCK)
    mask = pos < m
    k = tl.load(key_ptr + pos, mask=mask, other=0).to(tl.int64)
    oty = oval_ptr.dtype.element_ty
    for j in tl.range(0, dv):
        v = tl.load(acc_ptr + k * dv + j, mask=mask, other=0)
        if IS_BOOL:
            r = tl.where(v != 0, 1, 0).to(oty)
        else:
            r = v.to(oty)
        tl.store(oval_ptr + pos.to(tl.int64) * dv + j, r, mask=mask)
    kd = tl.full((), 1, tl.int64) * kstride
    if S > 0:
        tl.store(oidx_ptr + pos, (k // s0) % d0, mask=mask)
    if S > 1:
        tl.store(oidx_ptr + kd + pos, (k // s1) % d1, mask=mask)
    if S > 2:
        tl.store(oidx_ptr + 2 * kd + pos, (k // s2) % d2, mask=mask)
    if S > 3:
        tl.store(oidx_ptr + 3 * kd + pos, (k // s3) % d3, mask=mask)
    if S > 4:
        tl.store(oidx_ptr + 4 * kd + pos, (k // s4) % d4, mask=mask)


def _as_u8(t):
    """Zero-copy reinterpretation used for ``torch.bool`` storage."""
    return t.view(torch.uint8)


def _accel_device(values, idx):
    """Return the accelerator device the kernels should run on.

    torch_gcu's patched ``torch.sparse_coo_tensor`` tags the sparse wrapper as
    ``cpu`` even when the indices/values storage lives on the accelerator, and
    it does not move host-side indices/values to the requested device either,
    so the storage can sit on the accelerator, on the host, or mixed.  Prefer
    the storage's own accelerator device and fall back to the current one.
    """
    for t in (values, idx):
        if t.device.type in ("cuda", "gcu"):
            return t.device
    return torch.device("gcu", torch.gcu.current_device())


def _empty_out_like(shape, sp, dv, dtype, values, device):
    idx = torch.empty((sp, 0), dtype=torch.int64, device=device)
    val = torch.empty((0,) + tuple(values.shape[1:]), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, shape, is_coalesced=True)


def _coalesce_impl(self, out=None):
    """Coalesce a sparse COO tensor ``self`` (optionally into ``out``)."""
    if not isinstance(self, torch.Tensor) or self.layout != torch.sparse_coo:
        raise RuntimeError(
            "_coalesce only supports sparse coordinate (coo) tensors, "
            f"but got layout {getattr(self, 'layout', None)}"
        )
    # Parity with the native op, which asserts ``!self.is_coalesced()``: the
    # caller reaches this entry point only for uncoalesced inputs, because
    # ``Tensor.coalesce()`` short-circuits on already coalesced tensors.
    if self.is_coalesced():
        raise RuntimeError(
            "!self.is_coalesced() INTERNAL ASSERT FAILED: _coalesce expects an "
            "uncoalesced sparse COO tensor"
        )
    dtype = self.dtype
    if dtype not in _ACC_DTYPE:
        raise RuntimeError(f"_coalesce does not support values dtype {dtype}")

    sp = self.sparse_dim()
    shape = tuple(self.shape)
    values = self._values()
    idx = self._indices()
    n = self._nnz()
    dense_shape = tuple(values.shape[1:])
    dv = 1
    for d in dense_shape:
        dv *= d

    # torch_gcu redirects CUDA entry points to the GCU backend, so accelerator
    # tensors report device.type == "gcu" here, and the sparse wrapper reports
    # "cpu" regardless of where its indices/values storage lives.  The result is
    # therefore built with an explicit ``device=run_device`` so that the sparse
    # storage stays on the accelerator: the wrapper still reports ``cpu`` (same
    # as the native op), but no device->host copy of indices/values is needed -
    # which matters because the emitted buffers are O(m * sparse_dim) bytes and
    # a host round trip costs milliseconds at the benchmark sizes.  This mirrors
    # the delivery convention of the sibling ``coalesce`` implementation.
    run_device = _accel_device(values, idx)
    # gcu400 has no fp64 arithmetic: reinterpret fp64 values at fp32 precision
    # before the launch (the platform-wide convention for this backend; the
    # emitted values are widened back to the declared dtype afterwards).
    if dtype == torch.float64:
        values = values.to(torch.float32)
    if values.device != run_device:
        values = values.to(run_device)
    if idx.device != run_device:
        idx = idx.to(run_device)

    def _finish(res):
        if out is None:
            return res
        if not isinstance(out, torch.Tensor) or out.layout != torch.sparse_coo:
            raise RuntimeError("_coalesce.out requires a sparse COO out tensor")
        # Mirror the native ``.out`` contract: the caller buffer must already
        # carry the result dtype and shape, because resizing a sparse COO tensor
        # is not available on this backend (``aten::resize_`` has no sparse
        # implementation), which is exactly what the native op reports.
        if out.dtype != res.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {res.dtype}, but got "
                f"{out.dtype} instead"
            )
        if tuple(out.shape) != tuple(res.shape):
            raise NotImplementedError(
                "Could not run 'aten::resize_' with arguments from the "
                f"'SparseCPU' backend while resizing the out tensor from "
                f"{tuple(out.shape)} to {tuple(res.shape)}"
            )
        if out._values().device == res._values().device:
            out.copy_(res)
        else:
            # ``res`` is staged on the host (see the delivery convention at the
            # end of this function) while an accelerator-side ``out`` stays on
            # the device.  Sparse ``copy_`` does handle that cross-device
            # transfer here (it resizes ``out`` to the source nnz first), whereas
            # ``out.data = res`` rejects a host sparse source with
            # "incompatible tensor type".  Keep the buffer identity via
            # ``copy_`` and fall back to ``set_data`` only if the transfer is
            # rejected (mirrors the structural-resize path of
            # copy_sparse_to_sparse_).
            try:
                out.copy_(res)
            except RuntimeError:
                out.data = res
        out._coalesced_(True)
        return out

    if n == 0 or sp == 0:
        return _finish(_empty_out_like(shape, sp, dv, dtype, values, run_device))

    if sp > _MAX_SPARSE_DIM:
        raise NotImplementedError(
            f"_coalesce supports at most {_MAX_SPARSE_DIM} sparse dimensions "
            f"(got {sp})"
        )

    dims = list(shape[:sp])
    numel = 1
    for d in dims:
        numel *= d
    if numel == 0:
        return _finish(_empty_out_like(shape, sp, dv, dtype, values, run_device))

    strides = [1] * sp
    acc = 1
    for d in range(sp - 1, -1, -1):
        strides[d] = acc
        acc *= dims[d]

    # The dense key space is only materialised when it stays small relative
    # to the number of stored elements; larger key spaces fall back to the
    # native implementation.
    if numel > _DENSE_MAX or numel > 64 * n:
        raise NotImplementedError(
            f"_coalesce currently handles dense key spaces up to {_DENSE_MAX} "
            f"elements (got numel={numel}, nnz={n})"
        )

    is_bool = dtype == torch.bool
    acc_dtype = _ACC_DTYPE[dtype]
    if acc_dtype == torch.float64:
        acc_dtype = torch.float32
    vals = _as_u8(values) if is_bool else values

    s = strides + [0] * (_MAX_SPARSE_DIM - sp)
    d = dims + [1] * (_MAX_SPARSE_DIM - sp)
    s0, s1, s2, s3, s4 = s
    d0, d1, d2, d3, d4 = d

    # One accumulator per (occupied key, dense element) pair: a hybrid COO tensor
    # (sparse_dim < dim) stores ``dv`` payload elements for every coordinate.
    acc_buf = torch.zeros(numel * dv, dtype=acc_dtype, device=run_device)
    # Pad the occupancy map to a multiple of 4 so ``_keylist_kernel`` can read it in 32-bit words;
    # the padded bytes are always 0 and won't be treated as occupancy bits.
    flag_buf = torch.zeros((numel + 3) & ~3, dtype=torch.uint8, device=run_device)

    _scatter_kernel[(triton.cdiv(n, _SCAT_BLOCK),)](
        idx,
        vals,
        acc_buf,
        flag_buf,
        n,
        numel,
        dv,
        s0,
        s1,
        s2,
        s3,
        s4,
        S=sp,
        BLOCK=_SCAT_BLOCK,
        num_warps=_SCAT_WARPS,
    )

    nb = triton.cdiv(numel, _TILE)
    cnt = torch.empty(nb, dtype=torch.int32, device=run_device)
    mtens = torch.empty(1, dtype=torch.int32, device=run_device)
    key_buf = torch.empty(n, dtype=torch.int32, device=run_device)

    _count_kernel[(nb,)](flag_buf, cnt, numel, BLOCK=_TILE, num_warps=_TILE_WARPS)
    if nb <= _SCAN_MAX_LANES:
        base = torch.empty(nb, dtype=torch.int32, device=run_device)
        _scan_kernel[(1,)](
            cnt,
            base,
            mtens,
            nb,
            B=triton.next_power_of_2(nb),
            num_warps=8,
        )
        if numel >= _WORD_KEYLIST_MIN:
            _keylist_word_kernel[(nb,)](
                flag_buf.view(torch.int32),
                base,
                key_buf,
                numel,
                BLOCK=_TILE,
                num_warps=_TILE_WARPS,
            )
        else:
            _keylist_kernel[(nb,)](
                flag_buf,
                base,
                key_buf,
                numel,
                BLOCK=_TILE,
                num_warps=_TILE_WARPS,
            )
    else:
        _keylist_prefix_kernel[(nb,)](
            flag_buf,
            cnt,
            key_buf,
            mtens,
            numel,
            nb,
            PB=_PREFIX_LANES,
            BLOCK=_TILE,
            num_warps=_TILE_WARPS,
        )

    m = int(mtens.item())
    if m == 0:
        return _finish(_empty_out_like(shape, sp, dv, dtype, values, run_device))

    oidx = torch.empty((sp, m), dtype=torch.int64, device=run_device)
    oval_dtype = torch.float32 if dtype == torch.float64 else dtype
    oval = torch.empty((m,) + dense_shape, dtype=oval_dtype, device=run_device)
    oval_ptr = _as_u8(oval) if is_bool else oval

    _emit_kernel[(triton.cdiv(m, _EMIT_BLOCK),)](
        key_buf,
        acc_buf,
        oidx,
        oval_ptr,
        m,
        dv,
        m,
        s0,
        s1,
        s2,
        s3,
        s4,
        d0,
        d1,
        d2,
        d3,
        d4,
        S=sp,
        IS_BOOL=is_bool,
        BLOCK=_EMIT_BLOCK,
        num_warps=_EMIT_WARPS,
    )

    if oval.dtype != dtype:
        # Widen the fp32-reinterpreted values back to the declared dtype.
        oval = oval.to(dtype)
    # Deliver on the device that produced the buffers: the native op returns an
    # accelerator-resident sparse tensor here, and constructing one from
    # accelerator-resident indices/values is supported by this torch_gcu build
    # (``is_coalesced=True`` included).  Host staging would return a host sparse
    # tensor, which both breaks the device parity contract of ``aten::_coalesce``
    # and forces a device->host copy of the emitted coordinates.
    res = torch.sparse_coo_tensor(
        oidx, oval, shape, is_coalesced=True, device=run_device
    )
    return _finish(res)


def _coalesce(self: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """``aten::_coalesce``: fresh coalesced copy of a sparse COO tensor.

    Passing ``out=`` selects the ``.out`` overload: the coalesced result is
    written into the caller buffer and that object is returned.
    """
    if out is None:
        logger.debug("GEMS_ENFLAME _COALESCE")
        return _coalesce_impl(self)
    logger.debug("GEMS_ENFLAME _COALESCE_OUT")
    return _coalesce_impl(self, out=out)


def _coalesce_out(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """``aten::_coalesce.out``: coalesce ``self`` into ``out``."""
    logger.debug("GEMS_ENFLAME _COALESCE_OUT")
    return _coalesce_impl(self, out=out)
