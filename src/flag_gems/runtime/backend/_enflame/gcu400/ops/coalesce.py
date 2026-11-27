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

"""Triton implementation of ``aten::coalesce`` for COO sparse tensors.

Pipeline (all real device work stays inside triton kernels):

1. ``_coalesce_hist``     - zero the value accumulator and the unique-key
                           counter, mark the set of linear keys present in the
                           input and publish each element's linear key so the
                           scatter kernel can reload it coalesced.
2. ``_coalesce_scan``     - word-level popcount + inclusive scan, publish the
                           exclusive word prefix and the per-block totals.
3. ``_coalesce_scatter``  - rank every stored element, atomically accumulate its
                           value into the dense accumulator and record the
                           element's linear key at the ranked position.
4. ``_coalesce_finalize`` - decode the ranked linear keys back into per-dimension
                           coordinates and cast the dense accumulator to the
                           output dtype over the first ``r`` (final nnz) slots;
                           ``r`` is read from the device so the host never
                           drains the pipeline early.

GCU addressing constraint (drives the whole buffer layout below)
----------------------------------------------------------------
The GCU triton backend scales the offset of a *64-bit* element pointer by 16
bytes instead of 8, so any 64-bit buffer that is read or written past lane 0
returns/lands on the wrong element (``ptr + 32 + arange`` reads elements
``64..``).  Every kernel-visible buffer here is therefore 32-bit:

* the ``int64`` index rows are narrowed to ``int32`` once on the host (a
  coordinate never exceeds the dimension size),
* the packed ``(prefix << 32) | word_bits`` table is split into two ``int32``
  buffers,
* the ranking scratch and the coordinate output are ``int32``; the final COO
  wrapper widens the coordinates back to ``int64`` once, which is pure dtype
  marshalling and not part of the coalescing math.

64-bit *value* dtypes (``int64``/``float64``) cannot be resolved through a
ranked index on this backend, so they are forwarded to the generic triton
coalescing kernel, which never indexes a value buffer by rank.

Only the resulting COO tensor is returned; the input tensor is never mutated.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_I32_MAX = 1 << 31
_HIST_BLOCK = 1024
_SCAN_SB_LOG = 11
_SCAN_SB = 1 << _SCAN_SB_LOG
_SCAN_BLOCK = 2048
_FIN_BLOCK = 1024
# Launch parameters are managed centrally for easy item-by-item A/B testing (no dynamic adaptation, to keep behavior
# reproducible).
_HIST_WARPS = 16
_SCAN_WARPS = 8
_SCATTER_WARPS = 16
_FIN_WARPS = 8

_FP8 = tuple(
    d
    for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if d is not None
)

_F32_DT = (torch.float16, torch.bfloat16, torch.float32)
_F64_DT = (torch.float64,)
_I_DT = (torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
# Narrow integer values are accumulated in int32 (associative modulo 2**32, so
# truncating the full sum gives the same bits) and narrowed on the host.
_NARROW_I_DT = (torch.int8, torch.uint8, torch.int16)


@triton.jit
def _popc32(x):
    """Population count of a 32-bit pattern held in an int32 value."""
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x3F


@triton.jit
def _fold_key(
    idx_ptr,
    offs,
    m,
    n,
    S,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
):
    """Row-major linear key; ``Dm`` is shape[m] for m >= 1.

    The dims are constexpr, so a power-of-two shape folds into shifts and adds
    instead of runtime integer multiplies.
    """
    k = tl.load(idx_ptr + offs, mask=m, other=0)
    if S > 1:
        k = k * D1 + tl.load(idx_ptr + n + offs, mask=m, other=0)
    if S > 2:
        k = k * D2 + tl.load(idx_ptr + 2 * n + offs, mask=m, other=0)
    if S > 3:
        k = k * D3 + tl.load(idx_ptr + 3 * n + offs, mask=m, other=0)
    if S > 4:
        k = k * D4 + tl.load(idx_ptr + 4 * n + offs, mask=m, other=0)
    return k


@triton.jit
def _coalesce_hist(
    idx_ptr,
    bits_ptr,
    acc_ptr,
    cnt_ptr,
    key_out_ptr,
    n,
    S,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if pid == 0:
        tl.store(cnt_ptr, 0)
    # Zero this program's slice of the dense value accumulator; the scatter
    # kernel runs only after this kernel completes, so the plain stores are
    # race-free and replace a separate zeroing memset.
    tl.store(
        acc_ptr + offs,
        tl.zeros([BLOCK], dtype=acc_ptr.dtype.element_ty),
        mask=m,
    )
    k = _fold_key(idx_ptr, offs, m, n, S, D1, D2, D3, D4)
    # Publish the key so scatter can reload it coalesced rather than re-reading
    # every index row and re-folding it.
    tl.store(key_out_ptr + offs, k, mask=m)
    w = (k >> 5).to(tl.int32)
    bit = tl.full([BLOCK], 1, tl.int32) << (k & 31).to(tl.int32)
    # Relaxed orderings are sufficient across this whole pipeline: the scratch
    # buffers are private to one launch and strictly stream-ordered.
    tl.atomic_or(bits_ptr + w, bit, mask=m, sem="relaxed")


@triton.jit
def _coalesce_scan(
    bits_ptr,
    woff_ptr,
    blk_tot_ptr,
    cnt_ptr,
    NW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    woffs = pid * BLOCK + tl.arange(0, BLOCK)
    wm = woffs < NW
    v = tl.load(bits_ptr + woffs, mask=wm, other=0)
    pc = _popc32(v)
    inc = tl.cumsum(pc, axis=0)
    exc = inc - pc
    # The word bits stay in ``bits_ptr`` (never cleared), so only the exclusive
    # prefix has to be published; that keeps the table 32-bit addressable.
    tl.store(woff_ptr + woffs, exc, mask=wm)
    tot = tl.sum(pc, axis=0)
    tl.store(blk_tot_ptr + pid, tot)
    tl.atomic_add(cnt_ptr, tot, sem="relaxed")


@triton.jit
def _coalesce_blkoff(
    blk_tot_ptr,
    blk_off_ptr,
    NB2,
    NTP: tl.constexpr,
):
    idx = tl.arange(0, NTP)
    m = idx < NB2
    t = tl.load(blk_tot_ptr + idx, mask=m, other=0)
    inc = tl.cumsum(t, axis=0)
    tl.store(blk_off_ptr + idx, inc - t, mask=m)


@triton.jit
def _coalesce_scatter(
    keyin_ptr,
    vals_ptr,
    woff_ptr,
    bits_ptr,
    blk_off_ptr,
    acc_ptr,
    key_ptr,
    n,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    SINGLE_SB: tl.constexpr,
    SB_LOG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    k = tl.load(keyin_ptr + offs, mask=m, other=0)
    w = (k >> 5).to(tl.int32)
    exc = tl.load(woff_ptr + w, mask=m, other=0)
    bits = tl.load(bits_ptr + w, mask=m, other=0)
    lmask = (tl.full([BLOCK], 1, tl.int32) << (k & 31).to(tl.int32)) - 1
    if SINGLE_SB:
        # When the key space has only one superblock, the global prefix offset is always 0, avoiding an extra launch and
        # load.
        blk = tl.zeros([BLOCK], tl.int32)
    else:
        blk = tl.load(blk_off_ptr + (w >> SB_LOG), mask=m, other=0)
    pos = exc + _popc32(bits & lmask) + blk
    raw = tl.load(vals_ptr + offs, mask=m, other=0)
    val = tl.where(m, raw, 0).to(acc_ptr.dtype.element_ty)
    tl.atomic_add(acc_ptr + pos, val, mask=m)
    # Ranked linear key, decoded by ``_coalesce_finalize`` into coordinates.
    tl.store(key_ptr + pos, k, mask=m)


@triton.jit
def _coalesce_finalize(
    acc_ptr,
    out_ptr,
    cnt_ptr,
    key_ptr,
    out_idx_ptr,
    n,
    S,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    IS_BOOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < tl.load(cnt_ptr)
    v = tl.load(acc_ptr + offs, mask=m, other=0)
    if IS_BOOL:
        v = (v != 0).to(tl.int32)
    tl.store(out_ptr + offs, v.to(out_ptr.dtype.element_ty), mask=m)
    k = tl.load(key_ptr + offs, mask=m, other=0)
    if S == 1:
        tl.store(out_idx_ptr + offs, k, mask=m)
    elif S == 2:
        c0 = k // D1
        tl.store(out_idx_ptr + offs, c0, mask=m)
        tl.store(out_idx_ptr + n + offs, k - c0 * D1, mask=m)
    elif S == 3:
        c0 = k // (D1 * D2)
        rem = k - c0 * (D1 * D2)
        c1 = rem // D2
        tl.store(out_idx_ptr + offs, c0, mask=m)
        tl.store(out_idx_ptr + n + offs, c1, mask=m)
        tl.store(out_idx_ptr + 2 * n + offs, rem - c1 * D2, mask=m)
    elif S == 4:
        c0 = k // (D1 * D2 * D3)
        rem = k - c0 * (D1 * D2 * D3)
        c1 = rem // (D2 * D3)
        rem2 = rem - c1 * (D2 * D3)
        c2 = rem2 // D3
        tl.store(out_idx_ptr + offs, c0, mask=m)
        tl.store(out_idx_ptr + n + offs, c1, mask=m)
        tl.store(out_idx_ptr + 2 * n + offs, c2, mask=m)
        tl.store(out_idx_ptr + 3 * n + offs, rem2 - c2 * D3, mask=m)
    else:
        c0 = k // (D1 * D2 * D3 * D4)
        rem = k - c0 * (D1 * D2 * D3 * D4)
        c1 = rem // (D2 * D3 * D4)
        rem2 = rem - c1 * (D2 * D3 * D4)
        c2 = rem2 // (D3 * D4)
        rem3 = rem2 - c2 * (D3 * D4)
        c3 = rem3 // D4
        tl.store(out_idx_ptr + offs, c0, mask=m)
        tl.store(out_idx_ptr + n + offs, c1, mask=m)
        tl.store(out_idx_ptr + 2 * n + offs, c2, mask=m)
        tl.store(out_idx_ptr + 3 * n + offs, c3, mask=m)
        tl.store(out_idx_ptr + 4 * n + offs, rem3 - c3 * D4, mask=m)


def coalesce(self):
    """Coalesce a COO sparse tensor, matching ``aten::coalesce`` semantics."""
    logger.debug("GEMS_ENFLAME COALESCE")
    if self.layout != torch.sparse_coo:
        raise RuntimeError(
            "coalesce expected sparse coordinate tensor layout but got "
            f"{self.layout}"
        )
    if self.dense_dim() != 0:
        raise NotImplementedError("coalesce with dense dimensions is not implemented")

    values = self._values()
    vdt = values.dtype
    if vdt in _FP8 or (vdt not in _F32_DT and vdt not in _F64_DT and vdt not in _I_DT):
        raise NotImplementedError(f'"coalesce_sparse_cuda" not implemented for {vdt!r}')
    if self.is_coalesced() or self._nnz() == 0:
        return self

    shape = tuple(int(d) for d in self.shape)
    S = self.sparse_dim()
    n = self._nnz()
    K = 1
    for d in shape:
        K *= d

    if vdt in (torch.int64,) or vdt in _F64_DT:
        # The kernels below resolve value pointers through a ranked index, which
        # a 64-bit element pointer cannot do on this backend (see the module
        # docstring).  The sibling ``_coalesce`` kernel never indexes a value
        # buffer by rank, so route 64-bit values there instead of failing.
        # Imported lazily to keep the sibling module's import order intact.
        from ._coalesce import _coalesce as _enflame_coalesce

        try:
            return _enflame_coalesce(self)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"coalesce does not support 64-bit values with a {K}-element key "
                f"space on this backend ({exc})"
            ) from exc

    indices = self._indices()
    # torch_gcu's sparse wrapper may report ``self.device`` as cpu, while the storage of indices/values is
    # actually on the GCU; the scratch buffer must live on the real device holding indices, otherwise
    # the Triton kernel receives a CPU pointer ("Pointer argument cannot be accessed from Kurama").
    dev = indices.device

    # 32-bit kernel view of the index rows: the int64 layout is not addressable
    # past lane 0 on this backend.
    idx32 = indices.to(torch.int32)

    if vdt in _F32_DT:
        adt = torch.float32
    else:
        adt = torch.int32

    if K >= _I32_MAX:
        # The linear key (and therefore the bitset word index) needs more than 32
        # bits; the GCU backend cannot address 64-bit buffers, so refuse instead
        # of silently producing wrong coordinates.
        raise NotImplementedError(
            f"coalesce with a {K}-element key space needs 64-bit addressing, "
            "which this backend does not support"
        )
    NW = (K + 31) // 32
    NB2 = (NW + _SCAN_SB - 1) // _SCAN_SB
    # Single superblock: the global prefix sum is always 0, so scatter needs no prefix-table load (and no blkoff
    # launch).
    # Multiple superblocks: _coalesce_blkoff materializes the prefix table. The previous "in-register cumsum +
    # tl.gather"
    # path could not be legalized on the GCU backend (tt.gather illegal) and has been removed.
    single_sb = NB2 <= 1
    NTP = max(16, triton.next_power_of_2(NB2))
    dims = list(shape) + [0] * (5 - len(shape))
    D1, D2, D3, D4 = dims[1], dims[2], dims[3], dims[4]

    bits = torch.zeros(NW, dtype=torch.int32, device=dev)
    woff = torch.empty(NW, dtype=torch.int32, device=dev)
    cnt = torch.empty(1, dtype=torch.int32, device=dev)
    is_bool = vdt == torch.bool
    # The GCU backend cannot select a vector truncation narrowed to i8 (LLVM ERROR: Cannot select
    # v512i8 = truncate), so int8/uint8 are first written into an int32 buffer and finally converted by torch
    # once for dtype conversion (equivalent to the ATen reference accumulating in int64 and then narrowing).
    narrow_i = vdt in _NARROW_I_DT
    if is_bool or narrow_i:
        out_dtype = torch.int32
    else:
        out_dtype = vdt
    out_vals = torch.empty(n, dtype=out_dtype, device=dev)
    accbuf = torch.empty(n, dtype=adt, device=dev)
    # Two separate key buffers: the histogram publishes the element-indexed keys
    # and the scatter writes the rank-indexed ones, so they must not alias.
    keyin = torch.empty(n, dtype=torch.int32, device=dev)
    keys = torch.empty(n, dtype=torch.int32, device=dev)
    out_idx = torch.empty(S * n, dtype=torch.int32, device=dev)
    if single_sb:
        blk_tot = torch.empty(1, dtype=torch.int32, device=dev)
        blk_off = blk_tot
    else:
        blk = torch.empty(NB2 + NTP, dtype=torch.int32, device=dev)
        blk_tot = blk[:NB2]
        blk_off = blk[NB2:]

    grid_n = (triton.cdiv(n, _HIST_BLOCK),)
    _coalesce_hist[grid_n](
        idx32,
        bits,
        accbuf,
        cnt,
        keyin,
        n,
        S,
        D1,
        D2,
        D3,
        D4,
        BLOCK=_HIST_BLOCK,
        num_warps=_HIST_WARPS,
    )
    _coalesce_scan[(triton.cdiv(NW, _SCAN_BLOCK),)](
        bits, woff, blk_tot, cnt, NW, BLOCK=_SCAN_BLOCK, num_warps=_SCAN_WARPS
    )
    if not single_sb:
        _coalesce_blkoff[(1,)](blk_tot, blk_off, NB2, NTP=NTP)
    _coalesce_scatter[grid_n](
        keyin,
        values,
        woff,
        bits,
        blk_off,
        accbuf,
        keys,
        n,
        D1,
        D2,
        D3,
        D4,
        SINGLE_SB=single_sb,
        SB_LOG=_SCAN_SB_LOG,
        BLOCK=_HIST_BLOCK,
        num_warps=_SCATTER_WARPS,
    )
    _coalesce_finalize[grid_n](
        accbuf,
        out_vals,
        cnt,
        keys,
        out_idx,
        n,
        S,
        D1,
        D2,
        D3,
        D4,
        IS_BOOL=is_bool,
        BLOCK=_FIN_BLOCK,
        num_warps=_FIN_WARPS,
    )

    # Single trailing device->host read; every kernel above is already queued.
    r = int(cnt.item())
    if r <= 0:
        return self

    # Widen the 32-bit coordinates back to the int64 index dtype of the COO
    # wrapper (dtype marshalling only).
    out_indices = out_idx.view(S, n)[:, :r].to(torch.int64)
    out_values = out_vals[:r]
    if is_bool:
        out_values = out_values != 0
    elif narrow_i:
        out_values = out_values.to(vdt)
    return torch.sparse_coo_tensor(
        out_indices, out_values, shape, device=dev, is_coalesced=True
    )
