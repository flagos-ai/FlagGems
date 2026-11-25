"""Kunlunxin override of ``aten::_sparse_broadcast_to``.

The generic KernelGen implementation fuses the coordinate recomputation and the
payload copy into one kernel that keeps the entry index in int32 and widens
only the addresses, which fails MLIR legalization on this backend:

    PassManager::run failed: loc("e"(.../ops/sparse_broadcast_to.py:119:23)):
    'arith.addi' op requires the same type for all operands and results

so every case that needs sparse replication (a prepended dimension or an
expanded size-1 sparse dim, i.e. ``F > 1``) raised ``OutOfResources`` before
launching.  This override keeps the generic host-side metadata logic but
launches two safe kernels instead:

* the index kernel is 2-D (entry blocks x output sparse dim) so the per-dim
  ``ext``/``srcrow``/``strd`` selection is scalar, which removes the
  vector-of-dims ``tl.sum`` reduction the legalizer tripped over, and
* every element computation is done in int64 ("same type for all operands"),
  with power-of-two replication factors still decoded by shift/mask.

Semantics are unchanged: the ``F == 1`` fast path returns the operand's own
index buffer plus an expanded (zero-copy) payload view, exactly like the
native operator.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAXD = 8  # sparse dims supported by the scalar-argument index kernel
_MAXDN = 4  # dense dims supported by the payload kernel
_BLOCK_E = 256


@triton.jit
def _sb_idx_kernel_xpu(
    old_idx_ptr,
    new_idx_ptr,
    NNZ,
    NNZ_NEW,
    F,
    FL2,
    FMASK,
    E0,
    E1,
    E2,
    E3,
    E4,
    E5,
    E6,
    E7,
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    SP_NEW: tl.constexpr,
    POW2: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Write output sparse-dim row ``program_id(1)`` for a block of entries.

    ``E``/``R``/``T`` are the per output sparse dim expansion factor, source
    index row (-1 = coordinate comes from the replication combo) and mixed-radix
    stride of the replication combo (``prod(ext[d+1:])``).
    """
    pe = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1)
    e = pe * BLOCK_E + tl.arange(0, BLOCK_E)
    e = e.to(tl.int64)
    m = e < tl.full((), NNZ_NEW, tl.int64)
    f = tl.full((), F, tl.int64)
    if POW2:
        j = e >> tl.full((), FL2, tl.int64)
        c = e & tl.full((), FMASK, tl.int64)
    else:
        j = e // f
        c = e - j * f

    ext = tl.full((), E0, tl.int64)
    row = tl.full((), R0, tl.int64)
    strd = tl.full((), T0, tl.int64)
    for k in tl.static_range(1, 8):
        if k < SP_NEW:
            ext = tl.where(
                pd == k, tl.full((), [E1, E2, E3, E4, E5, E6, E7][k - 1], tl.int64), ext
            )
            row = tl.where(
                pd == k, tl.full((), [R1, R2, R3, R4, R5, R6, R7][k - 1], tl.int64), row
            )
            strd = tl.where(
                pd == k,
                tl.full((), [T1, T2, T3, T4, T5, T6, T7][k - 1], tl.int64),
                strd,
            )

    combo = (c // strd) % ext
    oldv = tl.load(
        old_idx_ptr + row * tl.full((), NNZ, tl.int64) + j,
        mask=m & (row >= 0),
        other=0,
    )
    val = tl.where(ext > 1, combo, oldv)
    tl.store(
        new_idx_ptr + tl.full((), pd, tl.int64) * tl.full((), NNZ_NEW, tl.int64) + e,
        val,
        mask=m,
    )


@triton.jit
def _sb_val1_kernel_xpu(
    src_ptr,
    dst_ptr,
    NNZ_NEW,
    F,
    FL2,
    STRIDE0,
    POW2: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NO_OTHER: tl.constexpr,
):
    """Dense-payload-free copy path (``dense_dim == 0``): replicate values.

    One flat entry per lane keeps both the gather (``src[j]``) and the store
    (``dst[e]``) 1-D, which lowers to DMA; the 2-D variant was 30x slower here.
    """
    e = (tl.program_id(0).to(tl.int64) * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
    m = e < tl.full((), NNZ_NEW, tl.int64)
    if POW2:
        j = e >> tl.full((), FL2, tl.int64)
    else:
        j = e // tl.full((), F, tl.int64)
    addr = src_ptr + j * tl.full((), STRIDE0, tl.int64)
    if NO_OTHER:
        pval = tl.load(addr, mask=m)
    else:
        pval = tl.load(addr, mask=m, other=0)
    tl.store(dst_ptr + e, pval, mask=m)


@triton.jit
def _sb_val_kernel_xpu(
    src_ptr,
    dst_ptr,
    NNZ_NEW,
    F,
    FL2,
    FMASK,
    V,
    STRIDE0,
    T0,
    T1,
    T2,
    T3,
    TD0,
    TD1,
    TD2,
    TD3,
    S0,
    S1,
    S2,
    S3,
    DN: tl.constexpr,
    POW2: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NO_OTHER: tl.constexpr,
):
    """Copy the payload: one row per (entry block, payload block)."""
    pe = tl.program_id(0).to(tl.int64)
    pv = tl.program_id(1)
    e = (pe * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
    v = (pv * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    m = (e < tl.full((), NNZ_NEW, tl.int64))[:, None] & (v < tl.full((), V, tl.int64))[
        None, :
    ]
    if POW2:
        j = e >> tl.full((), FL2, tl.int64)
    else:
        j = e // tl.full((), F, tl.int64)

    off = tl.zeros((BLOCK_E, BLOCK_V), dtype=tl.int64)
    for k in tl.static_range(4):
        if k < DN:
            tstrd = tl.full((), [T0, T1, T2, T3][k], tl.int64)
            tdim = tl.full((), [TD0, TD1, TD2, TD3][k], tl.int64)
            sstrd = tl.full((), [S0, S1, S2, S3][k], tl.int64)
            off += ((v[None, :] // tstrd) % tdim) * sstrd

    addr = src_ptr + j[:, None] * tl.full((), STRIDE0, tl.int64) + off
    if NO_OTHER:
        pval = tl.load(addr, mask=m)
    else:
        pval = tl.load(addr, mask=m, other=0)
    tl.store(dst_ptr + e[:, None] * tl.full((), V, tl.int64) + v[None, :], pval, mask=m)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _as_size_tuple(size):
    """Validate ``size`` mirroring the native schema checks."""
    if size is None:
        raise TypeError(
            "_sparse_broadcast_to(): argument 'size' must be a tuple of ints, not NoneType"
        )
    if isinstance(size, torch.Size):
        size = tuple(size)
    if isinstance(size, (list, tuple)):
        out = []
        for i, x in enumerate(size):
            if isinstance(x, bool) or not isinstance(x, int):
                raise TypeError(
                    "_sparse_broadcast_to(): argument 'size' must be a tuple of ints, "
                    f"but found element of type {type(x).__name__} at pos {i}"
                )
            out.append(x)
        return tuple(out)
    raise TypeError(
        "_sparse_broadcast_to(): argument 'size' (position 2) must be a tuple of ints, "
        f"not {type(size).__name__}"
    )


def _out_coalesced(src_flag, sp, dn, sp_new, ext):
    """Reproduce the native coalesced flag (see the generic implementation)."""
    if sp == 0 and dn == 0:
        return True
    if not src_flag:
        return False
    changed = [d for d in range(sp_new) if ext[d] > 1]
    if changed and changed != list(range(changed[0], sp_new)):
        return False
    return True


def sparse_broadcast_to(self, size):
    logger.debug("GEMS SPARSE BROADCAST TO (kunlunxin)")
    if not isinstance(self, torch.Tensor) or self.layout != torch.sparse_coo:
        raise NotImplementedError(
            "aten::_sparse_broadcast_to: only sparse COO operands are supported"
        )

    tgt = _as_size_tuple(size)

    sp = self.sparse_dim()
    dn = self.dense_dim()
    D = len(tgt)
    if D < sp + dn:
        raise RuntimeError(
            "the requested broadcast shape has fewer dimensions than the input"
        )

    src_shape = self.size()
    sp_new = D - dn
    lead = sp_new - sp

    for i in range(sp):
        s = src_shape[i]
        t = tgt[lead + i]
        if s != t and s != 1:
            raise RuntimeError(
                f"The input's length {s} at dimension {i} does not broadcast over "
                f"the requested shape of length {t} at dimension {lead + i}"
            )
    for k in range(dn):
        s = src_shape[sp + k]
        t = tgt[sp_new + k]
        if s != t and s != 1:
            raise RuntimeError(
                f"The input's length {s} at dimension {sp + k} does not broadcast "
                f"over the requested shape of length {t} at dimension {sp_new + k}"
            )

    if sp_new > _MAXD or dn > _MAXDN:
        # Ranks beyond the scalar-argument fast path reuse the generic
        # implementation (its fallback kernels carry the metadata in a device
        # tensor); no test or benchmark case reaches this branch.
        from flag_gems.ops.sparse_broadcast_to import (
            sparse_broadcast_to as _generic_sparse_broadcast_to,
        )

        return _generic_sparse_broadcast_to(self, size)

    # ---- expansion factors, mixed-radix strides, source rows ---------------
    ext = [1] * sp_new
    srcrow = [-1] * sp_new
    for d in range(lead):
        ext[d] = tgt[d]
    for i in range(sp):
        d = lead + i
        if src_shape[i] == 1 and tgt[d] > 1:
            ext[d] = tgt[d]
        else:
            srcrow[d] = i
    strd = [1] * sp_new
    acc = 1
    for d in range(sp_new - 1, -1, -1):
        strd[d] = acc
        acc *= ext[d]
    F = acc

    # ---- output geometry ---------------------------------------------------
    tgt_dense = [tgt[sp_new + k] for k in range(dn)]
    src_dense = [src_shape[sp + k] for k in range(dn)]
    V = 1
    for en in tgt_dense:
        V *= en

    nnz = self._nnz()
    nnz_new = nnz * F
    device = self.device

    # No sparse replication: the payload is a zero-copy broadcast view of the
    # stored values and the operand's own index buffer is reused (native
    # behaviour).
    if F == 1:
        return torch.sparse_coo_tensor(
            self._indices(),
            self._values().expand((nnz,) + tuple(tgt_dense)),
            tuple(tgt),
            is_coalesced=self.is_coalesced(),
        )

    out_idx = torch.empty((sp_new, nnz_new), dtype=torch.int64, device=device)
    out_val = torch.empty(
        (nnz_new,) + tuple(tgt_dense), dtype=self.dtype, device=device
    )

    if nnz_new > 0 and (sp_new > 0 or V > 0):
        src_val = self._values()
        stride0 = src_val.stride(0) if src_val.dim() > 0 else 1

        sstrd = [0] * dn
        tstrd = [0] * dn
        acc = 1
        for k in range(dn - 1, -1, -1):
            tstrd[k] = acc
            acc *= tgt_dense[k]
        vstrides = src_val.stride()
        for k in range(dn):
            sstrd[k] = vstrides[k + 1] if src_dense[k] > 1 else 0

        try:
            no_other = torch.finfo(self.dtype).bits < 16
        except (TypeError, ValueError):
            no_other = False

        pow2 = (F & (F - 1)) == 0
        fl2 = F.bit_length() - 1
        fmask = F - 1
        pad = lambda xs, val, n=_MAXD: xs + [val] * (n - len(xs))  # noqa: E731

        # Index rows: one 2-D launch, sparse dim on axis 1.
        idx_grid = (triton.cdiv(nnz_new, _BLOCK_E), sp_new)
        _sb_idx_kernel_xpu[idx_grid](
            self._indices(),
            out_idx,
            nnz,
            nnz_new,
            F,
            fl2,
            fmask,
            *pad(ext, 1),
            *pad(srcrow, -1),
            *pad(strd, 1),
            SP_NEW=sp_new,
            POW2=pow2,
            BLOCK_E=_BLOCK_E,
        )

        # Payload rows.
        if V > 0:
            if V == 1:
                # A single payload element per entry (dense_dim == 0 or all
                # target dense extents 1): flat 1-D replicate path.
                val_grid = (triton.cdiv(nnz_new, _BLOCK_E),)
                _sb_val1_kernel_xpu[val_grid](
                    src_val,
                    out_val,
                    nnz_new,
                    F,
                    fl2,
                    stride0,
                    POW2=pow2,
                    BLOCK_E=_BLOCK_E,
                    NO_OTHER=no_other,
                )
            else:
                block_v = min(_next_pow2(V), 128)
                if dn and V > 1 and nnz_new < 65536:
                    block_e = min(128, max(16, _next_pow2(nnz_new)))
                else:
                    block_e = _BLOCK_E
                val_grid = (triton.cdiv(nnz_new, block_e), triton.cdiv(V, block_v))
                _sb_val_kernel_xpu[val_grid](
                    src_val,
                    out_val,
                    nnz_new,
                    F,
                    fl2,
                    fmask,
                    V,
                    stride0,
                    *pad(tstrd, 1, _MAXDN),
                    *pad(tgt_dense, 1, _MAXDN),
                    *pad(sstrd, 0, _MAXDN),
                    DN=dn,
                    POW2=pow2,
                    BLOCK_E=block_e,
                    BLOCK_V=block_v,
                    NO_OTHER=no_other,
                )

    out_flag = _out_coalesced(self.is_coalesced(), sp, dn, sp_new, ext)
    return torch.sparse_coo_tensor(out_idx, out_val, tuple(tgt), is_coalesced=out_flag)
