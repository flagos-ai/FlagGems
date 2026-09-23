import importlib
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems." + __name__)

# NOT `import flag_gems.ops.linalg_lstsq as _generic`: flag_gems/ops/__init__.py
# does `from .linalg_lstsq import linalg_lstsq`, which rebinds that attribute on
# the package to the FUNCTION, and `import a.b as c` binds by attribute lookup --
# so _generic would be the function and every constant below would raise
# AttributeError at import. import_module resolves through sys.modules instead.
_generic = importlib.import_module("flag_gems.ops.linalg_lstsq")

# ---------------------------------------------------------------------------
# Three MetaX device facts, all measured on a C550. Everything below is applied
# by rebinding attributes on the generic module at import: the generic op reads
# each of them at call time, so the rebinding takes effect without duplicating
# its ~250-line compact-WY driver (a copy would drift the moment the generic
# side changes).
#
#   1. 64KB of shared memory per block. The tuned WY update config asks for
#      2*ubr*bc*es + ubr*P*es = 73728 bytes for BOTH dtypes, 8KB over, and 19
#      of the suite's cases died with OutOfResources.
#   2. float64 tl.dot is MISCOMPILED. Measured relerr ~1.0 at every operand
#      shape including 16x16x16, with and without input_precision="ieee",
#      while the same contraction written as rank-1 updates is bit-exact
#      (0.00e+00). fp32 tl.dot is fine (1.1e-07 with ieee).
#   3. 4KB/thread of PRIVATE (register-spill) memory -- a driver setting,
#      "insmod metax.ko pri_mem_sz=XXX" -- which the monolithic path's reduce
#      kernel exceeds at BLOCK_NC=128, surfacing as the misleading "memory size
#      or pointer value too large to fit in 32 bit".
# ---------------------------------------------------------------------------


def _smem_per_block() -> int:
    """Shared memory Triton will let one block use, in bytes.

    From Triton's driver, not torch's device properties: Triton opts into the
    larger dynamic limit, so on NVIDIA torch reports 49152 (the static default)
    while Triton enforces 164-228KB. Both agree at 65536 on C550, but taking
    the wrong one would matter if this file were ever reused elsewhere.
    """
    try:
        drv = triton.runtime.driver.active
        props = drv.utils.get_device_properties(drv.get_current_device())
        return int(props["max_shared_mem"])
    except Exception:
        pass
    try:
        v = torch.cuda.get_device_properties(0).shared_memory_per_block
        if v:
            return int(v)
    except Exception:
        pass
    logger.warning("GEMS_METAX LINALG_LSTSQ: smem limit unknown, assuming 64KB")
    return 64 * 1024


def _derive_bc(ubr: int, es: int) -> int:
    """Largest BLOCK_C leaving two blocks resident per SM.

    Fitting is not the same as fitting well: measured on C550, bc=16 (24576 B,
    two blocks per SM) ran the update in 0.0797 ms against bc=32 (40960 B, one
    block) at 0.0977 -- so the target is HALF the limit, not the whole of it.
    """
    bc = _generic._WY_BLOCK_C
    target = _smem_per_block() // 2
    while bc > 8 and 2 * ubr * bc * es + ubr * _generic._WY_PANEL * es > target:
        bc //= 2
    return bc


def _derive_stack_rows(es: int = 4) -> int:
    """Largest _TARGET_STACK_ROWS whose reduce tile still fits in shared memory.

    The monolithic path stacks G = max(2, _TARGET_STACK_ROWS // NC) R factors
    and reduces them in one tile of next_pow2(G*NC) x next_pow2(NC). At the
    upstream 256 that is 256x64x4 = 65536 B for NC=33 -- exactly a C550's
    limit, which is why it fit there and nowhere else: maca3720's Triton asks
    for 66560 (the tile plus ~1KB of launch overhead) and raises OutOfResources.

    Sitting on the limit is not fitting. Budget half of it, as the WY path
    does, so the tile has room for that overhead and two blocks stay resident.
    """
    sr = _generic._TARGET_STACK_ROWS
    budget = _smem_per_block() // 2
    # Worst case is the widest tile the monolithic path can be handed, which
    # the NC cap below bounds.
    bnc = _generic._next_pow2(_generic._TALL_MAX_NC_F32)
    while sr > 64 and _generic._next_pow2(sr) * bnc * es > budget:
        sr //= 2
    return sr


_BC = None


def _bc() -> int:
    """BLOCK_C for both dtypes, derived on FIRST USE -- never at import.

    Deriving this at import would query the Triton driver while flag_gems is
    still loading its vendor ops, i.e. before torch has touched the device.
    Under the flagtree Triton backend used in CI that left the MACA context in
    a state where the next launch -- an ordinary `torch.randn` -- failed with
    `mcErrorInvalidDeviceFunction`. Nothing here needs to run before the
    operator is first called, so nothing here does.
    """
    global _BC
    if _BC is None:
        bc32 = _derive_bc(min(_generic._WY_BLOCK_R, 128), 4)
        bc64 = _derive_bc(64, 8)
        # The generic driver computes the update grid from the MODULE constant
        # _WY_BLOCK_C while passing BLOCK_C from _wy_cfg. Those agree upstream,
        # so the mismatch is dormant there; rebinding both keeps them agreeing
        # WITHOUT touching generic code -- but only while the two dtypes derive
        # the same value. They do on C550 (both 16). Fail loudly rather than
        # silently under-update the trailing block if a device makes them
        # differ.
        if bc32 != bc64:
            raise RuntimeError(
                "metax linalg_lstsq: fp32 and fp64 derive different BLOCK_C "
                f"({bc32} vs {bc64}); the generic grid uses a single module "
                "constant, so they must match. Fix the grid in "
                "flag_gems/ops/linalg_lstsq.py to use the per-dtype value "
                "instead."
            )
        _BC = bc32
        _generic._WY_BLOCK_C = _BC  # keep the generic grid in step with _wy_cfg
        _generic._TARGET_STACK_ROWS = _derive_stack_rows()
        # The panel kernel's tile is block_m x next_pow2(NC), and
        # _choose_block_m sizes block_m so that product stays under
        # _TARGET_TILE_BYTES. Upstream's 96KB presumes an NVIDIA-sized shared
        # memory; on a 64KB part it yields 256x64x4 = 65536 B at NC=33 -- the
        # entire limit -- and maca3720 then asks for 66560 with launch
        # overhead. Budget half, exactly as the other two tiles do.
        _generic._TARGET_TILE_BYTES = min(
            _generic._TARGET_TILE_BYTES, _smem_per_block() // 2
        )
        logger.debug(
            "GEMS_METAX LINALG_LSTSQ: BLOCK_C=%d, TARGET_STACK_ROWS=%d, "
            "TARGET_TILE_BYTES=%d (smem %d B)",
            _BC,
            _generic._TARGET_STACK_ROWS,
            _generic._TARGET_TILE_BYTES,
            _smem_per_block(),
        )
    return _BC


@triton.jit
def _wy_update_metax(
    W_ptr,
    T_ptr,
    M,
    NC,
    J0,
    PW,
    swb,
    swi,
    swj,
    sTb,
    sTi,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    P: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    """trailing -= V @ (T^T @ (V^T @ trailing)), with fp64 avoiding tl.dot.

    Identical to the generic kernel except that each of the three contractions
    has a float64 form written as P rank-1 updates. The branch is on COMPUTE,
    a tl.constexpr, so it is resolved at compile time and fp32 keeps the
    tl.dot path unchanged -- it is correct here (1.1e-07 with ieee) and faster.
    """
    b = tl.program_id(0)
    cb = tl.program_id(1)
    wb = b * swb
    kk = tl.arange(0, P)
    piv = J0 + kk
    cols = J0 + PW + cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < NC

    # ---- pass 1: Wacc = V^T @ trailing   (P x BLOCK_C) ----
    Wacc = tl.zeros((P, BLOCK_C), dtype=COMPUTE)
    for rb in range(J0, M, BLOCK_R):
        rows = rb + tl.arange(0, BLOCK_R)
        rmask = rows < M
        vo = wb + rows[:, None] * swi + piv[None, :] * swj
        vm = rmask[:, None] & (kk[None, :] < PW)
        Vb = tl.load(W_ptr + vo, mask=vm & (rows[:, None] > piv[None, :]), other=0.0)
        Vb = tl.where((rows[:, None] == piv[None, :]) & vm, 1.0, Vb)
        Vb = tl.where(rows[:, None] < piv[None, :], 0.0, Vb)
        to = wb + rows[:, None] * swi + cols[None, :] * swj
        Tb = tl.load(W_ptr + to, mask=rmask[:, None] & cmask[None, :], other=0.0)
        if COMPUTE == tl.float64:
            for p in range(P):
                sel = (kk == p).to(COMPUTE)
                vp = tl.sum(Vb * sel[None, :], axis=1)
                Wacc += sel[:, None] * tl.sum(vp[:, None] * Tb, axis=0)[None, :]
        else:
            Wacc += tl.dot(tl.trans(Vb), Tb, input_precision="ieee")

    tl.debug_barrier()  # WAR: pass-1 reads before pass-2 overwrites

    # ---- Y = T^T @ Wacc ----
    tof = tl.load(
        T_ptr + b * sTb + kk[:, None] * sTi + kk[None, :],
        mask=(kk[:, None] < PW) & (kk[None, :] < PW),
        other=0.0,
    )
    if COMPUTE == tl.float64:
        Y = tl.zeros((P, BLOCK_C), dtype=COMPUTE)
        for p in range(P):
            sel = (kk == p).to(COMPUTE)
            trow = tl.sum(tof * sel[:, None], axis=0)
            wrow = tl.sum(Wacc * sel[:, None], axis=0)
            Y += trow[:, None] * wrow[None, :]
    else:
        Y = tl.dot(tl.trans(tof), Wacc, input_precision="ieee")

    # ---- pass 2: trailing -= V @ Y ----
    for rb in range(J0, M, BLOCK_R):
        rows = rb + tl.arange(0, BLOCK_R)
        rmask = rows < M
        vo = wb + rows[:, None] * swi + piv[None, :] * swj
        vm = rmask[:, None] & (kk[None, :] < PW)
        Vb = tl.load(W_ptr + vo, mask=vm & (rows[:, None] > piv[None, :]), other=0.0)
        Vb = tl.where((rows[:, None] == piv[None, :]) & vm, 1.0, Vb)
        Vb = tl.where(rows[:, None] < piv[None, :], 0.0, Vb)
        to = wb + rows[:, None] * swi + cols[None, :] * swj
        tm = rmask[:, None] & cmask[None, :]
        Tb = tl.load(W_ptr + to, mask=tm, other=0.0)
        if COMPUTE == tl.float64:
            upd = tl.zeros((BLOCK_R, BLOCK_C), dtype=COMPUTE)
            for p in range(P):
                sel = (kk == p).to(COMPUTE)
                vp = tl.sum(Vb * sel[None, :], axis=1)
                yp = tl.sum(Y * sel[:, None], axis=0)
                upd += vp[:, None] * yp[None, :]
        else:
            upd = tl.dot(Vb, Y, input_precision="ieee")
        tl.store(W_ptr + to, Tb - upd, mask=tm)


def _wy_cfg_metax(dt):
    """(panel BLOCK_R, update BLOCK_R, BLOCK_C, num_stages), BLOCK_C derived."""
    if dt == torch.float64:
        return 256, 64, _bc(), 2
    return _generic._WY_BLOCK_R, min(_generic._WY_BLOCK_R, 128), _bc(), 3


# ---- apply, once, at import (pure Python rebinds; no device access) ----
_generic._wy_cfg = _wy_cfg_metax
_generic._wy_update = _wy_update_metax
# The monolithic path's reduce kernel wants 5KB/thread of private memory at
# BLOCK_NC=128 against the 4KB cap. num_warps=8 does fit and is numerically
# correct (relerr 4.1e-07) but costs 14.9 ms per launch against 0.60 ms at
# BLOCK_NC=64 -- a 25x spill penalty -- and num_warps=16 exceeds the device's
# threads-per-block. Capping NC sends those shapes to compact-WY, which works
# here and is the no-ceiling path by design.
_generic._TALL_MAX_NC_F32 = 64


def linalg_lstsq(A, b, rcond=None, driver=None):
    """Metax specialization: generic kernels, device-derived configuration."""
    logger.debug("GEMS_METAX LINALG_LSTSQ")
    _bc()  # derive the device-dependent config now, on a live context
    return _generic.linalg_lstsq(A, b, rcond=rcond, driver=driver)
