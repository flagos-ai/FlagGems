import importlib
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems." + __name__)

# NOT `import flag_gems.ops.linalg_lstsq as _generic`: flag_gems/ops/__init__.py
# does `from .linalg_lstsq import linalg_lstsq`, which rebinds that attribute on
# the package to the FUNCTION, and `import a.b as c` binds by attribute lookup,
# so _generic would be the function. import_module resolves via sys.modules.
_generic = importlib.import_module("flag_gems.ops.linalg_lstsq")

# ---------------------------------------------------------------------------
# One Hygon device fact: 64KB of shared memory per block. Upstream's compact-WY
# update config asks for 2*BLOCK_R*BLOCK_C*esize + BLOCK_R*P*esize
# = 2*128*64*4 + 128*16*4 = 73728 bytes, 8KB over, and 14 of the suite's cases
# died with
#
#     triton.runtime.errors.OutOfResources: out of resource: shared memory,
#     Required: 73728, Hardware limit: 65536
#
# Every one of them is a compact-WY shape (square, underdetermined, tall
# blocked, rank-deficient square); the monolithic and blocked-TSQR paths fit and
# pass. So BLOCK_C is derived from the limit Triton actually enforces, which on
# this device is 65536 and yields 16 (24576 bytes).
#
# Nothing else is rebound. The other two shared-memory-sensitive tiles
# (_TARGET_TILE_BYTES for the panel kernel, _TARGET_STACK_ROWS for the reduce)
# fit here today and their shapes all pass, and a backend override is not the
# place to change configuration that has not been measured on the device.
#
# A SECOND fact, seen only in CI: that runner's Triton rejects float64 `tl.dot`
# in its own frontend --
#
#     triton/language/semantic.py:1445
#     AssertionError: Unsupported lhs dtype fp64
#
# -- which is a version-dependent allow-list in the Triton build, not a
# property of the silicon: a local Hygon box with a different Triton runs the
# same float64 tests through `tl.dot` and passes all 76. So `_wy_update` also
# carries a float64 form written as P rank-1 updates, branching on the COMPUTE
# constexpr so float32 keeps `tl.dot` untouched. It is the same kernel as the
# Iluvatar override's, for the same reason.
# ---------------------------------------------------------------------------


def _smem_per_block() -> int:
    """Shared memory Triton will let one block use, in bytes.

    From Triton's driver, not torch's device properties: Triton opts into the
    larger dynamic limit, so on NVIDIA torch reports 49152 (the static default)
    while Triton enforces 164-228KB. Both agree at 65536 here, but taking the
    wrong one would matter if this file were ever reused elsewhere.
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
    logger.warning("GEMS_HYGON LINALG_LSTSQ: smem limit unknown, assuming 64KB")
    return 64 * 1024


def _derive_bc(ubr: int, es: int) -> int:
    """Largest BLOCK_C leaving two blocks resident per SM.

    Fitting is not the same as fitting well: measured on a MetaX part with the
    same 64KB limit, bc=16 (24576 B, two blocks per SM) ran the update in
    0.0797 ms against bc=32 (40960 B, one block) at 0.0977 -- so the target is
    HALF the limit, not the whole of it.
    """
    bc = _generic._WY_BLOCK_C
    target = _smem_per_block() // 2
    while bc > 8 and 2 * ubr * bc * es + ubr * _generic._WY_PANEL * es > target:
        bc //= 2
    return bc


_BC = None


def _bc() -> int:
    """BLOCK_C for both dtypes, derived on FIRST USE -- never at import.

    Deriving it at import would query the Triton driver while flag_gems is
    still loading its vendor ops, i.e. before torch has touched the device.
    Nothing here needs to run before the operator is first called.
    """
    global _BC
    if _BC is None:
        bc32 = _derive_bc(min(_generic._WY_BLOCK_R, 128), 4)
        bc64 = _derive_bc(64, 8)
        # The generic driver computes the WY update grid from the MODULE
        # constant _WY_BLOCK_C while taking BLOCK_C from _wy_cfg -- and _wy_cfg
        # returns a LITERAL 64 for float64. Rebinding only the constant would
        # therefore leave float64 launching a 64-wide tile against a grid sized
        # for 16, silently under-updating the trailing block. Both are rebound,
        # which is only sound while the two dtypes derive the same value. They
        # do here (both 16). Fail loudly if a device ever makes them differ.
        if bc32 != bc64:
            raise RuntimeError(
                "hygon linalg_lstsq: fp32 and fp64 derive different BLOCK_C "
                f"({bc32} vs {bc64}); the generic grid uses a single module "
                "constant, so they must match. Fix the grid in "
                "flag_gems/ops/linalg_lstsq.py to use the per-dtype value "
                "instead."
            )
        _BC = bc32
        _generic._WY_BLOCK_C = _BC  # keep the generic grid in step with _wy_cfg
        logger.debug(
            "GEMS_HYGON LINALG_LSTSQ: BLOCK_C=%d (smem %d B)",
            _BC,
            _smem_per_block(),
        )
    return _BC


def _wy_cfg_hygon(dt):
    """(panel BLOCK_R, update BLOCK_R, BLOCK_C, num_stages), BLOCK_C derived."""
    if dt == torch.float64:
        return 256, 64, _bc(), 2
    return _generic._WY_BLOCK_R, min(_generic._WY_BLOCK_R, 128), _bc(), 3


@triton.jit
def _wy_update_hygon(
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
    """trailing -= V @ (T^T @ (V^T @ trailing)), with float64 avoiding tl.dot.

    Identical to the generic kernel except that each of the three contractions
    has a float64 form written as P rank-1 updates, P being the panel width
    (16) rather than a problem dimension -- so the loop is short and fully
    unrolled, and no tile grows.
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


# ---- apply, once, at import (pure Python rebinds; no device access) ----
_generic._wy_cfg = _wy_cfg_hygon
_generic._wy_update = _wy_update_hygon


def linalg_lstsq(A, b, rcond=None, driver=None):
    """Hygon specialization: derived WY block width, float64 without tl.dot."""
    logger.debug("GEMS_HYGON LINALG_LSTSQ")
    _bc()  # derive the device-dependent config now, on a live context
    return _generic.linalg_lstsq(A, b, rcond=rcond, driver=driver)
