import importlib
import logging

import triton
import triton.language as tl

logger = logging.getLogger("flag_gems." + __name__)

# NOT `import flag_gems.ops.linalg_lstsq as _generic`: flag_gems/ops/__init__.py
# does `from .linalg_lstsq import linalg_lstsq`, which rebinds that attribute on
# the package to the FUNCTION, and `import a.b as c` binds by attribute lookup,
# so _generic would be the function. import_module resolves via sys.modules.
_generic = importlib.import_module("flag_gems.ops.linalg_lstsq")

# ---------------------------------------------------------------------------
# One Iluvatar device fact, and it is a COMPILER limit rather than a numerical
# one: float64 `tl.dot` does not compile. On a BI-V150,
#
#     tests/test_linalg_lstsq.py::test_linalg_lstsq_tall_blocked_fp64
#     triton.compiler.errors.CompilationError: at 39:16:
#         Wacc += tl.dot(tl.trans(Vb), Tb, input_precision="ieee")
#                 ^
#
# The device also reports `support_fp64 = False` and its driver warns that
# torch.double has "limited support", yet every other fp64 case in the suite
# passes: the monolithic and blocked-TSQR paths hold no `tl.dot` at all, so
# they compile and run. Only compact-WY contracts with one, and all three of
# the generic operator's `tl.dot` calls live in that single kernel.
#
# So the whole port is one kernel: `_wy_update` with a float64 form that
# expresses each contraction as P rank-1 updates. The branch is on COMPUTE, a
# tl.constexpr, so it resolves at compile time and float32 keeps `tl.dot`
# unchanged -- that path compiles and is faster.
#
# Nothing else is rebound. Unlike the MetaX port, no shared-memory pressure
# was observed here (the other 74 cases pass with upstream's block sizes), and
# configuration that has not been measured on the device does not belong in a
# backend override.
#
# NOTE ON COVERAGE: this device declares `support_fp64 = False`, and it means
# it -- `torch.matmul` reports "gemm of double is not supported on CoreX",
# cuSOLVER has neither Dormqr nor Dorgqr, and a 256x256 float64 solve returns
# NaN even with this kernel in place. The suite therefore SKIPS float64 here,
# so the branch below is not exercised in CI. It is kept because it is correct
# and because the compile error it removes is real: if Iluvatar gains usable
# float64, the branch is what makes compact-WY work, and until then it costs
# nothing -- COMPUTE is a tl.constexpr, so float32 compiles exactly as before.
# ---------------------------------------------------------------------------


@triton.jit
def _wy_update_iluvatar(
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


# ---- apply, once, at import (a pure Python rebind; no device access) ----
_generic._wy_update = _wy_update_iluvatar


def linalg_lstsq(A, b, rcond=None, driver=None):
    """Iluvatar specialization: generic kernels, float64 compact-WY sans tl.dot."""
    logger.debug("GEMS_ILUVATAR LINALG_LSTSQ")
    return _generic.linalg_lstsq(A, b, rcond=rcond, driver=driver)
