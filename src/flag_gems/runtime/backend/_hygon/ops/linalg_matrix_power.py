"""linalg_matrix_power override for the hygon (HIP) backend.

Hygon runs the generic NV implementation unchanged except that its HIP
devices expose only 64 KB of shared memory per block, so the fp64-stored
TRSM update gemm must use a 2-stage software pipeline instead of the
3-stage default.  The shared hosts in flag_gems.ops.linalg_matrix_power are
hooked to the 2-stage solve below and the generic dispatch is re-exported.
(amd reuses this file verbatim.)

The two public entry points are carried locally rather than re-exported: the
bodies below are verbatim copies of the generic implementation.  Everything
they call - the LU/TRSM kernels, the df64 helpers, the matmuls and the tuning
constants - is still imported from the generic module, so the only behavioural
difference from the generic path remains the 2-stage solve hooked in above.
"""

import importlib
import logging

import torch
import triton

import flag_gems
from flag_gems.ops.linalg_matrix_power import (  # noqa: E402, F401
    SINGLE_TILE_MAX,
    TILE,
    TILED_MAX,
    TRITON_THRESHOLD,
    _eye_like,
    _grid_sync_kernel,
    _inverse,
    _matmul,
    _resolve_linalg_matrix_power_out_args,
    _single_tile_kernel,
    _trsm_kernel,
)

logger = logging.getLogger(__name__)

# NB: bind the *module* explicitly - ``import flag_gems.ops.linalg_matrix_power
# as _generic`` would resolve through the package attribute, which the
# re-exported ``linalg_matrix_power`` function shadows.
_generic = importlib.import_module("flag_gems.ops.linalg_matrix_power")


def _trsm_solve_2d(A_tri, B, upper: bool, unitriangular: bool):
    """Same blocked triangular solve as the generic one, at a 2-stage
    software-pipeline depth (64 KB shared memory, see the module docstring)."""
    n = A_tri.shape[0]
    k = B.shape[1]
    K_SLICE = 8
    BM = 128
    num_kslices = (k + K_SLICE - 1) // K_SLICE
    if unitriangular:
        inv = B  # INV_ptr is only dereferenced when UNIT is false
        unit_flag = True
    else:
        inv = torch.zeros(num_kslices * 32, dtype=A_tri.dtype, device=A_tri.device)
        unit_flag = False
    _trsm_kernel[(num_kslices,)](
        A_tri,
        B,
        inv,
        n,
        k,
        A_tri.stride(0),
        B.stride(0),
        32,
        K_SLICE,
        BM,
        upper,
        unit_flag,
        num_warps=1,
        # fp64 dot operands are staged through shared memory (8 bytes/element);
        # the 3-stage default needs ~73.7 KB, over the 64 KB HIP limit.
        num_stages=2,
    )
    return B


# Hook the shared hosts (linalg_lu_solve / _inverse) so every triangular solve
# on this backend runs at the 2-stage depth.  One backend per process, and the
# generic module is fully imported before this override is applied, so the
# rebind is safe.
#
# `_inverse` below resolves `_trsm_solve_2d` as a global in the *generic*
# module at call time, so this rebind is what makes the local entry points pick
# up the 2-stage solve too.
_generic._trsm_solve_2d = _trsm_solve_2d


def linalg_matrix_power(
    A: torch.Tensor,
    n: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    logger.debug("GEMS_HYGON LINALG_MATRIX_POWER")

    # ---- validation ----
    shape = A.shape
    if len(shape) < 2:
        raise RuntimeError(
            f"linalg_matrix_power: A must be at least 2-D, got shape {shape}"
        )
    m, k = shape[-2], shape[-1]
    if m != k:
        raise RuntimeError(f"linalg_matrix_power: A must be square, got ({m}, {k})")
    if not isinstance(n, int):
        raise TypeError(f"linalg_matrix_power: n must be int, got {type(n).__name__}")
    if A.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only float32 and float64, "
            f"got {A.dtype}"
        )

    # ---- n == 0 ----
    if n == 0:
        eye = _eye_like(A)
        if out is not None:
            out.copy_(eye)
            return out
        return eye

    # ---- n == 1: A¹ = A — plain copy, no computation / kernel launch ----
    if n == 1:
        if out is not None:
            out.copy_(A)
            return out
        return A.clone()

    # ---- flag_gems Triton kernels are CUDA-only; n==0/n==1 above work on any
    # device, every computational path below requires the flag_gems device. ----
    if A.device.type != flag_gems.device:
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only {flag_gems.device}, "
            f"got {A.device}"
        )

    # ---- negative n ----
    upcast = False
    if n < 0:
        # f32 negatives: matrices above 512 use the f32-compute path — the
        # inverse is computed in f32 (fp32 storage + fp64 accumulation in the
        # LU/TRSM updates, fp64-accumulation Newton refinement), which is faster
        # once the barrier-bound LU is amortised.  Smaller f32 matrices keep the
        # fp64 upcast (the refinement overhead is not amortised there).
        # The power (A⁻¹)^|n| always runs in fp64 and the result is cast to f32.
        upcast = A.dtype == torch.float32
        if upcast and A.shape[-1] <= 512:
            A = A.double()
        A = _inverse(A)
        n = -n
        if upcast:
            A = A.double()

    # ---- n == 2, 3: fast paths for large M ----
    # mm/bmm kernels take at most 3-D inputs, so flatten deeper batch dims
    # (e.g. (b1, b2, M, M)) and reshape the result back to the input shape.
    if A.dim() > 3:
        fast_A = A.reshape(-1, m, m)
    else:
        fast_A = A
    if n == 2 and m > TRITON_THRESHOLD:
        r = _matmul(fast_A, fast_A)
        if A.dim() > 3:
            r = r.reshape(shape)
        if upcast:
            r = r.float()
        if out is not None:
            out.copy_(r)
            return out
        return r
    if n == 3 and m > TRITON_THRESHOLD:
        r = _matmul(_matmul(fast_A, fast_A), fast_A)
        if A.dim() > 3:
            r = r.reshape(shape)
        if upcast:
            r = r.float()
        if out is not None:
            out.copy_(r)
            return out
        return r

    # ---- flatten batch ----
    if len(shape) > 2:
        A_flat = A.reshape(-1, m, m)
    else:
        A_flat = A.unsqueeze(0)
    batch_size = A_flat.shape[0]
    batch_stride = m * m

    if out is not None:
        if upcast:
            # fp64 compute buffer (the kernels produce fp64); cast to fp32 out
            # at the end.
            out_flat = torch.empty(
                batch_size, m, m, dtype=torch.float64, device=A.device
            )
        else:
            out_flat = out.reshape(-1, m, m)
    else:
        out_flat = torch.empty(batch_size, m, m, dtype=A.dtype, device=A.device)

    # ---- dispatch ----
    if m <= SINGLE_TILE_MAX and A.device.type == flag_gems.device:
        # Tier 1: single-program fused (M <= 32).  tl.dot in sweet spot.
        BLOCK = max(triton.next_power_of_2(m), 16)
        _single_tile_kernel[(batch_size,)](
            A_flat,
            out_flat,
            m,
            n,
            batch_stride,
            BLOCK=BLOCK,
        )

    elif m <= TILED_MAX and A.device.type == flag_gems.device:
        # Tier 2: single-tile (33 <= M <= 64).
        # Grid-sync barrier overhead (~5 us/barrier × 3) exceeds the
        # single-SM tl.dot(64,64) time for 4-tile grids.  CUDA graph
        # memcpy overhead (~5 us × 3 copies) also dominates for M≤64.
        BLOCK = max(triton.next_power_of_2(m), 16)
        _single_tile_kernel[(batch_size,)](
            A_flat,
            out_flat,
            m,
            n,
            batch_stride,
            BLOCK=BLOCK,
        )

    elif m <= 256 and A.device.type == flag_gems.device:
        # Tier 3: grid-level sync fused (65 <= M <= 256).
        TILES = triton.cdiv(m, TILE)
        # Fresh buffers per call: the kernel's Step 0 fully overwrites every
        # scratch slot it reads, and the round-based barrier logic works from a
        # zero-initialized counter (0 is a multiple of n_total), so nothing
        # needs to persist across calls.  Allocating fresh also keeps concurrent
        # calls on different streams from racing on shared buffers, and avoids
        # an ever-growing barrier counter that could overflow int32.
        scratch = torch.empty(4 * batch_size, m, m, dtype=A.dtype, device=A.device)
        barrier = torch.zeros(batch_size * 64, dtype=torch.int32, device=A.device)
        _grid_sync_kernel[(batch_size, TILES, TILES)](
            A_flat,
            out_flat,
            scratch,
            barrier,
            m,
            n,
            batch_stride,
            TILE_BLOCK=TILE,
            TILES=TILES,
        )

    else:
        # M > 256: host-side binary exponentiation with the flag_gems Triton
        # matmul kernels (mm for 2D, bmm for batched), one launch per step.
        is_batched = batch_size > 1
        z = A_flat if is_batched else A_flat.squeeze(0)
        result = None
        n_remaining = n
        while n_remaining > 0:
            if n_remaining & 1:
                result = z if result is None else _matmul(result, z)
            n_remaining >>= 1
            if n_remaining > 0:
                z = _matmul(z, z)
        if is_batched:
            out_flat.copy_(result)
        else:
            out_flat.squeeze_(0).copy_(result)

    # ---- reshape back ----
    if upcast:
        out_flat = out_flat.float()
    if len(shape) > 2:
        out_flat = out_flat.reshape(shape)
    else:
        out_flat = out_flat.squeeze(0)

    if out is not None:
        if upcast:
            out.copy_(out_flat)
        return out
    return out_flat


def linalg_matrix_power_out(
    A: torch.Tensor, n: int, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Out variant of :func:`linalg_matrix_power` (aten ``linalg_matrix_power.out``).

    ``linalg_matrix_power`` already writes into ``out`` in place and returns it,
    so this resolves the required ``out`` tensor and delegates — kept as a
    separate entry point so the ``*.out`` dispatcher key routes through flag_gems
    rather than falling back to torch's native (compute) implementation.
    """
    logger.debug("GEMS_HYGON LINALG_MATRIX_POWER_OUT")
    out_resolved = _resolve_linalg_matrix_power_out_args(out)
    return linalg_matrix_power(A, n, out=out_resolved)
