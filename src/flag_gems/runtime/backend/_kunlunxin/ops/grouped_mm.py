import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from . import group_gemm as group_gemm_mod
from . import mm as mm_mod
from .mm import mm_out as vendor_mm_out

logger = logging.getLogger(__name__)


def _set_fast_mode(dtype, M, K, N):
    return mm_mod._set_matmul_fast_mode(dtype, M, K, N)


def _restore_fast_mode(saved):
    mm_mod._restore_matmul_fast_mode(saved)


@libentry()
@triton.jit
def _grouped_mm_dense_kernel(
    A,
    B,
    C,
    tile_row,
    tile_g,
    K,
    N,
    stride_ak,
    NT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One (BLOCK_M, BLOCK_N) output tile per program.

    The tile is complete (host-guaranteed: rows [row0, row0+BLOCK_M) lie inside
    one group, row0+BLOCK_M <= group end <= M_total), so loads and the store
    are unmasked and provably in-bounds - no OOB lane can reach the backend's
    masked-memory mis-lowering (see mm.py, 2026-09-02).
    """
    pidx = ext.program_id(0) // NT
    ni = ext.program_id(0) % NT
    g = tl.load(tile_g + pidx).to(tl.int64)
    row0 = tl.load(tile_row + pidx)
    rm = row0 + tl.arange(0, BLOCK_M)
    rn = ni * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A_ptrs = A + (rm[:, None] * stride_ak + rk[None, :])
    B_ptrs = B + (g * K * N) + (rk[:, None] * N + rn[None, :])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptrs)
        b = tl.load(B_ptrs)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        A_ptrs += BLOCK_K
        B_ptrs += BLOCK_K * N
    acc = acc.to(C.dtype.element_ty)
    C_ptrs = C + (rm[:, None] * N + rn[None, :])
    tl.store(C_ptrs, acc)


@libentry()
@triton.jit
def _grouped_mm_tail_kernel(
    A,
    B,
    C,
    offs,
    M_total,
    K,
    N,
    stride_ak,
    NT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Partial last (BLOCK_M, BLOCK_N) tile of each ragged group.

    Rows [mg // BLOCK_M * BLOCK_M, mg) are real; the rest of the tile is
    garbage.  A rows are loaded wrapped (`% M_total`, the mm_kernel pattern)
    and the store keeps the per-row mask - every lane address (masked or not)
    is in-bounds of C after the wrap, so the known TritonXPU OOB masked-store
    issue cannot trigger (probed: masks are honoured, nbad=0).
    """
    g = ext.program_id(0)
    ni = ext.program_id(1)
    offs_g = tl.load(offs + g)
    offs_prev = tl.load(offs + g - 1) if g > 0 else 0
    mg = offs_g - offs_prev
    if mg % BLOCK_M != 0:
        mi = mg // BLOCK_M
        rm_local = mi * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = ni * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        ld_m = rm_local < mg
        rm = offs_prev + rm_local
        rm_w = rm % M_total
        A_ptrs = A + (rm_w[:, None] * stride_ak + rk[None, :])
        B_ptrs = B + g.to(tl.int64) * (K * N) + (rk[:, None] * N + rn[None, :])
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(A_ptrs)
            b = tl.load(B_ptrs)
            if a.dtype != b.dtype:
                a = a.to(C.dtype.element_ty)
                b = b.to(C.dtype.element_ty)
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
            A_ptrs += BLOCK_K
            B_ptrs += BLOCK_K * N
        acc = acc.to(C.dtype.element_ty)
        C_ptrs = C + (rm_w[:, None] * N + rn[None, :])
        tl.store(C_ptrs, acc, mask=ld_m[:, None])


def _vendor_loop_group_mm(A, B, offs, num_groups, offs_cpu, M, N):
    C = A.new_empty(M, N)
    s = 0
    for g in range(num_groups):
        e = int(offs_cpu[g])
        if e > s:
            vendor_mm_out(A[s:e], B[g], out=C[s:e])
        s = e
    return C


def group_mm(A: torch.Tensor, B: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Kunlunxin XPU grouped_mm (`aten::_grouped_mm`).

    The generic implementation (`src/flag_gems/ops/group_gemm.py`) launches one
    persistent kernel whose per-program group scan is slow on XPU, and the
    previous vendor override launched one `mm_out` 2D GEMM per group (G
    sequential launches, each with a padded buffer alloc + copy-back, plus a
    2x/4x K-pad copy when K < 256) - measured 0.48-0.63x of native.

    This implementation treats the group-rows as one contiguous M axis and
    launches exactly two kernels (no per-group padding, no copy-back):

    * `_grouped_mm_dense_kernel` - one program per complete (BLOCK_M,
      BLOCK_N) tile; the host builds a tile map (tile_row/tile_g) so the
      kernel body is a plain unmasked, unguarded GEMM (in-bounds by
      construction; a runtime guard on the tile index was measured to cost
      ~1.5-2x on this backend).
    * `_grouped_mm_tail_kernel` - the partial last tile of each ragged group
      (G x NT programs) with %-wrapped loads and a per-row store mask; every
      lane address stays in-bounds (verified: masks are honoured, nbad=0).

    Both kernels require K % BLOCK_K == 0 and N % BLOCK_N == 0 (no masked
    K/N loads); shapes outside that (odd K/N) fall back to the per-group
    `mm_out` loop below.

    Shapes in the large-M / small-K regime (K <= 256 and M >= 65536) are routed
    to the persistent-kernel implementation in `group_gemm.py`, which autotunes
    one config per shape and launches NUM_SMS * 4 programs.  On such shapes the
    many small tiles of the two kernels above do not hide memory latency well:
    measured on P800 (M=525986, N=2048, K=128, G=64) 5.21 ms vs 6.36 ms
    (1.05x vs 0.86x of the torch reference).
    """
    logger.debug("GEMS_KUNLUNXIN GROUP_MM")
    assert A.dim() == 2
    assert B.dim() == 3
    M, K = A.shape
    num_groups, BK, N = B.shape
    assert num_groups == offs.numel()
    if num_groups == 0:
        return A.new_empty(M, N)

    if K <= 256 and M >= 65536:
        # Large-M / small-K regime: persistent kernel with per-shape tuning.
        # Dispatched before the host copy below, whose device sync would
        # otherwise add ~1.3 ms to this path.
        return group_gemm_mod.group_mm(A, B, offs)
    offs_cpu = offs.detach().cpu()
    offs_l = offs_cpu.tolist()

    if K % 256 == 0:
        BK_T = 256
    elif K % 128 == 0:
        BK_T = 128
    else:
        BK_T = 0
    if N % 256 == 0:
        BN_T = 256
    elif N % 128 == 0:
        BN_T = 128
    else:
        BN_T = 0
    if BK_T == 0 or BN_T == 0:
        return _vendor_loop_group_mm(A, B, offs, num_groups, offs_cpu, M, N)
    BM_T = 256

    tile_row = []
    tile_g = []
    s = 0
    for g in range(num_groups):
        e = offs_l[g]
        mg = e - s
        complete = mg // BM_T
        tile_row.extend([s + t * BM_T for t in range(complete)])
        tile_g.extend([g] * complete)
        s = e
    NT = (N + BN_T - 1) // BN_T
    T = len(tile_row)

    C = A.new_empty(M, N)
    offs_t = offs if offs.dtype == torch.int32 else offs.to(torch.int32)
    tile_row_t = torch.tensor(tile_row, dtype=torch.int32, device=A.device)
    tile_g_t = torch.tensor(tile_g, dtype=torch.int32, device=A.device)

    saved = _set_fast_mode(A.dtype, M, K, N)
    try:
        if T > 0:
            _grouped_mm_dense_kernel[(T * NT,)](
                A,
                B,
                C,
                tile_row_t,
                tile_g_t,
                K,
                N,
                A.stride(0),
                NT=NT,
                BLOCK_M=BM_T,
                BLOCK_N=BN_T,
                BLOCK_K=BK_T,
                num_warps=8,
            )
        _grouped_mm_tail_kernel[(num_groups, NT)](
            A,
            B,
            C,
            offs_t,
            M,
            K,
            N,
            A.stride(0),
            NT=NT,
            BLOCK_M=BM_T,
            BLOCK_N=BN_T,
            BLOCK_K=BK_T,
            num_warps=8,
        )
    finally:
        _restore_fast_mode(saved)
    return C
