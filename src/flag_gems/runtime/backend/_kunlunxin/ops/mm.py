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

import logging
import os

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def heur_split_k(args):
    return 1


def heur_even_k(args):
    return args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0


def heur_group_m(args):
    if args["BLOCK_M"] > args["BLOCK_N"]:
        return 1
    else:
        return (args["M"] + args["BLOCK_M"] - 1) // args["BLOCK_M"]


autotune_decorator = triton.autotune(
    configs=[],
    generate_configs="mm",
    key=["M", "N", "K"],
)


# 2026-09-02 (XPU 3): the generate_configs="mm" autotune path is NOT safe on
# this backend, so it is no longer the default (same call as addmm, 2026-08-31:
# addmm_family_num_stages_fix_xpu4_20260831.md / addmm_out_xpu4_20260831).
#
# Root cause of the OOB (measured): the autotune configs are synthesised per
# shape from triton.runtime.autotuner.block_size_candidates, whose arch-3
# envelope is 80/64-multiples (BLOCK_M in {2..480}, BLOCK_N up to 512,
# BLOCK_K up to 512; e.g. M=4096 -> BLOCK_M in {240,320,400,480}, M=16 ->
# down to 2).  mm_kernel stores a full cdiv(M,BLOCK_M)*BLOCK_M x
# cdiv(N,BLOCK_N)*BLOCK_N tile *unmasked* (TritonXPU mis-lowers masked
# stores/loads whose addresses leave the allocation), so the host pad must
# equal the launched BLOCK exactly.  _padded_or_direct pads with the fixed
# 128/256 heuristic blocks, so any autotune BLOCK that does not divide the
# shape (virtually all of them: 80-multiples never divide a power-of-two M)
# leaves the padded allocation: measured OOB / NaN on the grouped_mm
# [64,2048,128] and mm [1023,255] shapes (autotune BLOCK_K=512 also leaves
# the K-padded A/B extent, kp=cdiv(K,256)*256=256).
#
# Fix: mm_kernel no longer autotunes -- it always runs the fixed _block_*
# tiles, passed explicitly by the launcher, so the C/K pad and the launched
# BLOCKs agree by construction in BOTH modes; under =1 the aligned fast path
# is additionally disabled (its autotuned BLOCKs cannot be verified against
# the gate).  Set KLX_USE_AUTOTUNE=1 only for tuning experiments.
KLX_USE_AUTOTUNE = os.environ.get("KLX_USE_AUTOTUNE", "0") == "1"

if not KLX_USE_AUTOTUNE:

    # XPU tile probe (2026-08-13, XPU 4, 7 unique shapes x 3 dtypes, direct
    # do_bench warm+rep medians): the 256^3 tile is the floor for M,N > 512 on
    # all dtypes (fp16 4096^3 0.68ms / 0.82x, fp32 1.37ms / 0.90x), while small
    # shapes (M,N <= 512) are launch-bound and prefer the 128-tile w4 config
    # (384^3: 0.014 -> 0.0087ms fp16, 0.021 -> 0.012ms bf16, 0.014 -> 0.011ms
    # fp32, ~1.05-1.12x vs torch). num_stages stays at backend default (2):
    # bf16 BK=256 collapses at s3 (1.82ms vs 1.32ms on 4096^3). BK is kept at
    # 256 for the 256-tile (fp16 needs BK=256: 1.01ms at BK=128 vs 0.68ms at
    # BK=256 on 4096^3).

    def heur_block_m(args):
        M = args["M"]
        if M <= 512:
            return 128
        return 256

    def heur_block_n(args):
        N = args["N"]
        if N <= 512:
            return 128
        return 256

    def heur_block_k(args):
        M = args["M"]
        N = args["N"]
        if M <= 512 and N <= 512:
            return 128
        return 256

    def heur_num_warps(args):
        M = args["M"]
        N = args["N"]
        if M <= 512 and N <= 512:
            return 4
        return 8

    autotune_decorator = triton.heuristics(
        {
            "BLOCK_M": heur_block_m,
            "BLOCK_N": heur_block_n,
            "BLOCK_K": heur_block_k,
            "num_warps": heur_num_warps,
        }
    )


@libentry()
# NOTE: no autotune_decorator here on purpose.  This kernel is only ever
# launched on host-padded buffers (_padded_or_direct -> _compute_padded),
# whose M/N/K extents are the _block_* multiples; the launcher therefore
# passes BLOCK_M/BLOCK_N/BLOCK_K (and num_warps) explicitly so the pad
# divisors and the launched tiles always agree.  Autotune-generated BLOCKs
# are shape-dependent and would silently leave those extents (see the
# KLX_USE_AUTOTUNE comment): the unmasked store must stay inside the C
# allocation, and the K-loop loads must stay inside the K-padded A/B.
@triton.heuristics(
    {
        "SPLIT_K": heur_split_k,
        "EVEN_K": heur_even_k,
        "GROUP_M": heur_group_m,
    }
)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # matrix multiplication
    pid = ext.program_id(0)
    pid_z = ext.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    #
    # In-bounds-by-construction addressing (2026-09-02, XPU 3): the callers
    # hand in A (M, K_pad), B (K_pad, N) with K_pad % BLOCK_K == 0 and C
    # (M_pad, N_pad) with M_pad = cdiv(M, BLOCK_M)*BLOCK_M, N_pad =
    # cdiv(N, BLOCK_N)*BLOCK_N.  Loads wrap rows/cols through % (so the last
    # partial tile stays inside the padded K extent) and no load/store mask is
    # used: TritonXPU mis-lowers both masked loads and masked stores whose
    # addresses leave the allocation (intermittent "illegal memory access",
    # status 700, and silently wrong values measured on fp16/bf16, e.g. the
    # 1023x255 self-transpose cases).  The previous combination of
    # `rm % M` + tl.max_contiguous/tl.multiple_of hints lied about contiguity
    # on the partial last tile and faulted the same way.  Garbage rows/cols
    # land in the C padding and are excluded by the host-side view.
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = rm % M
    rbn = rn % N
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        a = tl.load(A)
        b = tl.load(B)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers; C (M_pad, N_pad) covers them
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc)
    else:
        tl.atomic_add(C, acc)


@libentry()
@autotune_decorator
@triton.heuristics(
    {
        "SPLIT_K": heur_split_k,
        "EVEN_K": heur_even_k,
        "GROUP_M": heur_group_m,
    }
)
@triton.jit
def mm_kernel_aligned(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # Fast path for exactly tile-aligned shapes (M % BLOCK_M == 0,
    # N % BLOCK_N == 0, K % (BLOCK_K * SPLIT_K) == 0): every tile is complete,
    # so the `rm % M` wrap is the identity, the max_contiguous/multiple_of
    # hints are truthful (they enable the backend's block loads, ~1.2-1.7x on
    # small tiles) and no address can leave the allocations.  Ragged shapes
    # must go through mm_kernel + host-side padding: on TritonXPU the partial
    # last tile makes these hints lie and the lowering faults intermittently
    # (illegal memory access, status 700; measured on the 1023x255
    # self-transpose shapes).
    pid = ext.program_id(0)
    pid_z = ext.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]

_FAST_MODE_ENV = "XMLIR_MATMUL_FAST_MODE"


def _set_matmul_fast_mode(a_dtype, M, N, K):
    """XPU probe (2026-08-13, XPU 4): XMLIR_MATMUL_FAST_MODE=1 speeds the
    bf16 tl.dot lowering for large-K GEMMs (4096^3 1.32->0.81ms; 2048^3
    0.20->0.15ms; K=65536 1.86->1.44ms) while fp16/fp32 kernels are
    unaffected; on small bf16 shapes it regresses (64^3 0.012->0.018ms), so
    the flag is applied selectively: bf16 only, with K >= 2048 and both M, N
    >= 128."""
    if a_dtype == torch.bfloat16 and K >= 2048 and M >= 128 and N >= 128:
        saved = os.environ.get(_FAST_MODE_ENV)
        os.environ[_FAST_MODE_ENV] = "1"
        return saved
    return None


def _restore_matmul_fast_mode(saved):
    if saved is None:
        os.environ.pop(_FAST_MODE_ENV, None)
    else:
        os.environ[_FAST_MODE_ENV] = saved


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def _block_m(M):
    return 128 if M <= 512 else 256


def _block_n(N):
    return 128 if N <= 512 else 256


def _block_k(M, N):
    if M <= 512 and N <= 512:
        return 128
    return 256


def _pad_k(a, b, M, K, N, blk_k, device):
    """Materialize strided / K-unaligned inputs into contiguous buffers.

    ``mm_kernel`` loads full (BLOCK_M, BLOCK_K) / (BLOCK_K, BLOCK_N) tiles with
    no OOB mask (TritonXPU does not honour masked loads whose addresses leave
    the allocation; see the mm_kernel comment), so the K extent must cover
    every column/row the K-loop touches.  The copy goes through the native
    ``_copy_from`` engine (gems does not override it); ``x.contiguous()`` must
    not be used because the registered ``_to_copy`` override mis-handles
    strided sources (see the mm() comment), and the kernel itself is
    stride-generic, so row-major inputs are passed through untouched.
    """
    kp = triton.cdiv(K, blk_k) * blk_k
    if (a.stride(0), a.stride(1)) != (K, 1) or kp != K:
        ap = torch.zeros((M, kp), device=device, dtype=a.dtype)
        torch.ops.aten._copy_from(a, ap[:, :K], False)
        a = ap
    if (b.stride(0), b.stride(1)) != (N, 1) or kp != K:
        bp = torch.zeros((kp, N), device=device, dtype=b.dtype)
        torch.ops.aten._copy_from(b, bp[:K, :], False)
        b = bp
    return a, b, kp


def _launch_kernel(ker, a, b, c, M, N, K, dot_out_dtype, device, blocks=None):
    """Launch ``ker`` on the (possibly padded) buffers.

    ``blocks`` = (blk_m, blk_n, blk_k) is passed explicitly for ``mm_kernel``
    so the launched tiles equal the host pad divisors exactly: the unmasked
    store covers cdiv(M, BLOCK_M)*BLOCK_M rows x cdiv(N, BLOCK_N)*BLOCK_N
    columns and the K-loop loads cover cdiv(K, BLOCK_K)*BLOCK_K columns, all
    of which are in-bounds of the padded C = (cdiv(M,blk_m)*blk_m,
    cdiv(N,blk_n)*blk_n) / K-padded A,B when BLOCK_* == (blk_m, blk_n, blk_k).
    ``mm_kernel_aligned`` supplies BLOCK_* through its (heuristics) decorator
    and must only be used when those tiles divide the shape -- verify with
    the same ``_block_*`` values and pass blocks=None for it.
    """
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    saved = _set_matmul_fast_mode(a.dtype, M, N, K)
    try:
        with torch_device_fn.device(a.device):
            if blocks is None:
                ker[grid](
                    a,
                    b,
                    c,
                    M,
                    N,
                    K,
                    a.stride(0),
                    a.stride(1),
                    b.stride(0),
                    b.stride(1),
                    c.stride(0),
                    c.stride(1),
                    dot_out_dtype=dot_out_dtype,
                )
            else:
                blk_m, blk_n, blk_k = blocks
                ker[grid](
                    a,
                    b,
                    c,
                    M,
                    N,
                    K,
                    a.stride(0),
                    a.stride(1),
                    b.stride(0),
                    b.stride(1),
                    c.stride(0),
                    c.stride(1),
                    dot_out_dtype=dot_out_dtype,
                    BLOCK_M=blk_m,
                    BLOCK_N=blk_n,
                    BLOCK_K=blk_k,
                    num_warps=4 if (M <= 512 and N <= 512) else 8,
                )
    finally:
        _restore_matmul_fast_mode(saved)


def _compute_padded(a, b, c, M, K, N, blk_m, blk_n, blk_k, dot_out_dtype, device):
    """Run mm_kernel on K-padded buffers; ``c`` must be (M_pad, N_pad) so the
    unmasked store stays inside the allocation."""
    a, b, kp = _pad_k(a, b, M, K, N, blk_k, device)
    _launch_kernel(
        mm_kernel, a, b, c, M, N, kp, dot_out_dtype, device, (blk_m, blk_n, blk_k)
    )
    return c


def _padded_or_direct(a, b, dest, M, K, N, blk_m, blk_n, dot_out_dtype, device):
    """Dispatch between the aligned fast kernel and the padded safe kernel.

    mm_kernel_aligned is only sound when every tile is complete (M % blk_m ==
    0, N % blk_n == 0, K % blk_k == 0); its contiguity hints are truthful then
    and it was measured ~1.2-1.7x faster on small tiles.  Any ragged/partial
    dimension (or a strided input, since mm_kernel_aligned assumes row-major
    layouts via the same hints) goes through the K-/C-padding safe path, which
    is in-bounds by construction (see mm_kernel) and copies through the native
    ``_copy_from`` engine when the copy-back is needed.  Under
    KLX_USE_AUTOTUNE=1 the aligned kernel is never used: the autotune block
    generator produces 80/64-multiples (up to 480/512) that virtually never
    divide the shape, so ``M % blk_m == 0`` cannot certify complete tiles and
    the padded path (fixed _block_* tiles) runs instead.  Returns (c,
    needs_copy_back).
    """
    if (
        not KLX_USE_AUTOTUNE
        and (a.stride(0), a.stride(1)) == (K, 1)
        and (b.stride(0), b.stride(1)) == (N, 1)
        and M % blk_m == 0
        and N % blk_n == 0
        and K % _block_k(M, N) == 0
        and dest.stride(1) == 1
    ):
        _launch_kernel(mm_kernel_aligned, a, b, dest, M, N, K, dot_out_dtype, device)
        return dest, False
    mp = triton.cdiv(M, blk_m) * blk_m
    np_ = triton.cdiv(N, blk_n) * blk_n
    c = torch.empty((mp, np_), device=device, dtype=dest.dtype)
    _compute_padded(
        a, b, c, M, K, N, blk_m, blk_n, _block_k(M, N), dot_out_dtype, device
    )
    return c, True


def mm(a, b):
    logger.debug("GEMS_KUNLUNXIN MM")
    device = a.device
    # NOTE: no ``x.contiguous()`` here.  Inside use_gems()/enable() a strided
    # input (e.g. the column-major self-transpose view) dispatches through the
    # registered ``_to_copy`` override, whose flat-1D kernel mis-handles
    # non-contiguous strides (measured: intermittent "illegal memory access",
    # status 700, e.g. a (1023, 255) column-major fp16 view).  mm_kernel takes
    # runtime strides, and _pad_k copies through the native ``_copy_from``
    # engine which is not overridden, so strided inputs need no pre-copy.
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    c, needs_copy = _padded_or_direct(
        a, b, c, M, K, N, _block_m(M), _block_n(N), tl.float32, device
    )
    if not needs_copy:
        return c
    # torch.mm returns a contiguous (M, N) tensor; the padded buffer view is
    # strided (storage M_pad x N_pad), so materialise the exact shape.
    return c[:M, :N].contiguous()


def mm_out(a, b, *, out):
    logger.debug("GEMS_KUNLUNXIN MM_OUT")
    # NOTE: no ``x.contiguous()`` here - see the mm() comment (the registered
    # _to_copy override mis-handles strided inputs).
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    if out.dtype != c_dtype:
        # ATen mm.out rejects an out dtype that differs from the result dtype.
        raise RuntimeError(
            f"mm.out: expected out of dtype {c_dtype} but got {out.dtype}"
        )
    # The kernel always stores unmasked, so `out` is only written directly when
    # the shape is tile-aligned along both axes (every store address then stays
    # inside the (M, N) allocation); a ragged shape goes through a padded
    # buffer and the native strided-copy engine (gems does not override
    # _copy_from), which also covers strided / aliased `out` layouts.
    c, needs_copy = _padded_or_direct(
        a, b, out, M, K, N, _block_m(M), _block_n(N), tl.float32, out.device
    )
    if needs_copy:
        torch.ops.aten._copy_from(c[:M, :N], out, False)
    return out
