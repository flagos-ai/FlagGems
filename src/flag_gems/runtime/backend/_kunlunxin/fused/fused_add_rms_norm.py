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

import builtins
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Launch-bound fix (same pattern as rms_norm / skip_layer_norm). The default
# fused_add_rmsnorm_kernel launches one program per row (grid=(M,)); each row's 1D
# reduce runs at the XPU memory ceiling, so per-row is optimal when N is large.
# BUT when N is small (each row moves few elements) AND M is large (many programs),
# per-program launch latency (~0.6-0.9us) dominates: [100,65536,100] launches 6.5M
# programs -> ~7s (speedup 0.002). In that regime a 2D multi-row tile (each program
# owns TILE_M rows, reduces along axis=1) cuts the grid from M to cdiv(M, TILE_M) and
# amortizes launch cost. The 2D axis=1 reduce is slower per element, so gate it
# strictly to the launch-bound corner: N small AND M large. Otherwise keep per-row.
MULTIROW_N = 256  # masked multi-row tile only pays off for dims this small
MULTIROW_M = 4096  # ... and only once there are enough rows to be launch-bound
TILE_BUDGET = 8192  # rows * cols per 2D tile; keeps the tile in SRAM
MULTIROW_MIN_GRID = 64  # keep at least ~this many programs so cores stay busy

# --- Unmasked multi-row tile (the big win for mid-size N) ---------------------
# The masked 2D tile above is only a win for very small N. Measured on this XPU
# (M=32768, N=512, fp16) the row mask itself, not the 2D shape, is what costs:
#   per-row grid=(M,)      21.2 ms   (speedup 0.013)
#   2D tile, masked        43.5 ms   (speedup 0.006)  <- worse than per-row
#   2D tile, UNMASKED       0.67 ms  (speedup 0.39)   <- 63x faster than masked
# So for N in (MULTIROW_N, UNMASKED_N] we emit a fully unmasked kernel, which
# requires M % TILE_M == 0; TILE_M walks down by powers of two until it divides
# M. Everything else (launch kwargs, TILE_M beyond the budget, a per-row inner
# loop, flat load + reshape) measured as noise, slower, or uncompilable.
UNMASKED_N = 4096  # largest normalized dim that still fits an unmasked tile
# The tile is NOT only for huge M -- it is a large win as soon as there are
# enough rows to form a few tiles. Measured M=1024, N=1024 (per-row vs best
# tile): fp16 1.18 -> 5.83, fp32 1.35 -> 5.49, bf16 1.06 -> 4.17. So the row
# gate is just "enough rows to fill the grid", not the MULTIROW_M of the masked
# path above.
UNMASKED_M = 512
# TILE_M * N budget: the kernel holds the fp32 [TILE_M, N] tile live across the
# reduce, and bf16 needs ~2x the SRAM of fp16/fp32 for the same tile, so it gets
# a smaller budget or it fails to compile with `uni_sram`. Cap it per dtype
# here; do NOT instead catch OutOfResources at launch and retry a lighter
# kernel, because the failed compile is re-attempted on every call (that
# measured bf16 speedup 0.009 vs 2.59 for the capped budget).
UNMASKED_TILE_ELEMS = 32768
UNMASKED_TILE_ELEMS_BF16 = 8192
UNMASKED_MIN_GRID = 8  # below this the grid is too small to fill the cores


def _prev_pow2(n):
    n = builtins.max(1, int(n))
    return 1 << (n.bit_length() - 1)


def _pick_tile_m(M, N):
    # TILE_M must satisfy two competing constraints:
    #   * budget: TILE_M * N <= TILE_BUDGET so the [TILE_M, N] tile fits SRAM.
    #   * grid  : cdiv(M, TILE_M) stays large enough to keep the cores busy;
    #             a too-large TILE_M (tiny N) collapses the grid to a few
    #             programs and underutilizes the device.
    # A power-of-2 TILE_M is measurably faster on XPU than an odd value
    # (e.g. N=100 -> 8192//100=81 ran ~1.6x slower than the pow2 64), so round
    # each cap down to a power of two and take the smaller.
    budget_cap = _prev_pow2(TILE_BUDGET // N)
    grid_cap = _prev_pow2(builtins.max(1, M // MULTIROW_MIN_GRID))
    return builtins.max(1, builtins.min(budget_cap, grid_cap))


def _pick_unmasked_tile_m(M, N, dtype):
    """Largest power-of-2 TILE_M that fits the SRAM budget and divides M exactly.

    Returns None when no such TILE_M leaves a usable grid, in which case the
    caller must fall back to a masked path (the unmasked kernel would read out
    of bounds).
    """
    # bf16 holds ~2x the SRAM for the same tile, so cap its budget instead of
    # letting the keep-live kernel fail to compile. (Catching OutOfResources at
    # launch and retrying is NOT viable: the failed compile is re-attempted on
    # every call and costs ~40x -- measured bf16 speedup 0.009 that way.)
    budget = (
        UNMASKED_TILE_ELEMS_BF16 if dtype == torch.bfloat16 else UNMASKED_TILE_ELEMS
    )
    tile_m = builtins.min(_prev_pow2(budget // N), _prev_pow2(M))
    while tile_m > 1 and M % tile_m:
        tile_m //= 2
    if tile_m < 2 or (M // tile_m) < UNMASKED_MIN_GRID:
        return None
    return tile_m


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rmsnorm_kernel(
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,  # whether N is not a multiple of BLOCK_SIZE
):
    pid = ext.program_id(0)
    X += pid * x_stride_r
    R += pid * r_stride_r

    # NOTE (kunlunxin/XPU): when N == BLOCK_SIZE (power-of-2 normalized dim, e.g.
    # N=1024/4096) the `cols < N` mask is trivially all-true, but keeping the
    # masked tl.load/tl.store still forces the XPU slow masked-memory path
    # (~1.06x fp32 but up to ~2x fp16/bf16 slower, byte-identical output). Take an
    # unmasked fast path when N is divisible by BLOCK_SIZE, mirroring rms_norm's
    # tile kernel. Same numeric result (all lanes valid), pure memory-path opt.
    cols = tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = cols < N
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        tl.store(R + cols * r_stride_c, x, mask=mask)
        var = tl.sum(x * x / N, axis=0)
        rrms = 1 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0)
        y = (x * rrms).to(X.dtype.element_ty) * w
        tl.store(X + cols * x_stride_c, y, mask=mask)
    else:
        x = tl.load(X + cols * x_stride_c).to(tl.float32)
        r = tl.load(R + cols * r_stride_c).to(tl.float32)
        x += r
        tl.store(R + cols * r_stride_c, x.to(R.dtype.element_ty))
        var = tl.sum(x * x / N, axis=0)
        rrms = 1 / tl.sqrt(var + eps)
        w = tl.load(W + cols)
        y = (x * rrms).to(X.dtype.element_ty) * w
        tl.store(X + cols * x_stride_c, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rmsnorm_kernel_tile(
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weight
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,  # whether N is not a multiple of BLOCK_SIZE
):
    pid = tl.program_id(0)
    X += pid * x_stride_r
    R += pid * r_stride_r

    # Same masked-vs-unmasked XPU story as rms_norm's tile kernel: when N is a
    # multiple of BLOCK_SIZE (e.g. N=65536, BLOCK_SIZE=8192) every `cols < N` mask
    # is all-true, but the masked memory path is ~1.6x (fp16) / ~2x (bf16) slower
    # with byte-identical output. Take an unmasked fast path when divisible.
    _var_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
            r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        else:
            x = tl.load(X + cols).to(tl.float32)
            r = tl.load(R + cols).to(tl.float32)
        x += r
        _var_base += x * x / N
    var = tl.sum(_var_base)
    rrms = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
            r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
            x += r
            w = tl.load(W + cols, mask, other=0.0)
            y = (x * rrms).to(X.dtype.element_ty) * w
            # write back to residual and input
            tl.store(R + cols * r_stride_c, x, mask=mask)
            tl.store(X + cols * x_stride_c, y, mask=mask)
        else:
            x = tl.load(X + cols).to(tl.float32)
            r = tl.load(R + cols).to(tl.float32)
            x += r
            w = tl.load(W + cols)
            y = (x * rrms).to(X.dtype.element_ty) * w
            # write back to residual and input
            tl.store(R + cols * r_stride_c, x.to(R.dtype.element_ty))
            tl.store(X + cols * x_stride_c, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rmsnorm_multirow_kernel(
    X,  # pointer to the input (gets normalized output)
    R,  # pointer to the residual (gets x + r)
    W,  # pointer to the weight
    M,  # number of rows
    eps,  # epsilon to avoid division by zero
    TILE_M: tl.constexpr,
    N: tl.constexpr,  # number of columns (normalized dim), used as tile width
):
    # Each program owns a [TILE_M, N] tile: TILE_M consecutive rows, the whole
    # normalized dim as ONE contiguous column block. N is a constexpr and n_off
    # spans exactly [0, N) with NO power-of-2 padding, so the tile is one stride-1
    # contiguous block -> XPU OffsetAnalysis emits block DMA. (Runtime N or padded
    # TILE_N would force discrete access ~2x slower, per rms_norm ROUND 2.)
    pid = ext.program_id(0)

    n_off = tl.arange(0, N)
    w = tl.load(W + n_off).to(tl.float32)

    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_off < M
    offs = m_off[:, None] * N + n_off[None, :]

    # Out-of-range rows load garbage (XPU ignores `other=`) but their axis=1 reduce
    # is per-row independent and their stores are masked, so valid rows are unaffected.
    x = tl.load(X + offs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    r = tl.load(R + offs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    x += r
    # write the residual sum back to R
    tl.store(R + offs, x.to(R.dtype.element_ty), mask=m_mask[:, None])

    var = tl.sum(x * x, axis=1) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    y = (x * rrms[:, None]).to(X.dtype.element_ty) * w[None, :]
    # write the normalized output back to X
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=m_mask[:, None])


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rmsnorm_unmasked_kernel(
    X,  # pointer to the input (gets normalized output)
    R,  # pointer to the residual (gets x + r)
    W,  # pointer to the weight
    eps,  # epsilon to avoid division by zero
    TILE_M: tl.constexpr,
    N: tl.constexpr,  # number of columns (normalized dim), used as tile width
):
    # Each program owns a [TILE_M, N] tile with NO masks anywhere: the caller
    # guarantees M % TILE_M == 0, so every lane is in range. Dropping the row
    # mask is worth 63x here (see the UNMASKED_N note above) because a masked
    # 2D tile falls off the XPU block-DMA path entirely.
    #
    # N is a constexpr and n_off spans exactly [0, N) with no power-of-2
    # padding, so the tile is one stride-1 contiguous block.
    #
    # `x` is also deliberately reused for the normalized value instead of
    # keeping a separate `y`: fewer live fp32 [TILE_M, N] temporaries measured
    # ~5% faster and is what lets bf16 fit in uni_sram at all.
    pid = ext.program_id(0)

    n_off = tl.arange(0, N)
    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    offs = m_off[:, None] * N + n_off[None, :]

    x = tl.load(X + offs).to(tl.float32) + tl.load(R + offs).to(tl.float32)
    # write the residual sum back to R
    tl.store(R + offs, x.to(R.dtype.element_ty))

    rrms = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
    x = x * rrms[:, None]
    x = x * tl.load(W + n_off).to(tl.float32)[None, :]
    # write the normalized output back to X
    tl.store(X + offs, x.to(X.dtype.element_ty))


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.
    """
    logger.debug("GEMS_KUNLUNXIN FUSED_ADD_RMS_NORM")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = builtins.min(
        64 * 128, triton.next_power_of_2(N)
    )  # core_num * buffer_size_limit
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(x.device):
        if N > 64 * 128:
            need_mask = (N % BLOCK_SIZE) != 0
            fused_add_rmsnorm_kernel_tile[M,](
                x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE, need_mask
            )
        elif N <= MULTIROW_N and M >= MULTIROW_M:
            # Small N + many rows: the per-row kernel is launch-bound, so batch
            # TILE_M rows per program to cut the grid from M to cdiv(M, TILE_M).
            # Columns span exactly N (no padding) so the tile is one contiguous
            # block -> block DMA. N is a constexpr for the same reason.
            TILE_M = _pick_tile_m(M, N)
            grid = (triton.cdiv(M, TILE_M),)
            # The 2D multirow kernel does two masked stores (R then X); the
            # TritonXPUUnrollControl pass fails on that store pattern, so disable
            # it (same launch kwargs rms_norm's backward kernels use).
            fused_add_rmsnorm_multirow_kernel[grid](
                x,
                residual,
                weight,
                M,
                eps,
                TILE_M,
                N,
                isCloseUnrollControl=True,
                isCloseVectorization=True,
            )
        else:
            # Mid-size N (256 < N <= 4096) with enough rows is still badly
            # launch-bound at grid=(M,) (e.g. M=32768 N=512 fp16: 21.2 ms,
            # speedup 0.013). An UNMASKED multi-row tile fixes it (0.67 ms,
            # speedup 0.39) but needs M % TILE_M == 0, so only take it when such
            # a TILE_M exists; otherwise keep the correct per-row kernel.
            tile_m = (
                _pick_unmasked_tile_m(M, N, x.dtype)
                if N <= UNMASKED_N and M >= UNMASKED_M
                else None
            )
            if tile_m is not None:
                # The keep-live kernel is fastest for every dtype at the TILE_M
                # its budget allows (bf16's budget is capped so it still fits).
                fused_add_rmsnorm_unmasked_kernel[(M // tile_m,)](
                    x, residual, weight, eps, tile_m, N
                )
            else:
                need_mask = (N % BLOCK_SIZE) != 0
                fused_add_rmsnorm_kernel[M,](
                    x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE, need_mask
                )
    return x, residual
