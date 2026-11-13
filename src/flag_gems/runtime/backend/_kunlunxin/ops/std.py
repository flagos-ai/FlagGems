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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

# `tle.gpu` moves a tile with the XPU cluster DMA (GM -> LM -> registers) instead of
# per-lane pointer traffic.  Pointer reductions on this backend top out around
# 0.45x of aten (see the measured ladder in `ops/mean.py`); the DMA row-reduce in
# `ops/sum.py` lands at 0.8-1.05x.  std needs two accumulators (sum and sum of
# squares) in the same pass, so this file carries its own dual variant and reuses
# the dtype table and the row-tile geometry that sum already tuned.
try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    from .sum import _TLE_ACC_DTYPE, _TLE_AVAILABLE, _TLE_TL_DTYPE, _tle_row_geom

    _STD_TLE = _TLE_AVAILABLE
except ImportError:  # triton without the XPU tile-language extension
    _STD_TLE = False

logger = logging.getLogger(__name__)


_FLAT_CHUNK = 8192
# Flat reduce: reshape to [n // COLS, COLS] and row-reduce (same shape sum uses).
# Only worth it once the tensor is big enough that the DMA row-reduce's
# descriptor/launch setup is amortised: on 1M-element f32 (4 MB) the whole call
# is ~0.1 ms of host work against ~20 us of kernel, which makes the measured
# latency track CPU contention rather than the GPU (bench A/B at 1M: 0.29-0.79x,
# while at 16M/1G the same kernel is a stable 4-7x the pointer path).
_STD_TLE_FLAT_COLS = 8192
_STD_TLE_FLAT_MIN = 4 << 20


@libentry()
@triton.jit
def _std_flat_core_kernel(X, Tmp_sum, Tmp_sum_sq, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + off).to(tl.float32)
    tl.store(Tmp_sum + pid, tl.sum(x, axis=0))
    tl.store(Tmp_sum_sq + pid, tl.sum(x * x, axis=0))


@libentry()
@triton.jit
def _std_flat_tail_kernel(X, Tmp_sum, Tmp_sum_sq, start, NTAIL, TL: tl.constexpr):
    off = tl.arange(0, TL)
    x = tl.load(X + start + off, mask=off < NTAIL, other=0.0).to(tl.float32)
    tl.store(Tmp_sum, tl.sum(x, axis=0))
    tl.store(Tmp_sum_sq, tl.sum(x * x, axis=0))


@libentry()
@triton.jit
def _std_flat_tail_staged_kernel(X, Tmp_sum, Tmp_sum_sq, TL: tl.constexpr):
    off = tl.arange(0, TL)
    x = tl.load(X + off).to(tl.float32)
    tl.store(Tmp_sum, tl.sum(x, axis=0))
    tl.store(Tmp_sum_sq, tl.sum(x * x, axis=0))


@libentry()
@triton.jit
def _std_flat_tail_copy_kernel(Src, Dst, NTAIL, TL: tl.constexpr):
    off = tl.arange(0, TL)
    x = tl.load(Src + off, mask=off < NTAIL, other=0.0)
    tl.store(Dst + off, x)


@libentry()
@triton.jit
def _std_flat_merge_kernel(
    Tmp_sum, Tmp_sum_sq, Out, N, correction, nb, NLANES: tl.constexpr
):
    off = tl.arange(0, NLANES)
    mask = off < nb
    s = tl.load(Tmp_sum + off, mask=mask, other=0.0).to(tl.float32)
    sq = tl.load(Tmp_sum_sq + off, mask=mask, other=0.0).to(tl.float32)
    total_sum = tl.sum(s, axis=0)
    total_sum_sq = tl.sum(sq, axis=0)
    mean = total_sum / N
    var = (total_sum_sq / N) - (mean * mean)
    var = var * N / tl.maximum(N - correction, 1.0)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    tl.store(Out, std_dev.to(Out.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["correction"])
def _std_dim_row_kernel(
    Out, X, M, N, correction, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    rows_c = tl.where(rows < M, rows, M - 1)
    sum_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    sum_sq_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        a = tl.load(X + rows_c * N + cols, mask=mask, other=0.0).to(tl.float32)
        sum_acc += a
        sum_sq_acc += a * a
    total_sum = tl.sum(sum_acc, axis=1)[:, None]
    total_sum_sq = tl.sum(sum_sq_acc, axis=1)[:, None]
    mean = total_sum / N
    var = (total_sum_sq / N) - (mean * mean)
    var = var * N / tl.maximum(N - correction, 1.0)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    tl.store(Out + rows, std_dev.to(Out.dtype.element_ty), row_mask)


_TILE_BUDGET = 32768
_N_WIDE = 8192


@triton.jit(
    do_not_specialize=["N", "correction"],
    do_not_specialize_on_alignment=["a_desc", "c_desc"],
)
def _std_tle_row_kernel(
    a_desc,
    c_desc,
    N,
    correction,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """`c[m] = std(a[m, :])` over a [M, N] row-contiguous input."""
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))

    acc = tl.zeros([XBLOCK], ACC_DTYPE)
    acc_sq = tl.zeros([XBLOCK], ACC_DTYPE)
    for coff in tl.range(0, N, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(a_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        v = tl.load(a_ptrs).to(ACC_DTYPE)
        acc = acc + tl.sum(v, axis=1)
        acc_sq = acc_sq + tl.sum(v * v, axis=1)
    mean = acc / N
    var = acc_sq / N - mean * mean
    var = var * N / tl.maximum(N - correction, 1.0)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    tl.store(c_ptrs, std_dev.to(OUT_DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK], [row_off])


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["a_desc", "s_desc", "q_desc"],
)
def _std_tle_rowsin_kernel(
    a_desc,
    s_desc,
    q_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """`s[m] = sum(a[m, :])`, `q[m] = sum(a[m, :] ** 2)` for a [M, N] input.

    Same tile/loop as the row kernel above; the flat reduce is this kernel
    applied to a [n // COLS, COLS] view, and the two partial vectors are folded
    together by `_std_flat_merge_kernel`.
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    s_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)
    q_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    lane_ids = tl.arange(0, XBLOCK)
    s_ptrs = tle.gpu.local_ptr(s_lmem, (lane_ids,))
    q_ptrs = tle.gpu.local_ptr(q_lmem, (lane_ids,))

    acc = tl.zeros([XBLOCK], ACC_DTYPE)
    acc_sq = tl.zeros([XBLOCK], ACC_DTYPE)
    for coff in tl.range(0, N, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(a_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        v = tl.load(a_ptrs).to(ACC_DTYPE)
        acc = acc + tl.sum(v, axis=1)
        acc_sq = acc_sq + tl.sum(v * v, axis=1)
    tl.store(s_ptrs, acc.to(OUT_DTYPE))
    tl.store(q_ptrs, acc_sq.to(OUT_DTYPE))
    tle.gpu.copy(s_lmem, s_desc, [XBLOCK], [row_off])
    tle.gpu.copy(q_lmem, q_desc, [XBLOCK], [row_off])


def _std_tle_ready(x, out):
    return (
        _STD_TLE
        and x.dtype in _TLE_TL_DTYPE
        and out.dtype in _TLE_TL_DTYPE
        and x.is_contiguous()
    )


def _std_tle_geom(M, N, itemsize):
    """Tile for the dual-accumulator kernels: sum's ladder, half the tile.

    `_tle_row_geom` sizes a_lmem to the whole 4 KB/core budget because the sum
    kernel holds one loaded tile.  This kernel materializes the tile twice (the
    loaded values and their squares) and keeps two [XBLOCK] accumulators, which
    was measured over budget: fp16 (4096, 4096) fails to launch at sum's
    512x256 tile and compiles at 512x128.  Halving YBLOCK only lengthens the
    loop; the transfers stay 64-128 KB.
    """
    xblock, yblock, row_blocks = _tle_row_geom(M, N, itemsize)
    if yblock > 1:
        yblock = max(1, yblock // 2)
    return xblock, yblock, row_blocks


def _std_tle_row(x2d, out1d, M, N, correction):
    """std along a contiguous trailing axis; False when tle cannot express it."""
    if not _std_tle_ready(x2d, out1d) or N < 2 or M < 1:
        return False
    xblock, yblock, row_blocks = _std_tle_geom(M, N, x2d.element_size())
    with torch_device_fn.device(x2d.device):
        _std_tle_row_kernel[(row_blocks,)](
            TensorDescriptor.from_tensor(x2d, block_shape=[xblock, yblock]),
            TensorDescriptor.from_tensor(out1d, block_shape=[xblock]),
            N,
            correction,
            xblock,
            yblock,
            _TLE_TL_DTYPE[x2d.dtype],
            _TLE_ACC_DTYPE[x2d.dtype],
            _TLE_TL_DTYPE[out1d.dtype],
            N % yblock != 0,
        )
    return True


def _std_tle_row_partials(x2d, sum1d, sq1d, M, N):
    """sum / sum-of-squares along a contiguous trailing axis."""
    if not _std_tle_ready(x2d, sum1d) or N < 2 or M < 1:
        return False
    xblock, yblock, row_blocks = _std_tle_geom(M, N, x2d.element_size())
    with torch_device_fn.device(x2d.device):
        _std_tle_rowsin_kernel[(row_blocks,)](
            TensorDescriptor.from_tensor(x2d, block_shape=[xblock, yblock]),
            TensorDescriptor.from_tensor(sum1d, block_shape=[xblock]),
            TensorDescriptor.from_tensor(sq1d, block_shape=[xblock]),
            N,
            xblock,
            yblock,
            _TLE_TL_DTYPE[x2d.dtype],
            _TLE_ACC_DTYPE[x2d.dtype],
            _TLE_TL_DTYPE[sum1d.dtype],
            N % yblock != 0,
        )
    return True


def _std_block_n(N, dtype):
    """Reduction tile width, sized after the measured `mean_dim` ladder.

    A persisted [BLOCK_M, BLOCK_N] tile keeps both partial sums in registers, so
    its width is what feeds the convert/reduce pipe; the old
    `min(8192, next_pow2(N))` made the accumulator a 128 x 8192 f32 tile (4 MB,
    twice) and every launch spilled it (profiling: 163 GB/s on (4096, 4096)).
    """
    if N > _N_WIDE:
        return min(triton.next_power_of_2(N), 2048)
    if dtype == torch.float32:
        return min(triton.next_power_of_2(N), 512)
    return min(triton.next_power_of_2(N), 256)


def _std_block_m(M, dtype, block_n):
    cap = 128 if dtype != torch.float32 else 64
    return max(min(triton.next_power_of_2(M), cap, _TILE_BUDGET // block_n), 1)


def _std_dim_dispatch(out, x_contiguous, M, N, K, effective_correction):
    # x_contiguous is [M, N] with the reduced axis trailing (dim_compress).
    if K == 1:
        try:
            if _std_tle_row(x_contiguous.view(M, N), out, M, N, effective_correction):
                return
        except Exception as exc:  # noqa: BLE001 - any gap reuses the pointer path
            logger.debug("GEMS_KUNLUNXIN STD tle row unavailable (%s)", exc)
    BLOCK_N = _std_block_n(N, x_contiguous.dtype)
    BLOCK_M = _std_block_m(M, x_contiguous.dtype, BLOCK_N)
    with torch_device_fn.device(x_contiguous.device):
        grid = (triton.cdiv(M, BLOCK_M), 1, 1)
        _std_dim_row_kernel[grid](
            out,
            x_contiguous,
            M,
            N,
            effective_correction,
            BLOCK_M,
            BLOCK_N,
            buffer_size_limit=2048,
        )


def _launch_std_flat(x, out, N, correction):
    CHUNK = _FLAT_CHUNK
    if N >= _STD_TLE_FLAT_MIN:
        try:
            if _std_tle_flat(x, out, N, correction):
                return
        except Exception as exc:  # noqa: BLE001 - any gap reuses the pointer path
            logger.debug("GEMS_KUNLUNXIN STD tle flat unavailable (%s)", exc)
    nfull = N // CHUNK
    tail = N % CHUNK
    nb = nfull + (1 if tail else 0)
    tmp_sum = torch.empty((nb,), dtype=torch.float32, device=x.device)
    tmp_sum_sq = torch.empty((nb,), dtype=torch.float32, device=x.device)
    with torch_device_fn.device(x.device):
        if nfull:
            _std_flat_core_kernel[(nfull, 1, 1)](
                x, tmp_sum, tmp_sum_sq, CHUNK, buffer_size_limit=2048
            )
        if tail and tail <= 8192:
            _std_flat_tail_kernel[(1, 1, 1)](
                x,
                tmp_sum[nfull : nfull + 1],
                tmp_sum_sq[nfull : nfull + 1],
                nfull * CHUNK,
                tail,
                triton.next_power_of_2(tail),
            )
        elif tail:
            TL = triton.next_power_of_2(tail)
            staged = torch.zeros((TL,), dtype=x.dtype, device=x.device)
            _std_flat_tail_copy_kernel[(1, 1, 1)](
                x[nfull * CHUNK : N], staged, tail, TL
            )
            _std_flat_tail_staged_kernel[(1, 1, 1)](
                staged,
                tmp_sum[nfull : nfull + 1],
                tmp_sum_sq[nfull : nfull + 1],
                TL,
                buffer_size_limit=2048,
            )
        _std_flat_merge_kernel[(1, 1, 1)](
            tmp_sum,
            tmp_sum_sq,
            out,
            N,
            correction,
            nb,
            triton.next_power_of_2(nb),
        )


def _std_tle_flat(x, out, N, correction):
    """Whole-tensor reduce: reshape to [rows, cols] and DMA-row-reduce it.

    One launch reads the tensor once and leaves `rows` (sum, sum-of-squares)
    partial pairs; `_std_flat_merge_kernel` folds those.  Row count is pinned at
    1024 so the grid still covers the machine on medium sizes (and, being a
    power of two, leaves no ragged last program), while wide rows (COLS=8192)
    keep the transfers large on huge ones -- the same trade `_tle_sum_flat`
    makes with its 4096-column stages.
    """
    if not _STD_TLE or x.dtype not in _TLE_TL_DTYPE:
        return False
    if not x.is_contiguous():
        return False
    cols = min(_STD_TLE_FLAT_COLS, max(1, N // 1024))
    rows = N // cols
    if rows < 1:
        return False
    tail = N - rows * cols
    nb = rows + (1 if tail else 0)
    tmp_sum = torch.empty((nb,), dtype=torch.float32, device=x.device)
    tmp_sum_sq = torch.empty((nb,), dtype=torch.float32, device=x.device)
    flat = x.view(-1)
    if not _std_tle_row_partials(
        flat[: rows * cols].view(rows, cols),
        tmp_sum[:rows],
        tmp_sum_sq[:rows],
        rows,
        cols,
    ):
        return False
    with torch_device_fn.device(x.device):
        if tail:
            _std_flat_tail_kernel[(1, 1, 1)](
                flat,
                tmp_sum[rows : rows + 1],
                tmp_sum_sq[rows : rows + 1],
                rows * cols,
                tail,
                triton.next_power_of_2(tail),
            )
        _std_flat_merge_kernel[(1, 1, 1)](
            tmp_sum,
            tmp_sum_sq,
            out,
            N,
            correction,
            nb,
            triton.next_power_of_2(nb),
        )
    return True


def std(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN STD")
    effective_correction = 1.0 if correction is None else float(correction)
    original_shape = x.shape
    input_ndim = x.ndim

    if dim is None:
        N = x.numel()
        if N == 0 or N - effective_correction <= 0:
            return torch.full([], float("nan"), device=x.device, dtype=x.dtype)
        if N == 1 and effective_correction == 0.0:
            out = torch.zeros([], device=x.device, dtype=x.dtype)
            return out.view([1] * input_ndim) if keepdim else out

        out = torch.empty([], device=x.device, dtype=x.dtype)
        _launch_std_flat(x.contiguous(), out, N, effective_correction)
        return out.view([1] * input_ndim) if keepdim else out

    if isinstance(dim, int):
        dim_list = [dim]
    else:
        dim_list = list(dim)
    dim_list_normalized = [d % input_ndim for d in dim_list]

    # Route EVERY dim reduction (single-dim AND multi-dim) through dim_compress so
    # the reduced dims land on the trailing axis => it is always a contiguous
    # (M, N) inner reduction (K == 1). We only ever launch the @libentry-cached
    # _std_dim_kernel_inner. This (a) avoids the giant 2D tile + heuristic-supplied
    # launch param IR explosion of the old _std_fused_dim_kernel path
    # (ir-std-dev5.log = 7.7M lines) and (b) avoids the non_inner (K>1) softmax
    # kernel, which was numerically wrong on XPU (std ~sqrt(K)x too small).
    x_view = dim_compress(x, dim_list_normalized)
    N = 1
    for d in dim_list_normalized:
        N *= original_shape[d]
    M = x.numel() // N

    output_shape_kept = list(original_shape)
    for d in dim_list_normalized:
        output_shape_kept[d] = 1

    if M * N > 0 and (N - effective_correction <= 0):
        final_shape = [
            s for i, s in enumerate(original_shape) if i not in dim_list_normalized
        ]
        return torch.full(
            final_shape if not keepdim else output_shape_kept,
            float("nan"),
            device=x.device,
            dtype=x.dtype,
        )
    if N == 1 and effective_correction == 0.0:
        final_shape = [
            s for i, s in enumerate(original_shape) if i not in dim_list_normalized
        ]
        return torch.zeros(
            final_shape if not keepdim else output_shape_kept,
            device=x.device,
            dtype=x.dtype,
        )

    out = torch.empty(output_shape_kept, device=x.device, dtype=x.dtype)
    if M * N == 0:
        return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out

    _std_dim_dispatch(out.view(-1), x_view, M, N, 1, effective_correction)
    return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out
