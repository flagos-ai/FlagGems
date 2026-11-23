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
from functools import lru_cache, reduce

import torch
import triton
import triton.language as tl
from torch._prims_common import is_boolean_dtype, is_integer_dtype

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _num_sms() -> int:
    # torch is a CPU build on GCU, so torch.cuda.* raises "Torch not compiled
    # with CUDA enabled"; query the active backend instead.
    try:
        return torch_device_fn.get_device_properties(
            torch_device_fn.current_device()
        ).multi_processor_count
    except Exception:
        return 4


def _is_integral(dtype):
    return is_integer_dtype(dtype) or is_boolean_dtype(dtype)


def _nansum_out_dtype(inp_dtype):
    return torch.int64 if _is_integral(inp_dtype) else inp_dtype


@tl.constexpr
def _nansum_acc_type(inp_dtype: tl.dtype) -> tl.dtype:
    if inp_dtype.is_int():
        return tl.int64
    if inp_dtype.is_fp64():
        return tl.float64
    return tl.float32


@tl.constexpr
def _nansum_tile_acc_type_or(
    inp_dtype: tl.dtype, acc_mode: tl.constexpr
) -> tl.dtype:
    """Tile accumulator selected by ``acc_mode`` (0 default, 1 int32, 2 float32).

    Modes 1 and 2 are only requested by the host when the narrow type is exact
    for that input dtype and reduction length (see ``_nansum_heur_acc_mode``).
    """
    if acc_mode == 2:
        return tl.float32
    if acc_mode == 1:
        return tl.int32
    return _nansum_acc_type(inp_dtype)


@tl.constexpr
def _nansum_inner_acc_type(
    inp_dtype: tl.dtype, reduce_len: tl.constexpr
) -> tl.dtype:
    """Accumulator for the last-dim (inner) kernel.

    8-bit integers narrow to int32 in :func:`_nansum_tile_acc_type`, but the
    lane-wise byte -> word integer widening is what makes that path ~8x slower
    than the 16-bit one. A float32 tile is exact for 8-bit inputs (every partial
    sum is an integer with ``|sum| <= 255 * 32768 < 2**24``) and rides the much
    faster 8-bit -> float widening instead.
    """
    base: tl.constexpr = _nansum_tile_acc_type(inp_dtype, reduce_len)
    # Only 8-bit inputs: a 16-bit tile is *not* exactly representable in float32.
    if base == tl.int32 and inp_dtype.primitive_bitwidth == 8:
        return tl.float32
    return base


@tl.constexpr
def _nansum_tile_acc_type(inp_dtype: tl.dtype, reduce_len: tl.constexpr) -> tl.dtype:
    # torch widens integer sums to int64, but a 64-bit accumulator vector costs
    # 8 bytes per lane and spills the tile as soon as it gets wide. Narrow ints
    # cannot overflow a 32-bit tile accumulator (see bounds below), so tile-local
    # math stays in int32 and only the per-tile scalar is widened on store.
    if inp_dtype.is_int():
        if inp_dtype.primitive_bitwidth == 8 and reduce_len <= 32768:
            # |8-bit| * 32768 <= 4.2e6
            return tl.int32
        if inp_dtype.primitive_bitwidth == 16 and reduce_len <= 16384:
            # |16-bit| * 16384 <= 2.15e9 (uint16 stays <= INT32_MAX)
            return tl.int32
        return tl.int64
    if inp_dtype.is_fp64():
        return tl.float64
    return tl.float32


@libentry()
@triton.jit
def _nan_to_zero_kernel(
    inp,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    val = tl.load(inp + offsets, mask=mask, other=0.0)
    val = tl.where(val != val, 0.0, val)
    tl.store(out + offsets, val, mask=mask)


@libentry()
@triton.jit
def nansum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    cdtype: tl.constexpr = _nansum_tile_acc_type(inp.dtype.element_ty, BLOCK_SIZE)

    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    x = tl.load(inp_ptrs, mask=mask, other=0.0).to(cdtype)
    if tl.constexpr(cdtype.is_floating()):
        x = tl.where(x != x, 0.0, x)

    sum_val = tl.sum(x, axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("nansum_final_reduce"),
    key=["mid_size"],
)
@triton.jit
def nansum_kernel_2(
    mid,
    out,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    cdtype: tl.constexpr = _nansum_acc_type(mid.dtype.element_ty)

    _sum = tl.zeros((), dtype=cdtype)

    for start in range(0, mid_size, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < mid_size
        val = tl.load(mid + idx, mask=mask, other=0.0).to(cdtype)
        if tl.constexpr(cdtype.is_floating()):
            val = tl.where(val != val, 0.0, val)
        _sum += tl.sum(val, axis=0)

    tl.store(out, _sum)


def _nansum_global(inp, *, dtype=None):
    if dtype is None:
        dtype = _nansum_out_dtype(inp.dtype)
        if inp.dtype == torch.bool:
            inp = inp.to(torch.int64)

    if inp.numel() == 0:
        return torch.tensor(0, dtype=dtype, device=inp.device)

    if not inp.is_contiguous():
        inp = inp.contiguous()
    M = inp.numel()

    if M <= 32768:
        out = torch.zeros([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            nansum_kernel_2[(1,)](inp, out, M)
        return out

    if _is_integral(inp.dtype):
        compute_dtype = torch.int64
    elif inp.dtype == torch.float64 or dtype == torch.float64:
        compute_dtype = torch.float64
    else:
        compute_dtype = torch.float32
    # Size the tile from the amount of work *per program*, not from sqrt(M):
    # the per-CTA prologue and the reduction tree dominate while a tile holds
    # only a few thousand elements, so the widest legal tile wins as long as
    # enough programs remain to fill the device. Measured on 16.7M elements:
    # BLOCK=4096 -> 281us, BLOCK=32768 -> 51us (fp16 and fp32 alike, i.e. the
    # path is ALU/prologue bound, not bandwidth bound).
    target_programs = max(_num_sms() * 8, 1)
    block_size = triton.next_power_of_2(triton.cdiv(M, target_programs))
    # Grid.x is capped at 65535, so the tile must cover M / 65535 elements; the
    # 32K cap keeps a full int32 tile accumulator inside the register file.
    block_size = min(max(block_size, 4096), 32768)
    block_size = max(block_size, triton.next_power_of_2(triton.cdiv(M, _GRID_X_LIMIT)))
    mid_size = triton.cdiv(M, block_size)
    mid = torch.empty(mid_size, dtype=compute_dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        nansum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        out = torch.zeros([], dtype=dtype, device=inp.device)
        nansum_kernel_2[(1,)](mid, out, mid_size)

    return out


def nansum(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME NANSUM")
    if dim is None:
        result = _nansum_global(inp, dtype=dtype)
        return result.reshape([1] * inp.ndim) if keepdim else result
    return nansum_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)


def nansum_out(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    logger.debug("GEMS_ENFLAME NANSUM_OUT")
    result = nansum(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    out.copy_(result)
    return out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("nansum_dim"),
    key=["M", "N"],
)
@triton.jit
def nansum_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cdtype: tl.constexpr = _nansum_acc_type(inp.dtype.element_ty)

    pid = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        val = tl.load(inp + cols, mask, other=0.0).to(cdtype)
        if tl.constexpr(cdtype.is_floating()):
            val = tl.where(val != val, 0.0, val)
        _sum += val

    result = tl.sum(_sum, axis=1)[:, None]
    tl.store(out, result, row_mask)


_INT32_MAX = 2**31 - 1


def _nansum_tile_k_raw(args):
    MAX_TILE_K = 8192
    if args["input_ptr"].element_size() == 1:
        MAX_TILE_K = 256
    NUM_SMS = _num_sms()

    if args["M"] <= 4:
        MIN_TILE_K = 8
        if args["M"] <= 1 and args["K"] > 32768 and args["N"] > 256:
            if args["input_ptr"].dtype == torch.float32:
                TARGET_WAVES = 8
            else:
                # Prefer heavier blocks (TARGET_WAVES=4) unless iterations
                # per block would exceed 16, then use 8 for more parallelism.
                test_tk = MIN_TILE_K
                ub = min(args["K"], MAX_TILE_K)
                while test_tk <= ub:
                    n_blocks = args["M"] * triton.cdiv(args["K"], test_tk)
                    if n_blocks / NUM_SMS >= 4:
                        if test_tk * 2 <= ub:
                            test_tk *= 2
                        else:
                            break
                    else:
                        break
                test_tn = triton.cdiv(32768, test_tk)
                test_tn = min(test_tn, args["N"])
                test_tn = triton.next_power_of_2(test_tn + 1) // 2
                test_iters = triton.cdiv(args["N"], test_tn)
                TARGET_WAVES = 8 if test_iters > 16 else 4
        else:
            TARGET_WAVES = 4

        tile_k = MIN_TILE_K
        upper_bound = min(args["K"], MAX_TILE_K)
        best_tile_k = tile_k
        while tile_k <= upper_bound:
            num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
            num_waves = num_blocks / NUM_SMS
            if num_waves >= TARGET_WAVES:
                best_tile_k = tile_k
                if tile_k * 2 <= upper_bound:
                    tile_k *= 2
                else:
                    break
            else:
                break
        return best_tile_k
    elif args["M"] <= 64:
        TARGET_WAVES = 2
        tile_k = 1
        upper_bound = min(args["K"], MAX_TILE_K)
        prev_tile_k = tile_k
        while tile_k <= upper_bound:
            num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
            num_waves = num_blocks / NUM_SMS
            if num_waves > TARGET_WAVES:
                prev_tile_k = tile_k
                if tile_k * 2 <= upper_bound:
                    tile_k *= 2
                else:
                    break
            else:
                break
        # The k axis is the contiguous one, so a narrow TILE_K reduces every load
        # to a few contiguous elements per row. Keep at least 64 lanes of it
        # unless K itself is smaller (measured: (64, 512, 512) over dim 1, fp16
        # 106us at TILE_K=8).
        return max(prev_tile_k, min(64, triton.next_power_of_2(args["K"])))
    else:
        tile_k = 1
        upper_bound = min(args["K"], MAX_TILE_K)
        while tile_k <= upper_bound:
            num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
            num_waves = num_blocks / NUM_SMS
            if (num_waves > 1) and (tile_k * 2 <= upper_bound):
                tile_k *= 2
            else:
                break
        return tile_k


# gcu400 caps grid.y at 255, and the k axis is launched on grid.y.
_GRID_Y_LIMIT = 255
# gcu400 caps grid.x at 65535, and the row axis is launched on grid.x.
_GRID_X_LIMIT = 65535
# Element budget of one last-dim inner tile (TILE_M * TILE_N). Bigger tiles
# amortize the per-CTA prologue, which dominates while the row count is large;
# the cap keeps a 4-byte accumulator tile (plus the dsm copy the backend makes)
# below the 917504-byte dsm budget.
_INNER_TILE_BUDGET = 65536
# Width of one int64 reduction stage; wider int64 reductions fail to link
# ("relocation ... tops_redsum_v2_si64"), so 64-bit sums are reduced in two steps.
_INT64_REDUCE_CHUNK = tl.constexpr(64)


def _nansum_heur_tile_k(args):
    tile_k = _nansum_tile_k_raw(args)
    # Grow the tile until the k axis fits grid.y; TILE_N is derived from
    # TILE_K afterwards, so the per-CTA budget stays the same.
    min_tile_k = triton.cdiv(args["K"], _GRID_Y_LIMIT)
    if tile_k < min_tile_k:
        tile_k = triton.next_power_of_2(min_tile_k)
    return tile_k


def _nansum_heur_tile_n_non_inner(args):
    tile_budget = 32768 if args["M"] <= 1 else 8192
    tile_n = triton.cdiv(tile_budget, args["TILE_K"])
    # Never tile wider than the reduction axis: the surplus lanes are masked off
    # on every load and store, so a 2x-too-wide tile halves the useful work
    # (measured: (64, 512, 512) over dim 1, fp16 106us -> 61us).
    tile_n = min(tile_n, triton.next_power_of_2(args["N"]))
    if args["M"] <= 1:
        tile_n = min(tile_n, args["N"])
        return triton.next_power_of_2(tile_n + 1) // 2
    return tile_n


def _nansum_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def _nansum_heur_acc_mode(args):
    """Pick the narrowest *exact* tile accumulator for the non-inner kernel.

    The accumulator is ``TILE_N * TILE_K`` elements wide, so int64 doubles the
    dsm footprint and spills: measured on 16.7M int8 elements (64, 512, 512)
    reduced over dim 0, int64 -> 834us, int32 -> 70us. 1-byte inputs do even
    better in float32 (values stay below 2**24, so every partial sum is exact)
    because the byte -> float widening is several times faster than byte ->
    word; that alone takes the same case to ~55us and the int8 last-dim
    reduction from 8.1ms to 1.6ms.
    """
    dtype = args["input_ptr"].dtype
    if dtype.is_floating_point or dtype.itemsize > 2:
        return 0
    reduce_len = args["N"]
    if dtype.itemsize == 1:
        if reduce_len <= 32768:
            return 2
        if reduce_len <= 8_000_000:  # 255 * 8e6 < INT32_MAX
            return 1
        return 0
    return 1 if reduce_len <= 16384 else 0


def _nansum_heur_num_warps_non_inner(args):
    tile_size = args["TILE_N"] * args["TILE_K"]
    if args.get("ONE_TILE_PER_CTA") and args["M"] <= 1:
        return 4
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def _nansum_heur_tile_n_inner(args):
    if args["N"] <= (32 * 1024):
        return triton.next_power_of_2(args["N"])
    else:
        return 4096


@tl.constexpr
def _nansum_needs_split_int64_reduce(
    cdtype: tl.dtype, tile_n: tl.constexpr
) -> tl.constexpr:
    return cdtype == tl.int64 and tile_n > _INT64_REDUCE_CHUNK


def _nansum_acc_is_64bit(dtype):
    if dtype.is_floating_point:
        return dtype == torch.float64
    # 32/64-bit integers accumulate in int64 (see _nansum_tile_acc_type)
    return dtype.itemsize > 2


def _nansum_heur_tile_m_inner(args):
    NUM_SMS = _num_sms()

    if args["M"] <= NUM_SMS * 16:
        tile_m = 1
    else:
        TARGET_GRID = NUM_SMS * 6
        tile_m = triton.next_power_of_2(max(args["M"] // TARGET_GRID, 1))
        # The dsm budget is spent by the accumulator tile, not by the input, so
        # size the tile from the accumulator width: int64 for 32/64-bit ints.
        tile_budget = _INNER_TILE_BUDGET
        if _nansum_acc_is_64bit(args["input_ptr"].dtype):
            tile_budget //= 2
        max_tile_m = max(tile_budget // args["TILE_N"], 1)
        tile_m = min(tile_m, max_tile_m, 64)

    # grid.x is capped at 65535 on gcu400, so one row per CTA overflows the x
    # axis for tall reductions; grow TILE_M just enough to stay legal.
    rows_per_grid = triton.cdiv(args["M"], _GRID_X_LIMIT)
    if rows_per_grid > tile_m:
        tile_m = triton.next_power_of_2(rows_per_grid)
    return tile_m


def _nansum_inner_grid(M, TILE_M):
    blocks = triton.cdiv(M, TILE_M)
    if blocks <= _GRID_X_LIMIT:
        return (blocks, 1, 1)
    return (_GRID_X_LIMIT, triton.cdiv(blocks, _GRID_X_LIMIT), 1)


def _nansum_heur_num_warps_inner(args):
    tile_size = args["TILE_M"] * args["TILE_N"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


@libentry()
@triton.heuristics(
    values=dict(
        TILE_K=_nansum_heur_tile_k,
        TILE_N=_nansum_heur_tile_n_non_inner,
        ONE_TILE_PER_CTA=_nansum_heur_one_tile_per_cta,
        ACC_MODE=_nansum_heur_acc_mode,
        num_warps=_nansum_heur_num_warps_non_inner,
    ),
)
@triton.jit
def nansum_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
    ACC_MODE: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
):
    cdtype: tl.constexpr = _nansum_tile_acc_type_or(
        input_ptr.dtype.element_ty, ACC_MODE
    )

    # Element offsets reach numel, so they pass INT32_MAX for tensors above 2**31
    # elements. Widen the index math only for those: 64-bit address arithmetic
    # costs a few percent on the large-tile path and is wasted below that.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    if WIDE_INDEX:
        pid_m = pid_m.to(tl.int64)
        pid_k = pid_k.to(tl.int64)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        if WIDE_INDEX:
            n_offsets = n_offsets.to(tl.int64)
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=0.0).to(cdtype)
        if tl.constexpr(cdtype.is_floating()):
            inp = tl.where(inp != inp, 0.0, inp)
        out = tl.sum(inp, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)
    else:
        _sum = tl.zeros([TILE_N, TILE_K], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            if WIDE_INDEX:
                n_offsets = n_offsets.to(tl.int64)
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0.0).to(cdtype)
            if tl.constexpr(cdtype.is_floating()):
                inp = tl.where(inp != inp, 0.0, inp)
            _sum += inp
        out = tl.sum(_sum, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(
    values=dict(
        TILE_N=_nansum_heur_tile_n_inner,
        TILE_M=_nansum_heur_tile_m_inner,
        num_warps=_nansum_heur_num_warps_inner,
    ),
)
@triton.jit
def nansum_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    cdtype: tl.constexpr = _nansum_inner_acc_type(input_ptr.dtype.element_ty, N)

    # The M axis is flattened over a 2D launch because grid.x is capped at
    # 65535 on gcu400; keeping TILE_M small is what keeps the accumulator tile
    # in registers for wide rows.
    pid = (
        ext.program_id(0) + ext.program_id(1) * ext.num_programs(0)
    ).to(tl.int64)
    m_start = pid * TILE_M + tl.arange(0, TILE_M)[:, None]
    m_mask = m_start < M

    _sum = tl.zeros([TILE_M, TILE_N], dtype=cdtype)
    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)[None, :]
        inp_offsets = m_start * N + n_offsets
        mask = m_mask & (n_offsets < N)
        inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0.0).to(cdtype)
        if tl.constexpr(cdtype.is_floating()):
            inp = tl.where(inp != inp, 0.0, inp)
        _sum += inp

    if _nansum_needs_split_int64_reduce(cdtype, TILE_N):
        # Split the int64 row sum: the backend only links reductions of up to
        # _INT64_REDUCE_CHUNK lanes for 64-bit integers.
        chunked = tl.reshape(
            _sum, (TILE_M, TILE_N // _INT64_REDUCE_CHUNK, _INT64_REDUCE_CHUNK)
        )
        result = tl.sum(tl.sum(chunked, axis=2), axis=1)[:, None]
    else:
        result = tl.sum(_sum, axis=1)[:, None]
    tl.store(output_ptr + m_start, result, mask=m_mask)


def _normalize_dims(dim, ndim):
    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        return []
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        dim = [dim]
    span = max(ndim, 1)
    for d in dim:
        if d < -span or d >= span:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-span}, {span - 1}], but got {d})"
            )
    dims = [d + span if d < 0 else d for d in dim]
    seen = set()
    for d in dims:
        if d in seen:
            raise RuntimeError(f"dim {d} appears multiple times in the list of dims")
        seen.add(d)
    return sorted(dims, reverse=True)


def _squeeze_dims(result, dims):
    for d in sorted(dims, reverse=True):
        result = result.squeeze(dim=d)
    return result


def nansum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = _nansum_out_dtype(inp.dtype)
        if inp.dtype == torch.bool:
            inp = inp.to(torch.int64)

    dims = _normalize_dims(dim, inp.ndim)

    if inp.ndim == 0:
        return _nansum_global(inp, dtype=dtype)

    # dim=[] -> reduce all
    if len(dims) == 0:
        result = _nansum_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    # empty tensor
    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if keepdim:
            for d in dims:
                out_shape[d] = 1
        else:
            for d in dims:
                out_shape.pop(d)
        return torch.zeros(out_shape, dtype=dtype, device=inp.device)

    # full-dimensional reduction -> delegate to global
    if len(dims) == inp.ndim:
        result = _nansum_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    shape = list(inp.shape)

    # single-dim path
    if len(dims) == 1:
        dim = dims[0]
        if not inp.is_contiguous():
            inp = inp.contiguous()
        N = shape[dim]
        M = reduce(lambda x, y: x * y, shape[:dim], 1)
        K = inp.numel() // (M * N)

        out_shape = list(shape)
        out_shape[dim] = 1

        if N <= 1:
            if _is_integral(inp.dtype):
                # integers carry no NaN: the widening cast is the whole job
                out = inp.to(dtype)
            else:
                NANZERO_BLOCK = 4096
                grid = (triton.cdiv(inp.numel(), NANZERO_BLOCK),)
                out = torch.empty_like(inp, dtype=dtype)
                with torch_device_fn.device(inp.device):
                    _nan_to_zero_kernel[grid](
                        inp, out, inp.numel(), BLOCK_SIZE=NANZERO_BLOCK
                    )
            out = out.reshape(out_shape)
            return out if keepdim else _squeeze_dims(out, dims)

        out = torch.empty(out_shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                nansum_dim_kernel_non_inner[grid](
                    out, inp, M, N, K, WIDE_INDEX=inp.numel() > _INT32_MAX
                )
            else:
                grid = lambda meta: _nansum_inner_grid(M, meta["TILE_M"])
                nansum_dim_kernel_inner[grid](out, inp, M, N)

        return out if keepdim else _squeeze_dims(out, dims)

    # multi-dim path
    inp = dim_compress(inp, dims)
    N = 1
    for d in dims:
        N *= shape[d]
        shape[d] = 1
    M = inp.numel() // N

    if N <= 1:
        if _is_integral(inp.dtype):
            out = inp.to(dtype)
        else:
            NANZERO_BLOCK = 4096
            grid = (triton.cdiv(inp.numel(), NANZERO_BLOCK),)
            out = torch.empty_like(inp, dtype=dtype)
            with torch_device_fn.device(inp.device):
                _nan_to_zero_kernel[grid](
                    inp, out, inp.numel(), BLOCK_SIZE=NANZERO_BLOCK
                )
        out = out.reshape(shape)
        return out if keepdim else _squeeze_dims(out, dims)

    out = torch.empty(M, dtype=dtype, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        nansum_dim_kernel[grid](inp, out, M, N)

    out = out.reshape(shape)
    return out if keepdim else _squeeze_dims(out, dims)
