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
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)

# GCU400 (gcu400 family): a program keeps at most an accumulator tile plus one
# loaded tile live in DSM.  Staying within this budget keeps two programs
# resident per SP.  Measured per-program fixed cost (~68ns/CTA) makes few large
# programs far cheaper than many small ones for this streaming reduction.
AMIN_DSM_BUDGET = 384 * 1024

# Strided middle-dim reduction: launch at most this many programs and let each
# one grid-stride over whole (i, j-tile) slabs.  Measured on gcu400 for
# (1024,1024,512): 48-96 programs beat a full 1024-program launch by ~7%
# (few large programs amortize the per-CTA cost better).
AMIN_STRIDED_GRID = 96

# The fused in-place strided kernel only wins while each program owns exactly
# one slab (its broadcast store then overlaps other programs' reads); with a
# long grid-stride slab list the store serializes behind the reads.
AMIN_STRIDED_IP_MAX_WORK = 128

# The broadcast-combine kernel re-reduces `mid` in every program, so it is only
# used while mid stays tiny (redundant reads are then negligible).
AMIN_BCAST_MID_MAX = 256

# aten::amin/amin_ are implemented for every float and integral dtype.  The
# kernels below are dtype-generic (the identity element comes from
# get_dtype_max, and the running min is a plain minimum), so the guard only
# rejects the dtypes a min-reduction cannot express (bool is not ordered and
# complex has no total order).
AMIN_SUPPORTED_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@libentry()
@triton.jit
def amin_kernel_1(
    inp,
    mid,
    M,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Phase 1 of the full reduction: each program folds a CHUNK-sized span
    # into a running [BLOCK] minimum (elementwise only, one tree at the end)
    # and stores one scalar partial into mid.
    pid = tle.program_id(0)
    base = pid * CHUNK
    offs = tl.arange(0, BLOCK)
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    acc = tl.full([BLOCK], value=max_value, dtype=dtype)
    for off in range(0, CHUNK, BLOCK):
        ptrs = inp + base + off + offs
        if EVEN:
            inp_val = tl.load(ptrs)
        else:
            mask = base + off + offs < M
            inp_val = tl.load(ptrs, mask=mask, other=max_value)
        # tl.minimum promotes bf16 operands to fp32 (a convert per element,
        # measured ~10% slower); the compare/select form keeps the running
        # min in bf16.  Other dtypes stay on the single-op minimum.  Both
        # forms take the non-NaN operand, so behaviour is identical.
        if dtype is tl.bfloat16:
            acc = tl.where(inp_val < acc, inp_val, acc)
        else:
            acc = tl.minimum(acc, inp_val)
    amin_val = tl.min(acc)
    tl.store(mid + pid, amin_val)


@libentry()
@triton.jit
def amin_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    amin_val = tl.min(mid_val)
    tl.store(out, amin_val)


@libentry()
@triton.jit
def amin_kernel_2_bcast(
    mid, inp, M, mid_size, BLOCK: tl.constexpr, BLOCK_MID: tl.constexpr
):
    # Broadcast-combine for the in-place full reduction: every program
    # re-reduces the (tiny) partials in mid and writes one BLOCK-sized chunk of
    # the broadcast result straight into inp.  Replaces the separate combine
    # + aten copy_ pair with a single kernel; only used while mid_size is
    # small so the redundant mid reads stay negligible.
    pid = tle.program_id(0)
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    amin_val = tl.min(mid_val)
    out_offs = pid * BLOCK + tl.arange(0, BLOCK)
    out_mask = out_offs < M
    tl.store(inp + out_offs, amin_val, mask=out_mask)


@libentry()
@triton.jit
def amin_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # Row reduction over a (M, N) contiguous view: rows spaced by N, row
    # elements contiguous.  EVEN_M/EVEN_N are compile-time specializations
    # that drop all masks when the grid divides M and N exactly.  The outer
    # group loop bounds the launch grid: huge-M/small-N inputs would need
    # more than 65535 programs (grid.x hardware limit), so the host launches
    # at most 65535 programs and each one strides over its share of groups.
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    pid = tle.program_id(0)
    nprog = tl.num_programs(0)
    num_groups = tl.cdiv(M, BLOCK_M)

    for g in range(pid, num_groups, nprog):
        rows = g * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_ptrs = inp + rows * N
        out_ptrs = out + rows
        row_mask = rows < M
        _all = tl.full([BLOCK_M, BLOCK_N], value=max_value, dtype=dtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            if EVEN_M and EVEN_N:
                a = tl.load(row_ptrs + cols)
            else:
                col_mask = cols < N
                mask = row_mask and col_mask
                a = tl.load(row_ptrs + cols, mask, other=max_value)
            # dtype-conditional min (see amin_kernel_1): bf16 avoids the
            # fp32 promote with compare/select, others use single-op minimum.
            if dtype is tl.bfloat16:
                _all = tl.where(a < _all, a, _all)
            else:
                _all = tl.minimum(_all, a)
        all = tl.min(_all, axis=1)[:, None]
        if EVEN_M:
            tl.store(out_ptrs, all)
        else:
            tl.store(out_ptrs, all, row_mask)


@libentry()
@triton.jit
def amin_kernel_rows_ip(
    inp,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # In-place row reduction for amin_: same reduce as amin_kernel, but the
    # per-row minima are broadcast straight back into the rows this program
    # owns instead of being parked in a separate (M, 1) tensor.  Rows are
    # partitioned across programs, so the broadcast stores never race with
    # another program's loads, and the write phase overlaps other programs'
    # reads (measured (4096,4096) fp16: 32.6us fused vs 23.6+15.3 split).
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    pid = tle.program_id(0)
    nprog = tl.num_programs(0)
    num_groups = tl.cdiv(M, BLOCK_M)

    for g in range(pid, num_groups, nprog):
        rows = g * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_ptrs = inp + rows * N
        row_mask = rows < M
        _all = tl.full([BLOCK_M, BLOCK_N], value=max_value, dtype=dtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            if EVEN_M and EVEN_N:
                a = tl.load(row_ptrs + cols)
            else:
                col_mask = cols < N
                mask = row_mask and col_mask
                a = tl.load(row_ptrs + cols, mask, other=max_value)
            # dtype-conditional min (see amin_kernel_1).
            if dtype is tl.bfloat16:
                _all = tl.where(a < _all, a, _all)
            else:
                _all = tl.minimum(_all, a)
        all = tl.min(_all, axis=1)  # (BLOCK_M,)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            val = tl.broadcast_to(all[:, None], (BLOCK_M, BLOCK_N))
            if EVEN_M and EVEN_N:
                tl.store(row_ptrs + cols, val)
            else:
                col_mask = cols < N
                mask = row_mask and col_mask
                tl.store(row_ptrs + cols, val, mask=mask)


@libentry()
@triton.jit
def amin_kernel_strided(
    inp,
    out,
    K,
    J,
    total,
    num_jt,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_J: tl.constexpr,
):
    # Reduction of a middle dim on a contiguous tensor, viewed as (I, K, J):
    # the reduced span K is strided by J while each J span stays contiguous.
    # Each work item is one (i, j-tile) slab; the host launches at most
    # AMIN_STRIDED_GRID programs and every program grid-strides over whole
    # slabs (few large programs beat a full one-slab-per-program launch,
    # measured ~7% on (1024,1024,512)).  No dim_compress copy of the input.
    pid = tle.program_id(0)
    nprog = tl.num_programs(0)
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)

    for w in range(pid, total, nprog):
        pid_i = w // num_jt
        pid_j = w % num_jt
        base = inp + pid_i * (K * J)
        cols = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)[None, :]
        col_mask = cols < J
        _all = tl.full([BLOCK_K, BLOCK_J], value=max_value, dtype=dtype)
        for off in range(0, K, BLOCK_K):
            rows = off + tl.arange(0, BLOCK_K)[:, None]
            ptrs = base + rows * J + cols
            if EVEN_K and EVEN_J:
                a = tl.load(ptrs)
            else:
                mask = rows < K and col_mask
                a = tl.load(ptrs, mask, other=max_value)
            # dtype-conditional min (see amin_kernel_1): bf16 avoids the
            # fp32 promote with compare/select, others use single-op minimum.
            if dtype is tl.bfloat16:
                _all = tl.where(a < _all, a, _all)
            else:
                _all = tl.minimum(_all, a)
        all = tl.min(_all, axis=0)
        out_ptrs = out + pid_i * J + cols
        if EVEN_J:
            tl.store(out_ptrs, all)
        else:
            tl.store(out_ptrs, all, mask=col_mask)


@libentry()
@triton.jit
def amin_kernel_strided_ip(
    inp,
    K,
    J,
    num_jt,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_J: tl.constexpr,
):
    # In-place middle-dim reduction for amin_: each program owns exactly one
    # (i, j-tile) slab, reduces it, then broadcast-stores the result back over
    # the same slab (disjoint from every other program's slab, so no hazard).
    # Only used for small work lists (see AMIN_STRIDED_IP_MAX_WORK): there the
    # store phase overlaps other programs' reads and saves the aten copy_;
    # with a long slab list the extra pass makes it slower than the split
    # reduce + copy_ pair.
    pid = tle.program_id(0)
    pid_i = pid // num_jt
    pid_j = pid % num_jt
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    base = inp + pid_i * (K * J)
    cols = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)[None, :]
    col_mask = cols < J
    _all = tl.full([BLOCK_K, BLOCK_J], value=max_value, dtype=dtype)
    for off in range(0, K, BLOCK_K):
        rows = off + tl.arange(0, BLOCK_K)[:, None]
        ptrs = base + rows * J + cols
        if EVEN_K and EVEN_J:
            a = tl.load(ptrs)
        else:
            mask = rows < K and col_mask
            a = tl.load(ptrs, mask, other=max_value)
        # dtype-conditional min (see amin_kernel_1).
        if dtype is tl.bfloat16:
            _all = tl.where(a < _all, a, _all)
        else:
            _all = tl.minimum(_all, a)
    all = tl.min(_all, axis=0)[None, :]
    for off in range(0, K, BLOCK_K):
        rows = off + tl.arange(0, BLOCK_K)[:, None]
        ptrs = base + rows * J + cols
        val = tl.broadcast_to(all, (BLOCK_K, BLOCK_J))
        if EVEN_K and EVEN_J:
            tl.store(ptrs, val)
        else:
            mask = rows < K and col_mask
            tl.store(ptrs, val, mask=mask)


def _amin_full(inp, keepdim):
    # Full reduction over all elements: two-phase (partials + combine).
    # CHUNK targets ~32 programs (per-program fixed cost dominates when the
    # grid is large); BLOCK picks a 16KB tile per dtype.  Measured on gcu400:
    # 1M fp32 15.4us -> 5.15us, 1M fp16 -> 4.06us.
    M = inp.numel()
    dtype = inp.dtype
    block = max(4096, 16384 // inp.element_size())
    span = triton.next_power_of_2((M + 31) // 32)
    chunk = min(max(span, block), 262144)
    mid_size = triton.cdiv(M, chunk)
    block_mid = triton.next_power_of_2(mid_size)
    even = M % chunk == 0
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    if not keepdim:
        out = torch.empty([], dtype=dtype, device=inp.device)
    else:
        shape = [1] * inp.dim()
        out = torch.empty(shape, dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        amin_kernel_1[(mid_size, 1)](
            inp,
            mid,
            M,
            CHUNK=chunk,
            BLOCK=block,
            EVEN=even,
            num_warps=2,
        )
        amin_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
    return out


def _amin_full_inplace(inp):
    # In-place full reduction: phase 1 as usual, then the broadcast-combine
    # kernel re-reduces mid and writes the result over every element of inp.
    # Two kernels total and no aten copy_ on the hot path (measured 1M fp16:
    # 10.2us vs 11.2us for native amin_ + copy_).  Falls back to the split
    # combine + copy_ pair while mid is too large to re-read per program.
    M = inp.numel()
    dtype = inp.dtype
    block = max(4096, 16384 // inp.element_size())
    span = triton.next_power_of_2((M + 31) // 32)
    chunk = min(max(span, block), 262144)
    mid_size = triton.cdiv(M, chunk)
    block_mid = triton.next_power_of_2(mid_size)
    even = M % chunk == 0
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        amin_kernel_1[(mid_size, 1)](
            inp,
            mid,
            M,
            CHUNK=chunk,
            BLOCK=block,
            EVEN=even,
            num_warps=2,
        )
        if mid_size <= AMIN_BCAST_MID_MAX:
            bcast_block = 32768
            amin_kernel_2_bcast[(triton.cdiv(M, bcast_block), 1)](
                mid,
                inp,
                M,
                mid_size,
                BLOCK=bcast_block,
                BLOCK_MID=block_mid,
                num_warps=2,
            )
        else:
            out = torch.empty([], dtype=dtype, device=inp.device)
            amin_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
            inp.copy_(out)
    return inp


def _amin_rows_config(inp, M, N):
    # Tile shape per dtype, re-tuned on gcu400 for (M, N) = (4096, 4096):
    # 16-bit 128x512/nw4 = 23.4-23.6us (1.08-1.09x native), fp32 128x256/nw4
    # = 38.7us (1.05x native).  All keep (acc + load) DSM <= 256KB.  Small
    # tiles stay on nw2 (nw4 costs ~0.6us on the 8x64 tail tiles).
    esize = inp.element_size()
    if esize == 4:
        block_n = min(triton.next_power_of_2(N), 256)
        lane_target = 32768
    else:
        block_n = min(triton.next_power_of_2(N), 512)
        lane_target = 65536
    block_m = min(triton.next_power_of_2(M), lane_target // block_n, 128)
    while triton.cdiv(M, block_m) < 24 and block_m > 8:
        block_m //= 2
    even_m = M % block_m == 0
    even_n = N % block_n == 0
    num_warps = 4 if block_m * block_n >= 8192 else 2
    return block_m, block_n, even_m, even_n, num_warps


def _amin_rows_launch(inp, out, M, N):
    block_m, block_n, even_m, even_n, num_warps = _amin_rows_config(inp, M, N)
    grid = (min(triton.cdiv(M, block_m), 65535),)
    with torch_device_fn.device(inp.device):
        amin_kernel[grid](
            inp,
            out,
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            EVEN_M=even_m,
            EVEN_N=even_n,
            num_warps=num_warps,
        )


def _amin_rows_ip_launch(inp, M, N):
    # In-place row reduction: single fused kernel, no (M, 1) staging tensor
    # and no aten copy_ (measured (4096,4096) fp16: 32.6us = 1.20x native
    # amin_ + copy_; the broadcast stores overlap other programs' reads).
    block_m, block_n, even_m, even_n, num_warps = _amin_rows_config(inp, M, N)
    grid = (min(triton.cdiv(M, block_m), 65535),)
    with torch_device_fn.device(inp.device):
        amin_kernel_rows_ip[grid](
            inp,
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            EVEN_M=even_m,
            EVEN_N=even_n,
            num_warps=num_warps,
        )


def _amin_strided_config(inp, J, K):
    # BJ=512 / num_warps=2 / BLOCK_K=64 measured best on gcu400: BJ=1024
    # tiles run ~29% slower at equal data, nw=4 adds ~21%, and BK=128 tiles
    # lose at every grid size tested.
    esize = inp.element_size()
    block_j = min(triton.next_power_of_2(J), 512)
    block_k = 64
    while block_k > 8 and block_k * block_j * esize * 2 > AMIN_DSM_BUDGET:
        block_k //= 2
    even_k = K % block_k == 0
    even_j = J % block_j == 0
    num_jt = triton.cdiv(J, block_j)
    return block_j, block_k, even_k, even_j, num_jt


def _amin_strided_launch(inp, out, I, K, J):  # noqa: E741
    # Grid-stride launch: at most AMIN_STRIDED_GRID programs, each walking
    # whole (i, j-tile) slabs.  Measured (1024,1024,512) bf16: 566us with
    # grid=48-96 vs 605us for the full 1024-program launch; also removes the
    # old 65535-program grid cap for huge I.
    block_j, block_k, even_k, even_j, num_jt = _amin_strided_config(inp, J, K)
    total = I * num_jt
    grid = (min(total, AMIN_STRIDED_GRID),)
    with torch_device_fn.device(inp.device):
        amin_kernel_strided[grid](
            inp,
            out,
            K,
            J,
            total,
            num_jt,
            BLOCK_K=block_k,
            BLOCK_J=block_j,
            EVEN_K=even_k,
            EVEN_J=even_j,
            num_warps=2,
        )


def _amin_strided_ip_launch(inp, I, K, J):  # noqa: E741
    # In-place middle-dim reduction for small work lists: one slab per
    # program, fused broadcast store (measured (64,512,512) bf16: 46.5us vs
    # 48.4us for reduce + copy_).  Larger lists stay on the split path where
    # grid-stride reads win and the fused store would serialize.
    block_j, block_k, even_k, even_j, num_jt = _amin_strided_config(inp, J, K)
    grid = (I * num_jt,)
    with torch_device_fn.device(inp.device):
        amin_kernel_strided_ip[grid](
            inp,
            K,
            J,
            num_jt,
            BLOCK_K=block_k,
            BLOCK_J=block_j,
            EVEN_K=even_k,
            EVEN_J=even_j,
            num_warps=2,
        )


def _amin_classify(inp, dim):
    # Shared dispatch for amin/amin_.  Normalises dim and returns either
    # ("full", None) or ("dim", reduced_set, out_shape) plus the launch plan:
    # ("rows", M, N) / ("strided", I, K, J) / ("fallback", None, None).
    if isinstance(dim, int):
        dim = [dim]
    ndim = inp.ndim
    if dim is not None and len(dim) > 0:
        dim = [d % ndim for d in dim]
    if dim is None or len(dim) == 0 or len(set(dim)) == ndim:
        return "full", None, None, None

    shape = list(inp.shape)
    reduced = set(dim)
    out_shape = []
    for i in range(ndim):
        if i in reduced:
            out_shape.append(1)
        else:
            out_shape.append(shape[i])

    if inp.is_contiguous() and inp.numel() > 0:
        sorted_dims = sorted(reduced)
        lo, hi = sorted_dims[0], sorted_dims[-1]
        if sorted_dims == list(range(lo, hi + 1)):
            if lo == ndim - len(sorted_dims):
                # Reduced dims are trailing: plain 2D (M, N) row reduction.
                N = 1
                for d in reduced:
                    N *= shape[d]
                M = inp.numel() // N
                return "dim", reduced, out_shape, ("rows", M, N)
            J = 1
            for i in range(hi + 1, ndim):
                J *= shape[i]
            if J >= 64:
                # Middle-dim reduction: strided (I, K, J) tiles, no copy.  The
                # grid-stride kernel caps the launch grid itself, so there is
                # no 65535-program upper limit anymore.
                K = 1
                for d in reduced:
                    K *= shape[d]
                I = inp.numel() // (K * J)  # noqa: E741
                return "dim", reduced, out_shape, ("strided", I, K, J)

    return "dim", reduced, out_shape, ("fallback", None, None)


def amin(inp, dim=None, keepdim=False):
    logger.debug("GEMS AMIN")
    assert inp.dtype in AMIN_SUPPORTED_DTYPES, "amin supports float and int dtypes"
    kind, reduced, out_shape, plan = _amin_classify(inp, dim)
    if kind == "full":
        # The full reduction walks the buffer flat, so a strided view must be
        # materialized first (same latent-bug guard as the amax twin).
        if not inp.is_contiguous():
            inp = inp.contiguous()
        return _amin_full(inp, keepdim)

    dtype = inp.dtype
    if plan[0] == "rows":
        _, M, N = plan
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        _amin_rows_launch(inp, out, M, N)
    elif plan[0] == "strided":
        _, I, K, J = plan
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        _amin_strided_launch(inp, out, I, K, J)
    else:
        # Generic fallback: compress the reduced dims to the end, then
        # row-reduce.
        inp_c = dim_compress(inp, sorted(reduced))
        N = 1
        for d in reduced:
            N *= inp.shape[d]
        M = inp_c.numel() // N
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        _amin_rows_launch(inp_c, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def amin_(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ENFLAME AMIN_")
    assert inp.dtype in AMIN_SUPPORTED_DTYPES, "amin_ supports float and int dtypes"
    kind, reduced, out_shape, plan = _amin_classify(inp, dim)
    if kind == "full":
        # The fused paths write every element of inp, so they need the flat
        # layout; a strided view falls back to compute + copy_ into the view.
        if inp.is_contiguous():
            return _amin_full_inplace(inp)
        out = _amin_full(inp.contiguous(), True)
        inp.copy_(out)
        return inp

    if plan[0] == "rows":
        # Contiguous trailing-dim reduction: the row kernel broadcasts the
        # per-row minima straight back into the rows it owns (single kernel).
        _, M, N = plan
        _amin_rows_ip_launch(inp, M, N)
        return inp
    if plan[0] == "strided":
        # Middle-dim reduction: fuse the broadcast into the slab kernel while
        # each program owns exactly one slab; otherwise split reduce+copy_
        # (the grid-stride read kernel is faster and the fused store would
        # serialize behind long slab lists).
        _, I, K, J = plan
        block_j, _, _, _, num_jt = _amin_strided_config(inp, J, K)
        if I * num_jt <= AMIN_STRIDED_IP_MAX_WORK:
            _amin_strided_ip_launch(inp, I, K, J)
            return inp
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        _amin_strided_launch(inp, out, I, K, J)
        inp.copy_(out)
        return inp

    # Fallback (non-contiguous / non-adjacent dims): reduce on a compressed
    # copy, then broadcast back into the original strided inp via copy_.
    result = amin(inp, dim=dim, keepdim=True)
    inp.copy_(result)
    return inp
