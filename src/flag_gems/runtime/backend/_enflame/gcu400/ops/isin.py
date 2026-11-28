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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

from .eq import eq_scalar
from .ne import ne_scalar
from .sort import sort as flaggems_sort
from .unique import _unique2

logger = logging.getLogger(__name__)

_INT_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_NPROGS = 48  # GCU400 concurrent CTA slots (enflame codegen max_grid_size)
_SEARCH_GRID_LIMIT = 32768  # grid.x hard limit is 65535, but launches that
# close to the boundary are empirically pathological (P19); stay at half.
_UNIQUE_THRESHOLD = 4194304  # 1024 * 4096


# ---------------------------------------------------------------------------
# Tensor-Tensor, small inputs: brute-force comparison tile kernel.
#
# The generic int1 accumulator + custom combine_fn reduce (reduce_any/all)
# compiles to far slower code on GCU400, so the tile accumulator is int8 and
# the row combine is tl.max/tl.min (same lesson as ops/any.py). A persistent
# 48-CTA grid-stride wrapper amortizes the per-CTA setup cost; measured on
# (64, 64) int16: ~700us (old) -> ~22us (128x256 tile, 8 warps).
# ---------------------------------------------------------------------------


@triton.jit
def isin_by_comparation_impl(
    global_pid,
    in0_ravel_ptr: tl.tensor,
    in1_ravel_ptr: tl.tensor,  # in
    out_ptr: tl.tensor,  # out
    M: int,  # num_tasks
    N: int,  # num_tasks_1
    BLOCK_M: tl.constexpr,  # tile_size
    BLOCK_N: tl.constexpr,  # tile_size_1
    invert: tl.constexpr,
):
    rows = global_pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    in0 = tl.load(in0_ravel_ptr + rows, row_mask, other=0)
    acc = tl.full([BLOCK_M, BLOCK_N], (1 if invert else 0), dtype=tl.int8)
    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        in1 = tl.load(in1_ravel_ptr + cols, mask, other=0)
        if invert:
            miss = (in0 != in1) | (~col_mask)
            acc = acc & miss.to(tl.int8)
        else:
            hit = (in0 == in1) & col_mask
            acc = acc | hit.to(tl.int8)
    if invert:
        out = tl.min(acc, axis=1) > 0
    else:
        out = tl.max(acc, axis=1) > 0
    tl.store(out_ptr + rows, out[:, None].to(out_ptr.dtype.element_ty), row_mask)


@libentry()
@triton.jit
def isin_by_comparation_kernel(
    in0_ravel_ptr: tl.tensor,
    in1_ravel_ptr: tl.tensor,  # in
    out_ptr: tl.tensor,  # out
    M: int,  # num_tasks
    N: int,  # num_tasks_1
    BLOCK_M: tl.constexpr,  # tile_size
    BLOCK_N: tl.constexpr,  # tile_size_1
    tiles_per_cta: int,
    invert: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        isin_by_comparation_impl(
            global_pid,
            in0_ravel_ptr,
            in1_ravel_ptr,  # in
            out_ptr,  # out
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            invert,
        )


def isin_by_comparation(
    in0: torch.tensor,
    in1: torch.tensor,
    invert: bool,
):
    in0_ravel = in0.contiguous().ravel()
    in1_ravel = in1.contiguous().ravel()
    M = in0.numel()
    N = in1.numel()
    BLOCK_M = min(128, max(8, triton.next_power_of_2(triton.cdiv(M, _NPROGS))))
    # BLOCK_N never takes 128: the [BLOCK_M>=32, 128] int8 tile accumulator
    # segfaults the in-process compiler (P02,
    # flaggems_enflame_isin_cmp_tile_bn128_segv). 16/32/64/256/512 are all
    # validated safe, so next_power_of_2(N) >= 128 rounds up to 256.
    np_n = triton.next_power_of_2(N)
    BLOCK_N = 256 if np_n >= 128 else max(16, np_n)
    num_warps = 8
    num_tiles = triton.cdiv(M, BLOCK_M)
    ctas_num = min(_NPROGS, num_tiles)
    tiles_per_cta = triton.cdiv(num_tiles, ctas_num)
    grid = (ctas_num,)
    out = torch.empty_like(in0_ravel, dtype=torch.bool)
    with torch_device_fn.device(in0_ravel.device.index):
        isin_by_comparation_kernel[grid](
            in0_ravel,
            in1_ravel,  # in
            out,  # out
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            tiles_per_cta=tiles_per_cta,
            invert=invert,
            num_warps=num_warps,
        )
    return out.view_as(in0)


# ---------------------------------------------------------------------------
# Tensor-Tensor, large inputs: sort (or unique for narrow dtypes whose
# cardinality is bounded) + binary search.
#
# The sorted probe load is unmasked: `mid` stays inside [0, N-1] for active
# lanes (start <= mid < end <= N holds by construction) and collapses to 0
# for finished lanes, so dropping the predicate only re-reads element 0.
# ---------------------------------------------------------------------------


@triton.jit
def isin_by_search_impl(
    global_pid,
    in0_ravel_ptr: tl.tensor,
    in1_sorted_ptr: tl.tensor,  # in
    out_ptr: tl.tensor,  # out
    M: int,  # num_tasks
    N: int,  # num_tasks_1
    log_n: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile_size
    invert: tl.constexpr,
):
    r = tl.arange(0, BLOCK_M)
    i0 = global_pid * BLOCK_M + r
    mask = i0 < M

    # load in0_ravel
    in0_ravel = tl.load(in0_ravel_ptr + i0, mask=mask)

    # binary search: lower_bound
    out = tl.zeros_like(r).to(tl.int1)
    start = tl.zeros_like(r)
    end = start + N
    while_mask = start < end
    for i in range(log_n):
        mid = tl.where(while_mask, start + (end - start) // 2, 0)
        mid_val = tl.load(in1_sorted_ptr + mid)
        out = tl.where(while_mask, out | (mid_val == in0_ravel), out)  # found
        start = tl.where(while_mask & (mid_val < in0_ravel), mid + 1, start)
        end = tl.where(while_mask & (mid_val > in0_ravel), mid, end)
        while_mask = start < end

    # store out
    tl.store(out_ptr + i0, not out if invert else out, mask=mask)


@libentry()
@triton.jit
def isin_by_search_kernel(
    in0_ravel_ptr: tl.tensor,
    in1_sorted_ptr: tl.tensor,  # in
    out_ptr: tl.tensor,  # out
    M: int,  # num_tasks
    N: int,  # num_tasks_1
    log_n: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile_size
    pid_offset,
    invert: tl.constexpr,
):
    # One tile per program. The launcher walks the whole tile range in
    # ``ctas_num``-sized chunks and passes each chunk base as ``pid_offset``.
    # A grid-stride loop that carried the search state across tiles was
    # miscompiled on gcu400: every tile past the first grid chunk came back
    # with ~5% of its lanes wrong (measured on 16x128x64x1280, tiles >= 32768),
    # so the chunking is done on the host instead.
    global_pid = ext.program_id(0) + pid_offset
    isin_by_search_impl(
        global_pid,
        in0_ravel_ptr,
        in1_sorted_ptr,  # in
        out_ptr,  # out
        M,
        N,
        log_n,
        BLOCK_M,
        invert,
    )


def isin_by_search(
    in0: torch.tensor,
    in1: torch.tensor,
    invert: bool,
    unique_in1: bool,
):
    # narrow (<=2 byte) dtypes have bounded cardinality (<=65536 distinct
    # values), so _unique2 shrinks the probe array and shortens every binary
    # search; wide dtypes keep the plain sort because full-range i32/i64
    # inputs are almost all distinct and the compaction pass (plus its
    # host-side .item() sync) costs more than it saves.
    if unique_in1:
        in1_ravel, _, _ = _unique2(
            in1, sorted=True, return_inverse=False, return_counts=False
        )
    else:
        in1_ravel, _ = flaggems_sort(in1.contiguous().ravel())
    in0_ravel = in0.contiguous().ravel()
    M = in0_ravel.numel()
    N = in1_ravel.numel()
    if M <= 65536:
        BLOCK_M, num_warps = 1024, 4
    else:
        # 2048 lanes / 8 warps measured best at every large M (16.7M i32:
        # 17.6->18.3us vs 1024; 67M i32: 82ms vs a pathological 178ms), and
        # with the CTA cap below it also keeps the launch far from the
        # 65535 grid.x boundary, which is empirically pathological (P19).
        BLOCK_M, num_warps = 2048, 8
    log_n = int(math.log2(N)) + 1
    num_tiles = triton.cdiv(M, BLOCK_M)
    ctas_num = min(_SEARCH_GRID_LIMIT, num_tiles)
    grid = (ctas_num,)
    out = torch.empty_like(in0_ravel, dtype=torch.bool)
    with torch_device_fn.device(in0_ravel.device.index):
        for tile_base in range(0, num_tiles, ctas_num):
            isin_by_search_kernel[grid](
                in0_ravel,
                in1_ravel,  # in
                out,  # out
                M,
                N,
                log_n,
                BLOCK_M,
                pid_offset=tile_base,
                invert=invert,
                num_warps=num_warps,
            )
    return out.view_as(in0)


# ---------------------------------------------------------------------------
# Scalar-in-Tensor (aten::isin.Scalar_Tensor): a single membership test
# reduces to "any(x == s)" over the tensor, so the old path (wrap the scalar
# into a 0-dim tensor, then sort/unique the whole probe tensor and binary
# search one element) is replaced by a two-stage flat OR-reduction. invert is
# the final NOT of the same reduction. Measured on (4096, 4096) int16:
# ~25ms (old search path) -> ~47us.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["n"])
def isin_scalar_in_tensor_kernel(
    x_ptr: tl.tensor,
    mid_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n: int,
    S,  # scalar to test (int for int tensors, float for float tensors)
    BLOCK: tl.constexpr,
    FINAL: tl.constexpr,
    INVERT: tl.constexpr,
    FLOAT_CMP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.int8)
    arange = tl.arange(0, BLOCK)
    num_tiles = (n + BLOCK - 1) // BLOCK
    for tile_id in tl.range(pid, num_tiles, num_pids):
        off = tile_id * BLOCK + arange
        m = off < n
        x = tl.load(x_ptr + off, mask=m, other=0)
        # a float scalar against an int tensor promotes the tensor to float32
        # (ATen result_type semantics); Triton's implicit int-vs-float mix
        # truncates the scalar instead, so the cast is explicit.
        if FLOAT_CMP:
            hit = (x.to(tl.float32) == S) & m
        else:
            hit = (x == S) & m
        acc = acc | hit.to(tl.int8)
    found = tl.max(acc, axis=0) > 0
    if FINAL:
        tl.store(out_ptr, (not found) if INVERT else found)
    else:
        tl.store(mid_ptr + pid, found.to(tl.int8))


@libentry()
@triton.jit
def isin_scalar_in_tensor_reduce_kernel(
    mid_ptr: tl.tensor,
    out_ptr: tl.tensor,
    mid_size: int,
    BLOCK_MID: tl.constexpr,
    invert: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_MID)
    m = offs < mid_size
    v = tl.load(mid_ptr + offs, mask=m, other=0).to(tl.int8)
    found = tl.max(v, axis=0) > 0
    tl.store(out_ptr, (not found) if invert else found)


def _isin_scalar_in_tensor(x: torch.tensor, s, invert: bool) -> torch.Tensor:
    x_ravel = x.contiguous().ravel()
    n = x_ravel.numel()
    float_cmp = isinstance(s, float)
    if n == 0:
        # nothing is contained in an empty probe set; invert flips it
        return torch.full((), invert, dtype=torch.bool, device=x.device)
    if n <= 16384:
        block = max(triton.next_power_of_2(n), 1024)
    else:
        block = 16384
    num_blocks = triton.cdiv(n, block)
    ctas_num = min(_NPROGS, num_blocks)
    out = torch.empty([], dtype=torch.bool, device=x_ravel.device)
    with torch_device_fn.device(x_ravel.device.index):
        if ctas_num == 1:
            isin_scalar_in_tensor_kernel[(1,)](
                x_ravel,
                out,
                out,
                n,
                s,
                BLOCK=block,
                FINAL=True,
                INVERT=invert,
                FLOAT_CMP=float_cmp,
                num_warps=4,
            )
        else:
            mid = torch.empty((ctas_num,), dtype=torch.int8, device=x_ravel.device)
            isin_scalar_in_tensor_kernel[(ctas_num,)](
                x_ravel,
                mid,
                out,
                n,
                s,
                BLOCK=block,
                FINAL=False,
                INVERT=invert,
                FLOAT_CMP=float_cmp,
                num_warps=4,
            )
            isin_scalar_in_tensor_reduce_kernel[(1,)](
                mid,
                out,
                ctas_num,
                BLOCK_MID=triton.next_power_of_2(ctas_num),
                invert=invert,
                num_warps=1,
            )
    return out


# ---------------------------------------------------------------------------
# Tensor-vs-Scalar (aten::isin.Tensor_Scalar): elementwise eq/ne against the
# scalar. The comparison runs in the promoted type (Triton widens the tensor
# lanes to the scalar argument type), which is exact for every int width and
# matches ATen's result_type promotion, including out-of-range scalars such
# as isin(int16_tensor, 40000) -> all-False (no overflow error, same as ATen).
# The previous redirect to eq_scalar/ne_scalar cast both sides through fp32,
# which silently collides int32/int64 values above 2**24.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["n"])
def isin_tensor_scalar_kernel(
    x_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n: int,
    S,  # scalar to test (int for int tensors, float for float tensors)
    BLOCK: tl.constexpr,
    INVERT: tl.constexpr,
    FLOAT_CMP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_tiles = (n + BLOCK - 1) // BLOCK
    for tile_id in tl.range(pid, num_tiles, num_pids):
        off = tile_id * BLOCK + arange
        m = off < n
        x = tl.load(x_ptr + off, mask=m, other=0)
        # same ATen float32-promotion rule as isin_scalar_in_tensor_kernel
        if FLOAT_CMP:
            xf = x.to(tl.float32)
        else:
            xf = x
        r = (xf != S) if INVERT else (xf == S)
        tl.store(out_ptr + off, r.to(out_ptr.dtype.element_ty), mask=m)


def _isin_tensor_vs_scalar(x: torch.tensor, s, invert: bool) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros_like(x, dtype=torch.bool)
    x_ravel = x.contiguous().ravel()
    n = x_ravel.numel()
    out = torch.empty_like(x_ravel, dtype=torch.bool)
    if n <= 65536:
        block = max(256, min(16384, triton.next_power_of_2(n)))
        grid = (triton.cdiv(n, block),)
    else:
        block = 65536
        grid = (min(_NPROGS, triton.cdiv(n, block)),)
    with torch_device_fn.device(x_ravel.device.index):
        isin_tensor_scalar_kernel[grid](
            x_ravel,
            out,
            n,
            s,
            BLOCK=block,
            INVERT=invert,
            FLOAT_CMP=isinstance(s, float),
            num_warps=4,
        )
    return out.view_as(x)


def _scalar_compare_supported(t: torch.tensor, s) -> bool:
    if not torch.is_tensor(t) or isinstance(s, bool):
        return False
    if t.dtype in _INT_DTYPES or t.dtype in _FLOAT_DTYPES:
        # int scalars compare in the promoted int type (exact for every
        # width); float scalars promote an int tensor to float32 (ATen
        # result_type) and collapse to a float tensor's own dtype.
        return isinstance(s, (int, float))
    return False


def _scalar_kernel_arg(t: torch.tensor, s, brute_small: bool = False):
    # ATen isin scalar handling (verified against CPU ATen, 2026-11-18):
    # - int tensors keep the scalar's own width so the compare promotes
    #   exactly (int scalars), or promote the tensor to float32 (float
    #   scalars) - result_type semantics, at every input size;
    # - the Tensor_Scalar direction and the small-input brute-force branch
    #   of the Scalar_Tensor direction (numel <= 9) collapse a weak float
    #   scalar into a float16/bfloat16 tensor's own dtype (so e.g.
    #   isin(bf16_tensor, 3.14) matches bf16(3.14) = 3.140625), while the
    #   sort branch (numel >= 10) compares in float32 instead;
    # - float32 tensors compare in float32 either way.
    if t.dtype == torch.float16 or t.dtype == torch.bfloat16:
        if brute_small:
            return float(torch.tensor(s).to(t.dtype))
        return float(s)
    if isinstance(s, int):
        return s
    return float(s)


def isin(
    in0,
    in1,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME ISIN")
    if not torch.is_tensor(in0):
        if _scalar_compare_supported(in1, in0):
            # ATen's brute-force branch (test side numel <= 9) collapses the
            # scalar into a narrow float tensor's dtype; larger inputs compare
            # in float32 (see _scalar_kernel_arg).
            brute_small = torch.is_tensor(in1) and in1.numel() <= 9
            return _isin_scalar_in_tensor(
                in1, _scalar_kernel_arg(in1, in0, brute_small), invert
            )
        assert torch.is_tensor(in1)
        in0 = torch.tensor(in0, device=in1.device)
    elif not torch.is_tensor(in1):
        if _scalar_compare_supported(in0, in1):
            return _isin_tensor_vs_scalar(
                in0, _scalar_kernel_arg(in0, in1, brute_small=True), invert
            )
        assert torch.is_tensor(in0)
        # exotic dtypes (bool/complex/fp64): keep the legacy collapse of the
        # scalar into the tensor dtype and the local scalar compare ops.
        scalar_value = torch.tensor(in1).to(in0.dtype).item()
        if invert:
            return ne_scalar(in0, scalar_value)
        return eq_scalar(in0, scalar_value)
    if in0.numel() == 0 or in1.numel() == 0:
        if in1.numel() == 0 and invert:
            # every element is absent from an empty probe set
            return torch.ones_like(in0, dtype=torch.bool)
        return torch.zeros_like(in0, dtype=torch.bool)
    elif in0.numel() <= 12288 and in1.numel() <= 12288:  # 1024 * 12
        return isin_by_comparation(in0, in1, invert)
    else:
        use_unique = (
            (not assume_unique)
            and in1.numel() > _UNIQUE_THRESHOLD
            and in1.element_size() <= 2
        )
        return isin_by_search(in0, in1, invert, unique_in1=use_unique)


# ---------------------------------------------------------------------------
