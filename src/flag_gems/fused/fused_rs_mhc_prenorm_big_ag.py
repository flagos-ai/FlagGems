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

"""DeepSeek-V4 mHC boundary on rings: bulk RS + mhc_post + pre-norm GEMM + mhc_pre (with
the layer RMSNorm) + bulk AG, for pure TP.

Replaces ``x = all_reduce(x); res = mhc_post(x, res, post, comb);
post, comb, y = mhc_pre(res, fn, ..., norm_weight=w)``. Rank r owns rows
[r * chunk, min(T, (r + 1) * chunk)), chunk = ceil(T / W): the four residual streams and
post / comb of those rows stay on it (updated in place), only y [T, H] is gathered.

Ring protocol, slots and flags are the ones of ``fused_rs_norm_ag``; only the row mapping
and the owner step differ. RS/AG use BM-row stripes. SPLITS neighboring pids exchange
local post and prenorm results in a cooperative grid. Each computes one K slice for
all BM*SPLITS rows, then each reduces and sends its own BM rows. Owner, per tile:
  A  d = bf16(slot + x); read the 4 old streams, new_j = post_j d + sum_i comb[i, j] res_i,
     write the 4 new streams back
  G  mix[BM, 24] = new[BM, 4H] @ fn^T (tf32 dot), sq = sum(new^2)
  B  pre / post / comb from mix (sigmoid + Sinkhorn), post / comb written back
  C  z = sum_i pre_i new_i (fp32), inv = rsqrt(mean(z^2) + eps),
     y = bf16(bf16(z) * inv * w) -> my out rows + successor's AG slot
mode "d" stops after d (owner writes d instead of y): the ring alone, for tests / bench.
"""

import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from flag_gems.fused.fused_rs_norm_ag import (
    _bulk_st,
    _ld,
    _ld_hi,
    _ld_lo,
    _ldv,
    _ldv_hi,
    _ldv_lo,
    _pack,
    _rbulk_st,
    _RingComm,
    _signal,
    _unpack,
    _wait,
    build_bulk_lib,
    install_bulk_ir_fix,
)

logger = logging.getLogger(__name__)

HC = 4
MIX = HC * (2 + HC)  # 24
MIX_PAD = 32
SPLIT_WARPS = 4  # warps of a split-K owner program, see launch_shape
# fn rows in the padded GEMM: comb (16) first, then pre (4) and post (4), then 8 zeros, so
# the [BM, 32] result splits into two [BM, 4, 4] halves
_FN_ORDER = list(range(2 * HC, MIX)) + list(range(2 * HC))


@triton.jit
def _bf16_round(x):
    lo, hi = _pack(x)
    return _unpack(lo, hi)


@triton.jit
def _post_row(slot, xp, rp, post_ptr, comb_ptr, lr, rb, lane):
    """Step A for one row: returns nothing, writes the 4 new streams over the old ones."""
    d = _bf16_round(_ldv(slot + lane) + _ld(xp + lane))
    r0 = _ld(rp + lane)
    r1 = _ld(rp + rb + lane)
    r2 = _ld(rp + 2 * rb + lane)
    r3 = _ld(rp + 3 * rb + lane)
    pb = post_ptr + lr * 4
    cb = comb_ptr + lr * 16
    for j in tl.static_range(4):
        nj = tl.load(pb + j) * d
        nj += tl.load(cb + j) * r0
        nj += tl.load(cb + 4 + j) * r1
        nj += tl.load(cb + 8 + j) * r2
        nj += tl.load(cb + 12 + j) * r3
        lo, hi = _pack(nj)
        _bulk_st(rp + j * rb + lane, lo, hi)


@triton.jit
def _y_row(rp, w_addr, rb, lane, p0, p1, p2, p3, H: tl.constexpr, norm_eps):
    """Step C for one row: returns y as (lo, hi) bulk words."""
    z = p0 * _ld(rp + lane)
    z += p1 * _ld(rp + rb + lane)
    z += p2 * _ld(rp + 2 * rb + lane)
    z += p3 * _ld(rp + 3 * rb + lane)
    inv = tl.rsqrt(tl.sum(z * z) / H + norm_eps)
    return _pack(_bf16_round(z) * inv * _ld(w_addr + lane))


@triton.jit
def _sinkhorn(cm, eps, REPEAT: tl.constexpr):
    # cm [BM, 4, 4]: softmax over k, + eps, column norm, then REPEAT - 1 row / column rounds
    cm = tl.exp(cm - tl.max(cm, 2)[:, :, None])
    cm = cm / tl.sum(cm, 2)[:, :, None] + eps
    cm = cm / (tl.sum(cm, 1)[:, None, :] + eps)
    for _ in tl.static_range(REPEAT - 1):
        cm = cm / (tl.sum(cm, 2)[:, :, None] + eps)
        cm = cm / (tl.sum(cm, 1)[:, None, :] + eps)
    return cm


@triton.jit
def _group_ready(flags, tile, split, target, SPLITS: tl.constexpr):
    # One slot per (tile, producer), never recycled within a launch. Release all
    # threads' writes before advertising completion; acquire every producer.
    flags += tile * SPLITS
    tl.debug_barrier()
    tl.atomic_xchg(flags + split, target, sem="release", scope="gpu")
    for i in range(SPLITS):
        seen = tl.atomic_cas(flags + i, -1, -1, sem="acquire", scope="gpu")
        while seen != target:
            seen = tl.atomic_cas(flags + i, -1, -1, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def _quant_row(yrow, out, ag, scales, lr, gr, scale_offset, lane, H: tl.constexpr):
    # Use ordinary column order for quantization. Re-load packed q via the bulk
    # instruction before its first remote store (peer slots use bulk layout).
    h = tl.arange(0, H)
    y = tl.load(yrow.to(tl.pointer_type(tl.bfloat16)) + h, cache_modifier=".cg").to(tl.float32)
    a = tl.maximum(tl.max(tl.abs(y), 0), 1e-10)
    qrow = out + gr.to(tl.int64) * H
    tl.store(qrow.to(tl.pointer_type(tl.int8)) + h, libdevice.round(y * (127.0 / a)).to(tl.int8))
    tl.store(scales + gr, a / 127.0)
    tl.store((ag + scale_offset).to(tl.pointer_type(tl.float32)) + lr, a / 127.0, cache_modifier=".wt")
    tl.debug_barrier()
    _rbulk_st(ag + lr.to(tl.int64) * H + lane, _ld_lo(qrow + lane), _ld_hi(qrow + lane))


@triton.jit(
    do_not_specialize=[
        "x_addr", "res_addr", "w_addr", "out_addr", "num_tokens", "chunk", "slot_bytes",
        "par_off", "rank", "nchan", "rms_eps", "pre_eps", "sk_eps", "post_mult",
        "norm_eps", "fbase", "max_spin",
    ]
)
def rs_mhc_ag_ring_bulk_kernel(
    x_addr,  # int64 byte addresses, all 512B aligned: [T, H] partial sums
    res_addr,  # [Tl, 4, H] my rows, updated in place
    w_addr,  # [H] norm weight
    out_addr,  # [T, H] y (mode d: d)
    res_ptr,  # the residual again as a tensor, for the GEMM loads
    post_ptr,  # [Tl, 4] fp32, in / out
    comb_ptr,  # [Tl, 4, 4] fp32, in / out
    fn_ptr,  # [32, 4H] fp32, rows in _FN_ORDER then zeros
    scale_ptr,  # [3] fp32
    base_ptr,  # [24] fp32
    dump_ptr,  # [Tl, 25] fp32: raw mix (24) and sq, when DUMP
    partial, post_ready, mix_ready, out_scale,
    data_addrs,  # int64 [W], every rank's IPC data buffer
    flag_ptrs,  # int64 [W], every rank's IPC flag buffer (one int32 per program)
    order_ptr,  # int32 [C, W], ring of every channel
    pos_ptr,  # int32 [C], my position in every ring
    err_ptr,  # int32 [1], wait timeouts
    num_tokens,
    chunk,
    slot_bytes,
    par_off,  # parity * (2W - 2) * slot_bytes
    rank,
    nchan,
    rms_eps,
    pre_eps,
    sk_eps,
    post_mult,
    norm_eps,
    fbase,  # flag value at the start of this call (int32, wraps)
    max_spin,
    WORLD: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,  # threads per program; a row of H bf16 is exactly V * 16B
    BM: tl.constexpr,
    BK: tl.constexpr,
    REPEAT: tl.constexpr,
    D_ONLY: tl.constexpr,
    DUMP: tl.constexpr,
    SPLITS: tl.constexpr,
    QUANT: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = ((tl.arange(0, V) // 32) * 512).to(tl.int64)
    rb: tl.constexpr = H * 2  # bytes of one stream row
    sb = slot_bytes.to(tl.int64)

    c = pid % nchan
    p = tl.load(pos_ptr + c)
    ring = order_ptr + c * WORLD
    next = tl.load(ring + (p + 1) % WORLD)
    my_flag = tl.load(flag_ptrs + rank).to(tl.pointer_type(tl.int32)) + pid
    succ_flag = tl.load(flag_ptrs + next).to(tl.pointer_type(tl.int32)) + pid
    mine = tl.load(data_addrs + rank) + par_off.to(tl.int64)
    nxt = tl.load(data_addrs + next) + par_off.to(tl.int64)

    # RS step 0: my partial sums of chunk order[p - 1] -> succ slot 0
    k = tl.load(ring + (p + WORLD - 1) % WORLD)
    b = tl.minimum(k * chunk, num_tokens)
    n = tl.minimum((k + 1) * chunk, num_tokens) - b
    for t in range(pid, tl.cdiv(n, BM), G):
        for lr in range(t * BM, tl.minimum(t * BM + BM, n)):
            xp = x_addr + (b + lr).to(tl.int64) * rb + lane
            _rbulk_st(nxt + lr.to(tl.int64) * rb + lane, _ld_lo(xp), _ld_hi(xp))
    _signal(succ_flag, fbase + 1)

    # RS steps 1 .. W-2: add my partial sums to the received chunk, pass it on
    for s in range(1, WORLD - 1):
        _wait(my_flag, fbase + s, err_ptr, max_spin)
        k = tl.load(ring + (p + 2 * WORLD - s - 1) % WORLD)
        b = tl.minimum(k * chunk, num_tokens)
        n = tl.minimum((k + 1) * chunk, num_tokens) - b
        for t in range(pid, tl.cdiv(n, BM), G):
            for lr in range(t * BM, tl.minimum(t * BM + BM, n)):
                o = lr.to(tl.int64) * rb + lane
                acc = _ldv(mine + (s - 1) * sb + o) + _ld(x_addr + (b + lr).to(tl.int64) * rb + lane)
                lo, hi = _pack(acc)
                _rbulk_st(nxt + s * sb + o, lo, hi)
        _signal(succ_flag, fbase + s + 1)

    # owner: the last hop brings the sum of the other W - 1 ranks of my rows
    _wait(my_flag, fbase + WORLD - 1, err_ptr, max_spin)
    mb = tl.minimum(rank * chunk, num_tokens)
    mn = tl.minimum((rank + 1) * chunk, num_tokens) - mb
    last = mine + (WORLD - 2) * sb
    ag = nxt + (WORLD - 1) * sb
    rr = tl.arange(0, BM)
    i4 = tl.arange(0, 4)
    col = tl.arange(0, 32)
    orig = tl.where(col < 16, col + 8, tl.where(col < 24, col - 16, 0))  # mix column of col
    GM: tl.constexpr = BM * SPLITS
    split = pid % SPLITS
    for tile in range(pid // SPLITS, tl.cdiv(mn, GM), G // SPLITS):
        r0 = tile * GM + split * BM
        r1 = tl.minimum(r0 + BM, mn)
        if D_ONLY:
            for lr in range(r0, r1):
                o = lr.to(tl.int64) * rb + lane
                lo, hi = _pack(_ldv(last + o) + _ld(x_addr + (mb + lr).to(tl.int64) * rb + lane))
                _bulk_st(out_addr + (mb + lr).to(tl.int64) * rb + lane, lo, hi)
                _rbulk_st(ag + o, lo, hi)
        else:
            # A: mhc_post, the 4 new streams replace the old ones row by row
            for lr in range(r0, r1):
                rp = res_addr + lr.to(tl.int64) * 4 * rb
                xp = x_addr + (mb + lr).to(tl.int64) * rb
                _post_row(last + lr.to(tl.int64) * rb, xp, rp, post_ptr, comb_ptr, lr, rb, lane)
            if SPLITS > 1:
                _group_ready(post_ready, tile, split, fbase + 1, SPLITS)
            else:
                tl.debug_barrier()

            # G: pre-norm GEMM of the tile, rows r0 + rr
            rows = r0 + rr
            rmask = rows < r1
            grows = tile * GM + tl.arange(0, GM)
            part_mix = tl.full((GM, 32), 0, tl.float32)
            part_sq = tl.full((GM,), 0, tl.float32)
            for k0 in tl.range(0, 4 * H // SPLITS, BK, num_stages=2):
                ks = split * (4 * H // SPLITS) + k0 + tl.arange(0, BK)
                a = tl.load(
                    res_ptr + grows[:, None].to(tl.int64) * (4 * H) + ks[None, :],
                    mask=grows[:, None] < mn, other=0.0, cache_modifier=".cg",
                ).to(tl.float32)
                w = tl.load(fn_ptr + col[None, :].to(tl.int64) * (4 * H) + ks[:, None])
                part_mix += tl.dot(a, w, input_precision="tf32")
                part_sq += tl.sum(a * a, 1)
            if SPLITS > 1:
                storage = partial + tile.to(tl.int64) * SPLITS * GM * 25
                dst = storage + split * GM * 25 + tl.arange(0, GM) * 25
                tl.store(dst[:, None] + col[None, :], part_mix, mask=col[None, :] < 24)
                tl.store(dst + 24, part_sq)
                _group_ready(mix_ready, tile, split, fbase + 1, SPLITS)
                acc = tl.full((BM, 32), 0, tl.float32)
                sq = tl.full((BM,), 0, tl.float32)
                for j in range(SPLITS):
                    src = storage + j * GM * 25 + (split * BM + rr) * 25
                    acc += tl.load(src[:, None] + col[None, :], mask=col[None, :] < 24, other=0., cache_modifier=".cg")
                    sq += tl.load(src + 24, cache_modifier=".cg")
            else:
                acc = part_mix
                sq = part_sq
            if DUMP:
                tl.store(dump_ptr + rows[:, None] * 25 + orig[None, :], acc,
                         mask=rmask[:, None] & (col[None, :] < 24))
                tl.store(dump_ptr + rows * 25 + 24, sq, mask=rmask)

            # B: coefficients
            s0 = tl.load(scale_ptr)
            s1 = tl.load(scale_ptr + 1)
            s2 = tl.load(scale_ptr + 2)
            scl = tl.where(orig < 4, s0, tl.where(orig < 8, s1, s2))
            bse = tl.load(base_ptr + orig, mask=col < 24, other=0.0)
            rms = tl.rsqrt(sq / (4 * H) + rms_eps)
            m = (acc * rms[:, None]) * scl[None, :] + bse[None, :]
            cm, pp = tl.split(tl.permute(tl.reshape(m, (BM, 2, 16)), (0, 2, 1)))
            cm = _sinkhorn(tl.reshape(cm, (BM, 4, 4)), sk_eps, REPEAT)
            pp = tl.reshape(pp, (BM, 4, 4))  # [:, 0, :] pre, [:, 1, :] post
            pre = tl.sigmoid(tl.sum(tl.where(i4[None, :, None] == 0, pp, 0.0), 1)) + pre_eps
            post = tl.sigmoid(tl.sum(tl.where(i4[None, :, None] == 1, pp, 0.0), 1)) * post_mult
            tl.store(post_ptr + rows[:, None] * 4 + i4[None, :], post, mask=rmask[:, None])
            cidx = rows[:, None, None] * 16 + i4[None, :, None] * 4 + i4[None, None, :]
            tl.store(comb_ptr + cidx, cm, mask=rmask[:, None, None])

            # C: weighted sum + RMSNorm, y -> my out rows and the successor's AG slot
            for lr in range(r0, r1):
                pr = tl.where(rr[:, None] == lr - r0, pre, 0.0)
                p0 = tl.sum(tl.sum(tl.where(i4[None, :] == 0, pr, 0.0), 1), 0)
                p1 = tl.sum(tl.sum(tl.where(i4[None, :] == 1, pr, 0.0), 1), 0)
                p2 = tl.sum(tl.sum(tl.where(i4[None, :] == 2, pr, 0.0), 1), 0)
                p3 = tl.sum(tl.sum(tl.where(i4[None, :] == 3, pr, 0.0), 1), 0)
                rp = res_addr + lr.to(tl.int64) * 4 * rb
                lo, hi = _y_row(rp, w_addr, rb, lane, p0, p1, p2, p3, H, norm_eps)
                if QUANT:
                    # This pid has consumed its last RS slot; reuse its own row
                    # to convert from bulk register order to logical H order.
                    yrow = last + lr.to(tl.int64) * rb
                    _bulk_st(yrow + lane, lo, hi)
                    tl.debug_barrier()
                    qlane = ((tl.arange(0, H // 16) // 32) * 512).to(tl.int64)
                    _quant_row(yrow, out_addr, ag, out_scale, lr, mb + lr, sb // 2, qlane, H)
                else:
                    _bulk_st(out_addr + (mb + lr).to(tl.int64) * rb + lane, lo, hi)
                    _rbulk_st(ag + lr.to(tl.int64) * rb + lane, lo, hi)
    _signal(succ_flag, fbase + WORLD)

    # AG steps: chunk order[p - t] arrives in slot W - 2 + t; keep it, pass it on
    for t in range(1, WORLD):
        _wait(my_flag, fbase + WORLD - 1 + t, err_ptr, max_spin)
        k = tl.load(ring + (p + WORLD - t) % WORLD)
        b = tl.minimum(k * chunk, num_tokens)
        n = tl.minimum((k + 1) * chunk, num_tokens) - b
        for tt in range(pid, tl.cdiv(n, BM), G):
            for lr in range(tt * BM, tl.minimum(tt * BM + BM, n)):
                if QUANT:
                    width: tl.constexpr = H
                    qlane = ((tl.arange(0, H // 16) // 32) * 512).to(tl.int64)
                    soff = sb // 2 + lr.to(tl.int64) * 4
                    scale = tl.load((mine + (WORLD - 2 + t) * sb + soff).to(tl.pointer_type(tl.float32)), volatile=True)
                    tl.store(out_scale + b + lr, scale)
                    if t < WORLD - 1:
                        tl.store((nxt + (WORLD - 1 + t) * sb + soff).to(tl.pointer_type(tl.float32)), scale, cache_modifier=".wt")
                    o = lr.to(tl.int64) * width + qlane
                    dst = out_addr + (b + lr).to(tl.int64) * width + qlane
                else:
                    o = lr.to(tl.int64) * rb + lane
                    dst = out_addr + (b + lr).to(tl.int64) * rb + lane
                src = mine + (WORLD - 2 + t) * sb + o
                lo = _ldv_lo(src)
                hi = _ldv_hi(src)
                _bulk_st(dst, lo, hi)
                if t < WORLD - 1:
                    _rbulk_st(nxt + (WORLD - 1 + t) * sb + o, lo, hi)
        if t < WORLD - 1:
            _signal(succ_flag, fbase + WORLD + t)


def pad_fn(fn: torch.Tensor) -> torch.Tensor:
    """[24, 4H] fp32 -> [32, 4H] in the kernel's row order (comb, pre, post, zeros)."""
    out = torch.zeros(MIX_PAD, fn.shape[1], dtype=torch.float32, device=fn.device)
    out[:MIX] = fn[_FN_ORDER]
    return out


class fused_rs_mhc_prenorm_big_ag(_RingComm):
    """IPC buffers + launcher of ``rs_mhc_ag_ring_bulk_kernel`` for one TP group.

    max_tokens: largest T of a call, sizes the data buffer: 2 parities x (2W - 2) slots of
    ceil(max_tokens / W) rows of H bf16. Ring arguments: see ``_RingComm``.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        max_tokens: int,
        hidden_size: int = 4096,
        hc_mult: int = 4,
        cpu_group: Optional[dist.ProcessGroup] = None,
        rings: Optional[Sequence[Sequence[int]]] = None,
        nchannels: int = 28,
        max_spin: int = 1 << 22,
        block_m: Optional[int] = None,
        block_k: int = 64,
        max_groups: int = 4,
    ):
        H, W = hidden_size, dist.get_world_size(group)
        nv = H // 8
        if hc_mult != HC or H % 256 or nv & (nv - 1) or nv > 1024:
            raise ValueError(f"needs hc_mult 4 and a row of H bf16 = one bulk tile, got {hc_mult} {H}")
        self.hidden_size = H
        self.max_tokens = max_tokens
        # An explicit BM retains the independent-tile configuration for A/B tests.
        self.splits = 16 if block_m is None and H == 4096 else 1
        self.block_m = block_m or (8 if self.splits > 1 else 16)
        self.block_k = block_k
        if max_groups < 1:
            raise ValueError("max_groups must be positive")
        self.max_groups = max_groups
        if (self.block_m < 1 or self.block_m & (self.block_m - 1) or block_k < 16
                or block_k & (block_k - 1) or 4 * H % (self.splits * block_k)):
            raise ValueError("invalid owner tile or split-K width")
        cpu = cpu_group or group
        rank = dist.get_rank(group)
        for turn in (0, 1):  # rank 0 compiles the wrappers, the others read the cache
            if (rank == 0) == (turn == 0):
                self.bulk_lib = build_bulk_lib()
            dist.barrier(group=cpu)
        install_bulk_ir_fix()
        self.slot_elems = (max_tokens + W - 1) // W * H
        data_bytes = 2 * (2 * W - 2) * self.slot_elems * 2
        self._setup(group, device, data_bytes, cpu_group, rings, nchannels, max_spin)
        tiles = triton.cdiv(self.slot_elems // H, self.block_m * self.splits)
        slots = tiles * self.splits if self.splits > 1 else 1
        self.partial = torch.empty(slots * self.block_m * self.splits * 25 if self.splits > 1 else 1,
                                   dtype=torch.float32, device=self.device)
        self.post_ready = torch.zeros(slots, dtype=torch.int32, device=self.device)
        self.mix_ready = torch.zeros_like(self.post_ready)
        self._fn_cache = {}

    def launch_shape(self, num_tokens: int) -> Tuple[int, int]:
        """(channels, programs) of a call; the same on every rank."""
        chunk = (num_tokens + self.world_size - 1) // self.world_size
        if self.splits > 1:
            # The cooperative launch limit on 810E is far below the occupancy API once the
            # kernel uses MMA: measured 80 CTAs at 4 warps, 40 at 8, 20 at 16 (SPLITS=16,
            # 256 vregs). 4 warps therefore allows 4 groups = 64 CTAs, one per SM.
            groups = min(self.max_groups, self.max_programs // self.splits, max(1, triton.cdiv(chunk, self.block_m * self.splits)))
            if not groups:
                raise RuntimeError("split-K needs at least 16 resident blocks")
            grid = groups * self.splits
            return min(grid, self.nchannels), grid
        tiles = triton.cdiv(chunk, self.block_m)
        nchan = max(1, min(self.nchannels, tiles))
        per_chan = max(1, min(self.max_programs // nchan, triton.cdiv(tiles, nchan)))
        return nchan, nchan * per_chan

    def _fn(self, fn: torch.Tensor) -> torch.Tensor:
        key = (fn.data_ptr(), fn._version)
        if key not in self._fn_cache:
            self._fn_cache = {key: pad_fn(fn)}
        return self._fn_cache[key]

    def _launch(self, x, out, res, post, comb, fn, scale, base, w, dump, eps, repeat, d_only, out_scale=None):
        T, H = x.shape
        W = self.world_size
        parity, fbase = self._next_call(2 * W - 2)
        nchan, grid = self.launch_shape(T)
        for t in (x, out, res, w):
            if t.numel() and t.data_ptr() % 512:
                raise ValueError("bulk tensors must be 512B aligned")
        slot_bytes = self.slot_elems * 2
        V = H // 8
        with torch.cuda.device(x.device):
            self.last_kernel = rs_mhc_ag_ring_bulk_kernel[(grid,)](
                x.data_ptr(), res.data_ptr(), w.data_ptr(), out.data_ptr(),
                res, post, comb, fn, scale, base, dump,
                self.partial, self.post_ready, self.mix_ready, out_scale if out_scale is not None else self.partial,
                self.data_addrs, self.flag_ptrs, self.order, self.pos, self.err,
                T, (T + W - 1) // W, slot_bytes, parity * (2 * W - 2) * slot_bytes,
                self.rank, nchan, *eps, fbase, self.max_spin,
                WORLD=W, H=H, V=V, BM=self.block_m, BK=self.block_k, REPEAT=repeat,
                D_ONLY=d_only, DUMP=dump.numel() > 1,
                SPLITS=self.splits, QUANT=out_scale is not None,
                num_warps=SPLIT_WARPS if self.splits > 1 else V // 32,
                launch_cooperative_grid=self.splits > 1, extern_libs={"ppubulk": self.bulk_lib},
            )
        return out

    def _check(self, x, out, quant=False):
        assert not self.closed
        T, H = x.shape
        if H != self.hidden_size or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError(f"x must be contiguous [T, {self.hidden_size}] bf16")
        if T > self.max_tokens:
            raise ValueError(f"{T} tokens > max_tokens {self.max_tokens}")
        dtype = torch.int8 if quant else torch.bfloat16
        if out is None:
            out = torch.empty_like(x, dtype=dtype)
        elif out.shape != x.shape or out.dtype != dtype or out.device != x.device or not out.is_contiguous():
            raise ValueError(f"bad out {tuple(out.shape)}")
        return out

    def ring_d(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """d = bf16 sum of x over the ranks, through the same ring (RS + AG of d)."""
        out = self._check(x, out)
        if x.shape[0] == 0:
            return out
        z = torch.zeros(1, device=x.device, dtype=torch.float32)
        return self._launch(x, out, x, z, z, z, z, z, x, z, (0.0,) * 5, 1, True)

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        norm_eps: float,
        out: Optional[torch.Tensor] = None,
        dump: Optional[torch.Tensor] = None,
        *, quant: bool = False,
    ):
        """x: [T, H] partial sums. residual [Tl, 4, H] bf16, post [Tl, 4] / comb [Tl, 4, 4]
        fp32 are this rank's rows and get the new values in place. Returns y [T, H].
        quant=True returns (q[T,H] int8, scale[T,1] fp32) on every rank.
        dump: optional [Tl, 25] fp32 receiving the raw GEMM mix (24) and sq."""
        if quant and self.hidden_size % 512:
            raise ValueError("INT8 bulk AG needs a 512-byte aligned row")
        out = self._check(x, out, quant)
        T, H = x.shape
        out_scale = torch.empty((T, 1), dtype=torch.float32, device=x.device) if quant else None
        b, e = self.row_range(T)
        Tl = e - b
        if tuple(residual.shape) != (Tl, HC, H) or residual.dtype != torch.bfloat16:
            raise ValueError(f"residual must be [{Tl}, 4, {H}] bf16, got {tuple(residual.shape)}")
        if tuple(post.shape) != (Tl, HC) or tuple(comb.shape) != (Tl, HC, HC):
            raise ValueError(f"post must be [{Tl}, 4], comb [{Tl}, 4, 4]")
        for t in (residual, post, comb, norm_weight):
            if not t.is_contiguous():
                raise ValueError("tensors must be contiguous")
        if post.dtype != torch.float32 or comb.dtype != torch.float32:
            raise ValueError("post / comb must be fp32")
        if T == 0:
            return (out, out_scale) if quant else out
        if dump is None:
            dump = torch.zeros(1, device=x.device, dtype=torch.float32)
        if Tl == 0:  # zero-row owner: any valid addresses, never touched
            residual, post, comb = x, dump, dump
        eps = (rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, norm_eps)
        self._launch(
            x, out, residual, post, comb, self._fn(fn), hc_scale.float(), hc_base.float(),
            norm_weight.bfloat16(), dump, eps, sinkhorn_repeat, False, out_scale,
        )
        return (out, out_scale) if quant else out
