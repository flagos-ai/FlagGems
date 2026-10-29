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

"""Fused reduce-scatter + residual add + RMSNorm + all-gather for pure TP, on rings.

Replaces ``all_reduce(x); x, residual = fused_add_rms_norm(x, residual)``. Rank r owns
rows [r * chunk, min(T, (r + 1) * chunk)), chunk = ceil(T / W): the norm only runs on
those rows and the residual stays scattered (each rank keeps its own rows).

Like pccl, the data moves on C channels at once; channel c is a ring order[c] over the
ranks (on 810E the rings of pccl's 28-channel graph, so every hop is an ICN link).
Program pid belongs to channel pid % C and handles local rows pid, pid + G, ... of every
chunk. Ring position p, predecessor / successor in order[c]:
  RS step s = 0 .. W-2   send x + received partial of chunk order[p-s-1] -> succ slot s
  owner                  slot W-2 + my x + residual, RMSNorm, keep the residual
  AG step t = 0 .. W-2   forward my normed chunk (t = 0) or the one received -> succ slot W-1+t
Every send is followed by one flag increment on the successor (2W - 2 per call), so the
receiver waits for flag >= base + step. Flags only grow; the parity flips every call, so
a predecessor that runs ahead into call n + 1 never touches what call n still reads.
Every rank must make the same calls in the same order on one stream (eager only).

gather=False stops after the owner step (W - 1 signals): RS + add + RMSNorm, every rank
keeps its normed rows. fused_qkv_norm_ag is the matching other end: it norms q / kv of
the scattered qkv_a output and runs only the AG half of the ring (W - 1 signals).
"""

import ctypes
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language import core

logger = logging.getLogger(__name__)


# ---- PPU warp-level 512B bulk accesses (pccl's SIMPLE_BULK data path) ----
# All 32 lanes of a warp pass the same 512B-aligned base, lane L handles bytes [16L, 16L+16).
# A remote bulk store lays the 512B out lane-interleaved in the peer's memory and only a
# bulk load reads it back in register order, so every row access of the bulk kernel goes
# through these ops; registers then hold one fixed permutation of the row, which the
# element-wise math and the row sum do not care about.
# Triton cannot emit them: build_bulk_lib() compiles extern "C" wrappers into a bitcode
# library for extern_libs, install_bulk_ir_fix() restores the pointer type that Triton's
# opaque-pointer IR loses on the way to ppu-llc (else "Intrinsic has incorrect argument type").
_BULK_SRC = r"""
#include <cuda_runtime.h>
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ < 150
#define RMT_ST(p, v) __ppu_remote_store_bulk_ncp_b32x4((void*)(p), v)
#else
#define RMT_ST(p, v) __ppu_remote_store_bulk_ncp_b32x4((void*)(p), v, 0)
#endif
typedef unsigned long long u64;
#define U4(lo, hi) make_uint4((unsigned)(lo), (unsigned)((u64)(lo) >> 32), (unsigned)(hi), (unsigned)((u64)(hi) >> 32))
#define LO(v) (long long)(((u64)(v).y << 32) | (v).x)
#define HI(v) (long long)(((u64)(v).w << 32) | (v).z)
#define API extern "C" __device__ __attribute__((used))
API int ppu_rbulk_st(long long p, long long lo, long long hi) { RMT_ST(p, U4(lo, hi)); return 0; }
API int ppu_bulk_st(long long p, long long lo, long long hi) { __ppu_global_stcg_bulk_b32x4((void*)p, U4(lo, hi)); return 0; }
API long long ppu_bulk_ld_lo(long long p) { return LO(__ppu_global_ldcg_bulk_b32x4((const void*)p)); }
API long long ppu_bulk_ld_hi(long long p) { return HI(__ppu_global_ldcg_bulk_b32x4((const void*)p)); }
// written by peers: bypass L1
API long long ppu_bulk_ldv_lo(long long p) { return LO(__ppu_global_ldcv_bulk_b32x4((const void*)p)); }
API long long ppu_bulk_ldv_hi(long long p) { return HI(__ppu_global_ldcv_bulk_b32x4((const void*)p)); }
"""

_BULK_IR_FIX = r'''
import re, sys
p = sys.argv[1]
s = open(p).read()
NAME = re.compile(r"@llvm\.ppu\.[\w.]*bulk[\w.]*?(b32x4|v4i32|b32x2|v2i32)[\w.]*\(")
P8 = "i8 addrspace(1)*"
out, n = [], 0
for line in s.split("\n"):
    m = NAME.search(line)
    if P8 in line and m:
        P = "<%s x i32> addrspace(1)*" % ("4" if m.group(1) in ("b32x4", "v4i32") else "2")
        if line.lstrip().startswith("declare"):
            line = line.replace(P8, P)
        else:
            ind = line[: len(line) - len(line.lstrip())]
            for v in re.findall(re.escape(P8) + r" (%[\w.]+)", line):
                c = "%%bulkc%d" % n
                n += 1
                out.append("%s%s = bitcast %s %s to %s" % (ind, c, P8, v, P))
                line = line.replace("%s %s" % (P8, v), "%s %s" % (P, c), 1)
    out.append(line)
open(p, "w").write("\n".join(out))
'''


def _bulk_cache_dir() -> str:
    return os.environ.get("FLAGGEMS_PPU_BULK_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "flag_gems", "ppu_bulk"
    )


def build_bulk_lib() -> str:
    """Bitcode of the bulk wrappers (nvcc, cached by source hash). Returns its path."""
    d = _bulk_cache_dir()
    key = hashlib.sha256((_BULK_SRC + "alwaysinline-v4").encode()).hexdigest()[:16]
    out = os.path.join(d, f"ppu_bulk_{key}.bc")
    if os.path.exists(out):
        return out
    os.makedirs(d, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=d) as tmp:
        src = os.path.join(tmp, "ppu_bulk.cu")
        with open(src, "w") as f:
            f.write(_BULK_SRC)
        cmd = ["nvcc", "-arch=sm_80", "-rdc=true", "-c", src, "-o", os.path.join(tmp, "ppu_bulk.o")]
        subprocess.run(cmd + ["-keep", "-keep-dir", tmp], check=True, capture_output=True, cwd=tmp)
        bcs = [f for f in os.listdir(tmp) if f.endswith("-ppu001-opt.bc")]
        if not bcs:
            raise RuntimeError(f"nvcc left no ppu001 bitcode in {tmp}: {os.listdir(tmp)}")
        # the wrappers carry target-cpu/target-features, the Triton kernel does not, and
        # LLVM refuses to inline across that: every access stayed a function call
        sdk = "/usr/local/PPU_SDK/bin"
        ll = os.path.join(tmp, "ppu_bulk.ll")
        subprocess.run([f"{sdk}/llvm-dis", os.path.join(tmp, bcs[0]), "-o", ll], check=True)
        text = open(ll).read()
        text = re.sub(r' "target-(cpu|features)"="[^"]*"', "", text)
        # the O3 inliner still keeps them as calls: force it
        for g in set(re.findall(r"^define \S+ @ppu_\w+\([^)]*\) #(\d+)", text, re.M)):
            text = text.replace(f"attributes #{g} = {{ ", f"attributes #{g} = {{ alwaysinline ", 1)
        open(ll, "w").write(text)
        subprocess.run([f"{sdk}/llvm-as", ll, "-o", out + f".{os.getpid()}"], check=True)
        os.replace(out + f".{os.getpid()}", out)
    return out


_BULK_IR_FIX_INSTALLED = False


def install_bulk_ir_fix():
    """Wrap the PPU backend's irformatter so its output gets the bulk pointer types back.
    Only lines calling or declaring b32x4 bulk intrinsics change."""
    global _BULK_IR_FIX_INSTALLED
    if _BULK_IR_FIX_INSTALLED:
        return
    import triton.backends.ppu.compiler as ppu_compiler

    real = ppu_compiler.get_irformatter()
    d = _bulk_cache_dir()
    os.makedirs(d, exist_ok=True)
    fix = os.path.join(d, "bulk_ir_fix.py")
    wrap = os.path.join(d, f"irformatter_bulk_{os.getpid()}.sh")
    with open(fix + f".{os.getpid()}", "w") as f:
        f.write(_BULK_IR_FIX)
    os.replace(fix + f".{os.getpid()}", fix)
    with open(wrap, "w") as f:
        f.write(
            "#!/bin/bash\n"
            f'"{real}" "$@" || exit $?\n'
            'out=""; prev=""\n'
            'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
            f'exec "{sys.executable}" "{fix}" "$out"\n'
        )
    os.chmod(wrap, 0o755)
    ppu_compiler.get_irformatter = lambda: wrap
    _BULK_IR_FIX_INSTALLED = True


_I32, _I64 = core.dtype("int32"), core.dtype("int64")


def _ext(name, args, ret, _semantic):
    # is_pure=False: stores have no used result, peer-slot loads must stay after the wait
    sig = tuple(_I64 for _ in args)
    return core.extern_elementwise("", "", args, {sig: (name, ret)}, is_pure=False, _semantic=_semantic)


@core.extern
def _rbulk_st(p, lo, hi, _semantic=None):
    return _ext("ppu_rbulk_st", [p, lo, hi], _I32, _semantic)


@core.extern
def _bulk_st(p, lo, hi, _semantic=None):
    return _ext("ppu_bulk_st", [p, lo, hi], _I32, _semantic)


@core.extern
def _ld_lo(p, _semantic=None):
    return _ext("ppu_bulk_ld_lo", [p], _I64, _semantic)


@core.extern
def _ld_hi(p, _semantic=None):
    return _ext("ppu_bulk_ld_hi", [p], _I64, _semantic)


@core.extern
def _ldv_lo(p, _semantic=None):
    return _ext("ppu_bulk_ldv_lo", [p], _I64, _semantic)


@core.extern
def _ldv_hi(p, _semantic=None):
    return _ext("ppu_bulk_ldv_hi", [p], _I64, _semantic)


@triton.jit
def _bf16x2(w):
    # int32 holding 2 bf16 -> [.., 2] fp32
    return tl.join((w << 16).to(tl.float32, bitcast=True), (w & -65536).to(tl.float32, bitcast=True))


@triton.jit
def _unpack(lo, hi):
    # 16B as two int64 -> [V, 2, 2, 2] fp32; _pack inverts it
    a = tl.join(_bf16x2(lo.to(tl.int32)), _bf16x2((lo >> 32).to(tl.int32)))
    b = tl.join(_bf16x2(hi.to(tl.int32)), _bf16x2((hi >> 32).to(tl.int32)))
    return tl.join(a, b)


@triton.jit
def _word(x2):
    lo, hi = tl.split(x2)
    lo = lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    return lo | (hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16)


@triton.jit
def _q(x):
    a, b = tl.split(x)
    return (_word(a).to(tl.int64) & 0xFFFFFFFF) | (_word(b).to(tl.int64) << 32)


@triton.jit
def _pack(x):
    a, b = tl.split(x)
    return _q(a), _q(b)


@triton.jit
def _ld(p):
    return _unpack(_ld_lo(p), _ld_hi(p))


@triton.jit
def _ldv(p):
    return _unpack(_ldv_lo(p), _ldv_hi(p))


@triton.jit
def _signal(flag, value):
    # all stores of this program are issued before thread 0 releases the flag
    tl.debug_barrier()
    tl.atomic_xchg(flag, value, sem="release", scope="sys")


@triton.jit
def _wait(flag, target, err_ptr, max_spin):
    # cas(-1, -1) is an acquire read that is not hoisted out of the loop (an
    # atomic_add(0) is); the difference keeps the compare right across int32 wrap
    i = 0
    v = tl.atomic_cas(flag, -1, -1, sem="acquire", scope="sys")
    while ((v - target) < 0) & (i < max_spin):
        v = tl.atomic_cas(flag, -1, -1, sem="acquire", scope="sys")
        i += 1
    if i >= max_spin:
        tl.atomic_add(err_ptr, 1)
    tl.debug_barrier()


@triton.jit(
    do_not_specialize=[
        "x_addr", "res_addr", "w_addr", "out_addr", "num_tokens", "chunk", "slot_bytes",
        "par_off", "rank", "nchan", "eps", "base", "max_spin",
    ]
)
def rs_norm_ag_ring_bulk_kernel(
    x_addr,  # int64 byte addresses, all 512B aligned: [T, H] partial sums
    res_addr,  # [rows of this rank, H], updated in place
    w_addr,  # [H]
    out_addr,  # [T, H], may alias x
    data_addrs,  # int64 [W], every rank's IPC data buffer
    flag_ptrs,  # int64 [W], every rank's IPC flag buffer (one int32 per program)
    order_ptr,  # int32 [C, W], ring of every channel
    pos_ptr,  # int32 [C], my position in every ring
    err_ptr,  # int32 [1], wait timeouts
    num_tokens,
    chunk,
    H,
    slot_bytes,
    par_off,  # parity * (2W - 2) * slot_bytes
    rank,
    nchan,
    eps,
    base,  # flag value at the start of this call (int32, wraps)
    max_spin,
    WORLD: tl.constexpr,
    V: tl.constexpr,  # threads per program, one 16B per thread: a row is NT tiles of V * 16B
    NT: tl.constexpr,
    DO_AG: tl.constexpr,  # False: stop after the norm, out holds only my rows
):
    # every row access is a warp bulk op;
    # tiles cover the row exactly, so every warp is full and no lane is masked
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = ((tl.arange(0, V) // 32) * 512).to(tl.int64)
    rb = (H * 2).to(tl.int64)
    sb = slot_bytes.to(tl.int64)

    c = pid % nchan
    p = tl.load(pos_ptr + c)
    ring = order_ptr + c * WORLD
    succ = tl.load(ring + (p + 1) % WORLD)
    my_flag = tl.load(flag_ptrs + rank).to(tl.pointer_type(tl.int32)) + pid
    succ_flag = tl.load(flag_ptrs + succ).to(tl.pointer_type(tl.int32)) + pid
    mine = tl.load(data_addrs + rank) + par_off.to(tl.int64)
    nxt = tl.load(data_addrs + succ) + par_off.to(tl.int64)

    # RS step 0: my partial sums of chunk order[p - 1] -> succ slot 0
    k = tl.load(ring + (p + WORLD - 1) % WORLD)
    b = tl.minimum(k * chunk, num_tokens)
    n = tl.minimum((k + 1) * chunk, num_tokens) - b
    for lr in range(pid, n, G):
        for j in tl.static_range(NT):
            o = j * V * 16 + lane
            xp = x_addr + (b + lr).to(tl.int64) * rb + o
            _rbulk_st(nxt + lr.to(tl.int64) * rb + o, _ld_lo(xp), _ld_hi(xp))
    _signal(succ_flag, base + 1)

    # RS steps 1 .. W-2: add my partial sums to the received chunk, pass it on
    for s in range(1, WORLD - 1):
        _wait(my_flag, base + s, err_ptr, max_spin)
        k = tl.load(ring + (p + 2 * WORLD - s - 1) % WORLD)
        b = tl.minimum(k * chunk, num_tokens)
        n = tl.minimum((k + 1) * chunk, num_tokens) - b
        for lr in range(pid, n, G):
            for j in tl.static_range(NT):
                o = lr.to(tl.int64) * rb + j * V * 16 + lane
                acc = _ldv(mine + (s - 1) * sb + o) + _ld(x_addr + (b + lr).to(tl.int64) * rb + j * V * 16 + lane)
                lo, hi = _pack(acc)
                _rbulk_st(nxt + s * sb + o, lo, hi)
        _signal(succ_flag, base + s + 1)

    # owner: the last hop brings the sum of the other W - 1 ranks of my rows; pass 1 sums
    # the squares, pass 2 rebuilds x (same bits) and writes residual / out / succ
    _wait(my_flag, base + WORLD - 1, err_ptr, max_spin)
    mb = tl.minimum(rank * chunk, num_tokens)
    mn = tl.minimum((rank + 1) * chunk, num_tokens) - mb
    for lr in range(pid, mn, G):
        row = lr.to(tl.int64) * rb
        g = (mb + lr).to(tl.int64) * rb
        sq = 0.0
        for j in tl.static_range(NT):
            o = j * V * 16 + lane
            x = _ldv(mine + (WORLD - 2) * sb + row + o) + _ld(x_addr + g + o) + _ld(res_addr + row + o)
            sq += tl.sum(x * x)
        rrms = 1 / tl.sqrt(sq / H + eps)
        for j in tl.static_range(NT):
            o = j * V * 16 + lane
            x = _ldv(mine + (WORLD - 2) * sb + row + o) + _ld(x_addr + g + o) + _ld(res_addr + row + o)
            lo, hi = _pack(x)
            _bulk_st(res_addr + row + o, lo, hi)
            lo, hi = _pack(x * rrms * _ld(w_addr + o))
            if DO_AG:
                _bulk_st(out_addr + g + o, lo, hi)
                _rbulk_st(nxt + (WORLD - 1) * sb + row + o, lo, hi)
            else:
                _bulk_st(out_addr + row + o, lo, hi)
    if DO_AG:
        _signal(succ_flag, base + WORLD)

        # AG steps: chunk order[p - t] arrives in slot W - 2 + t; keep it, pass it on
        for t in range(1, WORLD):
            _wait(my_flag, base + WORLD - 1 + t, err_ptr, max_spin)
            k = tl.load(ring + (p + WORLD - t) % WORLD)
            b = tl.minimum(k * chunk, num_tokens)
            n = tl.minimum((k + 1) * chunk, num_tokens) - b
            for lr in range(pid, n, G):
                for j in tl.static_range(NT):
                    o = lr.to(tl.int64) * rb + j * V * 16 + lane
                    src = mine + (WORLD - 2 + t) * sb + o
                    lo = _ldv_lo(src)
                    hi = _ldv_hi(src)
                    _bulk_st(out_addr + (b + lr).to(tl.int64) * rb + j * V * 16 + lane, lo, hi)
                    if t < WORLD - 1:
                        _rbulk_st(nxt + (WORLD - 1 + t) * sb + o, lo, hi)
            if t < WORLD - 1:
                _signal(succ_flag, base + WORLD + t)


@triton.jit(
    do_not_specialize=[
        "num_tokens", "chunk", "slot_elems", "par_elems", "rank", "nchan", "q_eps", "kv_eps",
        "base", "max_spin",
    ]
)
def qkv_norm_ag_ring_kernel(
    x_ptr,  # [rows of this rank, D] qkv_a output: [q | kv | rest]
    qw_ptr,  # [Q]
    kvw_ptr,  # [KV]
    out_ptr,  # [T, D]
    data_addrs,  # int64 [W], every rank's IPC data buffer
    flag_ptrs,  # int64 [W], every rank's IPC flag buffer (one int32 per program)
    order_ptr,  # int32 [C, W], ring of every channel
    pos_ptr,  # int32 [C], my position in every ring
    err_ptr,  # int32 [1], wait timeouts
    num_tokens,
    chunk,
    slot_elems,
    par_elems,  # parity * (W - 1) * slot_elems
    rank,
    nchan,
    q_eps,
    kv_eps,
    base,  # flag value at the start of this call (int32, wraps)
    max_spin,
    WORLD: tl.constexpr,
    D: tl.constexpr,
    Q: tl.constexpr,
    KV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # owner: norm q and kv of my rows, keep them, send them to succ slot 0;
    # AG step t = 1 .. W-1: chunk order[p - t] arrives in slot t - 1, keep it, pass it on.
    # A row (D * 2B) is not a multiple of 512B, so plain accesses, no bulk ops.
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    in_row = cols < D
    is_q = cols < Q
    is_kv = (cols >= Q) & (cols < Q + KV)

    c = pid % nchan
    p = tl.load(pos_ptr + c)
    ring = order_ptr + c * WORLD
    succ = tl.load(ring + (p + 1) % WORLD)
    my_flag = tl.load(flag_ptrs + rank).to(tl.pointer_type(tl.int32)) + pid
    succ_flag = tl.load(flag_ptrs + succ).to(tl.pointer_type(tl.int32)) + pid
    pe = par_elems.to(tl.int64)
    se = slot_elems.to(tl.int64)
    mine = tl.load(data_addrs + rank).to(tl.pointer_type(tl.bfloat16)) + pe
    nxt = tl.load(data_addrs + succ).to(tl.pointer_type(tl.bfloat16)) + pe

    wq = tl.load(qw_ptr + cols, mask=is_q, other=0.0).to(tl.float32)
    wkv = tl.load(kvw_ptr + (cols - Q), mask=is_kv, other=0.0).to(tl.float32)
    w = tl.where(is_q, wq, wkv)

    mb = tl.minimum(rank * chunk, num_tokens)
    mn = tl.minimum((rank + 1) * chunk, num_tokens) - mb
    for lr in range(pid, mn, G):
        x = tl.load(x_ptr + lr.to(tl.int64) * D + cols, mask=in_row, other=0.0).to(tl.float32)
        xx = x * x
        rq = 1 / tl.sqrt(tl.sum(tl.where(is_q, xx, 0.0)) / Q + q_eps)
        rkv = 1 / tl.sqrt(tl.sum(tl.where(is_kv, xx, 0.0)) / KV + kv_eps)
        y = tl.where(is_q, x * rq * w, tl.where(is_kv, x * rkv * w, x)).to(tl.bfloat16)
        tl.store(out_ptr + (mb + lr).to(tl.int64) * D + cols, y, mask=in_row)
        dst = tl.multiple_of(nxt + lr.to(tl.int64) * D, 16)
        tl.store(dst + cols, y, mask=in_row)
    _signal(succ_flag, base + 1)

    for t in range(1, WORLD):
        _wait(my_flag, base + t, err_ptr, max_spin)
        k = tl.load(ring + (p + WORLD - t) % WORLD)
        b = tl.minimum(k * chunk, num_tokens)
        n = tl.minimum((k + 1) * chunk, num_tokens) - b
        for lr in range(pid, n, G):
            src = tl.multiple_of(mine + (t - 1) * se + lr.to(tl.int64) * D, 16)
            v = tl.load(src + cols, mask=in_row, cache_modifier=".cg")
            tl.store(out_ptr + (b + lr).to(tl.int64) * D + cols, v, mask=in_row)
            if t < WORLD - 1:
                dst = tl.multiple_of(nxt + t * se + lr.to(tl.int64) * D, 16)
                tl.store(dst + cols, v, mask=in_row)
        if t < WORLD - 1:
            _signal(succ_flag, base + t + 1)


# pccl's 810E graph (pccl/liuda/810e_topo/28channnel/ring.xml): 7 distinct rings,
# each used by 4 of the 28 channels. Every hop is an ICN link.
PPU810E_RINGS = [
    [0, 1, 2, 6, 5, 4, 7, 3, 10, 14, 13, 12, 15, 11, 8, 9],
    [0, 1, 2, 6, 5, 4, 7, 14, 13, 12, 15, 11, 8, 9, 10, 3],
    [0, 3, 7, 4, 5, 6, 2, 11, 15, 12, 13, 14, 10, 9, 8, 1],
    [0, 3, 7, 4, 5, 12, 8, 10, 11, 9, 13, 14, 15, 6, 2, 1],
    [0, 2, 3, 1, 8, 11, 15, 13, 9, 10, 14, 12, 5, 7, 6, 4],
    [0, 9, 11, 10, 8, 12, 13, 4, 6, 15, 14, 7, 5, 1, 3, 2],
    [0, 4, 13, 15, 12, 14, 10, 9, 8, 11, 2, 1, 5, 6, 7, 3],
]
# ICN neighbours of every PPU (topo_rank0000.xml): 2 links of 106 GB/s, 3 of 53
PPU810E_ICN = {
    0: (1, 2, 3, 4, 9),
    1: (0, 2, 3, 5, 8),
    2: (0, 1, 3, 6, 11),
    3: (0, 1, 2, 7, 10),
    4: (0, 5, 6, 7, 13),
    5: (1, 4, 6, 7, 12),
    6: (2, 4, 5, 7, 15),
    7: (3, 4, 5, 6, 14),
    8: (1, 9, 10, 11, 12),
    9: (0, 8, 10, 11, 13),
    10: (3, 8, 9, 11, 14),
    11: (2, 8, 9, 10, 15),
    12: (5, 8, 13, 14, 15),
    13: (4, 9, 12, 14, 15),
    14: (7, 10, 12, 13, 15),
    15: (6, 11, 12, 13, 14),
}


def ppu810e_rings(phys_ids: Sequence[int]) -> List[List[int]]:
    """pccl's 810E rings restricted to the PPUs of the group (rank i = phys_ids[i]).

    Keeps the rings whose every hop is still an ICN link after dropping the other PPUs;
    duplicates stay, so a ring keeps its pccl weight. Returns rank-space rings."""
    phys = list(phys_ids)
    where = {d: i for i, d in enumerate(phys)}
    rings, fallback = [], []
    for full in PPU810E_RINGS:
        r = [d for d in full if d in where]
        ok = all(r[(i + 1) % len(r)] in PPU810E_ICN[r[i]] for i in range(len(r)))
        (rings if ok else fallback).append([where[d] for d in r])
    if not rings:
        logger.warning("no pccl ring stays on ICN for PPUs %s, some hops go over PCIe", phys)
        rings = fallback
    return rings


class _CudaIpcHandle(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 64)]


class _CudaRT:
    """The libcudart torch already loaded, so allocations share its runtime state."""

    def __init__(self):
        path = None
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudart" in line:
                    path = line[line.index("/") :].strip()
                    break
        self.lib = ctypes.CDLL(path or "libcudart.so")
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, err):
        if err != 0:
            msg = self.lib.cudaGetErrorString(err).decode()
            raise RuntimeError(f"CUDA runtime error {err}: {msg}")

    def set_device(self, dev):
        self.check(self.lib.cudaSetDevice(ctypes.c_int(dev)))

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
        return ptr.value

    def memset(self, ptr, nbytes):
        self.check(
            self.lib.cudaMemset(ctypes.c_void_p(ptr), 0, ctypes.c_size_t(nbytes))
        )

    def free(self, ptr):
        self.check(self.lib.cudaFree(ctypes.c_void_p(ptr)))

    def ipc_get(self, ptr):
        handle = _CudaIpcHandle()
        self.check(self.lib.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(ptr)))
        return bytes(handle.internal)

    def ipc_open(self, raw):
        handle = _CudaIpcHandle()
        ctypes.memmove(handle.internal, raw, 64)
        ptr = ctypes.c_void_p()
        # 1 = cudaIpcMemLazyEnablePeerAccess
        self.check(self.lib.cudaIpcOpenMemHandle(ctypes.byref(ptr), handle, ctypes.c_uint(1)))
        return ptr.value

    def ipc_close(self, ptr):
        self.check(self.lib.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr)))


def row_range(num_tokens: int, world_size: int, rank: int) -> Tuple[int, int]:
    """[begin, end) rows of `rank`, chunk = ceil(T / W); trailing ranks may own 0 rows."""
    chunk = (num_tokens + world_size - 1) // world_size
    return min(num_tokens, rank * chunk), min(num_tokens, (rank + 1) * chunk)




def rs_norm_ag_buffer_bytes(
    world_size: int, max_tokens: int, hidden_size: int, dtype: torch.dtype
) -> int:
    """Data buffer size of one rank: 2 parities x (2W - 2) slots of ceil(T / W) rows."""
    elt = torch.empty((), dtype=dtype).element_size()
    W = world_size
    return 2 * (2 * W - 2) * math.ceil(max_tokens / W) * hidden_size * elt


def _wrap_i32(v: int) -> int:
    return (v + 2**31) % 2**32 - 2**31


class _RingComm:
    """IPC data / flag buffers and ring channels shared by the ring kernels below.

    rings: rank orders, one per distinct ring (e.g. ``ppu810e_rings(phys_ids)``); default
    is the identity ring. nchannels: channels per call, the rings are repeated to fill
    them (pccl uses 28 on 810E); a call runs min(nchannels, chunk) of them.
    cpu_group: a gloo group of the same ranks for the handle exchange (default: `group`).
    Creation runs collectives: call it on every rank with the same rings.

    Flags only grow: every call adds its number of signals to `_signals`, which is the
    same on every rank because every rank makes the same calls. The parity of the data
    slots flips every call.
    """

    def _setup(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        data_bytes: int,
        cpu_group: Optional[dist.ProcessGroup],
        rings: Optional[Sequence[Sequence[int]]],
        nchannels: int,
        max_spin: int,
    ):
        self.group = group
        self.cpu_group = cpu_group or group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.device = torch.device(device)
        self.max_spin = max_spin
        self._seq = 0
        self._signals = 0

        W = self.world_size
        if W < 2:
            raise ValueError(f"{type(self).__name__} needs at least 2 ranks")
        rings = [list(r) for r in (rings or [list(range(W))])]
        for r in rings:
            if sorted(r) != list(range(W)):
                raise ValueError(f"ring {r} is not a permutation of the {W} ranks")
        self.max_programs = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        # a program waits on the same pid of its predecessor: every program of the grid
        # must be resident at once, so at most one per SM
        self.nchannels = max(1, min(nchannels, self.max_programs))
        order = [rings[c % len(rings)] for c in range(self.nchannels)]
        self.rings = rings
        self.order = torch.tensor(order, dtype=torch.int32, device=self.device)
        self.pos = torch.tensor(
            [r.index(self.rank) for r in order], dtype=torch.int32, device=self.device
        )

        flag_bytes = self.max_programs * 4
        self._rt = _CudaRT()
        self._rt.set_device(self.device.index)
        self._data = self._rt.malloc(data_bytes)
        self._flags = self._rt.malloc(flag_bytes)
        self._rt.memset(self._flags, flag_bytes)
        torch.cuda.synchronize(self.device)

        mine = (
            self._rt.ipc_get(self._data),
            self._rt.ipc_get(self._flags),
            self.max_programs,
            order,
        )
        everyone = [None] * W
        dist.all_gather_object(everyone, mine, group=self.cpu_group)
        if len({e[2] for e in everyone}) != 1:
            raise RuntimeError("ranks have different SM counts, grids would not pair up")
        if any(e[3] != order for e in everyone):
            raise RuntimeError("ranks were given different rings")
        self._opened = []
        data_ptrs, flag_ptrs = [], []
        for r, (d, f, _, _) in enumerate(everyone):
            if r == self.rank:
                data_ptrs.append(self._data)
                flag_ptrs.append(self._flags)
            else:
                pd, pf = self._rt.ipc_open(d), self._rt.ipc_open(f)
                self._opened += [pd, pf]
                data_ptrs.append(pd)
                flag_ptrs.append(pf)
        assert all(p % 512 == 0 for p in data_ptrs), "IPC buffers must be 512B aligned"
        self.data_addrs = torch.tensor(data_ptrs, dtype=torch.int64, device=self.device)
        self.flag_ptrs = torch.tensor(flag_ptrs, dtype=torch.int64, device=self.device)
        self.err = torch.zeros(1, dtype=torch.int32, device=self.device)
        # every flag buffer is zeroed before any peer can signal into it
        dist.barrier(group=self.cpu_group)
        self.closed = False
        logger.info(
            "%s rank %d/%d: %.1f MiB data, %d channels on %d distinct rings",
            type(self).__name__,
            self.rank,
            W,
            data_bytes / 2**20,
            self.nchannels,
            len(rings),
        )

    def _next_call(self, signals: int) -> Tuple[int, int]:
        """(parity, flag base) of the next call, which signals `signals` times."""
        parity = self._seq & 1
        base = _wrap_i32(self._signals)
        self._seq += 1
        self._signals += signals
        return parity, base

    def row_range(self, num_tokens: int) -> Tuple[int, int]:
        return row_range(num_tokens, self.world_size, self.rank)

    def launch_shape(self, num_tokens: int) -> Tuple[int, int]:
        """(channels, programs) of a call with `num_tokens`; the same on every rank."""
        chunk = (num_tokens + self.world_size - 1) // self.world_size
        nchan = max(1, min(self.nchannels, chunk))
        per_chan = max(1, min(self.max_programs // nchan, triton.cdiv(chunk, nchan)))
        return nchan, nchan * per_chan

    def errors(self) -> int:
        """Wait timeouts so far (syncs the device)."""
        return int(self.err.item())

    def close(self):
        if getattr(self, "closed", True):
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu_group)
        for ptr in self._opened:
            self._rt.ipc_close(ptr)
        self._rt.free(self._data)
        self._rt.free(self._flags)
        self.closed = True


class fused_rs_add_rms_norm_ag(_RingComm):
    """IPC buffers + launcher of ``rs_norm_ag_ring_bulk_kernel`` for one TP group (bf16).

    gather=True (default): RS + add + RMSNorm + AG, replaces all_reduce + add norm.
    gather=False: RS + add + RMSNorm only, every rank keeps its normed rows (e.g. the
    next layer's input norm when the attention input stays scattered).
    max_tokens: largest T of a call, see ``rs_norm_ag_buffer_bytes``. Ring arguments: see
    ``_RingComm``.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        cpu_group: Optional[dist.ProcessGroup] = None,
        rings: Optional[Sequence[Sequence[int]]] = None,
        nchannels: int = 28,
        max_spin: int = 1 << 22,
    ):
        W, H = dist.get_world_size(group), hidden_size
        # one 16B per thread in whole warps
        if dtype != torch.bfloat16 or H % 256 != 0:
            raise ValueError(f"needs bf16 and H % 256 == 0, got {dtype} {H}")
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.max_tokens = max_tokens
        cpu = cpu_group or group
        # rank 0 compiles the wrappers, the others read the cache
        rank = dist.get_rank(group)
        for turn in (0, 1):
            if (rank == 0) == (turn == 0):
                self.bulk_lib = build_bulk_lib()
            dist.barrier(group=cpu)
        install_bulk_ir_fix()

        elt = torch.empty((), dtype=dtype).element_size()
        max_chunk = (max_tokens + W - 1) // W
        self.slot_elems = max_chunk * H
        self.parity_elems = (2 * W - 2) * self.slot_elems
        self._setup(
            group, device, 2 * self.parity_elems * elt, cpu_group, rings, nchannels, max_spin
        )

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: Optional[torch.Tensor] = None,
        gather: bool = True,
    ) -> torch.Tensor:
        """x: [T, H] partial sums, residual: this rank's rows (updated in place).
        gather=True: returns the normed full [T, H], written into x unless `out` is given.
        gather=False: returns the normed rows of this rank [e - b, H], written into
        x[b:e] unless `out` is given."""
        assert not self.closed
        T, H = x.shape
        W = self.world_size
        b, e = self.row_range(T)
        if H != self.hidden_size or x.dtype != self.dtype:
            raise ValueError(f"expected [*, {self.hidden_size}] {self.dtype}, got {x.shape} {x.dtype}")
        if T > self.max_tokens:
            raise ValueError(f"{T} tokens > max_tokens {self.max_tokens}")
        if tuple(residual.shape) != (e - b, H):
            raise ValueError(f"residual must hold rows [{b}, {e}), got {tuple(residual.shape)}")
        if not (x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
            raise ValueError("tensors must be contiguous")
        if out is None:
            out = x if gather else x[b:e]
        elif tuple(out.shape) != ((T, H) if gather else (e - b, H)):
            raise ValueError(f"bad out shape {tuple(out.shape)}")
        if T == 0:
            return out

        # RS + AG: 2W - 2 signals; RS only: the W - 1 RS hops
        parity, base = self._next_call(2 * W - 2 if gather else W - 1)
        chunk = (T + W - 1) // W
        nchan, grid = self.launch_shape(T)
        for t in (x, residual, weight, out):
            if t.numel() and t.data_ptr() % 512 != 0:
                raise ValueError("tensors must be 512B aligned (bulk ops)")
        slot_bytes = self.slot_elems * x.element_size()
        nv = H // 8
        V = min(1024, nv & -nv)  # largest power of 2 dividing the row's 16B count
        with torch.cuda.device(x.device):
            self.last_kernel = rs_norm_ag_ring_bulk_kernel[(grid,)](
                x.data_ptr(), residual.data_ptr() if residual.numel() else x.data_ptr(),
                weight.data_ptr(), out.data_ptr() if out.numel() else x.data_ptr(),
                self.data_addrs, self.flag_ptrs,
                self.order, self.pos, self.err, T, chunk, H, slot_bytes,
                parity * (2 * W - 2) * slot_bytes, self.rank, nchan, eps, base, self.max_spin,
                WORLD=W, V=V, NT=nv // V, DO_AG=gather, num_warps=V // 32,
                extern_libs={"ppubulk": self.bulk_lib},
            )
        return out


class fused_qkv_norm_ag(_RingComm):
    """IPC buffers + launcher of ``qkv_norm_ag_ring_kernel`` for one TP group (bf16).

    The qkv_a output of MLA attention when its input is scattered: every rank holds the
    rows row_range(T, W, rank) of [T, D] = [q | kv | rest], q has q_dim columns, kv
    kv_dim. RMSNorm q and kv of those rows (rest, e.g. k_pe, is copied as is) and
    all-gather them to every rank: replaces ``all_gather; q_a_layernorm; kv_a_layernorm``.
    Ring arguments: see ``_RingComm``.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        max_tokens: int,
        q_dim: int,
        kv_dim: int,
        row_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        cpu_group: Optional[dist.ProcessGroup] = None,
        rings: Optional[Sequence[Sequence[int]]] = None,
        nchannels: int = 28,
        max_spin: int = 1 << 22,
    ):
        W = dist.get_world_size(group)
        if dtype != torch.bfloat16:
            raise ValueError(f"needs bf16, got {dtype}")
        if q_dim <= 0 or kv_dim <= 0 or q_dim + kv_dim > row_dim:
            raise ValueError(f"bad split q {q_dim} + kv {kv_dim} of {row_dim}")
        # 16B aligned rows: whole-vector accesses
        if row_dim % 8 != 0:
            raise ValueError(f"row_dim {row_dim} must be a multiple of 8")
        self.q_dim, self.kv_dim, self.row_dim = q_dim, kv_dim, row_dim
        self.dtype = dtype
        self.max_tokens = max_tokens
        elt = torch.empty((), dtype=dtype).element_size()
        max_chunk = (max_tokens + W - 1) // W
        self.slot_elems = max_chunk * row_dim
        self.parity_elems = (W - 1) * self.slot_elems
        self._setup(
            group, device, 2 * self.parity_elems * elt, cpu_group, rings, nchannels, max_spin
        )

    def __call__(
        self,
        x: torch.Tensor,
        num_tokens: int,
        q_weight: torch.Tensor,
        q_eps: float,
        kv_weight: torch.Tensor,
        kv_eps: float,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [e - b, D] rows of this rank, num_tokens: T of the whole batch (same on
        every rank). Returns [T, D] with q and kv normed, in `out` if given."""
        assert not self.closed
        T, D, W = num_tokens, self.row_dim, self.world_size
        b, e = self.row_range(T)
        if tuple(x.shape) != (e - b, D) or x.dtype != self.dtype:
            raise ValueError(f"x must be rows [{b}, {e}) of [*, {D}] {self.dtype}, got {tuple(x.shape)} {x.dtype}")
        if T > self.max_tokens:
            raise ValueError(f"{T} tokens > max_tokens {self.max_tokens}")
        if q_weight.numel() != self.q_dim or kv_weight.numel() != self.kv_dim:
            raise ValueError("weights do not match q_dim / kv_dim")
        if out is None:
            out = torch.empty((T, D), dtype=self.dtype, device=x.device)
        elif tuple(out.shape) != (T, D) or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous [{T}, {D}]")
        if not (x.is_contiguous() and q_weight.is_contiguous() and kv_weight.is_contiguous()):
            raise ValueError("tensors must be contiguous")
        if T == 0:
            return out

        parity, base = self._next_call(W - 1)
        chunk = (T + W - 1) // W
        nchan, grid = self.launch_shape(T)
        BLOCK = triton.next_power_of_2(D)
        with torch.cuda.device(x.device):
            self.last_kernel = qkv_norm_ag_ring_kernel[(grid,)](
                x if x.numel() else out, q_weight, kv_weight, out,
                self.data_addrs, self.flag_ptrs, self.order, self.pos, self.err,
                T, chunk, self.slot_elems, parity * self.parity_elems, self.rank, nchan,
                q_eps, kv_eps, base, self.max_spin,
                WORLD=W, D=D, Q=self.q_dim, KV=self.kv_dim, BLOCK=BLOCK,
                num_warps=min(16, max(4, BLOCK // 256)),
            )
        return out
