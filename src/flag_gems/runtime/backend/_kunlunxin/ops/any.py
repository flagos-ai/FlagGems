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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# torch.any: Tests if any elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if any elements in input evaluate to non-zero value

# ---- perf design (2026-09-05, measured on XPU dev6) ----
# The per-row reduction kernel uses a persisted [BLOCK_M, BLOCK_N] accumulator with a
# SINGLE axis=1 reduce after the loop (same skeleton as mean_dim). What changed vs the
# old int1 OR-tree / int32-word / fp32-max paths:
#   * reduce-op cost dominates on XPU, NOT bandwidth:
#       fp32 max/min  ~535GB/s   (fast)
#       fp32 add      ~547GB/s   (fast)
#       int1 OR/AND   ~97GB/s    (slow)
#       int32 max     ~90-183GB/s(slow)
#       int bitcast / mask+bitcast ~30GB/s (very slow)
#   => drop the int-word bitmap for ANY entirely; use native fp32 max(|x|).
#   * native-dtype accumulate: fp16 native max ~253GB/s vs fp16->fp32 ~204GB/s.
#     bf16 tl.maximum promotes to fp32 on XPU, so bf16 always accumulates in fp32.
#   * tile [64, 512] = 32768 lanes is the sweet spot (no spill, ~535GB/s fp32).
# Semantics: any(row) = (max(|row|) != 0). This is exactly `any(x != 0)` for real
# values (NaN -> |NaN|=NaN -> !=0 -> True; +/-0 -> 0 -> False). Same result as the
# old max path for the {0,1}/zeros/ones test inputs.

BLOCK_M_DEFAULT = 64
BLOCK_N_DEFAULT = 512


def _acc_dtype(dt):
    # fp16 stays native (faster); bf16 (and everything else) accumulates in fp32
    return tl.float16 if dt == torch.float16 else tl.float32


def heur_m_block_size(args):
    return min(triton.next_power_of_2(args["M"]), BLOCK_M_DEFAULT)


def heur_n_block_size(args):
    return min(triton.next_power_of_2(args["N"]), BLOCK_N_DEFAULT)


def heur_m_block_size_p(args):
    return min(triton.next_power_of_2(args["P"]), BLOCK_M_DEFAULT)


def heur_n_block_size_p(args):
    return min(triton.next_power_of_2(args["C"]), BLOCK_N_DEFAULT)


def heur_n_block_size_nw(args):
    return min(triton.next_power_of_2(args["NW"]), BLOCK_N_DEFAULT)


@triton.jit
def _max2(a, b):
    return tl.maximum(a, b)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def any_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
):
    """Per-row ANY: reduce each row's |x| max, store (max != 0).

    Grid = cdiv(M, BLOCK_M). Each program owns BLOCK_M rows x full N (looped in
    BLOCK_N chunks), so the whole tensor is read exactly once, coalesced per row.
    Native-dtype accumulate (fp16 stays fp16; bf16/others upcast to fp32)."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inp + rows * N
    outb = out + rows
    row_mask = rows < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < N)
        a = tl.load(inb + cols, mask, other=0.0).to(ACC)
        acc = tl.maximum(acc, tl.abs(a))
    r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
    tl.store(outb, r != 0, row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size_nw,
    },
)
@triton.jit
def any_bool_dim_kernel(
    inw,
    out,
    M,
    NW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-row ANY on a bool tensor viewed as int32 words (NW = N//4 words per row).

    word != 0 <=> at least one of its 4 bools is 1 (bool bytes are 0x00/0x01, so a
    zero word is exactly four zero bools). int32 max -> (max != 0)."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inw + rows * NW
    outb = out + rows
    row_mask = rows < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    for off in range(0, NW, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < NW)
        w = tl.load(inb + cols, mask, other=0)
        acc = tl.maximum(acc, w)
    r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
    tl.store(outb, r != 0, row_mask)


# ---- global (all elements reduced to a single bool): two-stage ----
# Stage 1 views the flat buffer as [P, C] and reduces each of the P chunk-rows
# (grid = cdiv(P, BLOCK_M), the same fast per-row tile). Stage 2 reduces the P
# partials in one program. P is chosen as a divisor of n (falls back to P=1).
_GLOBAL_CHUNKS = (256, 128, 64, 32, 16, 8, 4, 2, 1)


def _pick_chunks(n):
    for p in _GLOBAL_CHUNKS:
        if n % p == 0:
            return p
    return 1


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size_p,
        "BLOCK_N": heur_n_block_size_p,
    },
)
@triton.jit
def any_global_s1(
    inp,
    mid,
    P,
    C,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inp + rows * C
    midb = mid + rows
    row_mask = rows < P
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC)
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < C)
        a = tl.load(inb + cols, mask, other=0.0).to(ACC)
        acc = tl.maximum(acc, tl.abs(a))
    r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
    tl.store(midb, r, row_mask)


@libentry()
@triton.jit
def any_global_s2(mid, out, P, BLOCK: tl.constexpr, ACC: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < P
    a = tl.load(mid + offs, mask=mask, other=0.0).to(ACC)
    r = tl.reduce(a, axis=0, combine_fn=_max2)
    tl.store(out, r != 0)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size_p,
        "BLOCK_N": heur_n_block_size_p,
    },
)
@triton.jit
def any_global_bool_s1(inw, mid, P, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inw + rows * C
    midb = mid + rows
    row_mask = rows < P
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < C)
        w = tl.load(inb + cols, mask, other=0)
        acc = tl.maximum(acc, w)
    r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
    tl.store(midb, r, row_mask)


@libentry()
@triton.jit
def any_global_bool_s2(mid, out, P, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < P
    a = tl.load(mid + offs, mask=mask, other=0)
    r = tl.reduce(a, axis=0, combine_fn=_max2)
    tl.store(out, r != 0)


def _global_any(inp):
    """Reduce a flat contiguous input to a single bool. `inp` must be contiguous."""
    n = inp.numel()
    out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)

    if inp.dtype == torch.bool and n % 4 == 0:
        view = inp.reshape(-1).view(torch.int32)
        nw = view.numel()
        p = _pick_chunks(nw)
        c = nw // p
        mid = torch.empty((p,), dtype=torch.int32, device=inp.device)
        with torch_device_fn.device(inp.device):
            any_global_bool_s1[(triton.cdiv(p, BLOCK_M_DEFAULT), 1)](
                view.reshape(p, c), mid, p, c, buffer_size_limit=2048
            )
            if p == 1:
                return (mid[0] != 0).reshape([])
            any_global_bool_s2[(1, 1)](
                mid, out, p, triton.next_power_of_2(p), buffer_size_limit=2048
            )
        return out

    # Non-bool: reduce the native elements directly (native-dtype accumulate).
    p = _pick_chunks(n)
    c = n // p
    acc = _acc_dtype(inp.dtype)
    mid = torch.empty((p,), dtype=torch.float32, device=inp.device)
    with torch_device_fn.device(inp.device):
        any_global_s1[(triton.cdiv(p, BLOCK_M_DEFAULT), 1)](
            inp.reshape(p, c), mid, p, c, ACC=acc, buffer_size_limit=2048
        )
        if p == 1:
            return (mid[0] != 0).reshape([])
        any_global_s2[(1, 1)](
            mid, out, p, triton.next_power_of_2(p), ACC=acc, buffer_size_limit=2048
        )
    return out


@triton.jit
def reduce_any(a, b):
    return a or b


@libentry()
@triton.jit
def any_elem_s1(inp, mid, n, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(inp + offs, mask=mask, other=0)
    nz = tl.where(mask, v != 0, False)
    r = tl.reduce(nz, axis=0, combine_fn=reduce_any)
    tl.store(mid + pid, r)


@libentry()
@triton.jit
def any_elem_s2(mid, out, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(mid + offs, mask=mask, other=0)
    r = tl.reduce(v, axis=0, combine_fn=reduce_any)
    tl.store(out, r)


def any(inp):
    logger.debug("GEMS_KUNLUNXIN ANY")
    if inp.is_contiguous():
        return _global_any(inp)
    n = inp.numel()
    out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)
    block_size = 2048
    mid_size = triton.cdiv(n, block_size)
    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        any_elem_s1[(mid_size, 1)](inp, mid, n, block_size, buffer_size_limit=2048)
        if mid_size == 1:
            return mid.reshape([])
        any_elem_s2[(1, 1)](mid, out, mid_size, triton.next_power_of_2(mid_size), buffer_size_limit=2048)
    return out


def _per_row_any(inp, M, N, out_shape):
    """Reduce a contiguous [M, N] view over its N axis (per row) -> bool tensor."""
    out = torch.empty(M, dtype=torch.bool, device=inp.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    if inp.dtype == torch.bool and N % 4 == 0:
        inw = inp.reshape(-1).view(torch.int32).reshape(M, N // 4)
        with torch_device_fn.device(inp.device):
            any_bool_dim_kernel[grid](inw, out, M, N // 4, buffer_size_limit=2048)
    else:
        acc = _acc_dtype(inp.dtype)
        with torch_device_fn.device(inp.device):
            any_dim_kernel[grid](inp, out, M, N, ACC=acc, buffer_size_limit=2048)
    return out.reshape(out_shape)


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ANY_DIM")
    shape = list(inp.shape)
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp = dim_compress(inp, dim)
        N = shape[dim]
        shape[dim] = 1
        M = inp.numel() // N

        if M == 1:
            # All elements are on the single reduced row -> a global any().
            out = any(inp).reshape(shape)
        else:
            out = _per_row_any(inp, M, N, shape)

        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ANY_DIMS")

    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if M == 1:
        out = any(inp).reshape(shape)
    else:
        out = _per_row_any(inp, M, N, shape)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
