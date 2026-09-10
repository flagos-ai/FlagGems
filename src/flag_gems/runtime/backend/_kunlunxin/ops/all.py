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

# torch.all: Tests if all elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if all elements in input evaluate to non-zero value

# ---- perf design (2026-09-05, measured on XPU dev6) ----
# Same skeleton as any.py (see the design note there): reduce-op cost dominates, so
# ALL is a native fp32 MIN(|x|) reduction; result = (min != 0). For real values,
# min(|x|) == 0  <=>  some element is +/-0  <=>  NOT all(x != 0), so (min != 0) is
# exactly torch.all (NaN -> |NaN|=NaN -> !=0 -> True, matching torch's nonzero rule).
# fp16 accumulates natively (~253GB/s); bf16 tl.minimum promotes to fp32 on XPU.
# The old per-element `val != 0` AND-tree (int1) was ~97GB/s; the int32-word paths
# were not viable for ALL (a nonzero int32 word does not imply all bytes nonzero).

BLOCK_M_DEFAULT = 64
BLOCK_N_DEFAULT = 512


def _acc_dtype(dt):
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
def _min2(a, b):
    return tl.minimum(a, b)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def all_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
):
    """Per-row ALL: reduce each row's |x| min, store (min != 0)."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inp + rows * N
    outb = out + rows
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], float("inf"), ACC)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < N)
        a = tl.load(inb + cols, mask, other=float("inf")).to(ACC)
        acc = tl.minimum(acc, tl.abs(a))
    r = tl.reduce(acc, axis=1, combine_fn=_min2)[:, None]
    tl.store(outb, r != 0, row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size_nw,
    },
)
@triton.jit
def all_bool_dim_kernel(
    inw,
    out,
    M,
    NW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-row ALL on a bool tensor viewed as int32 words.

    bool bytes are 0x00/0x01, so every word <= 0x01010101 and a word holds all-four
    True iff it equals 0x01010101. min over words == 0x01010101 <=> all elements True.
    (all_dim's official tests use FLOAT_DTYPES only; this covers torch.all(bool).)"""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inw + rows * NW
    outb = out + rows
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], 0x01010101, tl.int32)
    for off in range(0, NW, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < NW)
        w = tl.load(inb + cols, mask, other=0x01010101)
        acc = tl.minimum(acc, w)
    r = tl.reduce(acc, axis=1, combine_fn=_min2)[:, None]
    tl.store(outb, r == 0x01010101, row_mask)


# ---- global (all elements reduced to a single bool): two-stage ----
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
def all_global_s1(
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
    acc = tl.full([BLOCK_M, BLOCK_N], float("inf"), ACC)
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < C)
        a = tl.load(inb + cols, mask, other=float("inf")).to(ACC)
        acc = tl.minimum(acc, tl.abs(a))
    r = tl.reduce(acc, axis=1, combine_fn=_min2)[:, None]
    tl.store(midb, r, row_mask)


@libentry()
@triton.jit
def all_global_s2(mid, out, P, BLOCK: tl.constexpr, ACC: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < P
    a = tl.load(mid + offs, mask=mask, other=0.0).to(ACC)
    r = tl.reduce(a, axis=0, combine_fn=_min2)
    tl.store(out, r != 0)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size_p,
        "BLOCK_N": heur_n_block_size_p,
    },
)
@triton.jit
def all_global_bool_s1(inw, mid, P, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inw + rows * C
    midb = mid + rows
    row_mask = rows < P
    acc = tl.full([BLOCK_M, BLOCK_N], 0x01010101, tl.int32)
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < C)
        w = tl.load(inb + cols, mask, other=0x01010101)
        acc = tl.minimum(acc, w)
    r = tl.reduce(acc, axis=1, combine_fn=_min2)[:, None]
    tl.store(midb, r, row_mask)


@libentry()
@triton.jit
def all_global_bool_s2(mid, out, P, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < P
    a = tl.load(mid + offs, mask=mask, other=0x01010101)
    r = tl.reduce(a, axis=0, combine_fn=_min2)
    tl.store(out, r == 0x01010101)


def _global_all(inp):
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
            all_global_bool_s1[(triton.cdiv(p, BLOCK_M_DEFAULT), 1)](
                view.reshape(p, c), mid, p, c, buffer_size_limit=2048
            )
            if p == 1:
                return (mid[0] == 0x01010101).reshape([])
            all_global_bool_s2[(1, 1)](
                mid, out, p, triton.next_power_of_2(p), buffer_size_limit=2048
            )
        return out

    p = _pick_chunks(n)
    c = n // p
    acc = _acc_dtype(inp.dtype)
    mid = torch.empty((p,), dtype=torch.float32, device=inp.device)
    with torch_device_fn.device(inp.device):
        all_global_s1[(triton.cdiv(p, BLOCK_M_DEFAULT), 1)](
            inp.reshape(p, c), mid, p, c, ACC=acc, buffer_size_limit=2048
        )
        if p == 1:
            return (mid[0] != 0).reshape([])
        all_global_s2[(1, 1)](
            mid, out, p, triton.next_power_of_2(p), ACC=acc, buffer_size_limit=2048
        )
    return out


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.jit
def all_elem_s1(inp, mid, n, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(inp + offs, mask=mask, other=1)
    nz = tl.where(mask, v != 0, True)
    r = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, r)


@libentry()
@triton.jit
def all_elem_s2(mid, out, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(mid + offs, mask=mask, other=1)
    nz = tl.where(mask, v != 0, True)
    r = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(out, r)


@libentry()
@triton.jit
def all_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """Stage-2 global all reduction (shared with isclose.py)."""
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    val = tl.load(mid + offset, mask=mask, other=1)
    nz = tl.where(mask, val != 0, True)
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(out, result)


def all(inp):
    logger.debug("GEMS_KUNLUNXIN ALL")
    if inp.is_contiguous():
        return _global_all(inp)
    n = inp.numel()
    out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)
    block_size = 2048
    mid_size = triton.cdiv(n, block_size)
    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        all_elem_s1[(mid_size, 1)](inp, mid, n, block_size, buffer_size_limit=2048)
        if mid_size == 1:
            return mid.reshape([])
        all_elem_s2[(1, 1)](mid, out, mid_size, triton.next_power_of_2(mid_size), buffer_size_limit=2048)
    return out


def _per_row_all(inp, M, N, out_shape):
    """Reduce a contiguous [M, N] view over its N axis (per row) -> bool tensor."""
    out = torch.empty(M, dtype=torch.bool, device=inp.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    if inp.dtype == torch.bool and N % 4 == 0:
        inw = inp.reshape(-1).view(torch.int32).reshape(M, N // 4)
        with torch_device_fn.device(inp.device):
            all_bool_dim_kernel[grid](inw, out, M, N // 4, buffer_size_limit=2048)
    else:
        acc = _acc_dtype(inp.dtype)
        with torch_device_fn.device(inp.device):
            all_dim_kernel[grid](inp, out, M, N, ACC=acc, buffer_size_limit=2048)
    return out.reshape(out_shape)


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIM")
    shape = list(inp.shape)
    orig_ndim = inp.ndim

    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * orig_ndim)
        return out

    assert dim >= -orig_ndim and dim < orig_ndim, "Invalid dim"
    dim = dim % orig_ndim
    N = shape[dim]
    inp = dim_compress(inp, dim)
    shape[dim] = 1
    M = inp.numel() // N

    if M == 1:
        out = all(inp).reshape(shape)
    else:
        out = _per_row_all(inp, M, N, shape)

    if not keepdim and out.ndim > 0:
        out = out.squeeze(dim) if dim < out.ndim else out
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    orig_ndim = inp.ndim
    assert ((i >= -orig_ndim and i < orig_ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % orig_ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if M == 1:
        out = all(inp).reshape(shape)
    else:
        out = _per_row_all(inp, M, N, shape)

    if not keepdim:
        for d in sorted(dim, reverse=True):
            if out.ndim > 0:
                out = out.squeeze(dim=d)
    return out
