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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _iprm_tile(
    x_ptr,
    c_ptr,
    s_ptr,
    n_rows,
    n_tok,
    row0,
    ROW_STRIDE: tl.constexpr,
    ANG_TOK_STRIDE: tl.constexpr,
    ANG_LAST_STRIDE: tl.constexpr,
    SPAN_START: tl.constexpr,
    SUB: tl.constexpr,
    R: tl.constexpr,
    RP: tl.constexpr,
    HEADS: tl.constexpr,
    SHARED_ANG: tl.constexpr,
    MT: tl.constexpr,
    NEG: tl.constexpr,
    TOK_ZERO: tl.constexpr,
    TOK_MOD: tl.constexpr,
):
    """One row sub-tile: load the span, rotate it in place, store it back.

    ``NT`` is derived from the sub-tile itself (SUB // HEADS) so the token
    grouping stays correct however the rows are chunked on the host.
    """
    NT: tl.constexpr = SUB // HEADS
    rows = row0 + tl.arange(0, SUB)
    rmask = rows < n_rows
    k = tl.arange(0, RP)

    if R == RP:
        xmask = tl.broadcast_to(rmask[:, None], (SUB, RP))
    else:
        xmask = rmask[:, None] & (k < R)[None, :]

    off = rows[:, None] * ROW_STRIDE + SPAN_START + k[None, :]
    xv = tl.load(x_ptr + off, mask=xmask, other=0.0)

    # partner index p ^ 1, built in registers by splitting the tile into its
    # even/odd lanes and rejoining them in swapped order (one span load only).
    even = (k & 1) == 0
    xr = tl.reshape(xv, (SUB, RP // 2, 2))
    xa, xb = tl.split(xr)
    xpart = tl.reshape(tl.join(xb, xa), (SUB, RP))

    if SHARED_ANG:
        if MT:
            # sub-tile spans NT whole tokens: one angle row each, broadcast
            tokv = row0 // HEADS + tl.arange(0, NT)
            if TOK_ZERO:
                tokv = tokv * 0
            elif TOK_MOD:
                tokv = tokv % n_tok
            aoff = tokv[:, None] * ANG_TOK_STRIDE + k[None, :] * ANG_LAST_STRIDE
            if R == RP:
                cv = tl.load(c_ptr + aoff)
                sv = tl.load(s_ptr + aoff)
            else:
                amask = (k < R)[None, :]
                cv = tl.load(c_ptr + aoff, mask=amask, other=0.0)
                sv = tl.load(s_ptr + aoff, mask=amask, other=0.0)
        else:
            tok = row0 // HEADS
            aoff = tok * ANG_TOK_STRIDE + k * ANG_LAST_STRIDE
            if R == RP:
                cv = tl.load(c_ptr + aoff)
                sv = tl.load(s_ptr + aoff)
            else:
                amask = k < R
                cv = tl.load(c_ptr + aoff, mask=amask, other=0.0)
                sv = tl.load(s_ptr + aoff, mask=amask, other=0.0)
            cos_b = cv[None, :]
            sin_b = sv[None, :]
    else:
        if TOK_ZERO:
            tok = rows * 0
        elif TOK_MOD:
            tok = (rows // HEADS) % n_tok
        else:
            tok = rows // HEADS
        aoff = tok[:, None] * ANG_TOK_STRIDE + k[None, :] * ANG_LAST_STRIDE
        cos_b = tl.load(c_ptr + aoff, mask=xmask, other=0.0)
        sin_b = tl.load(s_ptr + aoff, mask=xmask, other=0.0)

    if MT:
        # keep the angle operands as a [NT, 1, RP] broadcast view so no full
        # [SUB, RP] fp32 angle tile has to be materialised, and fold the
        # parity sign into the sin operand (measured faster than two product
        # tiles plus a full-width parity select; bit-exact either way)
        xv3 = tl.reshape(xv, (NT, SUB // NT, RP))
        xp3 = tl.reshape(xpart, (NT, SUB // NT, RP))
        if NEG:
            ssn = tl.where(even[None, :], sv, -sv)
        else:
            ssn = tl.where(even[None, :], -sv, sv)
        out3 = (
            xv3.to(tl.float32) * cv[:, None, :] + xp3.to(tl.float32) * ssn[:, None, :]
        )
        out = tl.reshape(out3, (SUB, RP))
    else:
        if NEG:
            ssn = tl.where(even[None, :], sin_b, -sin_b)
        else:
            ssn = tl.where(even[None, :], -sin_b, sin_b)
        out = xv.to(tl.float32) * cos_b + xpart.to(tl.float32) * ssn

    tl.store(x_ptr + off, out.to(x_ptr.dtype.element_ty), mask=xmask)


@triton.jit
def _iprm_kernel(
    x_ptr,
    c_ptr,
    s_ptr,
    n_rows,
    n_tok,
    ROW_STRIDE: tl.constexpr,
    ANG_TOK_STRIDE: tl.constexpr,
    ANG_LAST_STRIDE: tl.constexpr,
    SPAN_START: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    R: tl.constexpr,
    RP: tl.constexpr,
    HEADS: tl.constexpr,
    SHARED_ANG: tl.constexpr,
    MT: tl.constexpr,
    NEG: tl.constexpr,
    TOK_ZERO: tl.constexpr,
    TOK_MOD: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid * BLOCK_ROWS

    if NCHUNK == 1:
        _iprm_tile(
            x_ptr,
            c_ptr,
            s_ptr,
            n_rows,
            n_tok,
            row0,
            ROW_STRIDE=ROW_STRIDE,
            ANG_TOK_STRIDE=ANG_TOK_STRIDE,
            ANG_LAST_STRIDE=ANG_LAST_STRIDE,
            SPAN_START=SPAN_START,
            SUB=BLOCK_ROWS,
            R=R,
            RP=RP,
            HEADS=HEADS,
            SHARED_ANG=SHARED_ANG,
            MT=MT,
            NEG=NEG,
            TOK_ZERO=TOK_ZERO,
            TOK_MOD=TOK_MOD,
        )
    else:
        # same rows per program and grid, walked as NCHUNK sub-tiles with a
        # pipelined loop so sub-tile i+1's span load overlaps sub-tile i's
        # compute+store
        for ci in tl.range(0, NCHUNK, num_stages=2):
            _iprm_tile(
                x_ptr,
                c_ptr,
                s_ptr,
                n_rows,
                n_tok,
                row0 + ci * (BLOCK_ROWS // NCHUNK),
                ROW_STRIDE=ROW_STRIDE,
                ANG_TOK_STRIDE=ANG_TOK_STRIDE,
                ANG_LAST_STRIDE=ANG_LAST_STRIDE,
                SPAN_START=SPAN_START,
                SUB=BLOCK_ROWS // NCHUNK,
                R=R,
                RP=RP,
                HEADS=HEADS,
                SHARED_ANG=SHARED_ANG,
                MT=MT,
                NEG=NEG,
                TOK_ZERO=TOK_ZERO,
                TOK_MOD=TOK_MOD,
            )


def _pick_block_rows(n_rows: int, heads: int, row_len: int, R: int):
    """Rows per program, angle-sharing mode, multi-token flag, tokens per tile.

    For the large-row regime the measured optimum depends on how much of each
    row the span covers: a dense span (at least half the row) prefers the larger
    descriptor (512 rows/program), a sparse one prefers 256; 128 and larger than
    512 are measured worse and 1024 exceeds the UB budget.
    """
    if n_rows <= 32:
        br = 1
    elif n_rows <= 128:
        br = 8
    elif n_rows <= 1024:
        br = 32
    elif n_rows <= 8192:
        br = 64
    elif 2 * R >= row_len:
        br = 1024
    else:
        br = 1024
    shared = heads >= 4
    mt = False
    nt = 1
    if shared:
        if br >= heads:
            # tile may span whole tokens (one angle row per token)
            br = max(heads, (br // heads) * heads)
            nt = br // heads
            mt = nt > 1
        else:
            # one tile must stay inside a single token: BLOCK_ROWS divides heads
            while br > 1 and (heads % br) != 0:
                br //= 2
    if br < 1:
        br = 1
    return br, shared, mt, nt


def _pick_nchunk(br: int, heads: int, mt: bool, row_len: int, R: int) -> int:
    """Sub-tiles per program for the pipelined schedule (1 = direct body).

    Every large band walks its rows as 256-row sub-tiles inside a num_stages=2
    loop: round 3 measured 2x256 as 3.2% faster than the monolithic 512-row form
    on the dense geometry, and round 4 measured the same form on the sparse
    geometry as 2% faster than its monolithic 256-row body and far better than
    either the monolithic 512-row form (+8.5%) or 2x128 (+8.9%).  So the working
    sub-tile height is 256 rows for both span densities and only the number of
    sub-tiles per program (i.e. the program count) is left to tune.
    """
    if br < 512 or not mt:
        return 1
    sub = 256
    if (br % sub) != 0 or sub < heads or (sub % heads) != 0:
        return 1
    return br // sub


def _launch(x, r1, r2, rotary_mode, partial_slice, negate_sin=False):
    L = x.shape[-1]
    start = int(partial_slice[0])
    end = int(partial_slice[1])
    R = end - start

    n_rows = x.numel() // L
    if n_rows <= 0 or R <= 0:
        return None

    row_stride = x.stride(-2) if x.dim() >= 2 else L
    heads = x.numel() // (int(x.shape[0]) * L)
    if heads < 1:
        heads = 1
    n_tok = int(r1.shape[0])

    br, shared, mt, nt = _pick_block_rows(n_rows, heads, L, R)
    nchunk = _pick_nchunk(br, heads, mt, L, R)

    tok_zero = n_tok <= 1
    tok_mod = (not tok_zero) and (n_tok != int(x.shape[0]))

    grid = ((n_rows + br - 1) // br,)
    _iprm_kernel[grid](
        x,
        r1,
        r2,
        n_rows,
        n_tok,
        ROW_STRIDE=row_stride,
        ANG_TOK_STRIDE=r1.stride(0),
        ANG_LAST_STRIDE=r1.stride(-1),
        SPAN_START=start,
        BLOCK_ROWS=br,
        R=R,
        RP=triton.next_power_of_2(R),
        HEADS=heads,
        SHARED_ANG=shared,
        MT=mt,
        NEG=bool(negate_sin),
        TOK_ZERO=tok_zero,
        TOK_MOD=tok_mod,
        NCHUNK=nchunk,
    )
    return None


def inplace_partial_rotary_mul(x, r1, r2, rotary_mode, partial_slice, negate_sin=False):
    """Interleaved partial rotary update of BF16 activations with FP32 sine and cosine values."""
    if (
        x.ndim != 4
        or x.shape[1] != 1
        or len(partial_slice) != 2
        or rotary_mode != "interleave"
    ):
        raise ValueError(
            "expected interleave mode, two bounds and [tokens,1,heads,width] input"
        )
    start, end = partial_slice
    if not (0 <= start < end <= x.shape[-1]) or end - start != 64 or (end - start) % 2:
        raise ValueError("expected an in-bounds 64-channel rotary span")
    if r1.shape != (x.shape[0], 1, 1, end - start) or r2.shape != r1.shape:
        raise ValueError("invalid angle table shape")
    if (
        x.dtype != torch.bfloat16
        or r1.dtype != torch.float32
        or r2.dtype != torch.float32
        or not all(t.is_contiguous() for t in (x, r1, r2))
    ):
        raise ValueError("expected contiguous BF16 activations and FP32 angles")
    if x.device.type != "npu" or r1.device != x.device or r2.device != x.device:
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(x.device):
        return _launch(x, r1, r2, rotary_mode, partial_slice, negate_sin)
