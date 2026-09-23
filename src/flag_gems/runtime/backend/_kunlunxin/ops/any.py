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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d
from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

# torch.any: Tests if any elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if any elements in input evaluate to non-zero value
# In triton function, test if any elements in input evaluate to non-zero value is ok.

cluster_num = 12
core_num = 64
buf_len_per_core = 2048
vector_size = 16

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def _any_permute_copy_pw(src):
    return src


def heur_m_block_size(args):
    return triton.next_power_of_2(min(triton.cdiv(args["M"], cluster_num), core_num))


def heur_n_block_size(args):
    return triton.next_power_of_2(min(args["N"], triton.cdiv(buf_len_per_core, 4)))


@triton.jit
def reduce_any(a, b):
    return a or b


@triton.jit
def reduce_or_i32(a, b):
    return a | b


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("any"), key=["M", "N"])
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def any_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0.0)
        _any = _any or (a != 0)
    any = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    tl.store(out, any[:, None], row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def max_kernel_dim(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    xoffset = tl.program_id(0) * BLOCK_M
    xindex = xoffset + tl.arange(0, BLOCK_M)[:, None]
    xmask = xindex < M
    rbase = tl.arange(0, BLOCK_N)[None, :]
    _max = tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32)
    for roffset in range(0, N, BLOCK_N):
        rindex = roffset + rbase
        rmask = rindex < N
        r1 = rindex
        inp = tl.load(
            in_ptr + (r1 + (N * xindex)), rmask & xmask, other=float("-inf")
        ).to(tl.float32)
        inpb = tl.broadcast_to(inp, [BLOCK_M, BLOCK_N])
        _max = tl.maximum(_max, inpb)
    tmp2 = tl.max(_max, axis=1, return_indices=False)[:, None]
    tl.store(out_ptr + xindex, tmp2, xmask)


@libentry()
@triton.jit
def any_word_stage1(in_ptr, mid, n_words, BLOCK_SIZE: tl.constexpr):
    """Stage 1 (int32-word bitmap path) of the global-any reduction.

    Reads the input as raw int32 words (valid whenever element_size divides 4:
    word != 0  <=>  at least one element in that word is nonzero, bit-exact).
    Maps the word to 0 (zero word) / INT32_MAX (nonzero word) with an integer
    select, then reduces each chunk with an int32 max -> INT32_MAX iff the
    chunk contains any nonzero element. Integer-only pipeline (no fcmp->i1
    per-element converts and no i1 OR-tree), which is markedly faster on XPU.
    Masked tail lanes load `other=0` (a zero word) and cannot create a false
    positive. `mid` receives INT32_MAX / 0 per chunk."""
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    w = tl.load(in_ptr + offs, mask=offs < n_words, other=0)
    m = tl.max(tl.where(w == 0, 0, 2147483647), axis=0)
    tl.store(mid + pid, m)


@libentry()
@triton.jit
def any_word_stage2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    """Stage 2: single program reduces the per-chunk int32 flags; masked
    lanes load 0 (matches the "zero chunk" encoding) and cannot flip the
    result. Outputs boolean (mx == INT32_MAX)."""
    offs = tl.arange(0, BLOCK_MID)
    m = tl.load(mid + offs, mask=offs < MID_SIZE, other=0)
    mx = tl.max(m, axis=0)
    tl.store(out, mx == 2147483647)


@libentry()
@triton.jit
def any_kernel_1(
    inp,
    mid,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 1 of the global-any reduction: each program reduces one
    BLOCK_SIZE-sized chunk of the flattened input into a single bool in `mid`.
    Reads the real elements and tests `!= 0`, so unlike the old uint8-view/
    byte-max hack it produces a canonical bool and scans every element (the
    hack passed numel as the byte count, silently scanning only the first
    numel/itemsize elements)."""
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    val = tl.load(inp + offset, mask=mask, other=0)
    nz = tl.where(mask, val != 0, False)
    any_val = tl.reduce(nz, axis=0, combine_fn=reduce_any)
    tl.store(mid + pid, any_val)


@libentry()
@triton.jit
def any_kernel_dim_v2(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            a = tl.load(inp + cols, (rows < M) and (cols < N), other=0.0)
        else:
            a = tl.load(inp + cols)
        _any = _any or (a != 0)
    any = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    if NEED_MASK:
        tl.store(out, any[:, None], rows < M)
    else:
        tl.store(out, any[:, None])


@libentry()
@triton.jit
def any_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    """Stage 2: a single program reduces the per-chunk bools from stage 1."""
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    val = tl.load(mid + offset, mask=mask, other=0)
    nz = tl.where(mask, val != 0, False)
    any_val = tl.reduce(nz, axis=0, combine_fn=reduce_any)
    tl.store(out, any_val)


@libentry()
@triton.jit
def any_row_word_stage1_kernel(
    in_ptr,
    mid,
    N_WORDS,
    N_CHUNKS,
    BLOCK_W: tl.constexpr,
    MAG: tl.constexpr,
    NEED_MASK: tl.constexpr,
    DIRECT: tl.constexpr,
    RAW_I32: tl.constexpr,
):
    """Per-row stage 1 on the int32-word bitmap (see `any_word_stage1`).

    Same idiom as the global-any fast path: OR-reduce the packed int32 words of
    one row.  `OR_w (w & MAG) == (OR_w w) & MAG`, so a zero result means every
    element of every reduced word was zero -- no per-element fcmp/i1 convert and
    no i1 OR-tree, which is markedly faster on XPU (`any_row_stage1_kernel`
    below is the slow i1 variant; the select+`tl.max` variant is ~2.8x slower).

    `MAG` clears the sign bits of each packed float lane (`0x7fff7fff` for
    16-bit elements, `0x7fffffff` for 32-bit, `-1` for integers/bool where a
    zero word already means "all elements zero"), so a `-0.0` lane still reads
    as zero and the result stays bit-exact with the elementwise `!= 0` test.

    `NEED_MASK` is False whenever `N_WORDS % BLOCK_W == 0`; the runtime
    `off < N_WORDS` predicate otherwise defeats the widest vectorisation on
    XPU and costs ~2x on the very case this fixes.

    `DIRECT` writes the finished result straight into `out` (the single-chunk
    case, `mid` is the output then) and skips the stage-2 launch, which is a
    pure fixed cost of ~90 us here -- larger than the reduction itself.

    `RAW_I32` keeps the raw int32 OR (instead of a bool) so the caller can
    split one word back into its 4 packed bytes (the 1-byte-dtype path, where
    the word axis groups a *kept* axis and each byte is a separate output).
    """
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    off = pid_c * BLOCK_W + tl.arange(0, BLOCK_W)
    if NEED_MASK:
        w = tl.load(in_ptr + pid_m * N_WORDS + off, mask=off < N_WORDS, other=0)
    else:
        w = tl.load(in_ptr + pid_m * N_WORDS + off)
    m = tl.reduce(w & MAG, axis=0, combine_fn=reduce_or_i32)
    if DIRECT:
        if RAW_I32:
            tl.store(mid + pid_m, m)
        else:
            tl.store(mid + pid_m, m != 0)
    else:
        tl.store(mid + pid_m * N_CHUNKS + pid_c, m)


@libentry()
@triton.jit
def any_row_word_stage2_kernel(
    mid, out, MID_N, BLOCK_MID: tl.constexpr, RAW_I32: tl.constexpr
):
    """Stage 2: fold the per-chunk int32 flags of one row into a bool."""
    pid_m = ext.program_id(0)
    off = tl.arange(0, BLOCK_MID)
    val = tl.load(mid + pid_m * MID_N + off, mask=off < MID_N, other=0)
    m = tl.reduce(val, axis=0, combine_fn=reduce_or_i32)
    if RAW_I32:
        tl.store(out + pid_m, m)
    else:
        tl.store(out + pid_m, m != 0)


@libentry()
@triton.jit
def any_row_stage1_kernel(inp, mid, N, N_CHUNKS, BLOCK_N: tl.constexpr):
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    offset = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    val = tl.load(inp + pid_m * N + offset, mask=mask, other=0)
    nz = tl.where(mask, val != 0, False)
    any_val = tl.reduce(nz, axis=0, combine_fn=reduce_any)
    tl.store(mid + pid_m * N_CHUNKS + pid_c, any_val)


@libentry()
@triton.jit
def any_row_stage2_kernel(mid, out, MID_N, BLOCK_MID: tl.constexpr):
    pid_m = ext.program_id(0)
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_N
    val = tl.load(mid + pid_m * MID_N + offset, mask=mask, other=0)
    nz = tl.where(mask, val != 0, False)
    any_val = tl.reduce(nz, axis=0, combine_fn=reduce_any)
    tl.store(out + pid_m, any_val)


_ROW_WORD_MAX = 32768


def _row_word_mag(dtype, elem_size):
    """Sign-clearing mask for the packed lanes of one int32 word, or None when
    the word bitmap cannot be made bit-exact with an elementwise `!= 0`.

    A raw `word != 0` test is exact for integer/bool payloads (the only zero
    pattern is all-zero bits) but not for floats, where `-0.0` has the sign bit
    set; masking the sign bit of every lane restores exactness.  `-1` is the
    no-op mask for the integer/bool case."""
    is_fp = dtype.is_floating_point
    if callable(is_fp):  # torch.dtype exposes it as a property in most builds
        is_fp = is_fp()
    if is_fp:
        if elem_size == 4:
            return 0x7FFFFFFF
        if elem_size == 2:
            return 0x7FFF7FFF
        return None  # f8 / f64: lane packing / sign layout not handled here
    if elem_size in (1, 2, 4):
        return -1
    return None


def _any_dims_reduce(inp, M, N, out_shape, raw_i32=False):
    """Reduce a contiguous [M, N] view over its N axis (per row), returning a bool
    tensor of shape `out_shape` (reduced dims already collapsed to 1).

    With `raw_i32` the per-row OR is returned as a raw int32 tensor of shape
    [M] instead of a bool (used by the 1-byte-dtype path, where each byte of
    the word is a distinct output)."""
    elem_size = inp.element_size()
    mag = (
        _row_word_mag(inp.dtype, elem_size)
        if inp.is_contiguous() and inp.data_ptr() % 4 == 0
        else None
    )
    if mag is not None and (N * elem_size) % 4 == 0:
        n_words = N * elem_size // 4
        BLOCK_W = min(triton.next_power_of_2(n_words), _ROW_WORD_MAX)
        n_chunks = triton.cdiv(n_words, BLOCK_W)
        need_mask = n_words % BLOCK_W != 0
        view = inp.reshape(-1).view(torch.uint8).view(torch.int32).reshape(M, n_words)
        out = torch.empty(
            M, dtype=torch.int32 if raw_i32 else torch.bool, device=inp.device
        )
        if n_chunks == 1:
            with torch_device_fn.device(inp.device):
                any_row_word_stage1_kernel[(M, 1)](
                    view,
                    out,
                    n_words,
                    1,
                    BLOCK_W=BLOCK_W,
                    MAG=mag,
                    NEED_MASK=need_mask,
                    DIRECT=True,
                    RAW_I32=raw_i32,
                    buffer_size_limit=2048,
                )
            return out.reshape(out_shape)
        mid = torch.empty((M, n_chunks), dtype=torch.int32, device=inp.device)
        with torch_device_fn.device(inp.device):
            any_row_word_stage1_kernel[(M, n_chunks)](
                view,
                mid,
                n_words,
                n_chunks,
                BLOCK_W=BLOCK_W,
                MAG=mag,
                NEED_MASK=need_mask,
                DIRECT=False,
                RAW_I32=raw_i32,
                buffer_size_limit=2048,
            )
            any_row_word_stage2_kernel[(M,)](
                mid,
                out,
                n_chunks,
                BLOCK_MID=triton.next_power_of_2(n_chunks),
                RAW_I32=raw_i32,
                buffer_size_limit=2048,
            )
        return out.reshape(out_shape)

    # generic i1 OR-tree path (any byte alignment / layout / dtype)
    BLOCK_N = 8192
    n_chunks = triton.cdiv(N, BLOCK_N)
    out = torch.empty(M, dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        if n_chunks == 1:
            any_row_stage1_kernel[(M, 1)](
                inp, out, N, 1, BLOCK_N=BLOCK_N, buffer_size_limit=2048
            )
        else:
            mid = torch.empty((M, n_chunks), dtype=torch.bool, device=inp.device)
            any_row_stage1_kernel[(M, n_chunks)](
                inp, mid, N, n_chunks, BLOCK_N=BLOCK_N, buffer_size_limit=2048
            )
            block_mid = triton.next_power_of_2(n_chunks)
            any_row_stage2_kernel[(M,)](
                mid, out, n_chunks, BLOCK_MID=block_mid, buffer_size_limit=2048
            )
    return out.reshape(out_shape)


def any(inp):
    logger.debug("GEMS_KUNLUNXIN ANY")
    n_elements = inp.numel()
    elem = inp.element_size()
    bytes_total = n_elements * elem

    if inp.is_contiguous() and bytes_total % 4 == 0:
        view = inp.reshape(-1).view(torch.uint8).view(torch.int32)
        n_words = view.numel()
        block_size = get_block_size_1d(n_words, 4)
        mid_size = triton.cdiv(n_words, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        # empty_strided (not registered by gems) -> native allocator,
        # avoids the per-call gems empty tax on the mid/out buffers.
        mid = torch.empty_strided(
            (mid_size,), (1,), dtype=torch.int32, device=inp.device
        )
        out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)
        with torch_device_fn.device(inp.device):
            any_word_stage1[(mid_size, 1)](
                view, mid, n_words, block_size, buffer_size_limit=2048
            )
            if mid_size == 1:
                return (mid == 2147483647).reshape([])
            any_word_stage2[(1, 1)](
                mid, out, mid_size, block_mid, buffer_size_limit=2048
            )
        return out

    # generic elementwise two-stage path (any byte alignment / layout)
    block_size = get_block_size_1d(n_elements, elem)
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty_strided((mid_size,), (1,), dtype=torch.bool, device=inp.device)
    out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        any_kernel_1[(mid_size, 1)](
            inp, mid, n_elements, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            return mid.reshape([])
        any_kernel_2[(1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
    return out


def _permute_contig(permuted):
    """Materialise a strided permute view as a contiguous tensor.

    Uses this file's tle idiom instead of `Tensor.contiguous()`: under
    `use_gems` a `contiguous()` on a permuted view dispatches to the vendor
    copy_, which moves it at ~1.4 GB/s (22.9 ms for the 33.5 MB
    (64,512,512) any_dims case), while tle_copy expresses the same transpose
    directly at ~880 GB/s (0.038 ms)."""
    new_shape = tuple(permuted.shape)
    strides = [1] * len(new_shape)
    for i in range(len(new_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * new_shape[i + 1]
    # empty_strided is not registered by gems -> native allocator.
    dst = torch.empty_strided(
        new_shape,
        tuple(strides),
        dtype=permuted.dtype,
        device=permuted.device,
    )
    if tle_copy(permuted, dst):
        return dst
    try:
        _any_permute_copy_pw(permuted, out0=dst)
        return dst
    except Exception:  # noqa: BLE001
        logger.warning(
            "GEMS_KUNLUNXIN ANY: tle/pointwise permute copy failed for "
            "shape=%s strides=%s dtype=%s; using contiguous()",
            tuple(permuted.shape),
            permuted.stride(),
            permuted.dtype,
        )
    return permuted.contiguous()


def _move_dim_last_contig(inp, dim):
    if dim == inp.ndim - 1 and inp.is_contiguous():
        return inp
    order = [i for i in range(inp.ndim) if i != dim] + [dim]
    permuted = inp.permute(order)
    if permuted.is_contiguous():
        return permuted
    return _permute_contig(permuted)


def _dims_last_contig(inp, dims):
    """Drop-in replacement for the shared `utils.dim_compress`.

    Same layout (batch dims first in their original order, reduced dims last
    sorted by descending stride, materialised contiguous), so the output is
    bit-identical to `dim_compress`, but the copy goes through `_permute_contig`
    (tle) instead of `Tensor.contiguous()`. The shared helper cannot be changed
    here -- it is used by many other operators -- and its `.contiguous()` is the
    22.9 ms/1.4 GB/s part of the (64,512,512) any_dims regression.

    The descending-stride order is load-bearing for throughput as well: it is
    what makes the tle transpose land on the fast path (an ascending order for
    the 3D reduced-prefix case was measured at 8.05 ms)."""
    dset = set(dims)
    order = [i for i in range(inp.ndim) if i not in dset]
    order += sorted(dims, key=lambda i: inp.stride()[i], reverse=True)
    permuted = inp.permute(order)
    if permuted.is_contiguous():
        return permuted
    return _permute_contig(permuted)


def _byte_word_compress(inp, dims):
    """1-byte-input variant of `_dims_last_contig` (see it for the layout).

    `tle_copy` refuses 1-byte-element transposes (its SDNN trans kernel faults
    on them -- see `tle_copy`), so a 1-byte `_permute_contig` drops to the
    pointwise kernel, measured at ~3.1 ms for the (64,512,512) dim=[0,1] case
    versus 0.038 ms for the same transpose in a 4-byte dtype.  Viewing the
    innermost (contiguous) axis as int32 words makes it a 4-byte transpose
    again -- the same trick the reduce already uses on the *reduced* axis.

    The innermost axis is always the last one.  When it is a *reduced* axis,
    packing four of its elements into a word is harmless (`word != 0` still
    means "some element non-zero"), so the word tensor feeds
    `_any_dims_reduce` unchanged.  When it is a *kept* axis -- the
    (64,512,512) dim=[0,1] case -- a word's four bytes are four distinct
    outputs, so the reduce must return the raw int32 OR and the caller splits
    it byte-wise: exact, because an int32 OR is a per-byte-position OR, and
    because the innermost kept axis varies fastest, `view(uint8)` reproduces
    the flat output order exactly.

    Returns `(word_tensor, expand)` or None when the word view is not
    expressible (leaving the caller on the generic path).
    """
    if inp.element_size() != 1 or inp.ndim == 0 or not inp.is_contiguous():
        return None
    last = inp.ndim - 1
    if inp.shape[last] % 4 != 0 or inp.storage_offset() % 4 != 0:
        return None
    if _row_word_mag(inp.dtype, 1) is None:
        return None
    words = inp.view(torch.int32)
    dset = set(dims)
    order = [i for i in range(words.ndim) if i not in dset]
    order += sorted(dims, key=lambda i: words.stride()[i], reverse=True)
    permuted = words.permute(order)
    if not permuted.is_contiguous():
        permuted = _permute_contig(permuted)
    return permuted, last not in dset


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
        N = shape[dim]
        shape[dim] = 1

        # Contiguous [M, N] view with the reduced dim last (see helper).
        if dim == inp.ndim - 1 and inp.is_contiguous():
            inpc = inp
        else:
            inpc = _move_dim_last_contig(inp, dim)
        M = inpc.numel() // N

        if inp.dtype == torch.bool:
            if N <= 512:
                block_m, block_n = 64, triton.next_power_of_2(N)
            elif N <= 4096:
                block_m, block_n = 64, 512
            else:
                block_m, block_n = 8, 4096
            need_mask = (M % block_m != 0) or (N % block_n != 0)
            out = torch.empty(shape, dtype=torch.bool, device=inp.device)
            grid = (triton.cdiv(M, block_m),)
            with torch_device_fn.device(inp.device):
                any_kernel_dim_v2[grid](
                    inpc,
                    out,
                    M,
                    N,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    NEED_MASK=need_mask,
                    buffer_size_limit=2048,
                )
        elif N >= vector_size * vector_size:
            outf = torch.empty(shape, dtype=torch.float, device=inp.device)
            block_m = triton.next_power_of_2(min(triton.cdiv(M, cluster_num), core_num))
            grid = (triton.cdiv(M, block_m),)
            with torch_device_fn.device(inp.device):
                max_kernel_dim[grid](inpc, outf, M, N, buffer_size_limit=2048)
            out = outf.to(torch.bool)
        else:
            out = torch.empty(shape, dtype=torch.bool, device=inp.device)
            block_m = triton.next_power_of_2(min(triton.cdiv(M, cluster_num), core_num))
            grid = (triton.cdiv(M, block_m),)
            with torch_device_fn.device(inp.device):
                any_kernel_dim[grid](inpc, out, M, N, buffer_size_limit=2048)

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
    n_red = 1
    for i in dim:
        n_red *= shape[i]
    byte_words = (
        _byte_word_compress(inp, dim)
        if inp.numel() > 0 and n_red > 0 and inp.numel() // n_red > 1
        else None
    )
    if byte_words is not None:
        words, expand = byte_words
        shape1 = list(shape)
        for i in dim:
            shape1[i] = 1
        n_red_w = n_red if expand else n_red // 4
        M = words.numel() // n_red_w
        if expand:
            o = _any_dims_reduce(words, M, n_red_w, [M], raw_i32=True)
            out = (o.view(torch.uint8) != 0).reshape(shape1)
        else:
            out = _any_dims_reduce(words, M, n_red_w, shape1)
    else:
        inp = _dims_last_contig(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        if M == 1:
            res = any(inp)
            out = res.reshape(shape)
        else:
            out = _any_dims_reduce(inp, M, N, shape)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
