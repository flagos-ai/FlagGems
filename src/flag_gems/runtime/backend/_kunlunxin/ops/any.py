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
from flag_gems.utils import dim_compress, libentry
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


def _any_dims_reduce(inp, M, N, out_shape):
    """Reduce a contiguous [M, N] view over its N axis (per row), returning a bool
    tensor of shape `out_shape` (reduced dims already collapsed to 1)."""
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


def _move_dim_last_contig(inp, dim):
    if dim == inp.ndim - 1 and inp.is_contiguous():
        return inp
    order = [i for i in range(inp.ndim) if i != dim] + [dim]
    permuted = inp.permute(order)
    if permuted.is_contiguous():
        return permuted
    new_shape = tuple(permuted.shape)
    strides = [1] * len(new_shape)
    for i in range(len(new_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * new_shape[i + 1]
    # empty_strided is not registered by gems -> native allocator/copy.
    dst = torch.empty_strided(
        new_shape, tuple(strides), dtype=inp.dtype, device=inp.device
    )
    # `permuted` is a strided permute view of `inp`; tle takes it as a TMA tile
    # when the permutation keeps the innermost axis contiguous and otherwise
    # transposes a tile on chip (see permute_copy). The pointwise kernel keeps
    # whatever tle cannot express -- no `torch.ops.aten._copy_from`, which
    # bypasses gems and dispatches to the vendor fallback.
    if not tle_copy(permuted, dst):
        _any_permute_copy_pw(permuted, out0=dst)
    return dst


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
    inp = dim_compress(inp, dim)
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
