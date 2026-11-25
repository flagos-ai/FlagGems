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

logger = logging.getLogger(__name__)


@triton.jit
def _i64_scalar(x, like):
    """Promote a runtime/constexpr scalar int to an i64 tensor broadcast.

    `x` may arrive as an i32 kernel scalar or as a Python int (Triton binds
    0/1-valued ints as constexpr), so `.to()` is not usable; adding a zero i64
    tensor handles both bindings.
    """
    return x + like * 0


@triton.jit
def _index_select_backward_scatter(
    grad_ptr,
    index_ptr,
    out_ptr,
    inner_size,
    idx_len,
    out_dim,
    NEED_CAST: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    """Scatter-add grad rows into out rows along `dim`.

    Grid: (ceil(inner_size / BLOCK_INNER), outer_size * idx_len)
      pid0 -> contiguous chunk of the trailing (inner) dims
      pid1 -> (outer, j): leading-dims position and selected index position

    out[outer, index[j], inner] += grad[outer, j, inner]
    """
    pid_inner = tl.program_id(0).to(tl.int64)
    pid_row = tl.program_id(1).to(tl.int64)

    offs = pid_inner * BLOCK_INNER + tl.arange(0, BLOCK_INNER).to(tl.int64)
    mask = offs < inner_size

    inner_size = _i64_scalar(inner_size, offs)
    idx_len = _i64_scalar(idx_len, offs)
    out_dim = _i64_scalar(out_dim, offs)

    j = pid_row % idx_len
    outer = pid_row // idx_len
    idx_val = tl.load(index_ptr + j).to(tl.int64)

    g_row = outer * (idx_len * inner_size) + j * inner_size
    g = tl.load(grad_ptr + g_row + offs, mask=mask, other=0.0)
    if NEED_CAST:
        g = g.to(tl.float32)

    o_row = outer * (out_dim * inner_size) + idx_val * inner_size
    tl.atomic_add(out_ptr + o_row + offs, g, mask=mask)


@triton.jit
def _index_select_backward_generic(
    grad_ptr,
    index_ptr,
    out_ptr,
    inner_size,
    idx_len,
    out_dim,
    total,
    NEED_CAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fallback 1D scatter-add used when the row grid exceeds grid limits."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < total

    g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    if NEED_CAST:
        g = g.to(tl.float32)

    inner_size = _i64_scalar(inner_size, offs)
    idx_len = _i64_scalar(idx_len, offs)
    out_dim = _i64_scalar(out_dim, offs)

    inner = offs % inner_size
    j = (offs // inner_size) % idx_len
    outer = offs // (inner_size * idx_len)
    idx = tl.load(index_ptr + j, mask=mask, other=0).to(tl.int64)
    out_off = outer * (out_dim * inner_size) + idx * inner_size + inner
    tl.atomic_add(out_ptr + out_off, g, mask=mask)


@triton.jit
def _cast_copy_kernel(src_ptr, dst_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < total
    v = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, v.to(dst_ptr.dtype.element_ty), mask=mask)


def index_select_backward(grad, self_sizes, dim, index):
    ndim = grad.ndim
    if dim < 0:
        dim += ndim
    sizes = tuple(int(s) for s in self_sizes)

    idx_len = index.numel()
    inner_size = 1
    for s in sizes[dim + 1 :]:
        inner_size *= s
    out_dim = sizes[dim]
    outer_size = 1
    for s in sizes[:dim]:
        outer_size *= s

    total = grad.numel()
    if total == 0:
        return torch.zeros(sizes, dtype=grad.dtype, device=grad.device)

    dtype = grad.dtype
    need_cast = dtype in (torch.float16, torch.bfloat16)

    # metax warps have 64 threads: size the row block to inner_size so no
    # thread is idle (64/128/256 lanes for inner 64/128/256).
    BLOCK_INNER = 64
    while BLOCK_INNER < inner_size and BLOCK_INNER < 256:
        BLOCK_INNER *= 2
    NUM_WARPS = BLOCK_INNER // 64
    BLOCK_FLAT = 1024

    if need_cast:
        # cast-copy kernel writes EVERY element of out, so out needs no zero fill
        out = torch.empty(sizes, dtype=grad.dtype, device=grad.device)
        scratch = torch.zeros(sizes, dtype=torch.float32, device=grad.device)
        if outer_size * idx_len <= 65535:
            grid = (triton.cdiv(inner_size, BLOCK_INNER), outer_size * idx_len)
            _index_select_backward_scatter[grid](
                grad,
                index,
                scratch,
                inner_size,
                idx_len,
                out_dim,
                NEED_CAST=True,
                BLOCK_INNER=BLOCK_INNER,
                num_warps=NUM_WARPS,
            )
        else:
            grid = (triton.cdiv(total, BLOCK_FLAT),)
            _index_select_backward_generic[grid](
                grad,
                index,
                scratch,
                inner_size,
                idx_len,
                out_dim,
                total,
                NEED_CAST=True,
                BLOCK=BLOCK_FLAT,
            )
        _cast_copy_kernel[(triton.cdiv(out.numel(), BLOCK_FLAT),)](
            scratch, out, out.numel(), BLOCK=BLOCK_FLAT
        )
    else:
        out = torch.zeros(sizes, dtype=grad.dtype, device=grad.device)
        if outer_size * idx_len <= 65535:
            grid = (triton.cdiv(inner_size, BLOCK_INNER), outer_size * idx_len)
            _index_select_backward_scatter[grid](
                grad,
                index,
                out,
                inner_size,
                idx_len,
                out_dim,
                NEED_CAST=False,
                BLOCK_INNER=BLOCK_INNER,
                num_warps=NUM_WARPS,
            )
        else:
            grid = (triton.cdiv(total, BLOCK_FLAT),)
            _index_select_backward_generic[grid](
                grad,
                index,
                out,
                inner_size,
                idx_len,
                out_dim,
                total,
                NEED_CAST=False,
                BLOCK=BLOCK_FLAT,
            )
    return out
