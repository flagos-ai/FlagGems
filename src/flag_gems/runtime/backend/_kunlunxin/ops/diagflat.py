# Copyright 2026, The FlagOS Contributors.
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

_BLOCK_FILL = 4096
_BLOCK_SCATTER = 512
_INT32_MAX = 2147483647

_COMPLEX_REAL = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def _fill_view(itemsize):
    """A same-itemsize integer view type used for the zero fill."""
    if itemsize >= 8:
        return torch.int64
    if itemsize == 4:
        return torch.int32
    if itemsize == 2:
        return torch.int16
    return torch.uint8


_HIDDEN_LANES = 64


def _block(count, limit):
    """Largest power of two <= min(count, limit); keeps ``tail_base >= 0``."""
    block = 1 << (int(count).bit_length() - 1)
    return block if block < limit else limit


@triton.jit
def _diagflat_zero_kernel(
    out_ptr,
    count,
    last_pid,
    tail_base,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
    CLAMP: tl.constexpr,
):
    pid = tl.program_id(0)
    if CLAMP:
        offs = tl.minimum(pid * BLOCK + tl.arange(0, BLOCK), count - 1)
    else:
        if WIDE:
            b = tl.where(pid == last_pid, tail_base, pid.to(tl.int64) * BLOCK)
        else:
            b = tl.where(pid == last_pid, tail_base, pid * BLOCK)
        offs = b + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty))


@triton.jit
def _diagflat_scatter_kernel(
    src_ptr,
    dst_ptr,
    count,
    last_pid,
    tail_base,
    sstride,
    dstride,
    base,
    jmul,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
    CLAMP: tl.constexpr,
):
    pid = tl.program_id(0)
    if CLAMP:
        offs = tl.minimum(pid * BLOCK + tl.arange(0, BLOCK), count - 1)
    else:
        b = tl.where(pid == last_pid, tail_base, pid * BLOCK)
        offs = b + tl.arange(0, BLOCK)
    val = tl.load(src_ptr + offs * sstride)
    if WIDE:
        addr = offs.to(tl.int64) * dstride + base - (offs & 1).to(tl.int64) * jmul
    else:
        addr = offs * dstride + base - (offs & 1) * jmul
    tl.store(dst_ptr + addr, val)


@triton.jit
def _diagflat_gather_kernel(
    src_ptr,
    dst_ptr,
    count,
    last_pid,
    tail_base,
    sstride,
    base,
    jmul,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
    CLAMP: tl.constexpr,
):
    """``dst[i] = src[i * sstride + base - (i & 1) * jmul]`` (autograd backward)."""
    pid = tl.program_id(0)
    if CLAMP:
        offs = tl.minimum(pid * BLOCK + tl.arange(0, BLOCK), count - 1)
    else:
        b = tl.where(pid == last_pid, tail_base, pid * BLOCK)
        offs = b + tl.arange(0, BLOCK)
    if WIDE:
        addr = offs.to(tl.int64) * sstride + base - (offs & 1).to(tl.int64) * jmul
    else:
        addr = offs * sstride + base - (offs & 1) * jmul
    val = tl.load(src_ptr + addr)
    tl.store(dst_ptr + offs, val)


def _zero_fill(out):
    buf = out.view(_fill_view(out.element_size()))
    count = buf.numel()
    if count == 0:
        return
    block = _block(count, _BLOCK_FILL)
    grid = (triton.cdiv(count, block),)
    _diagflat_zero_kernel[grid](
        buf,
        count,
        grid[0] - 1,
        count - block,
        BLOCK=block,
        WIDE=count > _INT32_MAX,
        CLAMP=block < _HIDDEN_LANES,
        num_warps=4,
    )


def _diagonal_params(out, k):
    m = out.shape[0]
    if out.is_complex():
        base = 2 * k if k >= 0 else 2 * (-k) * m
        return base, m, _COMPLEX_REAL[out.dtype]
    base = k if k >= 0 else (-k) * m
    return base, 0, None


def _forward(self, k):
    n = self.numel()
    m = n + (k if k >= 0 else -k)
    out = torch.empty((m, m), dtype=self.dtype, device=self.device)
    if out.numel() == 0:
        return out

    logger.debug("GEMS_KUNLUNXIN DIAGFLAT")
    _zero_fill(out)
    if n == 0:
        return out

    flat = self.reshape(-1)
    base, jmul, real_dtype = _diagonal_params(out, k)
    if real_dtype is not None:
        if not flat.is_contiguous():
            raise NotImplementedError(
                "diagflat: non-contiguous complex input is not supported"
            )
        src = flat.view(real_dtype)
        dst = out.view(real_dtype)
        count = 2 * n
        sstride = 1
    else:
        src = flat
        dst = out
        count = n
        sstride = flat.stride(0)

    block = _block(count, _BLOCK_SCATTER)
    grid = (triton.cdiv(count, block),)
    _diagflat_scatter_kernel[grid](
        src,
        dst,
        count,
        grid[0] - 1,
        count - block,
        sstride,
        m + 1,
        base,
        jmul,
        BLOCK=block,
        WIDE=out.numel() > _INT32_MAX,
        CLAMP=block < _HIDDEN_LANES,
        num_warps=4,
    )
    return out


def _backward(grad_out, shape, k):
    m = grad_out.shape[0]
    n = m - (k if k >= 0 else -k)
    grad_in = torch.empty(shape, dtype=grad_out.dtype, device=grad_out.device)
    if n <= 0 or grad_in.numel() == 0:
        return grad_in

    if grad_out.is_complex():
        real_dtype = _COMPLEX_REAL[grad_out.dtype]
        src = grad_out.view(real_dtype)
        dst = grad_in.view(real_dtype)
        count = 2 * n
        base = 2 * k if k >= 0 else 2 * (-k) * m
        jmul = m
    else:
        src = grad_out
        dst = grad_in
        count = n
        base = k if k >= 0 else (-k) * m
        jmul = 0

    block = _block(count, _BLOCK_SCATTER)
    grid = (triton.cdiv(count, block),)
    _diagflat_gather_kernel[grid](
        src,
        dst,
        count,
        grid[0] - 1,
        count - block,
        m + 1,
        base,
        jmul,
        BLOCK=block,
        WIDE=grad_out.numel() > _INT32_MAX,
        CLAMP=block < _HIDDEN_LANES,
        num_warps=4,
    )
    return grad_in


class _DiagFlat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, offset):
        ctx.input_shape = self.shape
        ctx.offset = offset
        return _forward(self, offset)

    @staticmethod
    def backward(ctx, grad_out):
        return _backward(grad_out, ctx.input_shape, ctx.offset), None


def diagflat(self, offset=0):
    if not isinstance(self, torch.Tensor):
        raise TypeError("diagflat(): argument 'self' must be a Tensor")
    if not isinstance(offset, int):
        raise TypeError("diagflat(): argument 'offset' must be an integer")
    k = int(offset)
    if self.requires_grad:
        return _DiagFlat.apply(self, k)
    return _forward(self, k)
