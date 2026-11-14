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

"""Kunlunxin override for ``aten::select_copy``.

The generic implementation has three XPU pathologies:

1. ``grid = lambda meta: (...)`` at ops/select_copy.py:145 and :170.  On the
   XPU backend the launch grid participates in the compilation cache key, so a
   fresh lambda per call forces a FULL recompile on every launch (measured
   ~133 ms per call for a 8192-element copy versus 0.019 ms with a constant
   tuple).  This alone explains the 0.0002 speedup of the generic path.
2. No ``@libentry`` -> the kernel is specialized per pointer alignment/shape.
3. ``torch.empty`` is intercepted by the gems ``empty`` override inside a
   gems context, adding an extra zeroing kernel and a second recompile per
   call.  ``torch.empty_strided`` is a plain native allocation.

The override keeps the same semantics (out-of-place copy of the selected
slice, contiguous result) but uses constant-tuple grids, ``@libentry`` and
affine addressing: for a contiguous input the selected slice along ``dim``
consists of ``pre`` contiguous runs of ``post`` elements, so a 2D grid
``(pre, cdiv(post, BLOCK))`` needs no per-element division at all.
"""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_MAX_RANK = 8
_BLOCK = 4096
_MIN_POST_FOR_ROWS = 256


@libentry()
@triton.jit
def _select_copy_flat_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Pure contiguous copy: NEED_MASK is a constexpr so the exact-multiple case
    # avoids the masked-memory path (roughly 2x more expensive on this backend).
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = off < n_elements
        v = tl.load(src_ptr + off, mask=mask)
        tl.store(dst_ptr + off, v, mask=mask)
    else:
        v = tl.load(src_ptr + off)
        tl.store(dst_ptr + off, v)


@libentry()
@triton.jit
def _select_copy_rows_kernel(
    src_ptr,
    dst_ptr,
    dim_size,
    post,
    index,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Contiguous input, dim > 0 with a wide tail: program (p, j) copies the
    # contiguous run out[p, j*BLOCK ...] from
    # in[p*dim_size*post + index*post + j*BLOCK ...].  No div/mod.
    p = tl.program_id(0)
    j = tl.program_id(1)
    off = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = off < post
        v = tl.load(src_ptr + p * dim_size * post + index * post + off, mask=mask)
        tl.store(dst_ptr + p * post + off, v, mask=mask)
    else:
        v = tl.load(src_ptr + p * dim_size * post + index * post + off)
        tl.store(dst_ptr + p * post + off, v)


@libentry()
@triton.jit
def _select_copy_narrow_post_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    dim_size,
    index,
    POST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Contiguous input, dim > 0 but the tail ``post`` is narrow (e.g. post == 1
    # for a row-select on a 2-D tensor).  ``POST`` is a constexpr, so the
    # div/mod pair lowers to a constant-divisor sequence; post == 1 folds away
    # entirely and leaves a strided gather.
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < n_elements
    p = off // POST
    q = off - p * POST
    src = p * (dim_size * POST) + index * POST + q
    v = tl.load(src_ptr + src, mask=mask)
    tl.store(dst_ptr + off, v, mask=mask)


@libentry()
@triton.jit
def _select_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    src_base,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    ND: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Arbitrary layout: decompose the flat output index over the slice
    # dimensions (row-major) and follow the input strides of those dimensions.
    # Unused slots are padded with (size 1, stride 0) on the host.
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < n_elements
    rem = off
    src = tl.full([BLOCK_SIZE], src_base, tl.int32)
    if ND > 7:
        c = rem % d7
        rem = rem // d7
        src += c * s7
    if ND > 6:
        c = rem % d6
        rem = rem // d6
        src += c * s6
    if ND > 5:
        c = rem % d5
        rem = rem // d5
        src += c * s5
    if ND > 4:
        c = rem % d4
        rem = rem // d4
        src += c * s4
    if ND > 3:
        c = rem % d3
        rem = rem // d3
        src += c * s3
    if ND > 2:
        c = rem % d2
        rem = rem // d2
        src += c * s2
    if ND > 1:
        c = rem % d1
        rem = rem // d1
        src += c * s1
    if ND > 0:
        c = rem % d0
        src += c * s0
    v = tl.load(src_ptr + src, mask=mask)
    tl.store(dst_ptr + off, v, mask=mask)


def _merge_dims(sizes, strides):
    """Drop size-1 dims and merge adjacent dims that are contiguous."""
    out_sizes, out_strides = [], []
    for size, stride in zip(sizes, strides):
        if size == 1:
            continue
        if out_sizes and out_strides[-1] == size * stride:
            out_sizes[-1] *= size
            out_strides[-1] = stride
        else:
            out_sizes.append(size)
            out_strides.append(stride)
    return out_sizes, out_strides


def _contig_strides(shape):
    stride = 1
    strides = [0] * len(shape)
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return tuple(strides)


def _pick_block(n):
    b = triton.next_power_of_2(n)
    if b > 8192:
        b = 8192
    if b < 1024:
        b = 1024
    return b


def _flat_copy(src_view, out, n, block):
    """Contiguous copy, widening to 8-byte lanes when the layout allows it."""
    es = out.element_size()
    if n * es % 8 == 0:
        try:
            src8 = src_view.reshape(-1).view(torch.int64)
            out8 = out.reshape(-1).view(torch.int64)
            n8 = n * es // 8
            grid = (triton.cdiv(n8, block),)
            _select_copy_flat_kernel[grid](
                src8,
                out8,
                n8,
                BLOCK_SIZE=block,
                NEED_MASK=n8 % block != 0,
            )
            return
        except RuntimeError:
            pass
    grid = (triton.cdiv(n, block),)
    _select_copy_flat_kernel[grid](
        src_view,
        out,
        n,
        BLOCK_SIZE=block,
        NEED_MASK=n % block != 0,
    )


def select_copy(self: torch.Tensor, dim: int, index: int):
    if self.dim() == 0:
        raise RuntimeError("select() cannot be applied to a 0-dim tensor.")

    dim_size = self.size(dim)
    if index < -dim_size or index >= dim_size:
        raise RuntimeError(
            f"select(): index {index} out of range for tensor of size "
            f"[{', '.join(str(s) for s in self.shape)}] at dimension {dim}"
        )
    dim = dim % self.dim()
    index = index % dim_size

    out_shape = list(self.shape)
    del out_shape[dim]
    out = torch.empty_strided(
        out_shape, _contig_strides(out_shape), dtype=self.dtype, device=self.device
    )
    n = out.numel()
    if n == 0:
        return out

    if self.is_contiguous():
        pre = 1
        for size in self.shape[:dim]:
            pre *= size
        post = 1
        for size in self.shape[dim + 1 :]:
            post *= size

        with torch_device_fn.device(self.device):
            if dim == 0:
                # The selected slice is itself contiguous (storage offset
                # included in the data pointer).
                _flat_copy(self.select(dim, index), out, n, _pick_block(n))
                return out
            if post >= _MIN_POST_FOR_ROWS:
                block = _pick_block(post)
                grid = (pre, triton.cdiv(post, block))
                _select_copy_rows_kernel[grid](
                    self,
                    out,
                    dim_size,
                    post,
                    index,
                    BLOCK_SIZE=block,
                    NEED_MASK=post % block != 0,
                )
                return out
            block = 1024
            _select_copy_narrow_post_kernel[(triton.cdiv(n, block),)](
                self,
                out,
                n,
                dim_size,
                index,
                POST=post,
                BLOCK_SIZE=block,
            )
            return out

    # Non-contiguous input: address the slice with its real strides.
    slice_sizes = list(self.shape)
    slice_strides = [int(s) for s in self.stride()]
    del slice_sizes[dim]
    del slice_strides[dim]
    slice_sizes, slice_strides = _merge_dims(slice_sizes, slice_strides)
    nd = len(slice_sizes)
    if nd > _MAX_RANK:
        raise RuntimeError(
            f"select_copy supports at most {_MAX_RANK} dimensions, got {self.dim()}"
        )
    slice_sizes += [1] * (_MAX_RANK - nd)
    slice_strides += [0] * (_MAX_RANK - nd)
    src_base = index * int(self.stride(dim))
    block = 1024
    with torch_device_fn.device(self.device):
        _select_copy_strided_kernel[(triton.cdiv(n, block),)](
            self,
            out,
            n,
            src_base,
            *slice_sizes,
            *slice_strides,
            ND=nd,
            BLOCK_SIZE=block,
        )
    return out
