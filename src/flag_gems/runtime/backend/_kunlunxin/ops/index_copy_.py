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

from flag_gems.utils import libentry


@libentry()
@triton.jit
def _index_copy_rank1(
    inp,
    index,
    src,
    n_elements,
    inp_size0,
    inp_stride0,
    src_stride0,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    indices = tl.load(index + offsets, mask=mask, other=0)
    tl.device_assert(
        (~mask) | ((indices >= 0) & (indices < inp_size0)),
        "index value out of bounds: 0 <= index < self.size(dim)",
    )
    src_values = tl.load(src + offsets * src_stride0, mask=mask)
    tl.store(inp + indices * inp_stride0, src_values, mask=mask)


@libentry()
@triton.jit
def _index_copy_rank2(
    inp,
    index,
    src,
    n_elements,
    dim,
    inp_size_dim,
    inp_shape0,
    inp_shape1,
    inp_stride0,
    inp_stride1,
    src_shape1,
    src_stride0,
    src_stride1,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    coord0 = offsets // src_shape1
    coord1 = offsets % src_shape1
    index_coord = tl.where(dim == 0, coord0, coord1)
    indices = tl.load(index + index_coord, mask=mask, other=0)
    tl.device_assert(
        (~mask) | ((indices >= 0) & (indices < inp_size_dim)),
        "index value out of bounds: 0 <= index < self.size(dim)",
    )
    out_coord0 = tl.where(dim == 0, indices, coord0)
    out_coord1 = tl.where(dim == 1, indices, coord1)
    src_offset = coord0 * src_stride0 + coord1 * src_stride1
    out_offset = out_coord0 * inp_stride0 + out_coord1 * inp_stride1
    src_values = tl.load(src + src_offset, mask=mask)
    tl.store(inp + out_offset, src_values, mask=mask)


@libentry()
@triton.jit
def _index_copy_rank3(
    inp,
    index,
    src,
    n_elements,
    dim,
    inp_size_dim,
    inp_shape0,
    inp_shape1,
    inp_shape2,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    src_shape1,
    src_shape2,
    src_stride0,
    src_stride1,
    src_stride2,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    coord0 = offsets // (src_shape1 * src_shape2)
    remainder = offsets % (src_shape1 * src_shape2)
    coord1 = remainder // src_shape2
    coord2 = remainder % src_shape2
    index_coord = tl.where(dim == 0, coord0, tl.where(dim == 1, coord1, coord2))
    indices = tl.load(index + index_coord, mask=mask, other=0)
    tl.device_assert(
        (~mask) | ((indices >= 0) & (indices < inp_size_dim)),
        "index value out of bounds: 0 <= index < self.size(dim)",
    )
    out_coord0 = tl.where(dim == 0, indices, coord0)
    out_coord1 = tl.where(dim == 1, indices, coord1)
    out_coord2 = tl.where(dim == 2, indices, coord2)
    src_offset = coord0 * src_stride0 + coord1 * src_stride1 + coord2 * src_stride2
    out_offset = (
        out_coord0 * inp_stride0 + out_coord1 * inp_stride1 + out_coord2 * inp_stride2
    )
    src_values = tl.load(src + src_offset, mask=mask)
    tl.store(inp + out_offset, src_values, mask=mask)


@libentry()
@triton.jit
def _clone_contig(
    inp,
    out,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    v = tl.load(inp + offsets, mask=mask)
    tl.store(out + offsets, v, mask=mask)


def _validate(inp, dim, index, src):
    # O(1) host-side checks only. The full-tensor index bounds pre-check was
    # removed: under the FlagGems registry every elementwise op it issues
    # (le/ge/bitwise_and/all) costs ~0.1 ms of Python dispatch, dwarfing the
    # kernel itself. Bounds are instead enforced by an in-kernel
    # tl.device_assert in the rank kernels below.
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"


def index_copy_(inp, dim, index, src):
    _validate(inp, dim, index, src)
    if index.numel() == 0 or src.numel() == 0:
        return inp
    dim %= inp.ndim
    n_elements = src.numel()
    block = 4096
    grid = (triton.cdiv(n_elements, block),)
    if inp.ndim == 1:
        _index_copy_rank1[grid](
            inp,
            index,
            src,
            n_elements,
            inp.size(0),
            inp.stride(0),
            src.stride(0),
            BLOCK=block,
            num_warps=8,
        )
    elif inp.ndim == 2:
        _index_copy_rank2[grid](
            inp,
            index,
            src,
            n_elements,
            dim,
            inp.size(dim),
            inp.size(0),
            inp.size(1),
            inp.stride(0),
            inp.stride(1),
            src.size(1),
            src.stride(0),
            src.stride(1),
            BLOCK=block,
            num_warps=8,
        )
    elif inp.ndim == 3:
        _index_copy_rank3[grid](
            inp,
            index,
            src,
            n_elements,
            dim,
            inp.size(dim),
            inp.size(0),
            inp.size(1),
            inp.size(2),
            inp.stride(0),
            inp.stride(1),
            inp.stride(2),
            src.size(1),
            src.size(2),
            src.stride(0),
            src.stride(1),
            src.stride(2),
            BLOCK=block,
            num_warps=8,
        )
    else:
        raise NotImplementedError("Kunlunxin index_copy_ supports ranks 1 through 3")
    return inp


def index_copy(inp, dim, index, src):
    _validate(inp, dim, index, src)
    # Functional variant: clone the input with a lightweight Triton copy
    # (avoids both the generic code-generated kernel, which produces wrong
    # results on large 3-D shapes, and `aten::_copy_from`, whose registry
    # dispatch costs ~0.2 ms/call under `use_gems`), then reuse the in-place
    # path above.
    out = torch.empty_like(inp, memory_format=torch.contiguous_format)
    n_elements = inp.numel()
    if n_elements > 0:
        if inp.is_contiguous():
            _clone_contig[(triton.cdiv(n_elements, 256),)](
                inp, out, n_elements, BLOCK=256, num_warps=4
            )
        else:
            torch.ops.aten._copy_from(inp, out, False)
    return index_copy_(out, dim, index, src)
