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

"""thead specialization of ``cat`` / ``cat_out``.

The generic kernel selects the source tensor of each program with an
``if/elif`` chain on ``program_id(1)``.  The PPU compiler lowers
``pid_y == 2`` into the ``pid_y == 1`` branch, so the third and fourth inputs
of a batch were silently skipped (their output region kept ``torch.empty``
contents).  ``tl.where`` on the same comparisons is compiled correctly, so
this version selects the per-tensor parameters with scalar ``tl.where`` and
issues four masked loads (three of them fully masked) instead of branching.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cat import (
    _cat_build_working_list,
    _cat_build_working_list_uint8_view,
    _should_use_uint8_view_path,
)

logger = logging.getLogger(__name__)


@triton.jit
def cat_copy_func_kernel_4(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    dim_size_in_a: tl.int64,
    dim_size_in_b: tl.int64,
    dim_size_in_c: tl.int64,
    dim_size_in_d: tl.int64,
    dim_size_out,
    dim_prod_post,
    dim_offset_a: tl.int64,
    dim_offset_b: tl.int64,
    dim_offset_c: tl.int64,
    dim_offset_d: tl.int64,
    total_elements_a: tl.int64,
    total_elements_b: tl.int64,
    total_elements_c: tl.int64,
    total_elements_d: tl.int64,
    BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    idx = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)

    is_a = pid_y == 0
    is_b = pid_y == 1
    is_c = pid_y == 2

    dim_size_in = tl.where(
        is_a,
        dim_size_in_a,
        tl.where(is_b, dim_size_in_b, tl.where(is_c, dim_size_in_c, dim_size_in_d)),
    )
    dim_offset = tl.where(
        is_a,
        dim_offset_a,
        tl.where(is_b, dim_offset_b, tl.where(is_c, dim_offset_c, dim_offset_d)),
    )
    total_elements = tl.where(
        is_a,
        total_elements_a,
        tl.where(
            is_b,
            total_elements_b,
            tl.where(is_c, total_elements_c, total_elements_d),
        ),
    )

    pre_idx = idx // (dim_size_in * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size_in
    post_idx = idx % dim_prod_post

    out_idx = (
        pre_idx * dim_size_out * dim_prod_post
        + (dim_idx + dim_offset) * dim_prod_post
        + post_idx
    )

    valid = idx < total_elements
    val_a = tl.load(in_ptr_a + idx, mask=valid & is_a)
    val_b = tl.load(in_ptr_b + idx, mask=valid & is_b)
    val_c = tl.load(in_ptr_c + idx, mask=valid & is_c)
    val_d = tl.load(in_ptr_d + idx, mask=valid & (pid_y == 3))
    data = tl.where(is_a, val_a, tl.where(is_b, val_b, tl.where(is_c, val_c, val_d)))
    tl.store(out_ptr + out_idx, data, mask=valid)


def _cat_run_kernel(A, dim, out_shape, out):
    BLOCK = 1024
    dim_size_out = out_shape[dim]
    dim_prod_post = 1
    for d in range(dim + 1, A[0].ndim):
        dim_prod_post *= A[0].shape[d]

    dim_offset = 0
    i = 0
    while i < len(A):
        tensors_in_batch = A[i : i + 4]
        num_tensors_in_batch = len(tensors_in_batch)

        tensors = []
        dim_sizes = []
        dim_offsets = []
        total_elements_list = []
        current_dim_offset = dim_offset

        for j in range(4):
            if j < num_tensors_in_batch:
                tensor = tensors_in_batch[j].contiguous()
                dim_size_in = tensor.shape[dim]
                total_elements = tensor.numel()
                tensors.append(tensor)
                # All batches pass dim_size_in for every slot because the
                # selection is branchless; never hand a zero divisor to the
                # index math (empty tensors are masked out anyway).
                dim_sizes.append(dim_size_in if dim_size_in > 0 else 1)
                dim_offsets.append(current_dim_offset)
                total_elements_list.append(total_elements)
                current_dim_offset += dim_size_in
            else:
                tensors.append(tensors_in_batch[0])
                dim_sizes.append(1)
                dim_offsets.append(0)
                total_elements_list.append(0)

        max_elements_in_batch = max(total_elements_list)
        if max_elements_in_batch > 0:
            grid = (triton.cdiv(max_elements_in_batch, BLOCK), num_tensors_in_batch)
            cat_copy_func_kernel_4[grid](
                out,
                tensors[0],
                tensors[1],
                tensors[2],
                tensors[3],
                dim_sizes[0],
                dim_sizes[1],
                dim_sizes[2],
                dim_sizes[3],
                dim_size_out,
                dim_prod_post,
                dim_offsets[0],
                dim_offsets[1],
                dim_offsets[2],
                dim_offsets[3],
                total_elements_list[0],
                total_elements_list[1],
                total_elements_list[2],
                total_elements_list[3],
                BLOCK_X=BLOCK,
            )

        dim_offset = current_dim_offset
        i += num_tensors_in_batch


def cat_out(A, dim=0, *, out):
    logger.debug("GEMS_THEAD CAT_OUT")
    A = list(A)
    if _should_use_uint8_view_path(A):
        mode, payload, original_dtype = _cat_build_working_list_uint8_view(A, dim)
        if mode == "single":
            t = payload.view(original_dtype)
            out.resize_(t.shape)
            if out.dtype != t.dtype:
                out.copy_(t.to(out.dtype))
            else:
                out.copy_(t)
            return out
        if mode == "empty":
            t = payload.view(original_dtype)
            out.resize_(t.shape)
            out.copy_(t)
            return out

        A_u8, dim, out_shape, _, _ = payload
        if out.dtype != original_dtype:
            raise RuntimeError(
                f"cat.out: expected out dtype {original_dtype}, got {out.dtype}"
            )
        if list(out.shape) != out_shape:
            out.resize_(out_shape)
        out_u8 = out.view(torch.uint8)
        _cat_run_kernel(A_u8, dim, out_shape, out_u8)
        return out

    mode, payload = _cat_build_working_list(A, dim)
    if mode == "single":
        t = payload
        out.resize_(t.shape)
        if out.dtype != t.dtype:
            out.copy_(t.to(out.dtype))
        else:
            out.copy_(t)
        return out
    if mode == "empty":
        t = payload
        out.resize_(t.shape)
        out.copy_(t)
        return out

    A, dim, out_shape, dtype, device = payload
    if out.dtype != dtype:
        raise RuntimeError(f"cat.out: expected out dtype {dtype}, got {out.dtype}")
    if list(out.shape) != out_shape:
        out.resize_(out_shape)
    _cat_run_kernel(A, dim, out_shape, out)
    return out


def cat(A, dim=0):
    logger.debug("GEMS_THEAD CAT")
    A = list(A)
    if _should_use_uint8_view_path(A):
        mode, payload, original_dtype = _cat_build_working_list_uint8_view(A, dim)
        if mode == "single":
            return payload.view(original_dtype)
        if mode == "empty":
            return payload.view(original_dtype)

        A_u8, dim, out_shape, _, device = payload
        out_u8 = torch.empty(out_shape, dtype=torch.uint8, device=device)
        _cat_run_kernel(A_u8, dim, out_shape, out_u8)
        return out_u8.view(original_dtype)

    mode, payload = _cat_build_working_list(A, dim)
    if mode == "single":
        return payload
    if mode == "empty":
        return payload

    A, dim, out_shape, dtype, device = payload
    out = torch.empty(out_shape, dtype=dtype, device=device)
    _cat_run_kernel(A, dim, out_shape, out)
    return out
