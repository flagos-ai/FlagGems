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
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _is_float8_e8m0fnu(dtype: torch.dtype) -> bool:
    return str(dtype) == "torch.float8_e8m0fnu"


def _should_use_uint8_view_path(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
) -> bool:
    if len(A) == 0:
        return False
    first_dtype = A[0].dtype
    if not _is_float8_e8m0fnu(first_dtype):
        return False
    if A[0].element_size() != 1:
        return False
    for tensor in A[1:]:
        if tensor.dtype != first_dtype or tensor.element_size() != 1:
            return False
    return True


def _cat_build_working_list_uint8_view(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    dim: int,
):
    original_dtype = A[0].dtype
    A_u8 = [tensor.view(torch.uint8) for tensor in A]
    mode, payload = _cat_build_working_list(A_u8, dim)
    return mode, payload, original_dtype


# GCU limits each grid dimension as follows (triton_gcu GcuLauncher):
#   grid.x <= 65535, grid.y <= 255, grid.z <= 255.
# Only grid.x can be large, so we always keep num_tensors (<= 4) on grid.y,
# keep the "rows" axis chunked on grid.z (with an in-kernel loop), and pack
# per-row blocks on grid.x (scaling BLOCK_X to stay within 65535).
_MAX_GRID_DIM = 65535
_MAX_GRID_DIM_Y = 255
_MAX_GRID_DIM_Z = 255
_BLOCK_MIN = 2048


@triton.jit
def cat_copy_contig_kernel_4(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    dim_offset_a: tl.int64,
    dim_offset_b: tl.int64,
    dim_offset_c: tl.int64,
    dim_offset_d: tl.int64,
    total_elements_a: tl.int64,
    total_elements_b: tl.int64,
    total_elements_c: tl.int64,
    total_elements_d: tl.int64,
    grid_x,
    BLOCK_X: tl.constexpr,
):
    # dim == 0: every input occupies a contiguous block of the output, so this
    # is a plain 1D masked copy with a per-tensor output base.
    # Grid: (grid_x, num_tensors, grid_z); block_id = pid_z*grid_x + pid_x.
    pid_x = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_z = tl.program_id(2)

    if pid_t == 0:
        in_ptr = in_ptr_a
        dim_offset = dim_offset_a
        total_elements = total_elements_a
    elif pid_t == 1:
        in_ptr = in_ptr_b
        dim_offset = dim_offset_b
        total_elements = total_elements_b
    elif pid_t == 2:
        in_ptr = in_ptr_c
        dim_offset = dim_offset_c
        total_elements = total_elements_c
    else:
        in_ptr = in_ptr_d
        dim_offset = dim_offset_d
        total_elements = total_elements_d

    block_start = (pid_z.to(tl.int64) * grid_x + pid_x) * BLOCK_X
    offsets = tl.arange(0, BLOCK_X)
    mask = block_start + offsets < total_elements

    data = tl.load(in_ptr + block_start + offsets, mask=mask)
    tl.store(out_ptr + block_start + dim_offset + offsets, data, mask=mask)


@triton.jit
def cat_copy_row_kernel_4(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    dim_size_in_a: tl.int64,
    dim_size_in_b: tl.int64,
    dim_size_in_c: tl.int64,
    dim_size_in_d: tl.int64,
    dim_offset_a: tl.int64,
    dim_offset_b: tl.int64,
    dim_offset_c: tl.int64,
    dim_offset_d: tl.int64,
    dim_size_out: tl.int64,
    dim_prod_post: tl.int64,
    rows,
    grid_z,
    BLOCK_X: tl.constexpr,
):
    # dim > 0: view each input as [pre, row_size] and the output as
    # [pre, dim_size_out*dim_prod_post]. Grid: (blocks_per_row, num_tensors,
    # grid_z). pid_col/pid_row0/pid_t map directly onto the axes -> NO division
    # at all; each program loops over the rows assigned to its z-slot.
    pid_col = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_row0 = tl.program_id(2)

    if pid_t == 0:
        in_ptr = in_ptr_a
        dim_size_in = dim_size_in_a
        dim_offset = dim_offset_a
    elif pid_t == 1:
        in_ptr = in_ptr_b
        dim_size_in = dim_size_in_b
        dim_offset = dim_offset_b
    elif pid_t == 2:
        in_ptr = in_ptr_c
        dim_size_in = dim_size_in_c
        dim_offset = dim_offset_c
    else:
        in_ptr = in_ptr_d
        dim_size_in = dim_size_in_d
        dim_offset = dim_offset_d

    row_size = dim_size_in * dim_prod_post
    out_row_size = dim_size_out * dim_prod_post

    offsets = tl.arange(0, BLOCK_X)
    col_start = pid_col.to(tl.int64) * BLOCK_X
    mask = col_start + offsets < row_size

    rows_per_prog = tl.cdiv(rows, grid_z)
    for r in range(rows_per_prog):
        row = (pid_row0 + r * grid_z).to(tl.int64)
        if row < rows:
            base_in = row * row_size + col_start
            base_out = row * out_row_size + dim_offset * dim_prod_post + col_start
            data = tl.load(in_ptr + base_in + offsets, mask=mask)
            tl.store(out_ptr + base_out + offsets, data, mask=mask)


def _cat_run_kernel(
    A: List[torch.Tensor],
    dim: int,
    out_shape: List[int],
    out: torch.Tensor,
):
    dim %= A[0].ndim
    dim_size_out = out_shape[dim]

    # pre: number of independent "rows" (product of leading dims)
    pre = 1
    for d in range(dim):
        pre *= A[0].shape[d]

    # dim_prod_post: number of contiguous elements after `dim`
    dim_prod_post = 1
    for d in range(dim + 1, A[0].ndim):
        dim_prod_post *= A[0].shape[d]

    dim_offset = 0
    i = 0
    while i < len(A):
        tensors_in_batch = A[i : i + 4]
        num_tensors_in_batch = len(tensors_in_batch)

        slots = []
        max_numel = 0
        max_row_size = 0
        current_dim_offset = dim_offset

        for j in range(4):
            if j < num_tensors_in_batch:
                tensor = tensors_in_batch[j].contiguous()
                total_elements = tensor.numel()
                dim_size_in = tensor.shape[dim]
                slots.append((tensor, dim_size_in, current_dim_offset, total_elements))
                max_numel = max(max_numel, total_elements)
                max_row_size = max(max_row_size, dim_size_in * dim_prod_post)
                current_dim_offset += dim_size_in
            else:
                # Placeholders for unused tensor slots (never dereferenced).
                slots.append((tensors_in_batch[0], 0, 0, 0))

        (
            tensor_a,
            dim_size_in_a,
            dim_offset_a,
            total_elements_a,
            tensor_b,
            dim_size_in_b,
            dim_offset_b,
            total_elements_b,
            tensor_c,
            dim_size_in_c,
            dim_offset_c,
            total_elements_c,
            tensor_d,
            dim_size_in_d,
            dim_offset_d,
            total_elements_d,
        ) = [item for slot in slots for item in slot]

        if dim == 0:
            # Contiguous block copy: one masked 1D copy per tensor.
            num_elems = max(1, max_numel)
            BLOCK = _BLOCK_MIN
            # Prefer a single x-axis (grid_z == 1) with fewer, larger blocks to
            # minimize block-scheduling overhead. Cap BLOCK at 32K and let the
            # grid_z axis absorb anything larger (tensors > ~2.1G elements).
            while triton.cdiv(num_elems, BLOCK) > _MAX_GRID_DIM and BLOCK < 32768:
                BLOCK *= 2
            total_blocks = triton.cdiv(num_elems, BLOCK)
            grid_x = min(total_blocks, _MAX_GRID_DIM)
            grid_z = triton.cdiv(total_blocks, grid_x)
            grid = (grid_x, num_tensors_in_batch, grid_z)

            cat_copy_contig_kernel_4[grid](
                out,
                tensor_a,
                tensor_b,
                tensor_c,
                tensor_d,
                # For dim == 0 the output base is the cumulative *numel* offset
                # (= dim_offset * dim_prod_post), not the dim-unit offset.
                dim_offset_a * dim_prod_post,
                dim_offset_b * dim_prod_post,
                dim_offset_c * dim_prod_post,
                dim_offset_d * dim_prod_post,
                total_elements_a,
                total_elements_b,
                total_elements_c,
                total_elements_d,
                grid_x,
                BLOCK_X=BLOCK,
            )
        else:
            # Row-based copy: grid = (blocks_per_row, num_tensors, grid_z),
            # no division, rows handled by the in-kernel loop.
            row_size = max(1, max_row_size)
            BLOCK = _BLOCK_MIN
            if row_size < BLOCK:
                # Shrink the block to fit a small row for better lane utilization.
                BLOCK = max(64, triton.next_power_of_2(row_size))
            while triton.cdiv(row_size, BLOCK) > _MAX_GRID_DIM:
                BLOCK *= 2
            blocks_per_row = triton.cdiv(row_size, BLOCK)
            grid_z = min(_MAX_GRID_DIM_Z, max(1, pre))
            grid = (blocks_per_row, num_tensors_in_batch, grid_z)

            cat_copy_row_kernel_4[grid](
                out,
                tensor_a,
                tensor_b,
                tensor_c,
                tensor_d,
                dim_size_in_a,
                dim_size_in_b,
                dim_size_in_c,
                dim_size_in_d,
                dim_offset_a,
                dim_offset_b,
                dim_offset_c,
                dim_offset_d,
                dim_size_out,
                dim_prod_post,
                pre,
                grid_z,
                BLOCK_X=BLOCK,
            )

        dim_offset = current_dim_offset
        i += num_tensors_in_batch


def _cat_build_working_list(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int
):
    """Returns (mode, payload) where mode is 'single'|'empty'|'multi'."""
    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return "single", A[0]

    device = A[0].device
    dtype = A[0].dtype
    A = list(A)
    for i in range(len(A) - 1, -1, -1):
        if A[i].shape == torch.Size([0]):
            A.pop(i)
    if len(A) == 0:
        return "empty", torch.tensor([], device=device, dtype=dtype)
    if len(A) == 1:
        return "single", A[0]

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    dim %= A[0].ndim

    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx != dim and length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    dtypes = [t.dtype for t in A]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)
    A = [t.to(dtype) if t.dtype != dtype else t for t in A]

    shapes = [t.shape for t in A]
    cat_dim_sizes = [s[dim] for s in shapes]
    out_shape = list(shapes[0])
    out_shape[dim] = sum(cat_dim_sizes)
    return "multi", (A, dim, out_shape, dtype, A[0].device)


def cat_out(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    dim: int = 0,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME CAT")
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


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME CAT")
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
