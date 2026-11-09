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

"""GCU300 specialization of ``stack``.

The generic copy kernel widens ``pid * BLOCK + arange`` to int64 before it
derives the pre/post index and the output offset.  This target miscompiles
lane-varying int64 arithmetic once a block covers 256 or more lanes, so roughly
half of each copy lands in the wrong place.

This module keeps the flat index and every value derived from it in int32 and
uses a second kernel for inputs whose element count reaches 2**31, where the
flat index no longer fits: that one owns a whole ``post`` run per program and
folds the run bases (int64 scalars) into the pointers before adding an int32
vector.
"""

import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_I32 = 1 << 31
_BLOCK = 1024


@triton.jit
def stack_copy_func_kernel_4(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    dim_size_out,
    dim_prod_post,
    dim_offset_a,
    dim_offset_b,
    dim_offset_c,
    dim_offset_d,
    total_elements_a,
    total_elements_b,
    total_elements_c,
    total_elements_d,
    BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    if pid_y == 0:
        in_ptr = in_ptr_a
        dim_offset = dim_offset_a
        total_elements = total_elements_a
    elif pid_y == 1:
        in_ptr = in_ptr_b
        dim_offset = dim_offset_b
        total_elements = total_elements_b
    elif pid_y == 2:
        in_ptr = in_ptr_c
        dim_offset = dim_offset_c
        total_elements = total_elements_c
    else:
        in_ptr = in_ptr_d
        dim_offset = dim_offset_d
        total_elements = total_elements_d

    # int32 from the start; see the module docstring.
    idx = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = idx < total_elements

    pre_idx = idx // dim_prod_post
    post_idx = idx - pre_idx * dim_prod_post
    out_idx = pre_idx * (dim_size_out * dim_prod_post) + dim_offset * dim_prod_post
    out_idx = out_idx + post_idx

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


@triton.jit
def stack_copy_run_kernel(
    out_ptr,
    in_ptr,
    dim_size_out,
    dim_prod_post,
    dim_offset,
    runs,
    BLOCK_X: tl.constexpr,
):
    """Copy one ``post`` run per program, for inputs of 2**31 elements or more.

    Program ``pid`` owns run ``pid`` of the flat input (run = one ``post``
    group).  The run bases are int64 scalars and are folded into the pointers;
    only the intra-run index, bounded by ``dim_prod_post``, is a 32-bit vector.
    """
    run = tl.program_id(0)
    if run < runs:
        in_base = run.to(tl.int64) * dim_prod_post
        pre_idx = run // dim_size_out
        out_base = (
            pre_idx.to(tl.int64) * dim_size_out * dim_prod_post
            + dim_offset.to(tl.int64) * dim_prod_post
        )
        src = in_ptr + in_base.to(tl.int64)
        dst = out_ptr + out_base
        for start in range(0, dim_prod_post, BLOCK_X):
            offs = start + tl.arange(0, BLOCK_X)
            mask = offs < dim_prod_post
            tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def _launch(out, tensors, dim, dim_prod_post):
    BLOCK = _BLOCK
    dim_size_out = len(tensors)
    i = 0
    while i < len(tensors):
        batch = tensors[i : i + 4]
        args = []
        totals = []
        for j in range(4):
            if j < len(batch):
                tensor = batch[j].contiguous()
                total = tensor.numel()
                args.extend([tensor, i + j, total])
                totals.append(total)
            else:
                args.extend([batch[0], 0, 0])
                totals.append(0)

        max_elements = batch[0].numel()
        if max_elements >= _MAX_I32:
            # Run-per-program fallback: the flat index no longer fits int32.
            for j in range(len(batch)):
                tensor = args[3 * j]
                runs = tensor.numel() // dim_prod_post
                stack_copy_run_kernel[(runs,)](
                    out,
                    tensor,
                    dim_size_out,
                    dim_prod_post,
                    i + j,
                    runs,
                    BLOCK_X=BLOCK,
                )
        else:
            grid = (triton.cdiv(max_elements, BLOCK), len(batch))
            stack_copy_func_kernel_4[grid](
                out,
                args[0],
                args[3],
                args[6],
                args[9],
                dim_size_out,
                dim_prod_post,
                args[1],
                args[4],
                args[7],
                args[10],
                args[2],
                args[5],
                args[8],
                args[11],
                BLOCK_X=BLOCK,
            )
        i += len(batch)
    return out


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME STACK")

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    for i, s in enumerate(inp_shapes[1:]):
        if (dim < -tensors[i + 1].dim() - 1) or (dim > tensors[i + 1].dim()):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -tensors[i + 1].dim() - 1, tensors[i + 1].dim(), dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                "stack expects each tensor to be equal size, but got "
                f"{inp0_shape} at entry 0 and {s} at entry {i + 1}"
            )

    if dim < 0:
        dim = dim + len(inp0_shape) + 1

    dtypes = [t.dtype for t in tensors]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)
    tensors = [t.to(dtype) if t.dtype != dtype else t for t in tensors]
    device = tensors[0].device
    out_shape = inp0_shape[:dim] + [len(tensors)] + inp0_shape[dim:]
    out = torch.empty(out_shape, dtype=dtype, device=device)

    dim_prod_post = 1
    for s in inp0_shape[dim:]:
        dim_prod_post *= s

    return _launch(out, tensors, dim, dim_prod_post)
