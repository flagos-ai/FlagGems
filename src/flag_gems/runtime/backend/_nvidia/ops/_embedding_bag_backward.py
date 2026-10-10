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

from flag_gems.ops._embedding_bag_backward import (
    _eb_backward_validate_body,
    _embedding_bag_backward_impl,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _validate_max(
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    _eb_backward_validate_body(
        indices,
        offsets,
        offset_to_bag,
        bag_size,
        maximum_indices,
        indices,
        indices,
        indices,
        indices,
        out,
        out,
        0,
        meta[0],
        meta[1],
        meta[2],
        meta[3],
        meta[4],
        meta[5],
        meta[8],
        meta[9],
        meta[10],
        meta[11],
        meta[12],
        meta[13],
        meta[14],
        2,
        False,
        False,
        False,
        True,
        True,
        0,
        256,
        False,
    )


@libentry()
@triton.jit(debug=True)
def max_owned_tiles(
    grad,  # output gradient
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    acc,  # accumulator buffer
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    _validate_max(indices, offsets, offset_to_bag, bag_size, maximum_indices, out, meta)
    num_bags: tl.constexpr = meta[1]
    embedding_dim: tl.constexpr = meta[2]
    num_weights: tl.constexpr = meta[3]
    pad: tl.constexpr = meta[5]
    sg0: tl.constexpr = meta[6]
    sg1: tl.constexpr = meta[7]
    sb: tl.constexpr = meta[11]
    sx0: tl.constexpr = meta[12]
    sx1: tl.constexpr = meta[13]
    fp64: tl.constexpr = meta[15]
    CAST: tl.constexpr = meta[16]
    BR: tl.constexpr = meta[17]
    bd: tl.constexpr = meta[18]
    BB: tl.constexpr = meta[19]
    pid = tl.program_id(0).to(tl.int64)
    nc: tl.constexpr = tl.cdiv(embedding_dim, bd)
    if pid < tl.cdiv(num_weights, BR) * nc:
        row0 = pid // nc * BR
        rows = row0 + tl.arange(0, BR)
        cols = pid % nc * bd + tl.arange(0, bd)
        positions = rows[:, None] * embedding_dim + cols[None, :]
        output_mask = (rows[:, None] < num_weights) & (cols[None, :] < embedding_dim)
        tl.store(acc + positions, 0, output_mask)
        # Each CTA owns all addresses in this output tile. No other CTA writes
        # its zeros, atomics, or cast, so a CTA barrier is sufficient here.
        tl.debug_barrier()
        if fp64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        for start in range(0, num_bags, BB):
            bags = start + tl.arange(0, BB)
            maximum = tl.load(
                maximum_indices + bags[:, None] * sx0 + cols[None, :] * sx1,
                (bags[:, None] < num_bags) & (cols[None, :] < embedding_dim),
                other=-1,
            ).to(tl.int64)
            sizes = tl.load(bag_size + bags * sb, bags < num_bags, other=0)
            active = (
                (bags[:, None] < num_bags)
                & (cols[None, :] < embedding_dim)
                & (maximum >= row0)
                & (maximum < row0 + BR)
                & (maximum < num_weights)
                & (maximum != pad)
                & (sizes[:, None] > 0)
            )
            values = tl.load(
                grad + bags[:, None] * sg0 + cols[None, :] * sg1,
                active,
                other=0,
            ).to(acc_dtype)
            tl.atomic_add(
                acc + maximum * embedding_dim + cols[None, :],
                values,
                active,
                sem="relaxed",
            )
        if CAST:
            tl.debug_barrier()
            values = tl.load(acc + positions, output_mask, other=0)
            tl.store(out + positions, values, output_mask)


@libentry()
@triton.jit(debug=True)
def max_shared_tiles(
    grad,  # output gradient
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    acc,  # accumulator buffer
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    _validate_max(indices, offsets, offset_to_bag, bag_size, maximum_indices, out, meta)
    num_bags: tl.constexpr = meta[1]
    embedding_dim: tl.constexpr = meta[2]
    num_weights: tl.constexpr = meta[3]
    pad: tl.constexpr = meta[5]
    sg0: tl.constexpr = meta[6]
    sg1: tl.constexpr = meta[7]
    sb: tl.constexpr = meta[11]
    sx0: tl.constexpr = meta[12]
    sx1: tl.constexpr = meta[13]
    fp64: tl.constexpr = meta[15]
    SMEM_ASM: tl.constexpr = meta[20]
    BR: tl.constexpr = meta[17]
    bd: tl.constexpr = meta[18]
    BB: tl.constexpr = meta[19]
    pid = tl.program_id(0).to(tl.int64)
    nc: tl.constexpr = tl.cdiv(embedding_dim, bd)
    if pid < tl.cdiv(num_weights, BR) * nc:
        row0 = pid // nc * BR
        rows = row0 + tl.arange(0, BR)
        cols = pid % nc * bd + tl.arange(0, bd)
        positions = rows[:, None] * embedding_dim + cols[None, :]
        output_mask = (rows[:, None] < num_weights) & (cols[None, :] < embedding_dim)
        base = tl.inline_asm_elementwise(
            SMEM_ASM,
            constraints="=r,r",
            args=[tl.full((), 0, tl.int32)],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
        smem_offsets = (
            tl.arange(0, BR)[:, None] * (bd + 1) + tl.arange(0, bd)[None, :]
        ) * 4
        tl.inline_asm_elementwise(
            "{ .reg .u32 a; add.u32 a, $1, $2; st.shared.u32 [a], 0; mov.u32 $0, 0; }",
            constraints="=r,r,r",
            args=[base, smem_offsets],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        if fp64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        for start in range(0, num_bags, BB):
            bags = start + tl.arange(0, BB)
            maximum = tl.load(
                maximum_indices + bags[:, None] * sx0 + cols[None, :] * sx1,
                (bags[:, None] < num_bags) & (cols[None, :] < embedding_dim),
                other=-1,
            ).to(tl.int64)
            sizes = tl.load(bag_size + bags * sb, bags < num_bags, other=0)
            active = (
                (bags[:, None] < num_bags)
                & (cols[None, :] < embedding_dim)
                & (maximum >= row0)
                & (maximum < row0 + BR)
                & (maximum < num_weights)
                & (maximum != pad)
                & (sizes[:, None] > 0)
            )
            values = tl.load(
                grad + bags[:, None] * sg0 + cols[None, :] * sg1,
                active,
                other=0,
            ).to(acc_dtype)
            smem_index = (
                (maximum - row0).to(tl.int32) * (bd + 1) + tl.arange(0, bd)[None, :]
            ) * 4
            tl.inline_asm_elementwise(
                "{ .reg .pred p; .reg .u32 a; .reg .f32 v; "
                "setp.ne.u32 p, $4, 0; add.u32 a, $1, $2; "
                "@p atom.shared.add.f32 v, [a], $3; mov.u32 $0, 0; }",
                constraints="=r,r,r,f,r",
                args=[base, smem_index, values.to(tl.float32), active.to(tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        tl.debug_barrier()
        values = tl.inline_asm_elementwise(
            "{ .reg .u32 a; add.u32 a, $1, $2; ld.shared.f32 $0, [a]; }",
            constraints="=f,r,r",
            args=[base, smem_offsets],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
        )
        tl.store(out + positions, values, output_mask)


def _compute_max(grad, indices, offsets, mapping, sizes, maximum, num_weights, padding):
    b, d = grad.shape
    n = indices.numel()
    # One CTA owns all output addresses in its row/feature tile. Reduced
    # dtypes accumulate in NVIDIA shared FP32 storage, then cast exactly once.
    # FP32/FP64 use the output allocation as their accumulation storage.
    block_rows, block_cols, block_bags = 512, 8, 256
    reduced = grad.dtype in (torch.float16, torch.bfloat16)
    if reduced:
        block_rows = 128 if b <= 32 else (1024 if b >= 512 else 512)
        # Inline PTX atomics execute per physical lane. Keep the bag dimension
        # at least the CTA lane count so Triton never replicates a contribution.
        block_bags = max(128, min(triton.next_power_of_2(max(b, 1)), 256))
    kernel = max_shared_tiles if reduced else max_owned_tiles
    math_blocks = triton.cdiv(num_weights, block_rows) * triton.cdiv(d, block_cols)
    validate_blocks = triton.cdiv(max(n, offsets.numel(), b, b * d, 1), 256)
    blocks = max(math_blocks, validate_blocks)
    output = torch.empty((num_weights, d), dtype=grad.dtype, device=grad.device)
    metadata = (
        n,
        b,
        d,
        num_weights,
        offsets.numel(),
        padding,
        *grad.stride(),
        indices.stride(0),
        offsets.stride(0),
        mapping.stride(0),
        sizes.stride(0),
        *maximum.stride(),
        mapping.numel() != 0 and d != 0,
        grad.dtype == torch.float64,
        False,
        block_rows,
        block_cols,
        block_bags,
        "{ .shared .align 4 .b8 eb_smem["
        + str(block_rows * (block_cols + 1) * 4)
        + "]; mov.u32 $0, eb_smem; }",
    )
    with torch_device_fn.device(grad.device):
        kernel[(blocks,)](
            grad,
            indices,
            offsets,
            mapping,
            sizes,
            maximum,
            output,
            output,
            metadata,
            debug=True,
        )
    return output


def _embedding_bag_backward(
    grad,
    indices,
    offsets,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights=None,
    padding_idx=-1,
):
    logger.debug("GEMS_NVIDIA _EMBEDDING_BAG_BACKWARD")
    return _embedding_bag_backward_impl(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        padding_idx,
        max_fn=_compute_max,
    )
