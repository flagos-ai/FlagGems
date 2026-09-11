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

from flag_gems.ops._embedding_bag_backward import (
    _eb_backward_validate_body,
    _embedding_bag_backward_impl,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@triton.jit
def _validate_max(
    INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, META: tl.constexpr
):
    _eb_backward_validate_body(
        INDICES,
        OFFSETS,
        OFFSET2BAG,
        BAG_SIZE,
        MAXIMUM,
        INDICES,
        INDICES,
        INDICES,
        INDICES,
        OUT,
        OUT,
        0,
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
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
    GRAD,
    INDICES,
    OFFSETS,
    OFFSET2BAG,
    BAG_SIZE,
    MAXIMUM,
    ACC,
    OUT,
    META: tl.constexpr,
):
    _validate_max(INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, META)
    B: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    V: tl.constexpr = META[3]
    PAD: tl.constexpr = META[5]
    SG0: tl.constexpr = META[6]
    SG1: tl.constexpr = META[7]
    SB: tl.constexpr = META[11]
    SX0: tl.constexpr = META[12]
    SX1: tl.constexpr = META[13]
    FP64: tl.constexpr = META[15]
    CAST: tl.constexpr = META[16]
    BR: tl.constexpr = META[17]
    BD: tl.constexpr = META[18]
    BB: tl.constexpr = META[19]
    pid = tl.program_id(0).to(tl.int64)
    nc: tl.constexpr = tl.cdiv(D, BD)
    if pid < tl.cdiv(V, BR) * nc:
        row0 = pid // nc * BR
        rows = row0 + tl.arange(0, BR)
        cols = pid % nc * BD + tl.arange(0, BD)
        positions = rows[:, None] * D + cols[None, :]
        output_mask = (rows[:, None] < V) & (cols[None, :] < D)
        tl.store(ACC + positions, 0, output_mask)
        # Each CTA owns all addresses in this output tile. No other CTA writes
        # its zeros, atomics, or cast, so a CTA barrier is sufficient here.
        tl.debug_barrier()
        if FP64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        for start in range(0, B, BB):
            bags = start + tl.arange(0, BB)
            maximum = tl.load(
                MAXIMUM + bags[:, None] * SX0 + cols[None, :] * SX1,
                (bags[:, None] < B) & (cols[None, :] < D),
                other=-1,
            ).to(tl.int64)
            sizes = tl.load(BAG_SIZE + bags * SB, bags < B, other=0)
            active = (
                (bags[:, None] < B)
                & (cols[None, :] < D)
                & (maximum >= row0)
                & (maximum < row0 + BR)
                & (maximum < V)
                & (maximum != PAD)
                & (sizes[:, None] > 0)
            )
            values = tl.load(
                GRAD + bags[:, None] * SG0 + cols[None, :] * SG1,
                active,
                other=0,
            ).to(acc_dtype)
            tl.atomic_add(
                ACC + maximum * D + cols[None, :], values, active, sem="relaxed"
            )
        if CAST:
            tl.debug_barrier()
            values = tl.load(ACC + positions, output_mask, other=0)
            tl.store(OUT + positions, values, output_mask)


@libentry()
@triton.jit(debug=True)
def max_shared_tiles(
    GRAD,
    INDICES,
    OFFSETS,
    OFFSET2BAG,
    BAG_SIZE,
    MAXIMUM,
    ACC,
    OUT,
    META: tl.constexpr,
):
    _validate_max(INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, META)
    B: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    V: tl.constexpr = META[3]
    PAD: tl.constexpr = META[5]
    SG0: tl.constexpr = META[6]
    SG1: tl.constexpr = META[7]
    SB: tl.constexpr = META[11]
    SX0: tl.constexpr = META[12]
    SX1: tl.constexpr = META[13]
    FP64: tl.constexpr = META[15]
    SMEM_ASM: tl.constexpr = META[20]
    BR: tl.constexpr = META[17]
    BD: tl.constexpr = META[18]
    BB: tl.constexpr = META[19]
    pid = tl.program_id(0).to(tl.int64)
    nc: tl.constexpr = tl.cdiv(D, BD)
    if pid < tl.cdiv(V, BR) * nc:
        row0 = pid // nc * BR
        rows = row0 + tl.arange(0, BR)
        cols = pid % nc * BD + tl.arange(0, BD)
        positions = rows[:, None] * D + cols[None, :]
        output_mask = (rows[:, None] < V) & (cols[None, :] < D)
        base = tl.inline_asm_elementwise(
            SMEM_ASM,
            constraints="=r,r",
            args=[tl.full((), 0, tl.int32)],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
        smem_offsets = (
            tl.arange(0, BR)[:, None] * (BD + 1) + tl.arange(0, BD)[None, :]
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
        if FP64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        for start in range(0, B, BB):
            bags = start + tl.arange(0, BB)
            maximum = tl.load(
                MAXIMUM + bags[:, None] * SX0 + cols[None, :] * SX1,
                (bags[:, None] < B) & (cols[None, :] < D),
                other=-1,
            ).to(tl.int64)
            sizes = tl.load(BAG_SIZE + bags * SB, bags < B, other=0)
            active = (
                (bags[:, None] < B)
                & (cols[None, :] < D)
                & (maximum >= row0)
                & (maximum < row0 + BR)
                & (maximum < V)
                & (maximum != PAD)
                & (sizes[:, None] > 0)
            )
            values = tl.load(
                GRAD + bags[:, None] * SG0 + cols[None, :] * SG1,
                active,
                other=0,
            ).to(acc_dtype)
            smem_index = (
                (maximum - row0).to(tl.int32) * (BD + 1) + tl.arange(0, BD)[None, :]
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
        tl.store(OUT + positions, values, output_mask)


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
