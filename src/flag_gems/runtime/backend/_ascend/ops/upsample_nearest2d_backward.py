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

import triton
import triton.language as tl

from flag_gems.utils import libentry


@libentry()
@triton.jit
def _nearest2d_backward_channels_last(
    GO,
    GI,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SCALE_H: tl.constexpr,
    SCALE_W: tl.constexpr,
    H_STEPS: tl.constexpr,
    W_STEPS: tl.constexpr,
    PIXELS: tl.constexpr,
    BLOCK: tl.constexpr,
    PATCH: tl.constexpr,
):
    # Each core owns consecutive tiles of input pixels. Read contiguous
    # channel runs into UB, then gather the short inverse windows locally.
    tiles: tl.constexpr = triton.cdiv(IW, PIXELS)
    tasks: tl.constexpr = N * IH * tiles
    count = tl.cdiv(tasks, tl.num_programs(0))
    first = tl.program_id(0) * count
    index = tl.arange(0, BLOCK)
    col = tl.floor(tl.div_rn(index.to(tl.float32), C))
    channel = (index.to(tl.float32) - col * C).to(tl.int32)
    patch_index = tl.arange(0, PATCH)
    for task in range(first, tl.minimum(first + count, tasks)):
        row = task // tiles
        batch = row // IH
        iy = row - batch * IH
        first_col = (task - row * tiles) * PIXELS
        ix = first_col + col
        # Keep a prefix mask so stores do not rewrite neighboring row tails.
        valid = index < tl.minimum(PIXELS, IW - first_col) * C
        h0 = tl.minimum(tl.ceil(iy.to(tl.float32) * SCALE_H), OH).to(tl.int32)
        h1 = tl.minimum(tl.ceil((iy + 1).to(tl.float32) * SCALE_H), OH).to(tl.int32)
        w0 = tl.minimum(tl.ceil(ix * SCALE_W), OW)
        w1 = tl.minimum(tl.ceil((ix + 1) * SCALE_W), OW)
        patch_col = tl.minimum(tl.ceil(first_col.to(tl.float32) * SCALE_W), OW).to(
            tl.int32
        )
        acc = tl.full((BLOCK,), 0, tl.float32)
        for dy in tl.static_range(H_STEPS):
            oy = h0 + dy
            # Clamp the source row and mask its contribution below. A scalar
            # FP32-derived height condition in the load mask fails to lower on
            # CANN 9.1.1; the contiguous column mask is sufficient for safety.
            values = tl.load(
                GO
                + ((batch * OH + tl.minimum(oy, OH - 1)) * OW + patch_col) * C
                + patch_index,
                patch_index < (OW - patch_col) * C,
                other=0,
            ).to(tl.float32)
            for dx in tl.static_range(W_STEPS):
                ox = w0 + dx
                local = (ox.to(tl.int32) - patch_col) * C + channel
                local = tl.minimum(tl.maximum(local, 0), PATCH - 1)
                if PATCH == 1:
                    value = tl.sum(values, 0)
                else:
                    value = tl.gather(values, local, 0)
                acc += tl.where(valid & (ox < w1) & (oy < h1), value, 0.0)
        tl.store(GI + row * IW * C + first_col * C + index, acc, valid)


@libentry()
@triton.jit
def _nearest2d_backward_strided_rows(
    GO,
    GI,
    PLANES: tl.constexpr,
    C: tl.constexpr,
    GO_N: tl.constexpr,
    GO_C: tl.constexpr,
    GO_W: tl.constexpr,
    IDENTITY: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SCALE_H: tl.constexpr,
    SCALE_W: tl.constexpr,
    H_STEPS: tl.constexpr,
    W_STEPS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    PATCH: tl.constexpr,
):
    # Read physical spans contiguously, including gaps between logical
    # pixels, and select the source values in UB. Keep stores contiguous;
    # strided destinations use a separate copy kernel to avoid incorrect
    # compiler lowering of a gather followed by a strided store.
    tiles: tl.constexpr = triton.cdiv(IH, ROWS)
    tasks: tl.constexpr = PLANES * tiles
    count = tl.cdiv(tasks, tl.num_programs(0))
    first = tl.program_id(0) * count
    index = tl.arange(0, BLOCK)
    index_f = index.to(tl.float32)
    row = tl.floor(tl.div_rn(index_f, IW))
    column = index_f - row * IW
    patch_index = tl.arange(0, PATCH)
    for task in range(first, tl.minimum(first + count, tasks)):
        plane = task // tiles
        batch = plane // C
        channel = plane - batch * C
        go_base = batch * GO_N + channel * GO_C
        first_row = (task - plane * tiles) * ROWS
        ih = first_row + row
        # An affine prefix lets the compiler shorten the physical store.
        # A row-coordinate mask instead reads and rewrites neighboring tiles.
        valid = index < tl.minimum(ROWS, IH - first_row) * IW
        if IDENTITY:
            values = tl.load(
                GO + go_base + first_row * IW * GO_W + patch_index,
                patch_index < (IH - first_row) * IW * GO_W - GO_W + 1,
                other=0,
            )
            if PATCH == 1:
                acc = values
            else:
                acc = tl.gather(values, index * GO_W, 0)
        else:
            h0 = tl.minimum(tl.ceil(ih * SCALE_H), OH)
            h1 = tl.minimum(tl.ceil((ih.to(tl.int32) + 1).to(tl.float32) * SCALE_H), OH)
            w0 = tl.minimum(tl.ceil(column * SCALE_W), OW)
            w1 = tl.minimum(
                tl.ceil((column.to(tl.int32) + 1).to(tl.float32) * SCALE_W), OW
            )
            patch_row = tl.minimum(tl.ceil(first_row.to(tl.float32) * SCALE_H), OH)
            if PATCH == 1:
                values = tl.load(GO + go_base).to(tl.float32)
            else:
                values = tl.load(
                    GO + go_base + patch_row.to(tl.int32) * OW * GO_W + patch_index,
                    patch_index < (OH - patch_row.to(tl.int32)) * OW * GO_W - GO_W + 1,
                    other=0,
                ).to(tl.float32)
            acc = tl.full((BLOCK,), 0, tl.float32)
            for dy in tl.static_range(H_STEPS):
                for dx in tl.static_range(W_STEPS):
                    oy = h0 + dy
                    ox = w0 + dx
                    local = (((oy - patch_row) * OW + ox) * GO_W).to(tl.int32)
                    local = tl.minimum(tl.maximum(local, 0), PATCH - 1)
                    if PATCH == 1:
                        value = values
                    else:
                        value = tl.gather(values, local, 0)
                    acc += tl.where(valid & (oy < h1) & (ox < w1), value, 0.0)
        tl.store(GI + plane * IH * IW + first_row * IW + index, acc, valid)


@libentry()
@triton.jit
def _nearest2d_backward_scatter(
    SRC, DST, TOTAL: tl.constexpr, STEP: tl.constexpr, BLOCK: tl.constexpr
):
    tasks: tl.constexpr = triton.cdiv(TOTAL, BLOCK)
    count = tl.cdiv(tasks, tl.num_programs(0))
    first = tl.program_id(0) * count
    index = tl.arange(0, BLOCK)
    for task in range(first, tl.minimum(first + count, tasks)):
        offset = task * BLOCK + index
        valid = index < tl.minimum(BLOCK, TOTAL - task * BLOCK)
        value = tl.load(SRC + offset, valid, other=0)
        tl.store(DST + offset * STEP, value, valid)
