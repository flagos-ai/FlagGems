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

"""pad candidate v4: explicit flat kernels for 1D/2D/3D constant each.

v2 used constexpr tuple + tl.static_range decomposition by dimension; measured 2D took the output stride
as the input stride (output (69,69) decomposed by 64), causing misaligned copies; 3D passed by chance.
v3 uses one kernel per rank with explicit scalar constexpr parameters, no tuple indexing.
v4 restores production's clamping of negative source coordinates: GCU masked load requires legal pointers.

The production codegen kernel builds an if_pad vector per element, per-dimension tl.where clamps,
and repeatedly accumulates load_cond; in constant mode these reduce to one valid mask,
so a single masked load does "copy or fill value".
Non-constant, non-contiguous and rank>3 fall back to production.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _pad_rank1_kernel(
    X,
    Out,
    N_OUT: tl.constexpr,
    VALUE: tl.constexpr,
    LEFT: tl.constexpr,
    N_IN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    src_coord = out_offset - LEFT
    valid = (out_offset < N_OUT) & (src_coord >= 0) & (src_coord < N_IN)
    value_or_input = tl.load(X + src_coord, mask=valid, other=VALUE)
    tl.store(Out + out_offset, value_or_input, mask=out_offset < N_OUT)


@triton.jit
def _pad_rank2_kernel(
    X,
    Out,
    N_OUT: tl.constexpr,
    VALUE: tl.constexpr,
    OUT_S0: tl.constexpr,
    IN_S0: tl.constexpr,
    PB0: tl.constexpr,
    PB1: tl.constexpr,
    IN_D0: tl.constexpr,
    IN_D1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    coord0 = out_offset // OUT_S0
    coord1 = out_offset - coord0 * OUT_S0
    src0 = coord0 - PB0
    src1 = coord1 - PB1
    valid = (
        (out_offset < N_OUT)
        & (src0 >= 0)
        & (src0 < IN_D0)
        & (src1 >= 0)
        & (src1 < IN_D1)
    )
    src_offset = src0 * IN_S0 + src1
    # GCU backend masked load still requires pointers in a legal range; negative offsets must be clamped first
    safe_src_offset = tl.where(valid, src_offset, 0)
    value_or_input = tl.load(X + safe_src_offset, mask=valid, other=VALUE)
    tl.store(Out + out_offset, value_or_input, mask=out_offset < N_OUT)


@triton.jit
def _pad_rank3_kernel(
    X,
    Out,
    N_OUT: tl.constexpr,
    VALUE: tl.constexpr,
    OUT_S0: tl.constexpr,
    OUT_S1: tl.constexpr,
    IN_S0: tl.constexpr,
    IN_S1: tl.constexpr,
    PB0: tl.constexpr,
    PB1: tl.constexpr,
    PB2: tl.constexpr,
    IN_D0: tl.constexpr,
    IN_D1: tl.constexpr,
    IN_D2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    coord0 = out_offset // OUT_S0
    rest0 = out_offset - coord0 * OUT_S0
    coord1 = rest0 // OUT_S1
    coord2 = rest0 - coord1 * OUT_S1
    src0 = coord0 - PB0
    src1 = coord1 - PB1
    src2 = coord2 - PB2
    valid = (
        (out_offset < N_OUT)
        & (src0 >= 0)
        & (src0 < IN_D0)
        & (src1 >= 0)
        & (src1 < IN_D1)
        & (src2 >= 0)
        & (src2 < IN_D2)
    )
    src_offset = src0 * IN_S0 + src1 * IN_S1 + src2
    safe_src_offset = tl.where(valid, src_offset, 0)
    value_or_input = tl.load(X + safe_src_offset, mask=valid, other=VALUE)
    tl.store(Out + out_offset, value_or_input, mask=out_offset < N_OUT)


def _pick_block(n_out):
    if n_out < 65536:
        return 1024, 2
    if n_out < (1 << 24):
        return 16384, 2
    return 32768, 4


def _launch(A, pad_before, pad_after, value):
    in_shape = tuple(A.shape)
    rank = len(in_shape)
    out_shape = tuple(in_shape[d] + pad_before[d] + pad_after[d] for d in range(rank))
    out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
    n_out = out.numel()
    BLOCK, num_warps = _pick_block(n_out)
    grid = (triton.cdiv(n_out, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n_out, BLOCK),)
    if rank == 1:
        _pad_rank1_kernel[grid](
            A,
            out,
            n_out,
            value,
            pad_before[0],
            in_shape[0],
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    elif rank == 2:
        _pad_rank2_kernel[grid](
            A,
            out,
            n_out,
            value,
            out_shape[1],
            in_shape[1],
            pad_before[0],
            pad_before[1],
            in_shape[0],
            in_shape[1],
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    else:
        _pad_rank3_kernel[grid](
            A,
            out,
            n_out,
            value,
            out_shape[1] * out_shape[2],
            out_shape[2],
            in_shape[1] * in_shape[2],
            in_shape[2],
            pad_before[0],
            pad_before[1],
            pad_before[2],
            in_shape[0],
            in_shape[1],
            in_shape[2],
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out


def pad(self, pad, mode="constant", value=None):
    """Same signature as the production entry: self is the input tensor, pad is a list of length 2*ndim."""
    logger.debug("GEMS_ENFLAME_GCU400 PAD")
    if value is None:
        value = 0.0
    rank = self.ndim
    if (
        mode == "constant"
        and self.is_contiguous()
        and rank <= 3
        and len(pad) == 2 * rank
    ):
        # #
        pad_before = [0] * rank
        pad_after = [0] * rank
        for dim_from_last in range(rank):
            dim = rank - dim_from_last - 1
            pad_before[dim] = int(pad[2 * dim_from_last])
            pad_after[dim] = int(pad[2 * dim_from_last + 1])
        return _launch(self, pad_before, pad_after, float(value))
    from flag_gems.runtime.backend._enflame.gcu400.ops.pad import pad as _public

    return _public(self, pad, mode, value)
