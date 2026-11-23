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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Fixed Grid on GCU400 (48 CTAs per dim). The generic pointwise_dynamic
# codegen launches cdiv(out_numel, BLOCK_SIZE) CTAs, which exceeds the
# 65535 grid.x hardware limit for large outputs (e.g. 16x7x57x32x29 ->
# >200k CTAs -> OutOfResources). A fixed grid keeps every launch within
# the limit.
_GRID_SIZE = 48
# Metadata lanes for the outer dims (all dims except the last). aten allows
# arbitrary rank; 8 lanes cover every shape in the test suite.
_MAX_OUTER = 8


# Persistent-grid programs for the flat high-rank pad path.  grid.x must stay
# <= 65535 on GCU400; a strided loop covers the remaining tiles.
_MAX_PROGRAMS = 2048


@triton.jit(do_not_specialize=["value"])
def _pad_flat_kernel(
    in_ptr,
    out_ptr,
    value,
    N,
    out_shape_ptr,
    out_stride_ptr,
    in_stride_ptr,
    valid_start_ptr,
    valid_end_ptr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    for base in tl.range(tl.program_id(0) * BLOCK, N, NPROG * BLOCK):
        offs = base + cols
        mask = offs < N
        valid = mask
        src_off = tl.full((BLOCK,), 0, tl.int32)
        for d in tl.static_range(NDIM):
            osh = tl.load(out_shape_ptr + d)
            ostr = tl.load(out_stride_ptr + d)
            istr = tl.load(in_stride_ptr + d)
            vst = tl.load(valid_start_ptr + d)
            ven = tl.load(valid_end_ptr + d)
            coord = (offs // ostr) % osh
            valid = valid & (coord >= vst) & (coord < ven)
            src_off = src_off + (coord - vst) * istr
        val = tl.load(in_ptr + src_off, mask=valid, other=value)
        tl.store(out_ptr + offs, val, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["value"])
def constant_pad_nd_row_kernel(
    in_ptr,
    out_ptr,
    shape0,
    num_rest,
    num_rest_blocks,
    num_col_tile,
    in_last,
    out_last,
    pad_left,
    in_strides_ptr,
    outer_shapes_ptr,
    suffix_prods_ptr,
    valid_start_ptr,
    valid_end_ptr,
    value,
    OUTER_RANK: tl.constexpr,
    NUM_REST_DIMS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 1D row path (BLOCK_R == 1, wide last dims). Kept as a SEPARATE
    # kernel from the 2D row-block path on purpose: compiling both paths
    # into one kernel was measured to slow the 2D path down 3.4x on GCU400
    # (register pressure / occupancy), so each path gets its own kernel.
    pid0 = ext.program_id(0)
    pid1 = ext.program_id(1)
    pid2 = ext.program_id(2)

    lanes = tl.arange(0, 8)  # _MAX_OUTER metadata lanes
    lane_mask = lanes < OUTER_RANK
    rest_lane_mask = (lanes >= 1) & (lanes < OUTER_RANK)
    vs_v = tl.load(valid_start_ptr + lanes, mask=lane_mask, other=0)
    ve_v = tl.load(valid_end_ptr + lanes, mask=lane_mask, other=1)
    is_v = tl.load(in_strides_ptr + lanes, mask=lane_mask, other=0)
    sh_v = tl.load(outer_shapes_ptr + lanes, mask=lane_mask, other=1)
    sp_v = tl.load(suffix_prods_ptr + lanes, mask=rest_lane_mask, other=1)

    # Dim-0 scalars extracted once (where-sum; no per-iteration loads).
    sel0 = lanes == 0
    vs0 = tl.sum(tl.where(sel0, vs_v, 0))
    ve0 = tl.sum(tl.where(sel0, ve_v, 0))
    is0 = tl.sum(tl.where(sel0, is_v, 0))
    if NUM_REST_DIMS == 1:
        # Dim-1 scalars (3D inputs): loop-invariant, so extract once here
        # instead of inside the row-block loop (per-tile where-sum reductions
        # were measured to dominate the 3D runtime).
        sel1 = lanes == 1
        vs1 = tl.sum(tl.where(sel1, vs_v, 0))
        ve1 = tl.sum(tl.where(sel1, ve_v, 0))
        is1 = tl.sum(tl.where(sel1, is_v, 0))

    cols = tl.arange(0, BLOCK_C)

    for d0 in tl.range(pid0, shape0, GRID_SIZE):
        s0 = tl.where(d0 < vs0, 0, d0 - vs0)
        valid0 = (d0 >= vs0) & (d0 < ve0)
        src0 = s0.to(tl.int64) * is0
        for blk in tl.range(pid1, num_rest_blocks, GRID_SIZE):
            # 1D row path: rest is a single row index (scalar decode).
            rest = blk
            if NUM_REST_DIMS == 1:
                # rest enumerates dim 1 directly: no division at all.
                # (vs1/ve1/is1 extracted once at kernel top.)
                s1 = tl.where(rest < vs1, 0, rest - vs1)
                valid_r = (rest >= vs1) & (rest < ve1)
                src_r = s1.to(tl.int64) * is1
            elif NUM_REST_DIMS >= 2:
                idx_v = (rest // sp_v) % sh_v
                valid_v = (idx_v >= vs_v) & (idx_v < ve_v)
                valid_r = (
                    tl.sum((valid_v.to(tl.int32)) * (rest_lane_mask.to(tl.int32)))
                    == NUM_REST_DIMS
                )
                clamped = tl.where(idx_v < vs_v, 0, idx_v - vs_v)
                src_r = tl.sum(
                    clamped.to(tl.int64)
                    * is_v.to(tl.int64)
                    * rest_lane_mask.to(tl.int64)
                )
            else:
                # OUTER_RANK <= 1: rest enumerates dim 0 (2D input folded
                # into the row space by the host) or is the single row of
                # a 1D input. Decode from the dim-0 metadata; for 1D
                # inputs those are the dummy lane-0 values (vs0=0,
                # ve0=1, is0=0), which yield valid=True/src=0.
                valid = (rest >= vs0) & (rest < ve0) & (rest < num_rest)
                src_r = tl.where(rest < vs0, 0, rest - vs0).to(tl.int64) * is0
            if NUM_REST_DIMS >= 1:
                # NRD==0 already defines `valid` above (self-contained).
                valid = valid0 & valid_r & (rest < num_rest)

            row_src = src0 + src_r
            if NUM_REST_DIMS == 0:
                row_base = rest.to(tl.int64) * out_last
            else:
                row_base = (d0.to(tl.int64) * num_rest + rest) * out_last
            for ct in tl.range(pid2, num_col_tile, GRID_SIZE, num_stages=NUM_STAGES):
                col = ct * BLOCK_C + cols
                col_mask = col < out_last
                src_col = col - pad_left
                in_valid = (col >= pad_left) & (col < pad_left + in_last)
                load_mask = col_mask & in_valid & valid
                x_val = tl.load(in_ptr + row_src + src_col, mask=load_mask, other=value)
                tl.store(out_ptr + row_base + col, x_val, mask=col_mask)


@libentry()
@triton.jit(do_not_specialize=["value"])
def constant_pad_nd_kernel(
    in_ptr,
    out_ptr,
    shape0,
    num_rest,
    num_rest_blocks,
    num_col_tile,
    in_last,
    out_last,
    pad_left,
    in_strides_ptr,
    outer_shapes_ptr,
    suffix_prods_ptr,
    valid_start_ptr,
    valid_end_ptr,
    value,
    OUTER_RANK: tl.constexpr,
    NUM_REST_DIMS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 2D row-block path (BLOCK_R > 1, skinny last dims). See
    # constant_pad_nd_row_kernel for why the paths are split.
    pid0 = ext.program_id(0)
    pid1 = ext.program_id(1)
    pid2 = ext.program_id(2)

    lanes = tl.arange(0, 8)  # _MAX_OUTER metadata lanes
    lane_mask = lanes < OUTER_RANK
    rest_lane_mask = (lanes >= 1) & (lanes < OUTER_RANK)
    vs_v = tl.load(valid_start_ptr + lanes, mask=lane_mask, other=0)
    ve_v = tl.load(valid_end_ptr + lanes, mask=lane_mask, other=1)
    is_v = tl.load(in_strides_ptr + lanes, mask=lane_mask, other=0)
    sh_v = tl.load(outer_shapes_ptr + lanes, mask=lane_mask, other=1)
    sp_v = tl.load(suffix_prods_ptr + lanes, mask=rest_lane_mask, other=1)

    # Dim-0 scalars extracted once (where-sum; no per-iteration loads).
    sel0 = lanes == 0
    vs0 = tl.sum(tl.where(sel0, vs_v, 0))
    ve0 = tl.sum(tl.where(sel0, ve_v, 0))
    is0 = tl.sum(tl.where(sel0, is_v, 0))
    if NUM_REST_DIMS == 1:
        # Dim-1 scalars (3D inputs): loop-invariant, so extract once here
        # instead of inside the row-block loop (per-tile where-sum reductions
        # were measured to dominate the 3D runtime).
        sel1 = lanes == 1
        vs1 = tl.sum(tl.where(sel1, vs_v, 0))
        ve1 = tl.sum(tl.where(sel1, ve_v, 0))
        is1 = tl.sum(tl.where(sel1, is_v, 0))

    cols = tl.arange(0, BLOCK_C)
    rows_in_block = tl.arange(0, BLOCK_R)

    for d0 in tl.range(pid0, shape0, GRID_SIZE):
        s0 = tl.where(d0 < vs0, 0, d0 - vs0)
        valid0 = (d0 >= vs0) & (d0 < ve0)
        src0 = s0.to(tl.int64) * is0
        for blk in tl.range(pid1, num_rest_blocks, GRID_SIZE):
            # 2D row-block path for skinny last dims.
            rows = blk * BLOCK_R + rows_in_block  # [BLOCK_R]
            row_mask = rows < num_rest
            if NUM_REST_DIMS == 0:
                # 2D input (OUTER_RANK == 1): the row space IS dim 0.
                # The host passes shape0=1 so the d0 loop runs once with
                # d0=0 (src0=0), and rows here enumerate dim 0 directly
                # -- decode from the dim-0 metadata (vs0/ve0/is0), no
                # division at all. Keeping dim 0 in a scalar d0 loop
                # instead would waste (BLOCK_R-1)/BLOCK_R of every 2D
                # tile (num_rest would be 1).
                valid = (rows >= vs0) & (rows < ve0) & row_mask
                src_r = tl.where(rows < vs0, 0, rows - vs0).to(tl.int64) * is0
            elif NUM_REST_DIMS == 1:
                # 3D input: the single rest dim IS dim 1, so rows decode
                # directly with no division. The vector-division decode
                # below costs ~128 int64 divisions per tile (~600 cycles
                # each on GCU) and dominated the 3D runtime.
                # (vs1/ve1/is1 extracted once at kernel top.)
                valid = (rows >= vs1) & (rows < ve1) & row_mask
                src_r = tl.where(rows < vs1, 0, rows - vs1).to(tl.int64) * is1
                valid = valid0 & valid
            else:
                idx2d = (rows[:, None] // sp_v[None, :]) % sh_v[None, :]
                valid2d = (idx2d >= vs_v[None, :]) & (idx2d < ve_v[None, :])
                valid_r = (
                    tl.sum(
                        (valid2d.to(tl.int32)) * (rest_lane_mask[None, :].to(tl.int32)),
                        axis=1,
                    )
                    == NUM_REST_DIMS
                )
                clamped = tl.where(idx2d < vs_v[None, :], 0, idx2d - vs_v[None, :])
                src_r = tl.sum(
                    clamped.to(tl.int64)
                    * is_v[None, :].to(tl.int64)
                    * rest_lane_mask[None, :].to(tl.int64),
                    axis=1,
                )  # [BLOCK_R]
                valid = valid0 & valid_r & row_mask

            row_src = src0 + src_r  # [BLOCK_R]
            if NUM_REST_DIMS == 0:
                row_base = rows.to(tl.int64) * out_last
            else:
                row_base = (d0.to(tl.int64) * num_rest + rows.to(tl.int64)) * out_last
            for ct in tl.range(pid2, num_col_tile, GRID_SIZE, num_stages=NUM_STAGES):
                col = ct * BLOCK_C + cols
                col_mask = col < out_last
                src_col = col - pad_left
                in_valid = (col >= pad_left) & (col < pad_left + in_last)
                load_mask = col_mask[None, :] & in_valid[None, :] & valid[:, None]
                x_val = tl.load(
                    in_ptr + row_src[:, None] + src_col[None, :],
                    mask=load_mask,
                    other=value,
                )
                tl.store(
                    out_ptr + row_base[:, None] + col[None, :],
                    x_val,
                    mask=col_mask[None, :] & row_mask[:, None],
                )


def constant_pad_nd(inp, pad_list, value=0):
    logger.debug("GEMS_ENFLAME CONSTANT_PAD_ND")
    ndim = inp.dim()
    pad_pairs = len(pad_list) // 2
    if pad_pairs == 0:
        return inp.clone()

    in_shape = inp.shape
    out_shape = list(in_shape)
    valid_start = [0] * ndim
    valid_end = [0] * ndim
    for i in range(ndim):
        valid_start[i] = 0
        valid_end[i] = in_shape[i]
    # aten pad order: [left_last, right_last, left_prev, right_prev, ...].
    # Negative pads crop; the same valid_start/valid_end formulas cover both.
    for i in range(pad_pairs):
        dim = ndim - 1 - i
        if dim < 0:
            break
        left = int(pad_list[2 * i])
        right = int(pad_list[2 * i + 1])
        out_shape[dim] = in_shape[dim] + left + right
        valid_start[dim] = left
        valid_end[dim] = left + in_shape[dim]

    out_numel = 1
    for d in out_shape:
        out_numel *= d
    if out_numel == 0:
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    src = inp.contiguous()

    # gcu400: the fixed-grid kernel below mis-indexes 4D+ multi-dimension
    # pads (measured ~47% wrong elements on a padded 16x128x64x60 input),
    # while the generic codegen pad launches cdiv(out_numel, 256) programs
    # and exceeds the 65535 grid.x limit for large random pads.  The flat
    # kernel below decodes every output coordinate directly and uses a
    # persistent grid, so it is exact and launch-size bounded for any rank.
    if ndim >= 4:
        out_shape_t = torch.tensor(out_shape, dtype=torch.int32, device=inp.device)
        out_stride = list(out.stride())
        out_stride_t = torch.tensor(out_stride, dtype=torch.int32, device=inp.device)
        in_stride_t = torch.tensor(
            list(src.stride()), dtype=torch.int32, device=inp.device
        )
        vs_t = torch.tensor(valid_start, dtype=torch.int32, device=inp.device)
        ve_t = torch.tensor(valid_end, dtype=torch.int32, device=inp.device)
        block = 1024
        nprog = min(triton.cdiv(out_numel, block), _MAX_PROGRAMS)
        _pad_flat_kernel[(nprog,)](
            src,
            out,
            value,
            out_numel,
            out_shape_t,
            out_stride_t,
            in_stride_t,
            vs_t,
            ve_t,
            NDIM=ndim,
            BLOCK=block,
            NPROG=nprog,
            num_warps=4,
        )
        return out
    outer_rank = ndim - 1
    if outer_rank > _MAX_OUTER:
        # Extremely high rank: fall back to the generic implementation.
        from flag_gems.ops.pad import pad as _generic_pad

        return _generic_pad(inp, pad_list, mode="constant", value=value)

    in_strides = src.stride()
    # int32 metadata keeps the decode divisions in int32; guard against
    # overflow (tensors with >= 2**31 elements or huge strides).
    meta_max = max(
        [abs(in_strides[i]) for i in range(ndim)]
        + [out_shape[i] for i in range(ndim)]
        + [valid_end[i] for i in range(ndim)]
    )
    if meta_max >= 2**31:
        from flag_gems.ops.pad import pad as _generic_pad

        return _generic_pad(inp, pad_list, mode="constant", value=value)

    # Outer-dim metadata padded to _MAX_OUTER lanes with dummies that decode
    # to "valid, zero source offset" (start 0 / end 1 / stride 0 / shape 1).
    pad_vs = [0] * _MAX_OUTER
    pad_ve = [1] * _MAX_OUTER
    pad_is = [0] * _MAX_OUTER
    pad_sh = [1] * _MAX_OUTER
    for i in range(outer_rank):
        pad_vs[i] = valid_start[i]
        pad_ve[i] = valid_end[i]
        pad_is[i] = in_strides[i]
        pad_sh[i] = out_shape[i]
    # Suffix products over the rest dims (1..outer_rank-1): sp[j] is the
    # product of out_shape[j+1 : outer_rank], so (row // sp[j]) % shape_j
    # recovers dim j. Dummy lanes use 1 (division by 1).
    suffix_prods = [1] * _MAX_OUTER
    acc = 1
    for j in range(outer_rank - 1, 0, -1):
        suffix_prods[j] = acc
        acc *= out_shape[j]

    if outer_rank == 1:
        # 2D input: fold dim 0 into the row space (see the kernel's
        # NUM_REST_DIMS == 0 paths). The d0 loop then runs exactly once
        # (d0=0, src0=0) and every 2D-tile row carries a real dim-0 row
        # instead of wasting (BLOCK_R-1)/BLOCK_R of the tile.
        shape0 = 1
        num_rest = out_shape[0]
    else:
        shape0 = out_shape[0] if outer_rank >= 1 else 1
        num_rest = 1
        for i in range(1, outer_rank):
            num_rest *= out_shape[i]
    num_rest_dims = max(outer_rank - 1, 0)
    in_last = in_shape[-1]
    out_last = out_shape[-1]
    pad_left = valid_start[-1]

    meta_kwargs = {"dtype": torch.int32, "device": inp.device}
    in_strides_t = torch.tensor(pad_is, **meta_kwargs)
    outer_shapes_t = torch.tensor(pad_sh, **meta_kwargs)
    suffix_prods_t = torch.tensor(suffix_prods, **meta_kwargs)
    valid_start_t = torch.tensor(pad_vs, **meta_kwargs)
    valid_end_t = torch.tensor(pad_ve, **meta_kwargs)

    # Tile shape: BLOCK_C covers the last dim (no masked-lane waste); BLOCK_R
    # batches rows only when the last dim is skinny, and never exceeds the
    # rest-space size (rows beyond num_rest would be pure waste). The 1D
    # row path (BLOCK_R == 1) keeps tiles 1D-shaped, which needs far less
    # DSM than [1, N] 2D tiles ([1, 32768] requires 1021952B > the 917504B
    # GCU400 limit).
    num_stages = 4
    block_c = min(16384, triton.next_power_of_2(out_last))
    rest_pow2 = 1
    while rest_pow2 * 2 <= num_rest:
        rest_pow2 *= 2
    block_r = min(rest_pow2, max(1, 16384 // block_c))
    num_warps = 4 if block_c >= 4096 else (2 if block_c >= 256 else 1)
    num_col_tile = triton.cdiv(out_last, block_c)
    num_rest_blocks = triton.cdiv(num_rest, block_r)
    grid = (
        min(_GRID_SIZE, shape0),
        min(_GRID_SIZE, num_rest_blocks),
        min(_GRID_SIZE, num_col_tile),
    )
    kernel = constant_pad_nd_row_kernel if block_r == 1 else constant_pad_nd_kernel
    with torch_device_fn.device(inp.device.index):
        kernel[grid](
            src,
            out,
            shape0,
            num_rest,
            num_rest_blocks,
            num_col_tile,
            in_last,
            out_last,
            pad_left,
            in_strides_t,
            outer_shapes_t,
            suffix_prods_t,
            valid_start_t,
            valid_end_t,
            value,
            OUTER_RANK=outer_rank,
            NUM_REST_DIMS=num_rest_dims,
            BLOCK_C=block_c,
            **({} if block_r == 1 else {"BLOCK_R": block_r}),
            GRID_SIZE=_GRID_SIZE,
            NUM_STAGES=num_stages,
            num_warps=num_warps,
        )
    return out
