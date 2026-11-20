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
from torch import Tensor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


def make_3d_for_bn(input: Tensor) -> Tensor:
    """
    Converts the input to a 3D view for batch normalization.

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)

    elif input.ndim >= 4:
        input = input.flatten(2, -1)

    return input


# NOTE: This part of the kernel code is copied and modified
# from the https://github.com/BobMcDear/attorch codebase.


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
    restore_value=["running_mean_pointer", "running_var_pointer"],
)
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit
def batch_norm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    momentum,
    eps,
    is_train: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    # traning mode default track_running_stat
    if is_train:
        mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        var = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        cnt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
        n_num_steps = tl.cdiv(spatial_dim, BLOCK_N)

        for m_step in range(0, m_num_steps):
            for n_step in range(0, n_num_steps):
                spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
                spatial_mask = spatial_offset < spatial_dim

                batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
                batch_mask = batch_offset < batch_dim

                curr_input_pointer = (
                    input_pointer
                    + input_feat_stride * feat_pid
                    + input_batch_stride * batch_offset[:, None]
                    + input_spatial_stride * spatial_offset[None, :]
                )

                mask = batch_mask[:, None] & spatial_mask[None, :]
                curr_input = tl.load(curr_input_pointer, mask=mask).to(tl.float32)

                step = m_step * n_num_steps + n_step + 1
                new_mean = tl.where(mask, mean + (curr_input - mean) / step, mean)
                new_var = tl.where(
                    mask, var + (curr_input - new_mean) * (curr_input - mean), var
                )
                cnt += mask.to(tl.int32)
                mean = new_mean
                var = new_var

        final_mean = tl.sum(mean * cnt) / (batch_dim * spatial_dim)
        var = tl.sum(var + cnt * (mean - final_mean) * (mean - final_mean)) / (
            batch_dim * spatial_dim
        )
        inv_std = rsqrt(var + eps)
        mean = final_mean

        tl.store(feat_pid + mean_pointer, mean)
        tl.store(feat_pid + inv_std_pointer, inv_std)

        running_mean_pointer += feat_pid
        running_var_pointer += feat_pid

        running_mean = tl.load(running_mean_pointer)
        running_var = tl.load(running_var_pointer)

        n = batch_dim * spatial_dim
        tl.store(running_mean_pointer, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            running_var_pointer,
            (1 - momentum) * running_var + momentum * var * n / (n - 1),
        )

    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        inv_std = rsqrt(tl.load(feat_pid + running_var_pointer) + eps)

    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
    if bias_pointer:
        bias = tl.load(feat_pid + bias_pointer).to(tl.float32)
    else:
        bias = 0.0

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_pointer = (
                output_pointer
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            output = weight * (curr_input - mean) * inv_std + bias

            tl.store(
                curr_output_pointer,
                output,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit
def batch_norm_backward_kernel(
    output_grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    weight_pointer,
    input_grad_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_feat_stride,
    output_grad_spatial_stride,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    input_grad_spatial_stride,
    input_grad_mask: tl.constexpr,
    weight_grad_mask: tl.constexpr,
    bias_grad_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    mean = tl.load(feat_pid + mean_pointer).to(tl.float32)
    inv_std = tl.load(feat_pid + inv_std_pointer).to(tl.float32)

    term1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    term2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_pointer = (
                output_grad_pointer
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )

            mask = batch_mask[:, None] & spatial_mask[None, :]
            curr_input = tl.load(curr_input_pointer, mask=mask).to(tl.float32)

            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(curr_output_grad_pointer, mask=mask).to(
                tl.float32
            )

            term1 += curr_pre_lin * curr_output_grad
            term2 += curr_output_grad

    term1 = tl.sum(term1)
    term2 = tl.sum(term2)

    if weight_grad_mask:
        tl.store(feat_pid + weight_grad_pointer, term1)
    if bias_grad_mask:
        tl.store(feat_pid + bias_grad_pointer, term2)

    if not input_grad_mask:
        return

    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0

    count = batch_dim * spatial_dim

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_pointer = (
                output_grad_pointer
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_input_grad_pointer = (
                input_grad_pointer
                + input_grad_feat_stride * feat_pid
                + input_grad_batch_stride * batch_offset[:, None]
                + input_grad_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(
                curr_output_grad_pointer,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            ).to(tl.float32)
            curr_input_grad = (
                inv_std
                * weight
                * (curr_output_grad - (term1 * curr_pre_lin + term2) / count)
            )
            tl.store(
                curr_input_grad_pointer,
                curr_input_grad,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


@libentry(cache_binding=True)
@triton.jit
def _bn_bwd_contig_kernel(
    go_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    var_ptr,
    gi_ptr,
    gwp_ptr,
    gbp_ptr,
    spatial,
    nchan,
    nchunk,
    eps,
    HAS_WEIGHT: tl.constexpr,
    NEED_GINPUT: tl.constexpr,
    NEED_GWEIGHT: tl.constexpr,
    NEED_GBIAS: tl.constexpr,
    FINALIZE: tl.constexpr,
    SAVED_INV_STD: tl.constexpr,
    ACC64: tl.constexpr,
    BATCHES: tl.constexpr,
    NUM_TILES: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chan = tl.program_id(0)
    kchunk = tl.program_id(1)
    pid_b = tl.program_id(2)

    mean = tl.load(mean_ptr + chan).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(tl.load(var_ptr + chan).to(tl.float32) + eps)
    if HAS_WEIGHT:
        scale = tl.load(weight_ptr + chan).to(tl.float32) * inv_std
    else:
        scale = inv_std

    chunk_size = (spatial + nchunk - 1) // nchunk
    chunk_start = kchunk * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, spatial)
    batch0 = pid_b * BATCHES
    plane = nchan * spatial

    offs = tl.arange(0, BLOCK)
    if ACC64:
        acc_w = tl.zeros([BLOCK], tl.float64)
        acc_b = tl.zeros([BLOCK], tl.float64)
    else:
        acc_w = tl.zeros([BLOCK], tl.float32)
        acc_b = tl.zeros([BLOCK], tl.float32)

    for nb in range(BATCHES):
        row = (batch0 + nb) * plane + chan * spatial + chunk_start
        gp = go_ptr + row
        ip = gi_ptr + row
        xp = x_ptr + row
        for t in tl.static_range(NUM_TILES):
            if MASKED:
                pos = chunk_start + t * BLOCK + offs
                m = pos < chunk_end
                g = tl.load(gp + t * BLOCK + offs, mask=m, other=0.0).to(tl.float32)
            else:
                g = tl.load(gp + t * BLOCK + offs).to(tl.float32)
            if NEED_GINPUT:
                if MASKED:
                    tl.store(
                        ip + t * BLOCK + offs,
                        (g * scale).to(gi_ptr.dtype.element_ty),
                        mask=m,
                    )
                else:
                    tl.store(
                        ip + t * BLOCK + offs, (g * scale).to(gi_ptr.dtype.element_ty)
                    )
            if NEED_GBIAS:
                if ACC64:
                    acc_b += g.to(tl.float64)
                else:
                    acc_b += g
            if NEED_GWEIGHT:
                if MASKED:
                    x = tl.load(xp + t * BLOCK + offs, mask=m, other=0.0).to(tl.float32)
                else:
                    x = tl.load(xp + t * BLOCK + offs).to(tl.float32)
                if ACC64:
                    acc_w += g.to(tl.float64) * (x.to(tl.float64) - mean.to(tl.float64))
                else:
                    acc_w += g * (x - mean)

    if NEED_GWEIGHT:
        w_sum = tl.sum(acc_w)
    if NEED_GBIAS:
        b_sum = tl.sum(acc_b)

    if FINALIZE:
        if NEED_GWEIGHT:
            tl.store(
                gwp_ptr + chan,
                (w_sum * inv_std).to(gwp_ptr.dtype.element_ty),
            )
        if NEED_GBIAS:
            tl.store(gbp_ptr + chan, b_sum.to(gbp_ptr.dtype.element_ty))
    else:
        p_off = chan * (nchunk * tl.num_programs(2)) + pid_b * nchunk + kchunk
        if NEED_GWEIGHT:
            tl.store(gwp_ptr + p_off, w_sum.to(tl.float64))
        if NEED_GBIAS:
            tl.store(gbp_ptr + p_off, b_sum.to(tl.float64))


@libentry(cache_binding=True)
@triton.jit
def _bn_bwd_strided_kernel(
    go_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    var_ptr,
    gi_ptr,
    gwp_ptr,
    gbp_ptr,
    spatial,
    chunk_size,
    nchunk,
    eps,
    go_b,
    go_f,
    go_s,
    in_b,
    in_f,
    in_s,
    gi_b,
    gi_f,
    gi_s,
    HAS_WEIGHT: tl.constexpr,
    NEED_GINPUT: tl.constexpr,
    NEED_GWEIGHT: tl.constexpr,
    NEED_GBIAS: tl.constexpr,
    SAVED_INV_STD: tl.constexpr,
    ACC64: tl.constexpr,
    BATCHES: tl.constexpr,
    NUM_TILES: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chan = tl.program_id(0)
    kchunk = tl.program_id(1)
    pid_b = tl.program_id(2)

    mean = tl.load(mean_ptr + chan).to(tl.float32)
    if SAVED_INV_STD:
        inv_std = tl.load(var_ptr + chan).to(tl.float32)
    else:
        inv_std = 1.0 / tl.sqrt(tl.load(var_ptr + chan).to(tl.float32) + eps)
    if HAS_WEIGHT:
        scale = tl.load(weight_ptr + chan).to(tl.float32) * inv_std
    else:
        scale = inv_std

    chunk_start = kchunk * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, spatial)
    batch0 = pid_b * BATCHES

    offs = tl.arange(0, BLOCK)
    if ACC64:
        acc_w = tl.zeros([BLOCK], tl.float64)
        acc_b = tl.zeros([BLOCK], tl.float64)
    else:
        acc_w = tl.zeros([BLOCK], tl.float32)
        acc_b = tl.zeros([BLOCK], tl.float32)

    for nb in range(BATCHES):
        bidx = batch0 + nb
        gb = go_ptr + bidx * go_b + chan * go_f
        xb = x_ptr + bidx * in_b + chan * in_f
        ib = gi_ptr + bidx * gi_b + chan * gi_f
        for t in tl.static_range(NUM_TILES):
            pos = chunk_start + t * BLOCK + offs
            m = pos < chunk_end
            if MASKED:
                g = tl.load(gb + pos * go_s, mask=m, other=0.0).to(tl.float32)
            else:
                g = tl.load(gb + pos * go_s).to(tl.float32)
            if NEED_GINPUT:
                if MASKED:
                    tl.store(
                        ib + pos * gi_s, (g * scale).to(gi_ptr.dtype.element_ty), mask=m
                    )
                else:
                    tl.store(ib + pos * gi_s, (g * scale).to(gi_ptr.dtype.element_ty))
            if NEED_GBIAS:
                if ACC64:
                    acc_b += g.to(tl.float64)
                else:
                    acc_b += g
            if NEED_GWEIGHT:
                if MASKED:
                    x = tl.load(xb + pos * in_s, mask=m, other=0.0).to(tl.float32)
                else:
                    x = tl.load(xb + pos * in_s).to(tl.float32)
                if ACC64:
                    acc_w += g.to(tl.float64) * (x.to(tl.float64) - mean.to(tl.float64))
                else:
                    acc_w += g * (x - mean)

    p_off = chan * (nchunk * tl.num_programs(2)) + pid_b * nchunk + kchunk
    if NEED_GWEIGHT:
        tl.store(gwp_ptr + p_off, tl.sum(acc_w).to(tl.float64))
    if NEED_GBIAS:
        tl.store(gbp_ptr + p_off, tl.sum(acc_b).to(tl.float64))


@libentry(cache_binding=True)
@triton.jit
def _bn_bwd_tile_kernel(
    go_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    var_ptr,
    gi_ptr,
    gw_ptr,
    gb_ptr,
    spatial,
    nchan,
    nbatch,
    eps,
    HAS_WEIGHT: tl.constexpr,
    NEED_GINPUT: tl.constexpr,
    NEED_GWEIGHT: tl.constexpr,
    NEED_GBIAS: tl.constexpr,
    SAVED_INV_STD: tl.constexpr,
    REDUCE64: tl.constexpr,
    NB_ITER: tl.constexpr,
    MASK_B: tl.constexpr,
    MASK_S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """One program per channel: a 2D [BT, BS] tile over (batch, spatial).

    The batch rows of a tile are independent parallel lanes instead of a serial
    walk, which shortens the loop-carried dependence chain and raises the warp
    count per program.  grid is (C, 1, 1), so the per-channel reduction is
    complete here and the whole call is ONE launch with a single final tree.
    """
    chan = tl.program_id(0)

    mean = tl.load(mean_ptr + chan).to(tl.float32)
    if SAVED_INV_STD:
        inv_std = tl.load(var_ptr + chan).to(tl.float32)
    else:
        inv_std = 1.0 / tl.sqrt(tl.load(var_ptr + chan).to(tl.float32) + eps)
    if HAS_WEIGHT:
        scale = tl.load(weight_ptr + chan).to(tl.float32) * inv_std
    else:
        scale = inv_std

    plane = nchan * spatial
    base = chan * spatial
    ob = tl.arange(0, BT)
    os_ = tl.arange(0, BS)
    mask_s = os_[None, :] < spatial

    acc_w = tl.zeros([BT, BS], tl.float32)
    acc_b = tl.zeros([BT, BS], tl.float32)

    for it in tl.static_range(NB_ITER):
        b = it * BT + ob
        row = b[:, None] * plane + base + os_[None, :]
        if MASK_B:
            m = mask_s & (b[:, None] < nbatch)
        elif MASK_S:
            m = mask_s
        if MASK_B or MASK_S:
            g = tl.load(go_ptr + row, mask=m, other=0.0).to(tl.float32)
        else:
            g = tl.load(go_ptr + row).to(tl.float32)
        if NEED_GINPUT:
            if MASK_B or MASK_S:
                tl.store(gi_ptr + row, (g * scale).to(gi_ptr.dtype.element_ty), mask=m)
            else:
                tl.store(gi_ptr + row, (g * scale).to(gi_ptr.dtype.element_ty))
        if NEED_GBIAS:
            acc_b += g
        if NEED_GWEIGHT:
            if MASK_B or MASK_S:
                x = tl.load(x_ptr + row, mask=m, other=0.0).to(tl.float32)
            else:
                x = tl.load(x_ptr + row).to(tl.float32)
            acc_w += g * (x - mean)

    if NEED_GWEIGHT:
        if REDUCE64:
            w_sum = tl.sum(tl.sum(acc_w.to(tl.float64), axis=1), axis=0)
        else:
            w_sum = tl.sum(tl.sum(acc_w, axis=1), axis=0)
        tl.store(gw_ptr + chan, (w_sum * inv_std).to(gw_ptr.dtype.element_ty))
    if NEED_GBIAS:
        if REDUCE64:
            b_sum = tl.sum(tl.sum(acc_b.to(tl.float64), axis=1), axis=0)
        else:
            b_sum = tl.sum(tl.sum(acc_b, axis=1), axis=0)
        tl.store(gb_ptr + chan, b_sum.to(gb_ptr.dtype.element_ty))


@libentry(cache_binding=True)
@triton.jit
def _bn_bwd_finalize_kernel(
    gwp_ptr,
    gbp_ptr,
    var_ptr,
    gw_ptr,
    gb_ptr,
    partial_dim,
    eps,
    NEED_GWEIGHT: tl.constexpr,
    NEED_GBIAS: tl.constexpr,
    SAVED_INV_STD: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold the fp64 per-chunk partials of one channel, then cast exactly once."""
    chan = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc_w = tl.zeros([BLOCK], tl.float64)
    acc_b = tl.zeros([BLOCK], tl.float64)
    for t in tl.static_range(NUM_TILES):
        pos = t * BLOCK + offs
        m = pos < partial_dim
        if NEED_GWEIGHT:
            acc_w += tl.load(gwp_ptr + chan * partial_dim + pos, mask=m, other=0.0)
        if NEED_GBIAS:
            acc_b += tl.load(gbp_ptr + chan * partial_dim + pos, mask=m, other=0.0)
    if NEED_GWEIGHT:
        if SAVED_INV_STD:
            inv_std = tl.load(var_ptr + chan).to(tl.float64)
        else:
            inv_std = 1.0 / tl.sqrt(tl.load(var_ptr + chan).to(tl.float64) + eps)
        tl.store(gw_ptr + chan, (tl.sum(acc_w) * inv_std).to(gw_ptr.dtype.element_ty))
    if NEED_GBIAS:
        tl.store(gb_ptr + chan, tl.sum(acc_b).to(gb_ptr.dtype.element_ty))


# Large-regime policy: enough programs to fill 132 SMs several times over.
_LARGE_NUMEL = 1_000_000
_LARGE_BLOCK = 1024
_LARGE_TILES = 4
_LARGE_STAGES = 3
# Per-channel term count above which an fp32 cross-lane tree reduction is the
# dominant reduction error (only visible when the output dtype is fp32).
_ACC64_TERMS = 262144
# Mid/small-regime 2D tile: BT batch rows x BS spatial lanes per program, kept
# within a register budget so at least two blocks stay resident per SM.
_MID_BT = 4
_MID_BS_MAX = 1024
_MID_TILE_MAX = 4096


def _warps_for(blk):
    w = blk // 256
    if w < 1:
        w = 1
    if w > 8:
        w = 8
    return w


def _batch_norm_backward_eval(
    input: Tensor,
    input_3d: Tensor,
    output_grad_3d: Tensor,
    weight,
    mean,
    inv_std_source,
    saved_inv_std,
    output_mask,
    eps,
):
    """Triton backward of batch normalization in inference mode.

    ``mean`` and ``inv_std_source`` follow the ATen ``train=False`` contract:
    saved (batch) statistics are reused when they are non-empty, otherwise the
    running estimates are the normalisation statistics.  ``saved_inv_std`` says
    whether ``inv_std_source`` already holds ``1/sqrt(var + eps)``.

    The streamed tensors are passed pre-flattened to 3D by the caller.  The
    kernel policy below picks between the 2D tiled one-launch form for compact
    shapes and the split-chunk form for wide ones, and falls back to the strided
    kernel when any operand is non-contiguous.
    """
    grad_out = output_grad_3d
    n_batch, n_chan, spatial = input_3d.shape
    has_w = weight is not None
    need_gi = bool(output_mask[0])
    need_gw = has_w and bool(output_mask[1])
    need_gb = has_w and bool(output_mask[2])
    needs_reduce = need_gw or need_gb
    dt = input.dtype
    device = input.device

    grad_input = (
        torch.empty(input_3d.shape, dtype=dt, device=device) if need_gi else None
    )
    if n_batch == 0 or spatial == 0:
        return (
            None if grad_input is None else grad_input.view_as(input),
            torch.zeros((n_chan,), dtype=dt, device=device) if need_gw else None,
            torch.zeros((n_chan,), dtype=dt, device=device) if need_gb else None,
        )

    grad_weight = torch.empty((n_chan,), dtype=dt, device=device) if need_gw else None
    grad_bias = torch.empty((n_chan,), dtype=dt, device=device) if need_gb else None
    contig = (
        grad_out.is_contiguous()
        and input_3d.is_contiguous()
        and (grad_input is None or grad_input.is_contiguous())
    )

    large = (
        input_3d.numel() >= _LARGE_NUMEL and spatial >= _LARGE_BLOCK and n_chan <= 65535
    )
    if large:
        acc64 = dt == torch.float32 and n_batch * spatial >= _ACC64_TERMS
        if acc64:
            blk = _LARGE_BLOCK
            target_tiles = _LARGE_TILES
            num_stages = _LARGE_STAGES
        else:
            blk = 2048
            target_tiles = 2
            num_stages = 2
        # Split so each program owns `target_tiles` blocks where possible, then
        # derive the real tile count from the resulting chunk: a chunk is never
        # allowed to be shorter than the tile budget assumes, otherwise the fixed
        # unrolled loop would run past the chunk end.
        nchunk = max(1, -(-spatial // (blk * target_tiles)))
        chunk_size = -(-spatial // nchunk)
        num_tiles = -(-chunk_size // blk)
        batches = 1
        nblocks = n_batch
        num_warps = _warps_for(blk)
    else:
        acc64 = False
        blk = triton.next_power_of_2(spatial)
        if blk > 4096:
            blk = 4096
        nchunk = 1
        chunk_size = spatial
        num_tiles = -(-spatial // blk)
        batches = n_batch
        nblocks = 1
        num_warps = _warps_for(blk)
        num_stages = 1

    masked = (chunk_size * nchunk != spatial) or (chunk_size % blk != 0)
    partial_dim = nchunk * nblocks
    # The strided kernel always writes raw fp64 partials, so it needs the finalize
    # pass even when there is only one partial (that pass applies inv_std and does
    # the single dtype cast). The contiguous kernels can write the channel
    # gradients directly when the reduction is complete in one program.
    tile_path = contig and not large
    direct_write = needs_reduce and partial_dim == 1 and (tile_path or contig)
    finalize = (not needs_reduce) or direct_write

    fallback = grad_input if need_gi else grad_out
    w_out = grad_weight if need_gw else fallback
    b_out = grad_bias if need_gb else fallback

    if needs_reduce and not finalize:
        scratch = torch.empty(
            (2, n_chan, partial_dim), dtype=torch.float64, device=device
        )
        w_part, b_part = scratch[0], scratch[1]
    else:
        w_part, b_part = w_out, b_out

    with torch_device_fn.device(device):
        if contig and not large:
            bs = triton.next_power_of_2(spatial)
            if bs > _MID_BS_MAX:
                bs = _MID_BS_MAX
            bt = _MID_BT
            if bt > n_batch:
                bt = triton.next_power_of_2(n_batch)
                if bt > _MID_BT:
                    bt = _MID_BT
            nb_iter = -(-n_batch // bt)
            while bt > 1 and bt * bs > _MID_TILE_MAX:
                bt //= 2
                nb_iter = -(-n_batch // bt)
            _bn_bwd_tile_kernel[(n_chan,)](
                grad_out,
                input_3d,
                weight if has_w else mean,
                mean,
                inv_std_source,
                fallback,
                w_out,
                b_out,
                spatial,
                n_chan,
                n_batch,
                eps,
                HAS_WEIGHT=has_w,
                NEED_GINPUT=need_gi,
                NEED_GWEIGHT=need_gw,
                NEED_GBIAS=need_gb,
                SAVED_INV_STD=saved_inv_std,
                REDUCE64=need_gw and dt == torch.float32,
                NB_ITER=nb_iter,
                MASK_B=(nb_iter * bt != n_batch),
                MASK_S=(bs != spatial) or (bt > n_batch),
                BT=bt,
                BS=bs,
                num_warps=_warps_for(bt * bs),
                num_stages=1,
            )
        elif contig:
            _bn_bwd_contig_kernel[(n_chan, nchunk, nblocks)](
                grad_out,
                input_3d,
                weight if has_w else mean,
                mean,
                inv_std_source,
                fallback,
                w_part,
                b_part,
                spatial,
                n_chan,
                nchunk,
                eps,
                HAS_WEIGHT=has_w,
                NEED_GINPUT=need_gi,
                NEED_GWEIGHT=need_gw,
                NEED_GBIAS=need_gb,
                FINALIZE=finalize,
                SAVED_INV_STD=saved_inv_std,
                ACC64=acc64,
                BATCHES=batches,
                NUM_TILES=num_tiles,
                MASKED=masked,
                BLOCK=blk,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            ndim = len(input_3d.shape)
            gi_st = grad_input.stride() if need_gi else grad_out.stride()
            _bn_bwd_strided_kernel[(n_chan, nchunk, nblocks)](
                grad_out,
                input_3d,
                weight if has_w else mean,
                mean,
                inv_std_source,
                fallback,
                w_part,
                b_part,
                spatial,
                chunk_size,
                nchunk,
                eps,
                grad_out.stride(0),
                grad_out.stride(1),
                grad_out.stride(ndim - 1),
                input_3d.stride(0),
                input_3d.stride(1),
                input_3d.stride(ndim - 1),
                gi_st[0],
                gi_st[1],
                gi_st[ndim - 1],
                HAS_WEIGHT=has_w,
                NEED_GINPUT=need_gi,
                NEED_GWEIGHT=need_gw,
                NEED_GBIAS=need_gb,
                SAVED_INV_STD=saved_inv_std,
                ACC64=acc64,
                BATCHES=batches,
                NUM_TILES=num_tiles,
                MASKED=masked,
                BLOCK=blk,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        if needs_reduce and not finalize:
            fblk = triton.next_power_of_2(partial_dim)
            if fblk > 1024:
                fblk = 1024
            _bn_bwd_finalize_kernel[(n_chan,)](
                w_part,
                b_part,
                inv_std_source,
                w_out,
                b_out,
                partial_dim,
                eps,
                NEED_GWEIGHT=need_gw,
                NEED_GBIAS=need_gb,
                SAVED_INV_STD=saved_inv_std,
                NUM_TILES=-(-partial_dim // fblk),
                BLOCK=fblk,
                num_warps=4,
                num_stages=1,
            )

    return (
        None if grad_input is None else grad_input.view_as(input),
        grad_weight,
        grad_bias,
    )


def batch_norm(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,  # self.running_mean if not self.training or self.track_running_state else None
    running_var=None,
    training=False,  # (self.running_mean is None) and (self.running_var is None)
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS BATCHNORM_FORWARD")

    input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

    running_mean = input if running_mean is None else running_mean
    running_var = input if running_var is None else running_var

    # Launches 1D grid where each program operates over one feature.
    with torch_device_fn.device(input.device):
        batch_norm_forward_kernel[(feat_dim,)](
            input_3d,
            weight,
            bias,
            mean,
            inv_std,
            output,
            running_mean,
            running_var,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            momentum,
            eps,
            is_train=training,
        )

    return output.view_as(input), mean, inv_std


def batch_norm_backward(
    grad_out,
    input,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_invstd=None,
    train=False,
    eps=1e-05,
    output_mask=None,
):
    logger.debug("GEMS BATCHNORM_BACKWARD")
    input_3d = make_3d_for_bn(input)
    output_grad_3d = make_3d_for_bn(grad_out)

    batch_dim, feat_dim, spatial_dim = input_3d.shape

    # ``train=False`` is the no-training / inference batch normalization: it
    # normalises with the running statistics instead of batch statistics, so it
    # takes a dedicated single-pass kernel.  The saved statistics are empty for
    # ``_native_batch_norm_legit_no_training`` (numel == 0) and then fall back to
    # running_mean / running_var, exactly like ATen's ``batch_norm_backward_cuda``.
    # A non-empty saved mean/invstd keeps the saved-statistics semantics used by
    # the eval path of ``cudnn_batch_norm``.
    if not train:
        use_saved_mean = save_mean is not None and save_mean.numel() != 0
        use_saved_invstd = save_invstd is not None and save_invstd.numel() != 0
        mean_source = save_mean if use_saved_mean else running_mean
        inv_std_source = save_invstd if use_saved_invstd else running_var
        if mean_source is None or inv_std_source is None:
            raise RuntimeError(
                "batch_norm_backward with train=False requires either weight or "
                "both running_mean and running_var"
            )
        return _batch_norm_backward_eval(
            input,
            input_3d,
            output_grad_3d,
            weight,
            mean_source,
            inv_std_source,
            use_saved_invstd,
            output_mask,
            eps,
        )

    if output_mask[0]:
        input_grad = torch.empty_like(input_3d)
    else:
        input_grad = None
    if output_mask[1]:
        weight_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    else:
        weight_grad = None
    if output_mask[2]:
        bias_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    else:
        bias_grad = None

    # Launches 1D grid where each program operates over one feature.
    with torch_device_fn.device(input.device):
        batch_norm_backward_kernel[(feat_dim,)](
            output_grad_3d,
            input_3d,
            save_mean,
            save_invstd,
            weight,
            input_grad,
            weight_grad,
            bias_grad,
            batch_dim,
            spatial_dim,
            *output_grad_3d.stride(),
            *input_3d.stride(),
            *input_grad.stride(),
            *output_mask,
        )

    # Pads output with None because a gradient is necessary for
    # all input arguments.
    return (
        input_grad.view_as(input),
        weight_grad,
        bias_grad,
    )
