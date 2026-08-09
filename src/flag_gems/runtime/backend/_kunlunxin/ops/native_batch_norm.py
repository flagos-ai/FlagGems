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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

# The accuracy test asserts on the GENERIC logger name
# ("flag_gems.ops.native_batch_norm"); native_layer_norm.py / native_group_norm.py
# in this directory do the same thing for the same reason.
logger = logging.getLogger("flag_gems.ops.native_batch_norm")
rsqrt = tl_extra_shim.rsqrt


def make_3d_for_bn(input: Tensor) -> Tensor:
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


# NOTE (kunlunxin / XPU, 2026-08-29): why this file exists at all.
#
# `src/flag_gems/ops/native_batch_norm.py:18` binds
# `from flag_gems.ops.batch_norm import batch_norm` at MODULE IMPORT TIME, so the
# reference is closed over inside `flag_gems.ops.native_batch_norm.__dict__`.
# `SpecOpRegistrar` only rebinds the `flag_gems` top-level globals, so the vendor
# `batch_norm` override was never reachable from `aten::native_batch_norm`: on XPU
# the op ran the GENERIC `flag_gems/ops/batch_norm.py` Welford 2D-tile kernel, which
# hard-fails to compile (`cnt += mask.to(tl.int32)` at batch_norm.py:107 ->
# `triton_xpu.convert_layout` shape mismatch -> `TritonXPUUnrollControl` ->
# wrapped as `out of resource: uni_sram`).  Registering a vendor
# `native_batch_norm` here is the fix.
#
# The vendor `batch_norm` in this directory cannot simply be delegated to, because
# `aten::native_batch_norm` has DIFFERENT running-stat semantics than what that
# implementation encodes for torch@XPU's `aten::batch_norm`:
#   * running_var must be folded with the UNBIASED batch variance
#     (var * count / (count - 1)); vendor batch_norm uses the biased one.
#   * running stats must be updated for float16/bfloat16 too; vendor batch_norm
#     restricts the update to float32.
# Both are required by `tests/test_batch_norm.py::test_native_batch_norm`, which
# compares against the CPU `aten::native_batch_norm` reference.
#
# Kernel structure.  Everything is 1D-tile only (TritonXPU rejects 2D `axis=0`
# reductions and silently miscompiles small 2D tiles) and every loop is a
# SINGLE level with a runtime bound: a NESTED runtime loop around a masked
# `tl.load` does not lower on this backend -- the first version of this file used
# `for n in range(batch_dim): for off in range(0, S, TILE_S)` and the compiler
# rejected it with
#   `'tt.addptr' op all non-scalar operands/results must have the same shape and
#    base type` -> `TritonXPUUnrollControl` -> wrapped as `uni_sram`
# (evidence: harness/results/functional/native_batch_norm_xpu3_20260829/
# func_post_r1.log).  Hence the per-channel reduction over N*S elements is split
# into a partial stage and an in-normalize combine:
#   stage 1  grid=(N*C,)  per-(n, c) partial sum / sum-of-squares.  Each slice is
#                         S CONTIGUOUS elements in the [N, C, S] layout, so the
#                         loop is block DMA.  Partials are written TRANSPOSED to
#                         [C, N] so that stage 2 reads them contiguously instead
#                         of through a stride-C gather.
#   stage 2  grid=(N*C,)  per-slice affine normalize.  Each program first folds
#                         its channel's N partials into mean / inv_std (a tiny
#                         contiguous [N] reduction), then streams the contiguous
#                         spatial run as block DMA.  The n == 0 program of each
#                         channel additionally writes save_mean / save_invstd and
#                         the running-stat update, so every address is written
#                         exactly once.  Same shape as the production-validated
#                         `_batch_norm_no_update_kernel` inference path.
# A dedicated grid=(C,) combine launch between the two stages was measured to
# cost a FLAT ~0.040 ms on every shape (pure launch overhead,
# harness/probe/nbn_stage_probe.py), which is why the combine lives inside
# stage 2 instead.
# Tile widths are always >= 64: TritonXPU silently miscompiles <= 32-wide tiles.


def _nbn_tile_s(spatial_dim):
    """1D tile policy for the spatial loops.

    Mirrors `_bn_train_tile_s` in batch_norm.py: on P800 a 64-lane tile costs
    almost the same as a 2048-lane tile (the per-program cost is fixed
    overhead), so use a flat 2048-lane masked tile up to S = 2048 and a
    pow2-capped-4096 tile above it.  Never below 64 lanes.
    """
    if spatial_dim <= 0:
        return 64, False
    if spatial_dim <= 2048:
        return 2048, (spatial_dim % 2048) != 0
    tile = min(triton.next_power_of_2(spatial_dim), 4096)
    return tile, (spatial_dim % tile) != 0


def _nbn_tile_n(batch_dim):
    """1D tile policy for the batch (partial-combine) loop.  Never below 64."""
    tile = min(max(64, triton.next_power_of_2(max(batch_dim, 1))), 2048)
    return tile, (batch_dim % tile) != 0


@libentry()
@triton.jit
def native_batch_norm_partial_stats_kernel(
    input_pointer,  # [N, C, S] contiguous, flattened
    part_sum_pointer,  # [C, N] f32 out
    part_sqsum_pointer,  # [C, N] f32 out
    batch_dim,
    feat_dim,
    spatial_dim,
    slice_offset,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = slice_offset + tl.program_id(axis=0)
    n = pid // feat_dim
    c = pid - n * feat_dim
    base = pid * spatial_dim

    acc = tl.zeros([TILE_S], dtype=tl.float32)
    acc_sq = tl.zeros([TILE_S], dtype=tl.float32)
    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        if NEED_MASK:
            m = idx < spatial_dim
            x = tl.load(input_pointer + base + idx, mask=m, other=0.0).to(tl.float32)
            # Do NOT rely on `other=` alone: without the explicit predication the
            # XPU masked tail can pull neighbouring memory into the reduction.
            # `tl.where` (not `mask.to(tl.float32)`) also avoids the arith.uitofp
            # uni_sram failure at TILE >= 256.
            x = tl.where(m, x, 0.0)
        else:
            x = tl.load(input_pointer + base + idx).to(tl.float32)
        acc += x
        acc_sq += x * x

    # Transposed [C, N] layout -> stage 2 reads a contiguous run per channel.
    out = c * batch_dim + n
    tl.store(part_sum_pointer + out, tl.sum(acc))
    tl.store(part_sqsum_pointer + out, tl.sum(acc_sq))


@libentry()
@triton.jit(do_not_specialize=["eps", "momentum", "var_correction"])
def native_batch_norm_normalize_kernel(
    input_pointer,  # [N, C, S] contiguous, flattened
    output_pointer,
    part_sum_pointer,  # [C, N] f32 (TRAINING) or unused alias
    part_sqsum_pointer,  # [C, N] f32 (TRAINING) or unused alias
    save_mean_pointer,  # [C] input-dtype out (TRAINING) or unused alias
    save_inv_std_pointer,  # [C] input-dtype out (TRAINING) or unused alias
    running_mean_pointer,  # [C] in/out (TRAINING) / in (inference), or alias
    running_var_pointer,  # [C] in/out (TRAINING) / in (inference), or alias
    weight_pointer,  # [C] or unused alias
    bias_pointer,  # [C] or unused alias
    batch_dim,
    feat_dim,
    spatial_dim,
    count,  # batch_dim * spatial_dim
    momentum,
    eps,
    var_correction,  # count / (count - 1), 1.0 when count <= 1
    slice_offset,
    TRAINING: tl.constexpr,
    HAS_RM: tl.constexpr,
    HAS_RV: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    TILE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    NEED_MASK_N: tl.constexpr,
):
    pid = slice_offset + tl.program_id(axis=0)
    n = pid // feat_dim
    c = pid - n * feat_dim
    base = pid * spatial_dim

    if TRAINING:
        # Combine this channel's N partials in-program.  The dedicated grid=(C,)
        # combine launch it replaces cost a FLAT ~0.040 ms on every shape
        # (measured, harness/probe/nbn_stage_probe.py) because it was pure launch
        # overhead; re-doing the tiny [N] reduction in each of the N*C programs is
        # far cheaper than paying that launch.  The running-stat update and the
        # returned save_mean / save_invstd are written by the n == 0 program only,
        # so every address is still written exactly once.
        pbase = c * batch_dim
        acc = tl.zeros([TILE_N], dtype=tl.float32)
        acc_sq = tl.zeros([TILE_N], dtype=tl.float32)
        for off in range(0, batch_dim, TILE_N):
            idx = off + tl.arange(0, TILE_N)
            if NEED_MASK_N:
                m = idx < batch_dim
                s = tl.load(part_sum_pointer + pbase + idx, mask=m, other=0.0)
                sq = tl.load(part_sqsum_pointer + pbase + idx, mask=m, other=0.0)
                s = tl.where(m, s, 0.0)
                sq = tl.where(m, sq, 0.0)
            else:
                s = tl.load(part_sum_pointer + pbase + idx)
                sq = tl.load(part_sqsum_pointer + pbase + idx)
            acc += s
            acc_sq += sq

        mean = tl.sum(acc) / count
        var = tl.sum(acc_sq) / count - mean * mean
        inv_std = rsqrt(var + eps)

        if n == 0:
            tl.store(save_mean_pointer + c, mean.to(save_mean_pointer.dtype.element_ty))
            tl.store(
                save_inv_std_pointer + c,
                inv_std.to(save_inv_std_pointer.dtype.element_ty),
            )
            if HAS_RM:
                running_mean = tl.load(running_mean_pointer + c).to(tl.float32)
                tl.store(
                    running_mean_pointer + c,
                    ((1.0 - momentum) * running_mean + momentum * mean).to(
                        running_mean_pointer.dtype.element_ty
                    ),
                )
            if HAS_RV:
                running_var = tl.load(running_var_pointer + c).to(tl.float32)
                # aten::native_batch_norm folds the UNBIASED batch variance into
                # running_var (this is what the CPU reference does).
                tl.store(
                    running_var_pointer + c,
                    (
                        (1.0 - momentum) * running_var
                        + momentum * var * var_correction
                    ).to(running_var_pointer.dtype.element_ty),
                )
    else:
        mean = tl.load(running_mean_pointer + c).to(tl.float32)
        inv_std = rsqrt(tl.load(running_var_pointer + c).to(tl.float32) + eps)

    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + c).to(tl.float32)
    else:
        bias = 0.0

    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        if NEED_MASK:
            m = idx < spatial_dim
            x = tl.load(input_pointer + base + idx, mask=m).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(
                output_pointer + base + idx,
                y.to(output_pointer.dtype.element_ty),
                mask=m,
            )
        else:
            x = tl.load(input_pointer + base + idx).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(output_pointer + base + idx, y.to(output_pointer.dtype.element_ty))


# grid cap used by the other batch-norm kernels in this directory.
NBN_MAX_PROGRAMS = 4096


def native_batch_norm(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    """aten::native_batch_norm on dedicated Kunlunxin kernels.

    See the NOTE above: the generic implementation binds the generic
    `batch_norm` at import time, so the vendor override never reached this op
    and the generic Welford 2D-tile kernel (which does not compile on XPU) was
    used instead.
    """
    logger.debug("GEMS NATIVE_BATCH_NORM")

    input_3d = make_3d_for_bn(input)  # [N, C, S]
    if not input_3d.is_contiguous():
        input_3d = input_3d.contiguous()
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    count = batch_dim * spatial_dim
    n_slices = batch_dim * feat_dim

    output = torch.empty_like(input_3d)
    # In inference mode aten never consumes save_mean / save_invstd, and the
    # generic implementation leaves them uninitialized too, so do not pay extra
    # launches to fill them.
    save_mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    save_inv_std = torch.empty_like(save_mean)

    training = bool(training)
    has_rm = running_mean is not None
    has_rv = running_var is not None
    if not training and not (has_rm and has_rv):
        # Nothing to normalize with; aten requires running stats in eval mode.
        return output.view_as(input), save_mean, save_inv_std
    if count == 0 or n_slices == 0:
        return output.view_as(input), save_mean, save_inv_std

    tile_s, need_mask = _nbn_tile_s(spatial_dim)
    tile_n, need_mask_n = _nbn_tile_n(batch_dim)
    input_flat = input_3d.reshape(-1)
    output_flat = output.reshape(-1)
    has_weight = weight is not None
    has_bias = bias is not None
    var_correction = (count / (count - 1)) if count > 1 else 1.0

    if training:
        # Stage 1 writes every one of the N*C partial slots it is responsible
        # for, so torch.empty is safe here (no zero-fill launch needed).
        part_sum = torch.empty(n_slices, device=input.device, dtype=torch.float32)
        part_sqsum = torch.empty_like(part_sum)
    else:
        part_sum = input_flat
        part_sqsum = input_flat

    with torch_device_fn.device(input.device):
        if training:
            for slice_offset in range(0, n_slices, NBN_MAX_PROGRAMS):
                slice_count = min(NBN_MAX_PROGRAMS, n_slices - slice_offset)
                native_batch_norm_partial_stats_kernel[(slice_count,)](
                    input_flat,
                    part_sum,
                    part_sqsum,
                    batch_dim,
                    feat_dim,
                    spatial_dim,
                    slice_offset,
                    TILE_S=tile_s,
                    NEED_MASK=need_mask,
                    num_warps=4,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
        for slice_offset in range(0, n_slices, NBN_MAX_PROGRAMS):
            slice_count = min(NBN_MAX_PROGRAMS, n_slices - slice_offset)
            native_batch_norm_normalize_kernel[(slice_count,)](
                input_flat,
                output_flat,
                part_sum,
                part_sqsum,
                save_mean,
                save_inv_std,
                running_mean if has_rm else save_mean,
                running_var if has_rv else save_inv_std,
                weight if has_weight else input_flat,
                bias if has_bias else input_flat,
                batch_dim,
                feat_dim,
                spatial_dim,
                count,
                momentum,
                eps,
                var_correction,
                slice_offset,
                TRAINING=training,
                HAS_RM=has_rm,
                HAS_RV=has_rv,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                TILE_S=tile_s,
                TILE_N=tile_n,
                NEED_MASK=need_mask,
                NEED_MASK_N=need_mask_n,
                num_warps=4,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )

    return output.view_as(input), save_mean, save_inv_std
