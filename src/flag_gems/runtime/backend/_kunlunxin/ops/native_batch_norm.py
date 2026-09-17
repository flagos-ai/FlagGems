import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger("flag_gems.ops.native_batch_norm")
rsqrt = tl_extra_shim.rsqrt


def make_3d_for_bn(input: Tensor) -> Tensor:
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


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


NBN_FUSED_S_MAX = 2048


def _nbn_fused_tile_s(spatial_dim):
    """Tile for the fused stats kernel: loop-carried accumulators only lower at
    TILE_S <= 128, and the masked variant needs TILE_S = 64 below 128 (a
    128-wide mostly-false mask fails to lower, e.g. for S = 1)."""
    if spatial_dim < 128:
        return 64, (spatial_dim % 64) != 0
    return 128, (spatial_dim % 128) != 0


def _nbn_exact_tile(spatial_dim):
    """Exact-fit tile for the fused normalize kernel (load/store only, so any
    tile width lowers).  512/1024 for short runs, pow2-capped-4096 above."""
    if spatial_dim <= 512:
        return 512, (spatial_dim % 512) != 0
    if spatial_dim <= 1024:
        return 1024, (spatial_dim % 1024) != 0
    tile = min(triton.next_power_of_2(spatial_dim), 4096)
    return tile, (spatial_dim % tile) != 0


@libentry()
@triton.jit(do_not_specialize=["momentum", "eps", "var_correction"])
def native_batch_norm_fused_stats_kernel(
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    save_mean_pointer,
    save_inv_std_pointer,
    running_mean_pointer,
    running_var_pointer,
    batch_dim,
    feat_dim,
    spatial_dim,
    count,
    momentum,
    eps,
    var_correction,
    HAS_RM: tl.constexpr,
    HAS_RV: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    c = tl.program_id(axis=0)
    acc = tl.zeros([TILE_S], dtype=tl.float32)
    acc_sq = tl.zeros([TILE_S], dtype=tl.float32)
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                m = idx < spatial_dim
                x = tl.load(input_pointer + base + idx, mask=m, other=0.0).to(
                    tl.float32
                )
                x = tl.where(m, x, 0.0)
                acc += x
                acc_sq += x * x
            else:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                acc += x
                acc_sq += x * x
    mean = tl.sum(acc) / count
    var = tl.sum(acc_sq) / count - mean * mean
    inv_std = rsqrt(var + eps)
    tl.store(mean_pointer + c, mean)
    tl.store(inv_std_pointer + c, inv_std)
    tl.store(save_mean_pointer + c, mean.to(save_mean_pointer.dtype.element_ty))
    tl.store(
        save_inv_std_pointer + c, inv_std.to(save_inv_std_pointer.dtype.element_ty)
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
        tl.store(
            running_var_pointer + c,
            ((1.0 - momentum) * running_var + momentum * var * var_correction).to(
                running_var_pointer.dtype.element_ty
            ),
        )


@libentry()
@triton.jit
def native_batch_norm_fused_normalize_kernel(
    input_pointer,
    output_pointer,
    mean_pointer,
    inv_std_pointer,
    weight_pointer,
    bias_pointer,
    batch_dim,
    feat_dim,
    spatial_dim,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    c = tl.program_id(axis=0)
    mean = tl.load(mean_pointer + c).to(tl.float32)
    inv_std = tl.load(inv_std_pointer + c).to(tl.float32)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + c).to(tl.float32)
    else:
        bias = 0.0
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
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
                tl.store(
                    output_pointer + base + idx, y.to(output_pointer.dtype.element_ty)
                )


@libentry()
@triton.jit
def native_batch_norm_partial_stats_kernel(
    input_pointer,
    part_sum_pointer,
    part_sqsum_pointer,
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
            x = tl.where(m, x, 0.0)
        else:
            x = tl.load(input_pointer + base + idx).to(tl.float32)
        acc += x
        acc_sq += x * x

    out = c * batch_dim + n
    tl.store(part_sum_pointer + out, tl.sum(acc))
    tl.store(part_sqsum_pointer + out, tl.sum(acc_sq))


@libentry()
@triton.jit(do_not_specialize=["eps", "momentum", "var_correction"])
def native_batch_norm_normalize_kernel(
    input_pointer,
    output_pointer,
    part_sum_pointer,
    part_sqsum_pointer,
    save_mean_pointer,
    save_inv_std_pointer,
    running_mean_pointer,
    running_var_pointer,
    weight_pointer,
    bias_pointer,
    batch_dim,
    feat_dim,
    spatial_dim,
    count,
    momentum,
    eps,
    var_correction,
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
                tl.store(
                    running_var_pointer + c,
                    (
                        (1.0 - momentum) * running_var + momentum * var * var_correction
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
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM")

    input_3d = make_3d_for_bn(input)
    if not input_3d.is_contiguous():
        input_3d = input_3d.contiguous()
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    count = batch_dim * spatial_dim
    n_slices = batch_dim * feat_dim

    output = torch.empty_like(input_3d)
    save_mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    save_inv_std = torch.empty_like(save_mean)

    training = bool(training)
    has_rm = running_mean is not None
    has_rv = running_var is not None
    if not training and not (has_rm and has_rv):
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

    if (
        training
        and (spatial_dim <= NBN_FUSED_S_MAX or batch_dim <= 1)
        and feat_dim <= NBN_MAX_PROGRAMS
    ):
        mean_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
        inv_f = torch.empty_like(mean_f)
        fused_tile_s, fused_need_m = _nbn_fused_tile_s(spatial_dim)
        exact_tile_s, exact_need_m = _nbn_exact_tile(spatial_dim)
        with torch_device_fn.device(input.device):
            native_batch_norm_fused_stats_kernel[(feat_dim,)](
                input_flat,
                mean_f,
                inv_f,
                save_mean,
                save_inv_std,
                running_mean if has_rm else save_mean,
                running_var if has_rv else save_inv_std,
                batch_dim,
                feat_dim,
                spatial_dim,
                count,
                momentum,
                eps,
                var_correction,
                HAS_RM=has_rm,
                HAS_RV=has_rv,
                TILE_S=fused_tile_s,
                NEED_MASK=fused_need_m,
                num_warps=4,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
            native_batch_norm_fused_normalize_kernel[(feat_dim,)](
                input_flat,
                output_flat,
                mean_f,
                inv_f,
                weight if has_weight else input_flat,
                bias if has_bias else input_flat,
                batch_dim,
                feat_dim,
                spatial_dim,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                TILE_S=exact_tile_s,
                NEED_MASK=exact_need_m,
                num_warps=4,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        return output.view_as(input), save_mean, save_inv_std

    if training:
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
