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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.nllloss import nll_loss_forward as nll_loss_2d
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def nll_loss_nd_forward_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    out_ptr,
    scratch_ptr,
    C,
    S,
    stride_in_n,
    stride_in_c,
    stride_in_s,
    stride_tgt_n,
    stride_tgt_s,
    ignore_index,
    HAS_WEIGHT: tl.constexpr,
    REDUCTION: tl.constexpr,
    BLOCK_S: tl.constexpr = 1024,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    tgt_offsets = pid_n * stride_tgt_n + s_offsets * stride_tgt_s
    t = tl.load(target_ptr + tgt_offsets, mask=mask_s, other=ignore_index).to(tl.int32)

    valid = mask_s & (t != ignore_index) & (t >= 0) & (t < C)

    in_offsets = pid_n * stride_in_n + t * stride_in_c + s_offsets * stride_in_s
    val = tl.load(input_ptr + in_offsets, mask=valid, other=0.0).to(tl.float32)

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + t, mask=valid, other=0.0).to(tl.float32)
        loss_val = tl.where(valid, -val * w, 0.0)
    else:
        w = tl.where(valid, 1.0, 0.0).to(tl.float32)
        loss_val = tl.where(valid, -val, 0.0)

    # none
    if REDUCTION == 0:
        out_offset = pid_n * S + s_offsets
        tl.store(
            out_ptr + out_offset, loss_val.to(out_ptr.dtype.element_ty), mask=mask_s
        )
    else:
        block_loss_sum = tl.sum(loss_val, axis=0)
        # mean
        if REDUCTION == 1:
            block_weight_sum = tl.sum(w, axis=0)

            tl.atomic_add(scratch_ptr, block_loss_sum, sem="relaxed")
            tl.atomic_add(scratch_ptr + 1, block_weight_sum, sem="relaxed")

            old_cnt = tl.atomic_add(scratch_ptr + 2, 1.0, sem="acq_rel")

            total_programs = tl.num_programs(0) * tl.num_programs(1)

            if old_cnt == total_programs - 1.0:
                total_loss = tl.load(scratch_ptr)
                total_weight = tl.load(scratch_ptr + 1)
                final_val = total_loss / total_weight
                tl.store(out_ptr, final_val.to(out_ptr.dtype.element_ty))
        # Sum
        else:
            tl.atomic_add(scratch_ptr, block_loss_sum, sem="relaxed")

            old_cnt = tl.atomic_add(scratch_ptr + 2, 1.0, sem="acq_rel")
            total_programs = tl.num_programs(0) * tl.num_programs(1)

            if old_cnt == total_programs - 1.0:
                total_loss = tl.load(scratch_ptr)
                tl.store(out_ptr, total_loss.to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit
def nll_loss_nd_backward_kernel(
    grad_out_ptr,
    target_ptr,
    weight_ptr,
    grad_in_ptr,
    total_weight_ptr,
    C,
    S,
    stride_in_n,
    stride_in_c,
    stride_in_s,
    stride_tgt_n,
    stride_tgt_s,
    stride_go_n,
    stride_go_s,
    ignore_index,
    HAS_WEIGHT: tl.constexpr,
    REDUCTION: tl.constexpr,
    BLOCK_S: tl.constexpr = 1024,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    tgt_offsets = pid_n * stride_tgt_n + s_offsets * stride_tgt_s
    t = tl.load(target_ptr + tgt_offsets, mask=mask_s, other=ignore_index).to(tl.int32)

    valid = mask_s & (t != ignore_index) & (t >= 0) & (t < C)

    if REDUCTION == 0:  # none
        out_grad_offsets = pid_n * stride_go_n + s_offsets * stride_go_s
        out_grad = tl.load(grad_out_ptr + out_grad_offsets, mask=valid, other=0.0).to(
            tl.float32
        )
    else:  # mean or sum
        out_grad = tl.load(grad_out_ptr).to(tl.float32)

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + t, mask=valid, other=0.0).to(tl.float32)
    else:
        w = tl.where(valid, 1.0, 0.0).to(tl.float32)

    if REDUCTION == 1:  # mean
        total_weight = tl.load(total_weight_ptr).to(tl.float32)
        grad_in_val = tl.where(total_weight != 0.0, -w * out_grad / total_weight, 0.0)
    else:  # sum or none
        grad_in_val = -w * out_grad

    in_offsets = pid_n * stride_in_n + t * stride_in_c + s_offsets * stride_in_s
    tl.store(
        grad_in_ptr + in_offsets,
        grad_in_val.to(grad_in_ptr.dtype.element_ty),
        mask=valid,
    )


def nll_loss_nd_forward(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: int = 1,
    ignore_index: int = -100,
):
    logger.debug("GEMS_HYGON NLL_LOSS_ND_FWD")
    if target.numel() == 0:
        shape = target.shape if reduction == 0 else ()
        out = torch.empty(shape, device=input.device, dtype=input.dtype)
        tw_dtype = input.dtype if input.ndim < 3 else torch.float32
        total_weight = torch.empty((), device=input.device, dtype=tw_dtype)
        with torch_device_fn.device(input.device):
            _nll_empty_forward_kernel[(1,)](
                out, total_weight, REDUCTION=reduction, num_warps=1
            )
        return out, total_weight
    if input.dim() < 3:
        out, total_weight = nll_loss_2d(
            input, target, weight=weight, reduction=reduction, ignore_index=ignore_index
        )
        return out, total_weight
    else:
        N = input.shape[0]
        C = input.shape[1]
        S = math.prod(input.shape[2:])

        inp = input.reshape(N, C, S)

        if target.numel() != N * S:
            raise ValueError(
                f"Target size {target.shape} doesn't match input size (N={N}, S={S})"
            )
        else:
            tgt = target.reshape(N, S)

        stride_in_n, stride_in_c, stride_in_s = inp.stride()
        stride_tgt_n, stride_tgt_s = tgt.stride()

        if weight is None:
            has_weight = False
            w = input
        else:
            has_weight = True
            if weight.numel() != C:
                raise ValueError(f"Weight shape {weight.shape} must be ({C},)")
            w = weight.contiguous()

        if reduction not in [0, 1, 2]:
            raise ValueError("reduction must be 0 ('none'), 1 ('mean'), or 2 ('sum')")

        grid = lambda meta: (triton.cdiv(S, meta["BLOCK_S"]), N)
        with torch_device_fn.device(input.device):
            if reduction == 0:
                out = torch.empty((N, S), device=input.device, dtype=input.dtype)
                scratch = torch.empty(1, device=input.device)

                nll_loss_nd_forward_kernel[grid](
                    inp,
                    tgt,
                    w,
                    out,
                    scratch,
                    C,
                    S,
                    stride_in_n,
                    stride_in_c,
                    stride_in_s,
                    stride_tgt_n,
                    stride_tgt_s,
                    ignore_index,
                    HAS_WEIGHT=has_weight,
                    REDUCTION=reduction,
                )

                if target.dim() == input.dim() - 1:
                    res = out.view_as(target)
                else:
                    res = out.reshape(target.shape)

                total_weight = torch.empty([], device=input.device, dtype=input.dtype)
                return res, total_weight

            else:
                out = torch.empty(1, device=input.device, dtype=input.dtype)
                scratch = torch.zeros(3, device=input.device, dtype=torch.float32)

                nll_loss_nd_forward_kernel[grid](
                    inp,
                    tgt,
                    w,
                    out,
                    scratch,
                    C,
                    S,
                    stride_in_n,
                    stride_in_c,
                    stride_in_s,
                    stride_tgt_n,
                    stride_tgt_s,
                    ignore_index,
                    HAS_WEIGHT=has_weight,
                    REDUCTION=reduction,
                )
                out = out[0]

                if reduction == 1:
                    total_weight = scratch[1]
                else:
                    total_weight = torch.empty(
                        [], device=input.device, dtype=input.dtype
                    )

                return out, total_weight


@triton.jit
def _nll_logical_offset(
    index, SHAPE: tl.constexpr, STRIDES: tl.constexpr, CONTIGUOUS: tl.constexpr
):
    if CONTIGUOUS:
        return index
    else:
        remaining = index.to(tl.int64)
        offset = tl.full(index.shape, 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            offset += (remaining % SHAPE[axis]) * STRIDES[axis]
            remaining //= SHAPE[axis]
        return offset


@triton.jit
def _nll_backward_value(
    G,
    W,
    TW,
    target,
    point,
    valid,
    GSHAPE: tl.constexpr,
    GSTRIDES: tl.constexpr,
    GCONTIG: tl.constexpr,
    WSTRIDE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    REDUCTION: tl.constexpr,
    FP64: tl.constexpr,
):
    compute: tl.constexpr = tl.float64 if FP64 else tl.float32
    if REDUCTION == 0:
        go_offset = _nll_logical_offset(point, GSHAPE, GSTRIDES, GCONTIG)
        grad = tl.load(G + go_offset, valid, other=0.0).to(compute)
    else:
        grad = tl.load(G).to(compute)
    if REDUCTION == 1:
        grad = grad / tl.load(TW).to(compute)
    if HAS_WEIGHT:
        weight = tl.load(W + target * WSTRIDE, valid, other=0.0).to(compute)
        return -grad * weight
    else:
        return -grad


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def _nll_backward_dense_kernel(
    G,
    T,
    W,
    Y,
    TW,
    ignore_index,
    C: tl.constexpr,
    S: tl.constexpr,
    TOTAL: tl.constexpr,
    TSHAPE: tl.constexpr,
    TSTRIDES: tl.constexpr,
    TCONTIG: tl.constexpr,
    GSHAPE: tl.constexpr,
    GSTRIDES: tl.constexpr,
    GCONTIG: tl.constexpr,
    WSTRIDE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    REDUCTION: tl.constexpr,
    FP64: tl.constexpr,
    INDEX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if INDEX64:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    point = offsets // (C * S) * S + offsets % S
    target_offset = _nll_logical_offset(point, TSHAPE, TSTRIDES, TCONTIG)
    target = tl.load(T + target_offset, mask, other=ignore_index)
    valid = mask & (target != ignore_index) & (target >= 0) & (target < C)
    value = _nll_backward_value(
        G,
        W,
        TW,
        target,
        point,
        valid,
        GSHAPE,
        GSTRIDES,
        GCONTIG,
        WSTRIDE,
        HAS_WEIGHT,
        REDUCTION,
        FP64,
    )
    channel = offsets // S % C
    result = tl.where(valid & (channel == target), value, 0.0)
    tl.store(Y + offsets, result, mask)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def _nll_backward_tiled_kernel(
    G,
    T,
    W,
    Y,
    TW,
    ignore_index,
    C: tl.constexpr,
    S: tl.constexpr,
    TSHAPE: tl.constexpr,
    TSTRIDES: tl.constexpr,
    TCONTIG: tl.constexpr,
    GSHAPE: tl.constexpr,
    GSTRIDES: tl.constexpr,
    GCONTIG: tl.constexpr,
    WSTRIDE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    REDUCTION: tl.constexpr,
    FP64: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    channel = tl.program_id(0) * BC + tl.arange(0, BC)
    spatial = tl.program_id(1) * BS + tl.arange(0, BS)
    point = batch * S + spatial
    target_offset = _nll_logical_offset(point, TSHAPE, TSTRIDES, TCONTIG)
    target = tl.load(T + target_offset, spatial < S, other=ignore_index)
    valid = (spatial < S) & (target != ignore_index) & (target >= 0) & (target < C)
    value = _nll_backward_value(
        G,
        W,
        TW,
        target,
        point,
        valid,
        GSHAPE,
        GSTRIDES,
        GCONTIG,
        WSTRIDE,
        HAS_WEIGHT,
        REDUCTION,
        FP64,
    )
    result = tl.where(
        valid[None, :] & (channel[:, None] == target[None, :]),
        value,
        0.0,
    )
    offsets = (batch * C + channel[:, None]) * S + spatial[None, :]
    tl.store(Y + offsets, result, (channel[:, None] < C) & (spatial[None, :] < S))


@libentry()
@triton.jit
def _nll_backward_zero_kernel(
    Y,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
    INDEX64: tl.constexpr,
):
    pid = tl.program_id(0)
    if INDEX64:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + offsets, 0.0, offsets < TOTAL)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def _nll_backward_sparse_kernel(
    G,
    T,
    W,
    Y,
    TW,
    ignore_index,
    C: tl.constexpr,
    S: tl.constexpr,
    POINTS: tl.constexpr,
    TSHAPE: tl.constexpr,
    TSTRIDES: tl.constexpr,
    TCONTIG: tl.constexpr,
    GSHAPE: tl.constexpr,
    GSTRIDES: tl.constexpr,
    GCONTIG: tl.constexpr,
    WSTRIDE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    REDUCTION: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    point = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    target_offset = _nll_logical_offset(point, TSHAPE, TSTRIDES, TCONTIG)
    target = tl.load(T + target_offset, point < POINTS, other=ignore_index)
    valid = (point < POINTS) & (target != ignore_index) & (target >= 0) & (target < C)
    value = _nll_backward_value(
        G,
        W,
        TW,
        target,
        point,
        valid,
        GSHAPE,
        GSTRIDES,
        GCONTIG,
        WSTRIDE,
        HAS_WEIGHT,
        REDUCTION,
        FP64,
    )
    offset = point // S * C * S + target * S + point % S
    tl.store(Y + offset, value, valid)


@libentry()
@triton.jit
def _nll_empty_forward_kernel(OUT, TW, REDUCTION: tl.constexpr):
    tl.store(TW, 0.0)
    if REDUCTION != 0:
        value = float("nan") if REDUCTION == 1 else 0.0
        tl.store(OUT, value)


def nll_loss_nd_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: int = 1,
    ignore_index: int = -100,
    total_weight: torch.Tensor = None,
    *,
    _launch_plan=None,
):
    if _launch_plan is not None:
        return _launch_plan.run(grad_output, target, total_weight)
    logger.debug("GEMS_HYGON NLL_LOSS_ND_BWD")
    if input.ndim < 1:
        raise ValueError(
            "nll_loss_nd_backward expects input with at least one dimension"
        )
    if reduction not in (0, 1, 2):
        raise ValueError("reduction must be 0 ('none'), 1 ('mean'), or 2 ('sum')")
    N = input.shape[0] if input.ndim > 1 else 1
    C = input.shape[1] if input.ndim > 1 else input.shape[0]
    S = math.prod(input.shape[2:]) if input.ndim > 2 else 1
    grad_input = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    total = grad_input.numel()
    if total == 0:
        return grad_input
    if target.numel() != N * S:
        raise ValueError(
            "target size does not match input batch and spatial dimensions"
        )
    if weight is not None and (weight.ndim != 1 or weight.numel() != C):
        raise ValueError(f"weight must have shape ({C},)")
    if reduction == 0 and grad_output.numel() != N * S:
        raise ValueError("unreduced grad_output must match target size")
    if reduction != 0 and grad_output.numel() != 1:
        raise ValueError("reduced grad_output must contain one element")
    if reduction == 1 and total_weight is None:
        _, total_weight = nll_loss_nd_forward(
            input, target, weight, reduction, ignore_index
        )

    tcontig = target.is_contiguous()
    gcontig = grad_output.is_contiguous()
    meta = dict(
        C=C,
        S=S,
        TSHAPE=() if tcontig else tuple(target.shape),
        TSTRIDES=() if tcontig else tuple(target.stride()),
        TCONTIG=tcontig,
        GSHAPE=() if gcontig else tuple(grad_output.shape),
        GSTRIDES=() if gcontig else tuple(grad_output.stride()),
        GCONTIG=gcontig,
        WSTRIDE=0 if weight is None else weight.stride(0),
        HAS_WEIGHT=weight is not None,
        REDUCTION=reduction,
        FP64=input.dtype == torch.float64,
    )
    index64 = total > 2**31 - 1
    with torch_device_fn.device(input.device):
        if (
            total >= 2**26
            and C >= 512
            and input.dtype in (torch.float16, torch.bfloat16)
        ):
            # For very large low-precision outputs, a pure streaming fill plus
            # sparse stores beats loading targets for every output tile.
            _nll_backward_zero_kernel[(triton.cdiv(total, 4096),)](
                grad_input, TOTAL=total, BLOCK=4096, INDEX64=index64, num_warps=8
            )
            _nll_backward_sparse_kernel[(triton.cdiv(N * S, 256),)](
                grad_output,
                target,
                weight,
                grad_input,
                total_weight,
                ignore_index,
                **meta,
                POINTS=N * S,
                BLOCK=256,
                num_warps=4,
            )
        elif total >= 2**26 and input.dtype == torch.float32:
            bc, bs, warps = (4096, 1, 8) if S == 1 else (64, 64, 8)
            _nll_backward_tiled_kernel[(triton.cdiv(C, bc), triton.cdiv(S, bs), N)](
                grad_output,
                target,
                weight,
                grad_input,
                total_weight,
                ignore_index,
                **meta,
                BC=bc,
                BS=bs,
                num_warps=warps,
            )
        elif total > 2**20 and S >= 64 and input.dtype == torch.float32:
            _nll_backward_tiled_kernel[(triton.cdiv(C, 32), triton.cdiv(S, 64), N)](
                grad_output,
                target,
                weight,
                grad_input,
                total_weight,
                ignore_index,
                **meta,
                BC=32,
                BS=64,
                num_warps=4,
            )
        else:
            if total >= 2**26:
                block, warps = 4096, 8
            elif total <= 2**18:
                block, warps = 256, 4
            elif total <= 2**20:
                block, warps = (2048, 4) if input.dtype == torch.float32 else (1024, 4)
            else:
                block, warps = (2048, 4) if input.dtype == torch.float32 else (4096, 4)
            _nll_backward_dense_kernel[(triton.cdiv(total, block),)](
                grad_output,
                target,
                weight,
                grad_input,
                total_weight,
                ignore_index,
                **meta,
                TOTAL=total,
                INDEX64=index64,
                BLOCK=block,
                num_warps=warps,
            )
    return grad_input


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def _nll_mean_dense_fast(
    G,
    T,
    Y,
    TW,
    ignore_index,
    C: tl.constexpr,
    S: tl.constexpr,
    TOTAL: tl.constexpr,
    INDEX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if INDEX64:
        pid = pid.to(tl.int64)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < TOTAL
    point = offset // (C * S) * S + offset % S
    target = tl.load(T + point, mask, other=ignore_index)
    value = (-tl.load(G).to(tl.float32) / tl.load(TW).to(tl.float32)).to(
        Y.dtype.element_ty
    )
    # The class index is in [0, C), so equality also checks the label range.
    match = (target == offset // S % C) & (target != ignore_index)
    result = tl.where(match, value, 0)
    tl.store(Y + offset, result, mask)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def _nll_mean_tiled_fast(
    G,
    T,
    Y,
    TW,
    ignore_index,
    C: tl.constexpr,
    S: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    channel = tl.program_id(0) * BC + tl.arange(0, BC)
    spatial = tl.program_id(1) * BS + tl.arange(0, BS)
    target = tl.load(T + batch * S + spatial, spatial < S, other=ignore_index)
    value = (-tl.load(G).to(tl.float32) / tl.load(TW).to(tl.float32)).to(
        Y.dtype.element_ty
    )
    # The store mask excludes padded classes and spatial positions.
    match = (channel[:, None] == target[None, :]) & (target != ignore_index)[None, :]
    result = tl.where(match, value, 0)
    offset = (batch * C + channel[:, None]) * S + spatial[None, :]
    tl.store(Y + offset, result, (channel[:, None] < C) & (spatial[None, :] < S))


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def _nll_mean_sparse_fast(
    G,
    T,
    Y,
    TW,
    ignore_index,
    C: tl.constexpr,
    S: tl.constexpr,
    POINTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    point = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    target = tl.load(T + point, point < POINTS, other=ignore_index)
    valid = (point < POINTS) & (target != ignore_index) & (target >= 0) & (target < C)
    value = (-tl.load(G).to(tl.float32) / tl.load(TW).to(tl.float32)).to(
        Y.dtype.element_ty
    )
    offset = point // S * C * S + target * S + point % S
    tl.store(Y + offset, value, valid)


class _NLLMeanBackwardPlan:
    """Cache only launch metadata in the forward context, never tensor data."""

    __slots__ = (
        "shape",
        "strides",
        "dtype",
        "device",
        "device_index",
        "mode",
        "launch",
        "args",
        "warps",
        "zero_launch",
        "zero_args",
    )

    def __init__(self, inp, ignore_index):
        self.shape = tuple(inp.shape)
        self.dtype, self.device = inp.dtype, inp.device
        self.device_index = inp.device.index
        ndim = len(self.shape)
        n = self.shape[0] if ndim > 1 else 1
        c = self.shape[1] if ndim > 1 else self.shape[0]
        s = math.prod(self.shape[2:]) if ndim > 2 else 1
        total = n * c * s
        stride = 1
        strides = []
        for size in reversed(self.shape):
            strides.append(stride)
            stride *= max(size, 1)
        self.strides = tuple(reversed(strides))
        index64 = total > 2**31 - 1
        self.zero_launch = self.zero_args = None
        if (
            total >= 2**26
            and c >= 512
            and self.dtype in (torch.float16, torch.bfloat16)
        ):
            self.mode = 2
            self.zero_launch = _nll_backward_zero_kernel[(triton.cdiv(total, 4096),)]
            self.zero_args = (total, 4096, index64)
            self.launch = _nll_mean_sparse_fast[(triton.cdiv(n * s, 256),)]
            self.args = (ignore_index, c, s, n * s, 256)
            self.warps = 4
        elif total >= 2**26 and self.dtype == torch.float32:
            self.mode = 1
            bc, bs, self.warps = (4096, 1, 8) if s == 1 else (64, 64, 8)
            self.launch = _nll_mean_tiled_fast[
                (triton.cdiv(c, bc), triton.cdiv(s, bs), n)
            ]
            self.args = (ignore_index, c, s, bc, bs)
        elif total > 2**20 and s >= 64 and self.dtype == torch.float32:
            self.mode = 1
            self.warps = 4
            self.launch = _nll_mean_tiled_fast[
                (triton.cdiv(c, 32), triton.cdiv(s, 64), n)
            ]
            self.args = (ignore_index, c, s, 32, 64)
        else:
            self.mode = 0
            if total >= 2**26:
                block, self.warps = 4096, 8
            elif total <= 2**18:
                block, self.warps = 256, 4
            elif total <= 2**20:
                block, self.warps = (
                    (2048, 4) if self.dtype == torch.float32 else (1024, 4)
                )
            else:
                block, self.warps = (
                    (2048, 4) if self.dtype == torch.float32 else (4096, 4)
                )
            self.launch = _nll_mean_dense_fast[(triton.cdiv(total, block),)]
            self.args = (ignore_index, c, s, total, index64, block)

    def _launch(self, grad_output, target, total_weight):
        result = torch.empty_strided(
            self.shape, self.strides, dtype=self.dtype, device=self.device
        )
        if self.mode == 2:
            self.zero_launch(result, *self.zero_args, num_warps=8)
        self.launch(
            grad_output,
            target,
            result,
            total_weight,
            *self.args,
            num_warps=self.warps,
        )
        return result

    def run(self, grad_output, target, total_weight):
        if torch_device_fn.current_device() == self.device_index:
            return self._launch(grad_output, target, total_weight)
        with torch_device_fn.device(self.device):
            return self._launch(grad_output, target, total_weight)


def _prepare_nll_mean_plan(inp, target, weight, reduction, ignore_index):
    if (
        reduction == 1
        and weight is None
        and inp.numel() > 0
        and target.is_contiguous()
        and inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return _NLLMeanBackwardPlan(inp, ignore_index)
    return None


class _HygonNLLLossForwardFunction(torch.autograd.Function):
    """ATen forward bridge whose autograd node uses the Hygon ND backward."""

    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index):
        output, total_weight = nll_loss_nd_forward(
            inp, target, weight, reduction, ignore_index
        )
        ctx.save_for_backward(inp, target, weight, total_weight)
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.backward_plan = _prepare_nll_mean_plan(
            inp, target, weight, reduction, ignore_index
        )
        ctx.mark_non_differentiable(total_weight)
        # The weight sum is nondifferentiable. Do not launch zeros_like to
        # materialize an unused gradient for it on each backward call.
        ctx.set_materialize_grads(False)
        return output, total_weight

    @staticmethod
    def backward(ctx, grad_output, grad_total_weight):
        if grad_output is None:
            return None, None, None, None, None
        inp, target, weight, total_weight = ctx.saved_tensors
        grad_input = nll_loss_nd_backward(
            grad_output,
            inp,
            target,
            weight,
            ctx.reduction,
            ctx.ignore_index,
            total_weight,
            _launch_plan=ctx.backward_plan,
        )
        return grad_input, None, None, None, None


def nll_loss_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    """Match aten::nll_loss_forward while keeping backend-aware autograd."""
    return _HygonNLLLossForwardFunction.apply(
        self, target, weight, reduction, ignore_index
    )


def nll_loss_nd(inp, target, weight=None, ignore_index=-100, reduction="mean"):
    """Return the loss for direct ND callers using the Hygon autograd path."""
    reduction_val = {"none": 0, "mean": 1, "sum": 2}[reduction]
    output, _ = nll_loss_forward(inp, target, weight, reduction_val, ignore_index)
    return output
