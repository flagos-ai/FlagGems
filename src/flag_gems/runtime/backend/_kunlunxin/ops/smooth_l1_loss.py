import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _l1_loss(input, target):
    return tl.abs(input.to(tl.float32) - target.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _smooth_loss(input, target, beta):
    diff = tl.abs(input.to(tl.float32) - target.to(tl.float32))
    return tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def _l1_backward(input, target, grad_output):
    diff = input.to(tl.float32) - target.to(tl.float32)
    grad = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    return grad * grad_output.to(tl.float32)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _l1_backward_scalar(input, target, grad_output):
    diff = input.to(tl.float32) - target.to(tl.float32)
    grad = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    return grad * grad_output


@pointwise_dynamic(
    is_tensor=[True, True, True, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def _smooth_backward(input, target, grad_output, beta):
    diff = input.to(tl.float32) - target.to(tl.float32)
    sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    grad = tl.where(tl.abs(diff) < beta, diff / beta, sign)
    return grad * grad_output.to(tl.float32)


@pointwise_dynamic(
    is_tensor=[True, True, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _smooth_backward_scalar(input, target, grad_output, beta):
    diff = input.to(tl.float32) - target.to(tl.float32)
    sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    grad = tl.where(tl.abs(diff) < beta, diff / beta, sign)
    return grad * grad_output


@libentry()
@triton.jit(do_not_specialize=["grad_scale", "rcp_beta"])
def _smooth_backward_scalar_clamp_kernel(
    in0,
    in1,
    out,
    M,
    grad_scale,
    rcp_beta,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = off < M
        a = tl.load(in0 + off, mask=mask).to(tl.float32)
        b = tl.load(in1 + off, mask=mask).to(tl.float32)
        diff = a - b
        v = tl.minimum(tl.maximum(diff * rcp_beta, -1.0), 1.0)
        tl.store(out + off, (v * grad_scale).to(out.dtype.element_ty), mask=mask)
    else:
        a = tl.load(in0 + off).to(tl.float32)
        b = tl.load(in1 + off).to(tl.float32)
        diff = a - b
        v = tl.minimum(tl.maximum(diff * rcp_beta, -1.0), 1.0)
        tl.store(out + off, (v * grad_scale).to(out.dtype.element_ty))


_SMOOTH_BWD_BLOCK = 8192


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        return {"none": 0, "mean": 1, "sum": 2}[reduction]
    return reduction


def _broadcast_inputs(input, target):
    shape = torch.broadcast_shapes(input.shape, target.shape)
    if input.numel() == 0 or target.numel() == 0:
        return shape, None, None
    return shape, input, target


def _loss_values(input, target, beta):
    if beta == 0.0:
        return _l1_loss(input, target)
    return _smooth_loss(input, target, beta)


@libentry()
@triton.jit
def _smooth_l1_loss_full_block_kernel(
    inp,
    target,
    mid,
    M,
    beta: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Unmasked stage-1 program over a FULL BLOCK_SIZE-lane block. On this
    # backend a masked load does not reliably keep out-of-range lanes out of
    # the reduction, and a masked stage-1 tile at BLOCK >= 32768 runs at a
    # fraction of the unmasked bandwidth (masked-memory path), so the masked
    # variant is avoided entirely -- see mse_loss.py / sum.py. The host
    # guarantees the grid covers only in-bounds full blocks; any remainder is
    # reduced by the tail kernel below without masks either.
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_val = tl.load(inp + offset).to(tl.float32)
    target_val = tl.load(target + offset).to(tl.float32)
    diff = tl.abs(inp_val - target_val)
    if beta == 0.0:
        loss = diff
    else:
        loss = tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    if reduction == 1:
        sum_val = tl.sum(loss) / M
    else:
        sum_val = tl.sum(loss)
    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def _smooth_l1_loss_tail_kernel(
    inp,
    target,
    mid,
    M,
    base,
    n_keep,
    g,
    beta: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Reduces the in-bounds window [M - BLOCK_SIZE, M). Its trailing `n_keep`
    # lanes are the remainder past the full blocks; the leading lanes were
    # already counted by the full-block kernel and are zeroed BY VALUE (a value
    # predicate, never an address mask). Exact for any tensor and never reads
    # out of range.
    lane = tl.arange(0, BLOCK_SIZE)
    inp_val = tl.load(inp + base + lane).to(tl.float32)
    target_val = tl.load(target + base + lane).to(tl.float32)
    diff = tl.abs(inp_val - target_val)
    if beta == 0.0:
        loss = diff
    else:
        loss = tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    loss = tl.where(lane >= BLOCK_SIZE - n_keep, loss, 0.0)
    if reduction == 1:
        sum_val = tl.sum(loss) / M
    else:
        sum_val = tl.sum(loss)
    tl.store(mid + g, sum_val)


@libentry()
@triton.jit
def _sum_full_block_kernel(inp, mid, M, MEAN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    v = tl.load(inp + offset).to(tl.float32)
    sum_val = tl.sum(v)
    if MEAN:
        sum_val = sum_val / M
    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def _sum_tail_kernel(
    inp, mid, M, base, n_keep, g, MEAN: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    lane = tl.arange(0, BLOCK_SIZE)
    v = tl.load(inp + base + lane).to(tl.float32)
    v = tl.where(lane >= BLOCK_SIZE - n_keep, v, 0.0)
    sum_val = tl.sum(v)
    if MEAN:
        sum_val = sum_val / M
    tl.store(mid + g, sum_val)


@libentry()
@triton.jit
def _smooth_l1_loss_final_sum_kernel(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    # `mid` has BLOCK_MID slots but only the first `mid_size` hold partials;
    # the padding lanes are dropped BY VALUE before the sum -- a value
    # predicate, never an address mask on the load.
    offset = tl.arange(0, BLOCK_MID)
    mid_val = tl.load(mid + offset).to(tl.float32)
    mid_val = tl.where(offset < mid_size, mid_val, 0.0)
    tl.store(out, tl.sum(mid_val))


# Largest exact tl.sum tile on this XPU with buffer_size_limit=2048: fully
# 131072-divisible tensors get the unmasked block-DMA path, measured +27..36%
# over the 32768-lane tile on this 2-load + reduce pattern (see mse_loss.py).
_FULL_BLOCK = 131072
# Secondary exact tile: 2 programs of 32768 lanes beat 1 program of 65536 on
# the 65536-element shapes (cluster scheduling).
_V2_BLOCK = 32768
# Legacy full-block size for the big non-divisible shapes.
_LEGACY_BLOCK = 65536
# Stage-2 must stay inside the 32768-lane tl.sum ceiling; grow stage-1 blocks
# if the grid would exceed it (legacy MAX_MID rule).
_MAX_MID = 32768
# buffer_size_limit for every kernel launch: the largest exact tl.sum tile
# ceiling on this backend.
_BUFFER_SIZE_LIMIT = 2048


def _smooth_l1_loss_reduce_fused(input, target, beta, reduction):
    input = input.contiguous()
    target = target.contiguous()
    M = input.numel()
    dtype = input.dtype

    block_size = get_block_size_1d(M, input.element_size() * 2)

    if (M > _FULL_BLOCK) and (M % _FULL_BLOCK == 0):
        # Fully divisible by the largest exact tl.sum tile: the unmasked
        # stage-1 path skips the masked-memory penalty entirely.
        mid_size = M // _FULL_BLOCK
        if mid_size <= _MAX_MID:
            block_mid = triton.next_power_of_2(mid_size)
            mid = torch.empty((block_mid,), dtype=torch.float32, device=input.device)
            out = torch.empty([], dtype=dtype, device=input.device)
            with torch_device_fn.device(input.device):
                _smooth_l1_loss_full_block_kernel[(mid_size,)](
                    input,
                    target,
                    mid,
                    M,
                    beta,
                    reduction,
                    _FULL_BLOCK,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
                _smooth_l1_loss_final_sum_kernel[(1, 1, 1)](
                    mid,
                    out,
                    mid_size,
                    block_mid,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
            return out

    if (M > _V2_BLOCK) and (M % _V2_BLOCK == 0):
        # Divisible by 32768 but not by 131072 (e.g. 65536 = 2 tiles): keep the
        # smaller unmasked tile, which beats one 65536-lane program.
        mid_size = M // _V2_BLOCK
        if mid_size <= _MAX_MID:
            block_mid = triton.next_power_of_2(mid_size)
            mid = torch.empty((block_mid,), dtype=torch.float32, device=input.device)
            out = torch.empty([], dtype=dtype, device=input.device)
            with torch_device_fn.device(input.device):
                _smooth_l1_loss_full_block_kernel[(mid_size,)](
                    input,
                    target,
                    mid,
                    M,
                    beta,
                    reduction,
                    _V2_BLOCK,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
                _smooth_l1_loss_final_sum_kernel[(1, 1, 1)](
                    mid,
                    out,
                    mid_size,
                    block_mid,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
            return out

    # Legacy path: stage-1 blocks sized by get_block_size_1d (or the measured
    # 65536-lane optimum for the big non-divisible shapes), fp32 mid
    # accumulation. Stage 1 is fully unmasked: full blocks via the full-block
    # kernel and one in-bounds window program for the remainder (tail kernel);
    # the mid padding lanes are dropped by value in the final-sum kernel.
    if M > _LEGACY_BLOCK:
        block_size = _LEGACY_BLOCK
    mid_size = triton.cdiv(M, block_size)
    if mid_size > _MAX_MID:
        block_size = triton.next_power_of_2(triton.cdiv(M, _MAX_MID))
        mid_size = triton.cdiv(M, block_size)
    n_full = M // block_size
    rem = M - n_full * block_size
    mid_size = n_full + (1 if rem else 0)
    block_mid = triton.next_power_of_2(mid_size)

    # BLOCK_MID slots: stage 2 sums the whole tile and drops the padding lanes
    # by value, so the slots past `mid_size` must be readable.
    mid = torch.empty((block_mid,), dtype=torch.float32, device=input.device)
    out = torch.empty([], dtype=dtype, device=input.device)

    with torch_device_fn.device(input.device):
        _smooth_l1_loss_full_block_kernel[(n_full, 1, 1)](
            input,
            target,
            mid,
            M,
            beta,
            reduction,
            block_size,
            buffer_size_limit=_BUFFER_SIZE_LIMIT,
        )
        if rem:
            _smooth_l1_loss_tail_kernel[(1, 1, 1)](
                input,
                target,
                mid,
                M,
                M - block_size,
                rem,
                n_full,
                beta,
                reduction,
                block_size,
                buffer_size_limit=_BUFFER_SIZE_LIMIT,
            )
        if mid_size == 1:
            return mid.reshape([]).to(dtype)
        _smooth_l1_loss_final_sum_kernel[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=_BUFFER_SIZE_LIMIT
        )

    return out


def _sum_reduce(loss, mean):
    """Two-stage fp32 sum of a 1-D tensor (replaces ``torch.sum``).

    The broadcast path materializes the loss with the generic pointwise codegen
    and reduces it here; stage-1 is shared logic with
    ``_smooth_l1_loss_reduce_fused`` (unmasked full blocks plus an in-bounds
    tail, fp32 accumulation, and the ``mean`` division inside stage-1 so the
    returned 0-d is already the mean).
    """
    loss = loss.contiguous().reshape(-1)
    M = loss.numel()
    block_size = get_block_size_1d(M, loss.element_size())

    if (M > _FULL_BLOCK) and (M % _FULL_BLOCK == 0):
        mid_size = M // _FULL_BLOCK
        if mid_size <= _MAX_MID:
            block_mid = triton.next_power_of_2(mid_size)
            mid = torch.empty((block_mid,), dtype=torch.float32, device=loss.device)
            out = torch.empty([], dtype=loss.dtype, device=loss.device)
            with torch_device_fn.device(loss.device):
                _sum_full_block_kernel[(mid_size,)](
                    loss,
                    mid,
                    M,
                    mean,
                    _FULL_BLOCK,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
                _smooth_l1_loss_final_sum_kernel[(1, 1, 1)](
                    mid,
                    out,
                    mid_size,
                    block_mid,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
            return out

    if (M > _V2_BLOCK) and (M % _V2_BLOCK == 0):
        mid_size = M // _V2_BLOCK
        if mid_size <= _MAX_MID:
            block_mid = triton.next_power_of_2(mid_size)
            mid = torch.empty((block_mid,), dtype=torch.float32, device=loss.device)
            out = torch.empty([], dtype=loss.dtype, device=loss.device)
            with torch_device_fn.device(loss.device):
                _sum_full_block_kernel[(mid_size,)](
                    loss,
                    mid,
                    M,
                    mean,
                    _V2_BLOCK,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
                _smooth_l1_loss_final_sum_kernel[(1, 1, 1)](
                    mid,
                    out,
                    mid_size,
                    block_mid,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
            return out

    if M > _LEGACY_BLOCK:
        block_size = _LEGACY_BLOCK
    mid_size = triton.cdiv(M, block_size)
    if mid_size > _MAX_MID:
        block_size = triton.next_power_of_2(triton.cdiv(M, _MAX_MID))
        mid_size = triton.cdiv(M, block_size)
    n_full = M // block_size
    rem = M - n_full * block_size
    mid_size = n_full + (1 if rem else 0)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((block_mid,), dtype=torch.float32, device=loss.device)
    out = torch.empty([], dtype=loss.dtype, device=loss.device)

    with torch_device_fn.device(loss.device):
        _sum_full_block_kernel[(n_full, 1, 1)](
            loss, mid, M, mean, block_size, buffer_size_limit=_BUFFER_SIZE_LIMIT
        )
        if rem:
            _sum_tail_kernel[(1, 1, 1)](
                loss,
                mid,
                M,
                M - block_size,
                rem,
                n_full,
                mean,
                block_size,
                buffer_size_limit=_BUFFER_SIZE_LIMIT,
            )
        if mid_size == 1:
            return mid.reshape([]).to(loss.dtype)
        _smooth_l1_loss_final_sum_kernel[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=_BUFFER_SIZE_LIMIT
        )

    return out


def smooth_l1_loss(input, target, reduction=1, beta: float = 1.0):
    logger.debug("GEMS_KUNLUNXIN SMOOTH_L1_LOSS")
    reduction = _normalize_reduction(reduction)
    beta = float(beta)
    if beta < 0:
        raise RuntimeError("smooth_l1_loss does not support negative values for beta.")

    shape, input_expanded, target_expanded = _broadcast_inputs(input, target)
    if input_expanded is None:
        if reduction == 0:
            return torch.empty(shape, device=input.device, dtype=input.dtype)
        if reduction == 1:
            return torch.full((), float("nan"), device=input.device, dtype=input.dtype)
        return torch.zeros((), device=input.device, dtype=input.dtype)

    if reduction == 0:
        return _loss_values(input_expanded, target_expanded, beta)

    if input_expanded.shape == target_expanded.shape:
        return _smooth_l1_loss_reduce_fused(
            input_expanded, target_expanded, beta, reduction
        )

    loss = _loss_values(input_expanded, target_expanded, beta)
    return _sum_reduce(loss, mean=(reduction == 1))


def smooth_l1_loss_out(input, target, reduction=1, beta: float = 1.0, *, out):
    logger.debug("GEMS_KUNLUNXIN SMOOTH_L1_LOSS_OUT")
    result = smooth_l1_loss(input, target, reduction, beta)
    out.resize_(result.shape)
    out.copy_(result)
    return out


def smooth_l1_loss_backward(grad_output, input, target, reduction, beta: float):
    logger.debug("GEMS_KUNLUNXIN SMOOTH_L1_LOSS_BACKWARD")
    reduction = _normalize_reduction(reduction)
    beta = float(beta)
    if beta < 0:
        raise RuntimeError("smooth_l1_loss does not support negative values for beta.")

    shape = torch.broadcast_shapes(input.shape, target.shape)
    if input.numel() == 0 or target.numel() == 0:
        return torch.empty(shape, device=input.device, dtype=input.dtype)

    if grad_output.numel() == 1:
        grad_scale = grad_output.item()
        if reduction == 1:
            grad_scale /= input.numel()
        if beta == 0.0:
            return _l1_backward_scalar(input, target, grad_scale)
        if (
            input.shape == target.shape
            and input.is_contiguous()
            and target.is_contiguous()
        ):
            M = input.numel()
            out = torch.empty_like(input)
            with torch_device_fn.device(input.device):
                _smooth_backward_scalar_clamp_kernel[
                    (triton.cdiv(M, _SMOOTH_BWD_BLOCK),)
                ](
                    input,
                    target,
                    out,
                    M,
                    grad_scale,
                    1.0 / beta,
                    BLOCK=_SMOOTH_BWD_BLOCK,
                    NEED_MASK=(M % _SMOOTH_BWD_BLOCK != 0),
                    num_warps=4,
                )
            return out
        return _smooth_backward_scalar(input, target, grad_scale, beta)

    if beta == 0.0:
        return _l1_backward(input, target, grad_output)
    return _smooth_backward(input, target, grad_output, beta)
