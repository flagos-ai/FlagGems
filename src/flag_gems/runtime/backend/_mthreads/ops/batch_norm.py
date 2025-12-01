import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
rsqrt = tl_extra_shim.rsqrt


def _make_3d_for_bn(input: torch.Tensor) -> torch.Tensor:
    if input.ndim == 2:
        return input.unsqueeze(-1)
    if input.ndim >= 4:
        return input.flatten(2, -1)
    return input


def _block_size(numel: int) -> int:
    if numel >= 524288:
        return 512
    if numel >= 1024:
        return 256
    if numel >= 256:
        return 128
    return 64


def _num_warps(block: int) -> int:
    if block >= 512:
        return 8
    if block >= 256:
        return 4
    if block >= 128:
        return 4
    return 2


_REDUCE_BLOCK = 256
_FALLBACK_ELEMENTS = 4096
_NATIVE_SWITCH_ELEMENTS = 32768
_NATIVE_CACHE = {}


def _get_temp_stats(device, dtype, feat_dim):
    key = (device, dtype, feat_dim)
    cached = _NATIVE_CACHE.get(key)
    if cached is None or cached[0].numel() != feat_dim:
        rm = torch.zeros((feat_dim,), device=device, dtype=dtype)
        rv = torch.ones((feat_dim,), device=device, dtype=dtype)
        _NATIVE_CACHE[key] = (rm, rv)
    return _NATIVE_CACHE[key]


@triton.jit
def _bn_forward_stats_stage1(
    input_ptr,
    partial_sum_ptr,
    partial_sq_ptr,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    num_blocks,
    BLOCK: tl.constexpr,
):
    feat = tl.program_id(0)
    block_id = tl.program_id(1)

    offset = block_id * BLOCK + tl.arange(0, BLOCK)
    total = batch_dim * spatial_dim
    mask = offset < total

    batch_idx = offset // spatial_dim
    spatial_idx = offset - batch_idx * spatial_dim

    ptrs = (
        input_ptr
        + feat * input_feat_stride
        + batch_idx * input_batch_stride
        + spatial_idx * input_spatial_stride
    )
    values = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    tl.store(partial_sum_ptr + feat * num_blocks + block_id, tl.sum(values, axis=0))
    tl.store(
        partial_sq_ptr + feat * num_blocks + block_id,
        tl.sum(values * values, axis=0),
    )


@triton.jit
def _bn_reduce_partial_kernel(
    partial_sum_ptr,
    partial_sq_ptr,
    sum_ptr,
    sum_sq_ptr,
    num_blocks,
    BLOCK: tl.constexpr,
):
    feat = tl.program_id(0)
    block_id = tl.program_id(1)

    offset = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num_blocks

    partial_sum = tl.load(
        partial_sum_ptr + feat * num_blocks + offset, mask=mask, other=0.0
    )
    partial_sq = tl.load(
        partial_sq_ptr + feat * num_blocks + offset, mask=mask, other=0.0
    )

    tl.atomic_add(sum_ptr + feat, tl.sum(partial_sum, axis=0))
    tl.atomic_add(sum_sq_ptr + feat, tl.sum(partial_sq, axis=0))


@triton.jit
def _bn_fused_train_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    inv_std_ptr,
    running_mean_ptr,
    running_var_ptr,
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
    update_running: tl.constexpr,
    BLOCK: tl.constexpr,
):
    feat = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    total = batch_dim * spatial_dim
    num_tiles = tl.cdiv(total, BLOCK)

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq_val = tl.zeros((), dtype=tl.float32)

    for tile in range(0, num_tiles):
        idx = tile * BLOCK + offsets
        mask = idx < total
        batch_idx = idx // spatial_dim
        spatial_idx = idx - batch_idx * spatial_dim
        ptrs = (
            input_ptr
            + feat * input_feat_stride
            + batch_idx * input_batch_stride
            + spatial_idx * input_spatial_stride
        )
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(vals, axis=0)
        sum_sq_val += tl.sum(vals * vals, axis=0)

    total_f = tl.full((), total, tl.float32)
    mean = sum_val / total_f
    var = tl.maximum(sum_sq_val / total_f - mean * mean, 0.0)
    inv_std = rsqrt(var + eps)

    tl.store(mean_ptr + feat, mean)
    tl.store(inv_std_ptr + feat, inv_std)

    if update_running:
        running_mean = tl.load(running_mean_ptr + feat)
        running_var = tl.load(running_var_ptr + feat)
        unbiased_var = var * total_f / tl.maximum(total_f - 1, 1.0)
        tl.store(
            running_mean_ptr + feat, (1 - momentum) * running_mean + momentum * mean
        )
        tl.store(
            running_var_ptr + feat,
            (1 - momentum) * running_var + momentum * unbiased_var,
        )

    weight = tl.load(weight_ptr + feat).to(tl.float32) if weight_ptr else 1.0
    bias = tl.load(bias_ptr + feat).to(tl.float32) if bias_ptr else 0.0

    for tile in range(0, num_tiles):
        idx = tile * BLOCK + offsets
        mask = idx < total
        batch_idx = idx // spatial_dim
        spatial_idx = idx - batch_idx * spatial_dim
        input_ptrs = (
            input_ptr
            + feat * input_feat_stride
            + batch_idx * input_batch_stride
            + spatial_idx * input_spatial_stride
        )
        output_ptrs = (
            output_ptr
            + feat * output_feat_stride
            + batch_idx * output_batch_stride
            + spatial_idx * output_spatial_stride
        )
        vals = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
        out = (vals - mean) * inv_std * weight + bias
        tl.store(output_ptrs, out, mask=mask)


@triton.jit
def _bn_forward_finalize_kernel(
    sum_ptr,
    sum_sq_ptr,
    mean_ptr,
    inv_std_ptr,
    running_mean_ptr,
    running_var_ptr,
    total_elems,
    momentum,
    eps,
    update_running: tl.constexpr,
):
    feat = tl.program_id(0)
    sum_val = tl.load(sum_ptr + feat)
    sum_sq_val = tl.load(sum_sq_ptr + feat)

    total = tl.full((), total_elems, tl.float32)
    mean = sum_val / total
    var = tl.maximum(sum_sq_val / total - mean * mean, 0.0)
    inv_std = rsqrt(var + eps)

    tl.store(mean_ptr + feat, mean)
    tl.store(inv_std_ptr + feat, inv_std)

    if update_running:
        if running_mean_ptr and running_var_ptr:
            running_mean = tl.load(running_mean_ptr + feat)
            running_var = tl.load(running_var_ptr + feat)
            unbiased_var = var * total / tl.maximum(total - 1.0, 1.0)
            tl.store(
                running_mean_ptr + feat,
                (1 - momentum) * running_mean + momentum * mean,
            )
            tl.store(
                running_var_ptr + feat,
                (1 - momentum) * running_var + momentum * unbiased_var,
            )


@triton.jit
def _bn_forward_apply_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    inv_std_ptr,
    output_ptr,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    BLOCK: tl.constexpr,
):
    feat = tl.program_id(0)
    block_id = tl.program_id(1)

    offset = block_id * BLOCK + tl.arange(0, BLOCK)
    total = batch_dim * spatial_dim
    mask = offset < total

    batch_idx = offset // spatial_dim
    spatial_idx = offset - batch_idx * spatial_dim

    mean = tl.load(mean_ptr + feat).to(tl.float32)
    inv_std = tl.load(inv_std_ptr + feat).to(tl.float32)

    weight = tl.load(weight_ptr + feat).to(tl.float32) if weight_ptr else 1.0
    bias = tl.load(bias_ptr + feat).to(tl.float32) if bias_ptr else 0.0

    input_ptrs = (
        input_ptr
        + feat * input_feat_stride
        + batch_idx * input_batch_stride
        + spatial_idx * input_spatial_stride
    )
    output_ptrs = (
        output_ptr
        + feat * output_feat_stride
        + batch_idx * output_batch_stride
        + spatial_idx * output_spatial_stride
    )

    values = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
    output = (values - mean) * inv_std * weight + bias
    tl.store(output_ptrs, output, mask=mask)


@triton.jit
def _bn_backward_reduce_kernel(
    output_grad_ptr,
    input_ptr,
    mean_ptr,
    inv_std_ptr,
    partial_sum_ptr,
    partial_sum_xhat_ptr,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_feat_stride,
    output_grad_spatial_stride,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    num_blocks,
    BLOCK: tl.constexpr,
):
    feat = tl.program_id(0)
    block_id = tl.program_id(1)

    offset = block_id * BLOCK + tl.arange(0, BLOCK)
    total = batch_dim * spatial_dim
    mask = offset < total

    batch_idx = offset // spatial_dim
    spatial_idx = offset - batch_idx * spatial_dim

    mean = tl.load(mean_ptr + feat).to(tl.float32)
    inv_std = tl.load(inv_std_ptr + feat).to(tl.float32)

    grad_ptrs = (
        output_grad_ptr
        + feat * output_grad_feat_stride
        + batch_idx * output_grad_batch_stride
        + spatial_idx * output_grad_spatial_stride
    )
    input_ptrs = (
        input_ptr
        + feat * input_feat_stride
        + batch_idx * input_batch_stride
        + spatial_idx * input_spatial_stride
    )

    dy = tl.load(grad_ptrs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * inv_std

    tl.store(partial_sum_ptr + feat * num_blocks + block_id, tl.sum(dy, axis=0))
    tl.store(
        partial_sum_xhat_ptr + feat * num_blocks + block_id,
        tl.sum(dy * x_hat, axis=0),
    )


@triton.jit
def _bn_backward_reduce_partial_kernel(
    partial_sum_ptr,
    partial_sum_xhat_ptr,
    sum_dy_ptr,
    sum_dy_xhat_ptr,
    num_blocks,
    BLOCK: tl.constexpr,
):
    feat = tl.program_id(0)
    block_id = tl.program_id(1)

    offset = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offset < num_blocks

    partial_sum = tl.load(
        partial_sum_ptr + feat * num_blocks + offset, mask=mask, other=0.0
    )
    partial_sum_xhat = tl.load(
        partial_sum_xhat_ptr + feat * num_blocks + offset, mask=mask, other=0.0
    )

    tl.atomic_add(sum_dy_ptr + feat, tl.sum(partial_sum, axis=0))
    tl.atomic_add(sum_dy_xhat_ptr + feat, tl.sum(partial_sum_xhat, axis=0))


@triton.jit
def _bn_backward_input_kernel(
    output_grad_ptr,
    input_ptr,
    mean_ptr,
    inv_std_ptr,
    weight_ptr,
    sum_dy_ptr,
    sum_dy_xhat_ptr,
    input_grad_ptr,
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
    BLOCK: tl.constexpr,
):
    feat = tl.program_id(0)
    block_id = tl.program_id(1)

    offset = block_id * BLOCK + tl.arange(0, BLOCK)
    total = batch_dim * spatial_dim
    mask = offset < total

    batch_idx = offset // spatial_dim
    spatial_idx = offset - batch_idx * spatial_dim

    mean = tl.load(mean_ptr + feat).to(tl.float32)
    inv_std = tl.load(inv_std_ptr + feat).to(tl.float32)
    sum_dy = tl.load(sum_dy_ptr + feat)
    sum_dy_xhat = tl.load(sum_dy_xhat_ptr + feat)
    count = tl.full((), total, tl.float32)

    weight = tl.load(weight_ptr + feat).to(tl.float32) if weight_ptr else 1.0

    grad_ptrs = (
        output_grad_ptr
        + feat * output_grad_feat_stride
        + batch_idx * output_grad_batch_stride
        + spatial_idx * output_grad_spatial_stride
    )
    input_ptrs = (
        input_ptr
        + feat * input_feat_stride
        + batch_idx * input_batch_stride
        + spatial_idx * input_spatial_stride
    )
    input_grad_ptrs = (
        input_grad_ptr
        + feat * input_grad_feat_stride
        + batch_idx * input_grad_batch_stride
        + spatial_idx * input_grad_spatial_stride
    )

    dy = tl.load(grad_ptrs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * inv_std

    term = (dy - sum_dy / count - x_hat * sum_dy_xhat / count) * inv_std * weight
    tl.store(input_grad_ptrs, term, mask=mask)


@triton.jit
def _bn_backward_param_kernel(
    sum_dy_ptr,
    sum_dy_xhat_ptr,
    weight_grad_ptr,
    bias_grad_ptr,
    weight_grad_mask: tl.constexpr,
    bias_grad_mask: tl.constexpr,
):
    feat = tl.program_id(0)
    if weight_grad_mask:
        tl.store(weight_grad_ptr + feat, tl.load(sum_dy_xhat_ptr + feat))
    if bias_grad_mask:
        tl.store(bias_grad_ptr + feat, tl.load(sum_dy_ptr + feat))


def _get_launch_config(batch_dim: int, spatial_dim: int) -> Tuple[int, int, int]:
    total = batch_dim * spatial_dim
    block = _block_size(total)
    num_blocks = triton.cdiv(total, block)
    return block, num_blocks, _num_warps(block)


def batch_norm(
    input: torch.Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05,
):
    logger.debug("GEMS_MTHREADS BATCHNORM FORWARD")
    input_3d = _make_3d_for_bn(input)
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    total = batch_dim * spatial_dim

    if total <= _NATIVE_SWITCH_ELEMENTS:
        rm = running_mean
        rv = running_var
        if rm is None or rv is None:
            rm, rv = _get_temp_stats(input.device, input.dtype, feat_dim)
        with torch_device_fn.device(input.device):
            return torch.ops.aten._native_batch_norm_legit.default(
                input, weight, bias, rm, rv, training, momentum, eps
            )
    output = torch.empty_like(input_3d)
    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    inv_std = torch.empty_like(mean)

    need_stats = training or running_mean is None or running_var is None
    update_running = training and running_mean is not None and running_var is not None

    small_training = total <= _FALLBACK_ELEMENTS and (
        training or running_mean is None or running_var is None
    )

    if small_training:
        block = _block_size(total)
        num_warps = _num_warps(block)
        with torch_device_fn.device(input.device):
            _bn_fused_train_kernel[(feat_dim,)](
                input_3d,
                weight,
                bias,
                output,
                mean,
                inv_std,
                running_mean if running_mean is not None else mean,
                running_var if running_var is not None else inv_std,
                batch_dim,
                spatial_dim,
                *input_3d.stride(),
                *output.stride(),
                momentum,
                eps,
                update_running=update_running,
                BLOCK=block,
                num_warps=num_warps,
            )
        return output.view_as(input), mean, inv_std

    block, num_blocks, num_warps = _get_launch_config(batch_dim, spatial_dim)

    with torch_device_fn.device(input.device):
        if need_stats:
            partial_shape = (feat_dim, num_blocks)
            partial_sum = torch.empty(
                partial_shape, device=input.device, dtype=torch.float32
            )
            partial_sq = torch.empty_like(partial_sum)

            _bn_forward_stats_stage1[(feat_dim, num_blocks)](
                input_3d,
                partial_sum,
                partial_sq,
                batch_dim,
                spatial_dim,
                *input_3d.stride(),
                num_blocks,
                BLOCK=block,
                num_warps=num_warps,
            )

            if num_blocks == 1:
                sum_buf = partial_sum[:, 0].contiguous()
                sum_sq_buf = partial_sq[:, 0].contiguous()
            else:
                sum_buf = torch.zeros(
                    (feat_dim,), device=input.device, dtype=torch.float32
                )
                sum_sq_buf = torch.zeros_like(sum_buf)
                reduce_blocks = triton.cdiv(num_blocks, _REDUCE_BLOCK)
                _bn_reduce_partial_kernel[(feat_dim, reduce_blocks)](
                    partial_sum,
                    partial_sq,
                    sum_buf,
                    sum_sq_buf,
                    num_blocks,
                    BLOCK=_REDUCE_BLOCK,
                    num_warps=_num_warps(_REDUCE_BLOCK),
                )

            _bn_forward_finalize_kernel[(feat_dim,)](
                sum_buf,
                sum_sq_buf,
                mean,
                inv_std,
                running_mean,
                running_var,
                total,
                momentum,
                eps,
                update_running=update_running,
                num_warps=1,
            )
        else:
            if running_mean is None or running_var is None:
                raise RuntimeError(
                    "running_mean and running_var are required in eval mode"
                )
            mean.copy_(running_mean)
            inv_std.copy_((running_var + eps).rsqrt())

        _bn_forward_apply_kernel[(feat_dim, num_blocks)](
            input_3d,
            weight,
            bias,
            mean,
            inv_std,
            output,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            BLOCK=block,
            num_warps=num_warps,
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
    train: bool = False,
    eps: float = 1e-05,
    output_mask=None,
):
    logger.debug("GEMS_MTHREADS BATCHNORM BACKWARD")

    input_3d = _make_3d_for_bn(input)
    output_grad_3d = _make_3d_for_bn(grad_out)
    batch_dim, feat_dim, spatial_dim = input_3d.shape

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

    block, num_blocks, num_warps = _get_launch_config(batch_dim, spatial_dim)

    with torch_device_fn.device(input.device):
        partial_shape = (feat_dim, num_blocks)
        partial_sum = torch.empty(
            partial_shape, device=input.device, dtype=torch.float32
        )
        partial_sum_xhat = torch.empty_like(partial_sum)

        _bn_backward_reduce_kernel[(feat_dim, num_blocks)](
            output_grad_3d,
            input_3d,
            save_mean,
            save_invstd,
            partial_sum,
            partial_sum_xhat,
            batch_dim,
            spatial_dim,
            *output_grad_3d.stride(),
            *input_3d.stride(),
            num_blocks,
            BLOCK=block,
            num_warps=num_warps,
        )

        if num_blocks == 1:
            sum_dy = partial_sum[:, 0].contiguous()
            sum_dy_xhat = partial_sum_xhat[:, 0].contiguous()
        else:
            sum_dy = torch.zeros((feat_dim,), device=input.device, dtype=torch.float32)
            sum_dy_xhat = torch.zeros_like(sum_dy)
            reduce_blocks = triton.cdiv(num_blocks, _REDUCE_BLOCK)
            _bn_backward_reduce_partial_kernel[(feat_dim, reduce_blocks)](
                partial_sum,
                partial_sum_xhat,
                sum_dy,
                sum_dy_xhat,
                num_blocks,
                BLOCK=_REDUCE_BLOCK,
                num_warps=_num_warps(_REDUCE_BLOCK),
            )

        if output_mask[0]:
            _bn_backward_input_kernel[(feat_dim, num_blocks)](
                output_grad_3d,
                input_3d,
                save_mean,
                save_invstd,
                weight,
                sum_dy,
                sum_dy_xhat,
                input_grad,
                batch_dim,
                spatial_dim,
                *output_grad_3d.stride(),
                *input_3d.stride(),
                *input_grad.stride(),
                BLOCK=block,
                num_warps=num_warps,
            )

        if output_mask[1] or output_mask[2]:
            _bn_backward_param_kernel[(feat_dim,)](
                sum_dy,
                sum_dy_xhat,
                weight_grad if weight_grad is not None else sum_dy,
                bias_grad if bias_grad is not None else sum_dy,
                weight_grad_mask=output_mask[1],
                bias_grad_mask=output_mask[2],
                num_warps=1,
            )

    return (
        input_grad.view_as(input) if input_grad is not None else None,
        weight_grad,
        bias_grad,
    )
