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

from .batch_norm import batch_norm_backward, make_3d_for_bn

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _bn_impl_index_rsqrt_kernel(
    var_pointer, invstd_pointer, eps, n_elements, BLOCK: tl.constexpr
):
    """invstd = 1 / sqrt(var + eps) written in the destination dtype."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    var = tl.load(var_pointer + offsets, mask=mask).to(tl.float32)
    tl.store(
        invstd_pointer + offsets,
        tl.rsqrt(var + eps).to(invstd_pointer.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _bn_impl_index_eval_backward_kernel(
    grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    weight_pointer,
    input_grad_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    feat_dim,
    spatial_dim,
    HAS_WEIGHT: tl.constexpr,
    IG_MASK: tl.constexpr,
    WG_MASK: tl.constexpr,
    BG_MASK: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Inference-mode batch-norm backward: no mean/variance correction.

    ``grad_input = grad_output * weight * inv_std`` while
    ``grad_weight = sum((input - mean) * inv_std * grad_output)`` and
    ``grad_bias = sum(grad_output)`` over the (batch, spatial) reduction.
    The reduction is still needed for the affine gradients, so this mirrors the
    vendor training fused kernel minus the ``-(term1*pre_lin + term2)/count``
    correction term (which only exists when the forward computed statistics
    from this same batch).
    """
    c = tl.program_id(axis=0)
    mean = tl.load(mean_pointer + c).to(tl.float32)
    inv_std = tl.load(inv_std_pointer + c).to(tl.float32)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0

    t1 = tl.zeros([TILE_S], dtype=tl.float32)
    t2 = tl.zeros([TILE_S], dtype=tl.float32)
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                mask = idx < spatial_dim
                x = tl.load(input_pointer + base + idx, mask=mask, other=0.0).to(
                    tl.float32
                )
                dy = tl.load(grad_pointer + base + idx, mask=mask, other=0.0).to(
                    tl.float32
                )
                pre_lin = (x - mean) * inv_std
                t1 += tl.where(mask, pre_lin * dy, 0.0)
                t2 += tl.where(mask, dy, 0.0)
            else:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                dy = tl.load(grad_pointer + base + idx).to(tl.float32)
                pre_lin = (x - mean) * inv_std
                t1 += pre_lin * dy
                t2 += dy

    if WG_MASK:
        tl.store(
            weight_grad_pointer + c,
            tl.sum(t1).to(weight_grad_pointer.dtype.element_ty),
        )
    if BG_MASK:
        tl.store(
            bias_grad_pointer + c,
            tl.sum(t2).to(bias_grad_pointer.dtype.element_ty),
        )
    if not IG_MASK:
        return

    scale = inv_std * weight
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                mask = idx < spatial_dim
                dy = tl.load(grad_pointer + base + idx, mask=mask).to(tl.float32)
                tl.store(
                    input_grad_pointer + base + idx,
                    (scale * dy).to(input_grad_pointer.dtype.element_ty),
                    mask=mask,
                )
            else:
                dy = tl.load(grad_pointer + base + idx).to(tl.float32)
                tl.store(
                    input_grad_pointer + base + idx,
                    (scale * dy).to(input_grad_pointer.dtype.element_ty),
                )


def _eval_backward(input, grad_output, weight, mean, invstd, output_mask):
    input_3d = make_3d_for_bn(input)
    grad_3d = make_3d_for_bn(grad_output)
    if not input_3d.is_contiguous():
        input_3d = input_3d.contiguous()
    if not grad_3d.is_contiguous():
        grad_3d = grad_3d.contiguous()

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    count = batch_dim * spatial_dim

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

    if batch_dim == 0 or count == 0:
        if weight_grad is not None:
            weight_grad.zero_()
        if bias_grad is not None:
            bias_grad.zero_()
        return (
            input_grad.view_as(input) if input_grad is not None else input_grad,
            weight_grad,
            bias_grad,
        )

    input_flat = input_3d.reshape(-1)
    grad_flat = grad_3d.reshape(-1)
    ig_flat = input_grad.reshape(-1) if output_mask[0] else grad_flat

    # Loop-carried accumulators only lower at TILE_S <= 128 on XPU3; the masked
    # variant needs 64 lanes below 128 (see native_batch_norm._nbn_fused_tile_s).
    if spatial_dim < 128:
        tile_s, need_mask = 64, (spatial_dim % 64) != 0
    else:
        tile_s, need_mask = 128, (spatial_dim % 128) != 0

    with torch_device_fn.device(input.device):
        _bn_impl_index_eval_backward_kernel[(feat_dim,)](
            grad_flat,
            input_flat,
            mean,
            invstd,
            weight if weight is not None else input_flat,
            ig_flat,
            weight_grad if output_mask[1] else grad_flat,
            bias_grad if output_mask[2] else grad_flat,
            batch_dim,
            feat_dim,
            spatial_dim,
            weight is not None,
            output_mask[0],
            output_mask[1],
            output_mask[2],
            TILE_S=tile_s,
            NEED_MASK=need_mask,
            num_warps=4,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return (
        input_grad.view_as(input) if input_grad is not None else input_grad,
        weight_grad,
        bias_grad,
    )


def _batch_norm_impl_index_backward(
    impl_index: int,
    input,
    grad_output,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_var_transform=None,
    train: bool = False,
    eps: float = 1e-5,
    output_mask=(True, True, True),
    reservedSpace=None,
):
    """Kunlunxin/XPU override of ``aten::_batch_norm_impl_index_backward``.

    The generic implementation's 2D-tile ``_batch_norm_impl_index_backward_kernel``
    does not lower on XPU3 (``triton_xpu.convert_layout`` shape mismatch /
    ``TritonXPUUnrollControl`` uni_sram OOM), so every non-empty call fails at
    compile time. This override keeps the generic *semantics* but routes to the
    vendor kernels:

    * ``train=True``  -> vendor ``batch_norm_backward`` (single-launch
      ``tle.raw`` / Triton fused / 3-stage paths). Every impl_index stores the
      inverse standard deviation in the ``save_var`` slot, so the same native
      backward formula covers impl_index 0/1/2.
    * ``train=False`` -> a dedicated 1D per-feature kernel. In inference mode
      there is no batch-statistics correction term: the gradient is
      ``grad_output * weight * invstd`` and the affine gradients still reduce
      ``(input - mean) * invstd * grad_output`` / ``grad_output``.

    The saved-statistics resolution matches the generic implementation: saved
    stats win when present, otherwise the running statistics rebuild the
    inverse standard deviation ``1/sqrt(running_var + eps)``.
    """
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_IMPL_INDEX_BACKWARD")

    if save_mean is not None and save_mean.numel() != 0:
        mean = save_mean
    else:
        mean = running_mean

    if save_var_transform is not None and save_var_transform.numel() != 0:
        invstd = save_var_transform
    else:
        invstd = torch.empty_like(running_var)
        block = 256
        with torch_device_fn.device(input.device):
            _bn_impl_index_rsqrt_kernel[(triton.cdiv(running_var.numel(), block),)](
                running_var, invstd, eps, running_var.numel(), BLOCK=block
            )

    if train:
        return batch_norm_backward(
            grad_output,
            input,
            weight=weight,
            running_mean=running_mean,
            running_var=running_var,
            save_mean=mean,
            save_invstd=invstd,
            train=True,
            eps=eps,
            output_mask=output_mask,
        )

    return _eval_backward(input, grad_output, weight, mean, invstd, output_mask)
