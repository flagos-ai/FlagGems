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

"""Hygon baddbmm entry points and kernel dispatch."""

import logging

import torch
import triton
from torch.autograd.function import once_differentiable

from flag_gems.ops.sum import sum_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.triton_version_utils import HAS_TLE

from ..bmm.bmm import check_out_autograd, copy_tensor
from ..bmm.splitk import launch_small_splitk
from .kernels import _TUNERS, _bias_kernel, _pack_b

logger = logging.getLogger(__name__)


def _algorithm(batch, m, n, k, device=None, dtype=None):
    if (
        dtype in (torch.float16, torch.bfloat16)
        and batch == 1
        and m >= 4096
        and n >= 1024
        and k >= 1024
    ):
        return "packed"
    if (
        HAS_TLE
        and dtype == torch.bfloat16
        and batch > 1
        and min(m, n) >= 1024
        and k >= 512
        and k % 64 == 0
        and n % 8 == 0
        and batch * triton.cdiv(m, 256) * triton.cdiv(n, 256) >= 24
        and 2 * m * n >= triton.cdiv(m, 256) * triton.cdiv(n, 256) * 256**2
        and max(batch * m * k, batch * k * n, batch * m * n) * 2 < 2**31
        and triton.cdiv(m, 256) * 256 * n * 2 < 2**31
        and getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx936"
    ):
        return "ring"
    return "matmul"


def _validate(bias, a, b):
    a_shape, b_shape = a.shape, b.shape
    if (
        len(a_shape) != 3
        or len(b_shape) != 3
        or a_shape[0] != b_shape[0]
        or a_shape[2] != b_shape[1]
    ):
        raise RuntimeError("baddbmm expects compatible batches of matrices")
    dtype = a.dtype
    if dtype != b.dtype or dtype != bias.dtype:
        raise RuntimeError("baddbmm expects all inputs to have the same dtype")
    device = a.device
    if device != b.device or device != bias.device:
        raise RuntimeError("baddbmm expects all inputs on the same device")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "Hygon baddbmm supports float16, bfloat16 and float32"
        )
    shape = (a_shape[0], a_shape[1], b_shape[2])
    bias_shape = bias.shape
    if bias_shape == shape:
        return shape, bias.stride()
    if len(bias_shape) > 3:
        raise RuntimeError("baddbmm bias is not broadcastable to the output")
    sizes = (1,) * (3 - len(bias_shape)) + tuple(bias_shape)
    strides = (0,) * (3 - len(bias_shape)) + bias.stride()
    if any(size != 1 and size != target for size, target in zip(sizes, shape)):
        raise RuntimeError("baddbmm bias is not broadcastable to the output")
    strides = tuple(0 if size == 1 else stride for size, stride in zip(sizes, strides))
    return shape, strides


def _launch(bias, a, b, beta, alpha, out, bias_strides):
    batch, m, k = a.shape
    n = b.shape[2]
    if not batch or not m or not n:
        return out
    with torch_device_fn.device(a.device):
        if k == 0 or alpha == 0:
            _bias_kernel[(triton.cdiv(m * n, 256), batch)](
                bias, out, m, n, *bias_strides, *out.stride(), beta, 256
            )
            return out
        if launch_small_splitk(a, b, out, bias.broadcast_to(out.shape), alpha, beta):
            return out
        algorithm = _algorithm(batch, m, n, k, a.device, a.dtype)
        if algorithm == "ring":
            if all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (a, b, out)):
                _TUNERS["ring"][(triton.cdiv(m, 256) * triton.cdiv(n, 256), batch)](
                    a, b, bias, out, m, n, k, *bias_strides, alpha, beta, BATCH=batch
                )
                return out
            algorithm = "matmul"
        if algorithm == "packed":
            packed = torch.empty((batch, n, k), device=b.device, dtype=b.dtype)
            _pack_b[(triton.cdiv(k, 64), triton.cdiv(n, 64), batch)](
                b, packed, n, k, *b.stride(), num_warps=4
            )
            b = packed.transpose(1, 2)
        _TUNERS[algorithm][
            lambda meta: (
                triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
                batch,
            )
        ](
            a,
            b,
            bias,
            out,
            m,
            n,
            k,
            *a.stride(),
            *b.stride(),
            *bias_strides,
            *out.stride(),
            alpha,
            beta,
            BATCH=batch,
        )
    return out


def _forward_checked(bias, a, b, beta, alpha, shape, bias_strides):
    out = (
        torch.empty_like(bias, memory_format=torch.contiguous_format)
        if bias.shape == shape
        else torch.empty(shape, dtype=a.dtype, device=a.device)
    )
    return _launch(bias, a, b, beta, alpha, out, bias_strides)


def _bias_gradient(grad, bias_shape, beta):
    out = torch.empty(bias_shape, device=grad.device, dtype=grad.dtype)
    if grad.numel() == 0:
        return copy_tensor(out, out, zero=True)

    shape = (1,) * (grad.ndim - len(bias_shape)) + bias_shape
    dims = [i for i, (src, dst) in enumerate(zip(grad.shape, shape)) if src != dst]
    if dims:
        kept = [i for i in range(grad.ndim) if i not in dims]
        grad = grad.permute(*kept, *dims)
        if not grad.is_contiguous():
            grad = copy_tensor(
                grad, torch.empty(grad.shape, device=grad.device, dtype=grad.dtype)
            )
        rows = grad.view(out.numel(), grad.numel() // out.numel())
        grad = sum_dim(rows, dim=(1,), dtype=torch.float32)

    return copy_tensor(grad.view(bias_shape), out, beta)


def _matmul_gradient(a, b, alpha):
    out = torch.empty(
        (a.shape[0], a.shape[1], b.shape[2]), device=a.device, dtype=a.dtype
    )
    # A zero beta makes the bias pointer unused. Apply alpha before casting.
    return _launch(out, a, b, 0, alpha, out, (0, 0, 0))


class _Baddbmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, a, b, beta, alpha, metadata):
        ctx.save_for_backward(a, b)
        ctx.bias_shape = tuple(bias.shape)
        ctx.alpha, ctx.beta = alpha, beta
        return _forward_checked(bias, a, b, beta, alpha, *metadata)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        need_bias, need_a, need_b = ctx.needs_input_grad[:3]
        dbias = _bias_gradient(grad, ctx.bias_shape, ctx.beta) if need_bias else None
        da = _matmul_gradient(grad, b.transpose(1, 2), ctx.alpha) if need_a else None
        db = _matmul_gradient(a.transpose(1, 2), grad, ctx.alpha) if need_b else None
        return dbias, da, db, None, None, None


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    logger.debug("GEMS BADDBMM")
    metadata = _validate(bias, A, B)
    if torch.is_grad_enabled() and (
        bias.requires_grad or A.requires_grad or B.requires_grad
    ):
        return _Baddbmm.apply(bias, A, B, beta, alpha, metadata)
    return _forward_checked(bias, A, B, beta, alpha, *metadata)


def baddbmm_out(bias, A, B, *, beta=1.0, alpha=1.0, out):
    logger.debug("GEMS BADDBMM_OUT")
    shape, bias_strides = _validate(bias, A, B)
    check_out_autograd("baddbmm", bias, A, B, out)
    if out.shape != shape or out.dtype != A.dtype or out.device != A.device:
        raise RuntimeError(
            "baddbmm.out expects the output shape, dtype and device to match"
        )
    if any(torch._C._overlaps(out, x) for x in (bias, A, B)):
        result = _forward_checked(bias, A, B, beta, alpha, shape, bias_strides)
        return copy_tensor(result, out)
    return _launch(bias, A, B, beta, alpha, out, bias_strides)
