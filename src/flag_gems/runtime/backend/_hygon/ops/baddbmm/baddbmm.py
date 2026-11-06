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

"""Hygon baddbmm public entry, dispatch and autograd."""

import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.triton_version_utils import HAS_TLE

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
    if (
        a.ndim != 3
        or b.ndim != 3
        or a.shape[0] != b.shape[0]
        or a.shape[2] != b.shape[1]
    ):
        raise RuntimeError("baddbmm expects compatible batches of matrices")
    if a.dtype != b.dtype or a.dtype != bias.dtype:
        raise RuntimeError("baddbmm expects all inputs to have the same dtype")
    if a.device != b.device or a.device != bias.device:
        raise RuntimeError("baddbmm expects all inputs on the same device")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "Hygon baddbmm supports float16, bfloat16 and float32"
        )
    shape = (a.shape[0], a.shape[1], b.shape[2])
    if bias.ndim > 3:
        raise RuntimeError("baddbmm bias is not broadcastable to the output")
    sizes = (1,) * (3 - bias.ndim) + tuple(bias.shape)
    strides = (0,) * (3 - bias.ndim) + bias.stride()
    if any(size != 1 and size != target for size, target in zip(sizes, shape)):
        raise RuntimeError("baddbmm bias is not broadcastable to the output")
    return tuple(0 if size == 1 else stride for size, stride in zip(sizes, strides))


def _launch(bias, a, b, beta, alpha, out, bias_strides=None):
    if bias_strides is None:
        bias_strides = _validate(bias, a, b)
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


class _Baddbmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, a, b, beta, alpha):
        ctx.save_for_backward(a, b)
        ctx.bias_shape, ctx.alpha, ctx.beta = bias.shape, alpha, beta
        bias_strides = _validate(bias, a, b)
        out = torch.empty(
            (a.shape[0], a.shape[1], b.shape[2]), device=a.device, dtype=a.dtype
        )
        return _launch(bias, a, b, beta, alpha, out, bias_strides=bias_strides)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        from flag_gems.ops.copy import copy_
        from flag_gems.ops.mul import mul
        from flag_gems.ops.sum_to_size import sum_to_size

        def matmul_grad(left, right):
            dtype = left.dtype
            if dtype == torch.float16:
                left_fp32 = torch.empty_like(left, dtype=torch.float32)
                right_fp32 = torch.empty_like(right, dtype=torch.float32)
                copy_(left_fp32, left)
                copy_(right_fp32, right)
                left, right = left_fp32, right_fp32
            # Reuse this operator's strided Triton product with a fused scale.
            # beta=0 keeps the uninitialized scalar bias unread.
            bias = torch.empty((), device=left.device, dtype=left.dtype)
            result = torch.empty(
                (left.shape[0], left.shape[1], right.shape[2]),
                device=left.device,
                dtype=left.dtype,
            )
            _launch(bias, left, right, 0.0, ctx.alpha, result)
            if dtype == torch.float16:
                output = torch.empty_like(result, dtype=dtype)
                return copy_(output, result)
            return result

        dbias = (
            sum_to_size(mul(grad, ctx.beta), ctx.bias_shape)
            if ctx.needs_input_grad[0]
            else None
        )
        da = matmul_grad(grad, b.transpose(1, 2)) if ctx.needs_input_grad[1] else None
        db = matmul_grad(a.transpose(1, 2), grad) if ctx.needs_input_grad[2] else None
        return dbias, da, db, None, None


def _call(bias, a, b, beta, alpha):
    if torch.is_grad_enabled() and any(x.requires_grad for x in (bias, a, b)):
        return _Baddbmm.apply(bias, a, b, beta, alpha)
    bias_strides = _validate(bias, a, b)
    output_shape = (a.shape[0], a.shape[1], b.shape[2])
    if bias.ndim == 3 and tuple(bias.shape) == output_shape and bias.is_contiguous():
        out = torch.empty_like(bias)
    else:
        out = torch.empty(output_shape, dtype=a.dtype, device=a.device)
    return _launch(bias, a, b, beta, alpha, out, bias_strides=bias_strides)


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    logger.debug("GEMS BADDBMM")
    return _call(bias, A, B, beta, alpha)


def baddbmm_out(bias, A, B, *, beta=1.0, alpha=1.0, out):
    logger.debug("GEMS BADDBMM_OUT")
    bias_strides = _validate(bias, A, B)
    if torch.is_grad_enabled() and any(x.requires_grad for x in (bias, A, B, out)):
        raise RuntimeError("baddbmm.out does not support automatic differentiation")
    if (
        out.shape != (A.shape[0], A.shape[1], B.shape[2])
        or out.dtype != A.dtype
        or out.device != A.device
    ):
        raise RuntimeError(
            "baddbmm.out expects the output shape, dtype and device to match"
        )
    if any(torch._C._overlaps(out, x) for x in (bias, A, B)):
        result = _call(bias, A, B, beta, alpha)
        from flag_gems.ops.copy import copy_

        copy_(out, result)
        return out
    return _launch(bias, A, B, beta, alpha, out, bias_strides=bias_strides)
