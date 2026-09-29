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
import warnings

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mm import get_higher_dtype
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def tensordot_mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    # Tiled matrix multiplication C = A @ B, where A is (M, K) and B is (K, N).
    # The contracted axes of tensordot are collapsed into K before launch.
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)

    if IS_FP64:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = rk < K
        a = tl.load(
            A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn),
            mask=mask_k[:, None],
            other=0.0,
        )
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if IS_FP64:
            acc += tl.dot(a, b, allow_tf32=False)
        else:
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(C.dtype.element_ty)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


def _matmul_2d(a, b, out=None):
    # Contract two 2D tensors (M, K) x (K, N) -> (M, N) with the Triton kernel.
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=c_dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        tensordot_mm_kernel[grid](
            a,
            b,
            out,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            GROUP_M=8,
            IS_FP64=a.dtype == torch.float64,
        )
    return out


def _normalize_dims(dims, ndim):
    # Map (possibly negative) dim indices into the range [0, ndim).
    # Raise error for out-of-range dimensions.
    normalized = []
    for d in dims:
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})"
            )
        normalized.append(d if d >= 0 else d + ndim)
    return normalized


def _invert_permutation(perm):
    # Invert a permutation so a permuted tensor can be mapped back to its
    # original axis order. perm[i] is the source axis for destination axis i.
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _resolve_contraction(self, other, dims_self, dims_other):
    # Normalize contracted dims and reconcile their sizes following the ATen
    # tensordot contract: a contracted axis may be size 1 on either side, in
    # which case it is broadcast (and later reduced) rather than rejected. Only
    # unequal, non-1 sizes are an error. Returns normalized dim lists together
    # with the (possibly expanded) tensors that share matching contracted sizes.
    dims_self = _normalize_dims(list(dims_self), self.ndim)
    dims_other = _normalize_dims(list(dims_other), other.ndim)

    if len(dims_self) != len(dims_other):
        raise RuntimeError(
            "tensordot: number of contracted dims must match, but got "
            f"{len(dims_self)} and {len(dims_other)}"
        )

    self_shape = list(self.shape)
    other_shape = list(other.shape)
    for ds, do in zip(dims_self, dims_other):
        s1 = self.shape[ds]
        s2 = other.shape[do]
        if s1 == s2:
            continue
        if s1 == 1:
            self_shape[ds] = s2
        elif s2 == 1:
            other_shape[do] = s1
        else:
            raise RuntimeError(
                "tensordot: contracted dimensions need to match, but got "
                f"size {s1} in dim {ds} of self and size {s2} in dim {do} of "
                "other"
            )

    self_e = self.expand(self_shape) if self_shape != list(self.shape) else self
    other_e = other.expand(other_shape) if other_shape != list(other.shape) else other
    return dims_self, dims_other, self_e, other_e


def _copy_tensordot_out(result: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Validate, resize, and copy a computed tensordot result into an out tensor."""
    if out.device != result.device:
        raise RuntimeError(
            "tensordot: Expected the output tensor to be on the same device as "
            f"the result ({result.device}), but got {out.device}"
        )
    if out.dtype != result.dtype:
        raise RuntimeError(
            f"tensordot: Expected the output tensor to have dtype {result.dtype}, "
            f"but got {out.dtype}"
        )
    if tuple(out.shape) != tuple(result.shape):
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(result.shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=3,
            )
        out.resize_(result.shape)
    out.copy_(result)
    return out


def _tensordot_impl(self, other, dims_self, dims_other, out=None):
    # Validate supported dtypes - only floating point types
    supported_dtypes = {torch.float16, torch.float32, torch.bfloat16}
    if runtime.device.support_fp64:
        supported_dtypes.add(torch.float64)

    if self.dtype not in supported_dtypes:
        raise RuntimeError(
            f"tensordot: unsupported dtype {self.dtype}. "
            f"Only floating point dtypes are supported."
        )

    if other.dtype not in supported_dtypes:
        raise RuntimeError(
            f"tensordot: unsupported dtype {other.dtype}. "
            f"Only floating point dtypes are supported."
        )

    # Validate dtype and device match
    if self.dtype != other.dtype:
        raise RuntimeError(
            f"tensordot: expected self and other to have the same dtype, "
            f"but got {self.dtype} and {other.dtype}"
        )
    if self.device != other.device:
        raise RuntimeError(
            f"tensordot: expected self and other to be on the same device, "
            f"but got {self.device} and {other.device}"
        )

    dims_self, dims_other, self, other = _resolve_contraction(
        self, other, dims_self, dims_other
    )

    # Free (non-contracted) dims, preserving their original order.
    free_self = [d for d in range(self.ndim) if d not in dims_self]
    free_other = [d for d in range(other.ndim) if d not in dims_other]

    free_self_sizes = [self.shape[d] for d in free_self]
    free_other_sizes = [other.shape[d] for d in free_other]

    M = math.prod(free_self_sizes) if free_self_sizes else 1
    N = math.prod(free_other_sizes) if free_other_sizes else 1
    K = math.prod([self.shape[d] for d in dims_self]) if dims_self else 1

    # Reorder so contracted dims are collapsed into a single axis:
    #   self  -> (free_self..., contracted...)  reshaped to (M, K)
    #   other -> (contracted..., free_other...) reshaped to (K, N)
    a2d = self.permute(free_self + dims_self).contiguous().reshape(M, K)
    b2d = other.permute(dims_other + free_other).contiguous().reshape(K, N)

    result_shape = free_self_sizes + free_other_sizes

    if out is not None:
        # Compute result and copy it into out without reshaping out itself, so a
        # non-contiguous out never creates a temporary tensor. resize_ + copy_
        # matches the aten out= contract (resize when the shape differs, raise on
        # dtype/device mismatch).
        res = _matmul_2d(a2d, b2d).reshape(result_shape)
        return _copy_tensordot_out(res, out)

    res = _matmul_2d(a2d, b2d)
    return res.reshape(result_shape)


class TensordotFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, other, dims_self, dims_other):
        logger.debug("GEMS TENSORDOT FORWARD")

        # Save the raw inputs and their original contracted-dim sizes so the
        # backward pass can broadcast for the matmul and then reduce the
        # gradient back to any size-1 contracted axis, matching ATen.
        dims_self_n, dims_other_n, _, _ = _resolve_contraction(
            self, other, dims_self, dims_other
        )
        ctx.dims_self = dims_self_n
        ctx.dims_other = dims_other_n
        ctx.save_for_backward(self, other)

        return _tensordot_impl(self, other, dims_self, dims_other)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS TENSORDOT BACKWARD")

        self, other = ctx.saved_tensors
        dims_self = ctx.dims_self
        dims_other = ctx.dims_other

        # Original (pre-broadcast) contracted sizes, used to reduce the grad
        # back to a size-1 axis after the matmul over the broadcast size.
        self_orig_sizes = {d: self.shape[d] for d in dims_self}
        other_orig_sizes = {d: other.shape[d] for d in dims_other}

        # Broadcast contracted dims so both operands share matching sizes.
        _, _, self_e, other_e = _resolve_contraction(self, other, dims_self, dims_other)

        grad_self = None
        grad_other = None

        free_self = [d for d in range(self_e.ndim) if d not in dims_self]
        free_other = [d for d in range(other_e.ndim) if d not in dims_other]

        free_self_sizes = [self_e.shape[d] for d in free_self]
        free_other_sizes = [other_e.shape[d] for d in free_other]
        M = math.prod(free_self_sizes) if free_self_sizes else 1
        N = math.prod(free_other_sizes) if free_other_sizes else 1
        K = math.prod([self_e.shape[d] for d in dims_self]) if dims_self else 1

        grad_2d = grad_output.reshape(M, N)

        if ctx.needs_input_grad[0]:
            b2d = other_e.permute(dims_other + free_other).contiguous().reshape(K, N)
            grad_a2d = _matmul_2d(grad_2d, b2d.transpose(0, 1))
            grad_self = grad_a2d.reshape(self_e.permute(free_self + dims_self).shape)
            grad_self = grad_self.permute(
                _invert_permutation(free_self + dims_self)
            ).contiguous()
            # Reduce over any contracted axis that was size 1 on self.
            for d in dims_self:
                if self_orig_sizes[d] == 1 and self_e.shape[d] != 1:
                    grad_self = grad_self.sum(d, keepdim=True)

        if ctx.needs_input_grad[1]:
            a2d = self_e.permute(free_self + dims_self).contiguous().reshape(M, K)
            grad_b2d = _matmul_2d(a2d.transpose(0, 1), grad_2d)
            grad_other = grad_b2d.reshape(
                other_e.permute(dims_other + free_other).shape
            )
            grad_other = grad_other.permute(
                _invert_permutation(dims_other + free_other)
            ).contiguous()
            # Reduce over any contracted axis that was size 1 on other.
            for d in dims_other:
                if other_orig_sizes[d] == 1 and other_e.shape[d] != 1:
                    grad_other = grad_other.sum(d, keepdim=True)

        return grad_self, grad_other, None, None


def tensordot(self, other, dims_self, dims_other):
    logger.debug("GEMS TENSORDOT")
    return TensordotFunction.apply(self, other, dims_self, dims_other)


def tensordot_out(self, other, dims_self, dims_other, *, out):
    logger.debug("GEMS TENSORDOT_OUT")
    _tensordot_impl(self, other, dims_self, dims_other, out=out)
    return out
