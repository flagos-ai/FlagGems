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

"""Element-wise ``aten::minimum`` with a fused Triton backward.

``aten::minimum`` has no dedicated ``*_backward`` ATen schema. Its gradient is
the derivative formula

    self:  where(self == other, grad / 2, grad).masked_fill(self > other, 0)
    other: where(self == other, grad / 2, grad).masked_fill(self < other, 0)

which ATen materialises as several elementwise kernels. The FlagGems entry owns
the whole differentiable path (see ``_Minimum``) and replaces that decomposition
with one fused pass over ``grad_output``, ``x`` and ``y``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.sum import sum_dim
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

_INT32_SAFE = 1 << 31


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 0, "DEFAULT")])
@triton.jit
def minimum_kernel(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)
    return tl.minimum(X, Y)


@pointwise_dynamic(
    is_tensor=[True, True, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def minimum_backward_self_kernel(grad, X, Y):
    """grad wrt ``X`` for the broadcast / strided fallback."""
    grad = grad.to(tl.float32)
    X = X.to(tl.float32)
    Y = Y.to(tl.float32)
    tie = X == Y
    return tl.where(X > Y, 0.0, tl.where(tie, grad * 0.5, grad))


@pointwise_dynamic(
    is_tensor=[True, True, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def minimum_backward_other_kernel(grad, X, Y):
    """grad wrt ``Y`` for the broadcast / strided fallback."""
    grad = grad.to(tl.float32)
    X = X.to(tl.float32)
    Y = Y.to(tl.float32)
    tie = X == Y
    return tl.where(Y > X, 0.0, tl.where(tie, grad * 0.5, grad))


# Tile/warp geometry, plus the two memory hints that measured best on H800:
# ``LH=1`` bypasses L1 on the loads (three read streams with no reuse) and
# ``SW=1`` marks the stores as streaming. The plain path stays in the candidate
# set, so a shape that prefers it still gets it. The list is written inline
# because ``pointwise_dynamic`` copies every standalone ``@triton.jit`` function
# of this module into its generated cache file, which would not carry a
# module-level constant with it.
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024, "LH": 0, "SW": 0}, num_warps=4),
        triton.Config({"BLOCK": 1024, "LH": 0, "SW": 1}, num_warps=4),
        triton.Config({"BLOCK": 1024, "LH": 1, "SW": 0}, num_warps=4),
        triton.Config({"BLOCK": 2048, "LH": 0, "SW": 0}, num_warps=4),
        triton.Config({"BLOCK": 2048, "LH": 0, "SW": 1}, num_warps=4),
        triton.Config({"BLOCK": 2048, "LH": 1, "SW": 0}, num_warps=4),
        triton.Config({"BLOCK": 4096, "LH": 0, "SW": 0}, num_warps=8),
        triton.Config({"BLOCK": 4096, "LH": 0, "SW": 1}, num_warps=8),
        triton.Config({"BLOCK": 4096, "LH": 1, "SW": 0}, num_warps=8),
        triton.Config({"BLOCK": 8192, "LH": 0, "SW": 0}, num_warps=8),
        triton.Config({"BLOCK": 8192, "LH": 0, "SW": 1}, num_warps=8),
        triton.Config({"BLOCK": 8192, "LH": 1, "SW": 1}, num_warps=16),
    ],
    key=["n", "dtype_key"],
)
@triton.jit(do_not_specialize=["n"])
def minimum_backward_fused_kernel(
    grad_ptr,
    x_ptr,
    y_ptr,
    gx_ptr,
    gy_ptr,
    n,
    dtype_key,
    NEED_GX: tl.constexpr,
    NEED_GY: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LH: tl.constexpr,
    SW: tl.constexpr,
):
    """Fused same-shape gradient of ``aten::minimum``.

    Both operands come from one pass: ``x`` decides ``grad_x``, ``y`` decides
    ``grad_y``, and the tie half is shared. ``dtype_key`` is never read; it only
    gives the autotuner a per-dtype cache entry next to ``n``.

    The arithmetic stays in the element dtype. ``+/-0.0`` compare equal, so they
    take the tie branch, and every comparison against NaN is false, which leaves
    the full gradient on both sides -- exactly what the native formula does.
    """
    pid = tl.program_id(0)
    if WIDE:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)

    if (n % BLOCK) == 0:
        # Every lane of every tile is in range: drop the predicates so 16-bit
        # accesses keep their full 128-bit vector width.
        if LH == 0:
            grad = tl.load(grad_ptr + offs)
            xv = tl.load(x_ptr + offs)
            yv = tl.load(y_ptr + offs)
        else:
            grad = tl.load(grad_ptr + offs, cache_modifier=".cg")
            xv = tl.load(x_ptr + offs, cache_modifier=".cg")
            yv = tl.load(y_ptr + offs, cache_modifier=".cg")
        tie = xv == yv
        gv = tl.where(tie, grad * 0.5, grad)
        if NEED_GX:
            gx = tl.where(xv > yv, 0.0, gv)
            if SW == 0:
                tl.store(gx_ptr + offs, gx)
            else:
                tl.store(gx_ptr + offs, gx, cache_modifier=".cs")
        if NEED_GY:
            gy = tl.where(yv > xv, 0.0, gv)
            if SW == 0:
                tl.store(gy_ptr + offs, gy)
            else:
                tl.store(gy_ptr + offs, gy, cache_modifier=".cs")
    else:
        mask = offs < n
        if LH == 0:
            grad = tl.load(grad_ptr + offs, mask=mask, other=0.0)
            xv = tl.load(x_ptr + offs, mask=mask, other=0.0)
            yv = tl.load(y_ptr + offs, mask=mask, other=0.0)
        else:
            grad = tl.load(grad_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
            xv = tl.load(x_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
            yv = tl.load(y_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
        tie = xv == yv
        gv = tl.where(tie, grad * 0.5, grad)
        if NEED_GX:
            gx = tl.where(xv > yv, 0.0, gv)
            if SW == 0:
                tl.store(gx_ptr + offs, gx, mask=mask)
            else:
                tl.store(gx_ptr + offs, gx, mask=mask, cache_modifier=".cs")
        if NEED_GY:
            gy = tl.where(yv > xv, 0.0, gv)
            if SW == 0:
                tl.store(gy_ptr + offs, gy, mask=mask)
            else:
                tl.store(gy_ptr + offs, gy, mask=mask, cache_modifier=".cs")


# Autotune cache discriminator: fp16, bf16 and fp32 have very different vector
# widths, so they must not share a tuned config.
_minimum_backward_fixed_kernel = libentry()(minimum_backward_fused_kernel.jit_function)
_DTYPE_KEY = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}

# Below this size the backward is a handful of CTAs, i.e. launch bound; it runs
# a fixed tile instead of paying the autotuner's per-call key lookup.
_SMALL_INPUT = 1 << 16
_SMALL_BLOCK = 1024


def _sum_to(tensor, shape):
    """Sum a broadcasted gradient back down to ``shape`` (right-aligned)."""
    shape = tuple(shape)
    while tensor.dim() > len(shape):
        tensor = sum_dim(tensor, dim=[0], keepdim=False)
    for i, size in enumerate(shape):
        if size == 1 and tensor.shape[i] != 1:
            tensor = sum_dim(tensor, dim=[i], keepdim=True)
    return tensor.reshape(shape)


def _fused_supported(grad, x, y):
    """The one-pass kernel covers the benchmark's same-shape contract."""
    return (
        grad.shape == x.shape
        and x.shape == y.shape
        and grad.dtype == x.dtype
        and x.dtype == y.dtype
        and x.dtype in _DTYPE_KEY
        and grad.is_contiguous()
        and x.is_contiguous()
        and y.is_contiguous()
    )


def _minimum_backward_fused(grad, x, y, needs):
    need_gx, need_gy = needs
    gx = torch.empty_like(x) if need_gx else None
    gy = torch.empty_like(y) if need_gy else None
    n = grad.numel()
    if n == 0:
        return gx, gy
    # The kernel takes both pointers; an unneeded side aliases the other one and
    # its store is compiled out by the constexpr flag.
    gx_ptr = gx if need_gx else gy
    gy_ptr = gy if need_gy else gx
    args = (grad, x, y, gx_ptr, gy_ptr, n, _DTYPE_KEY[x.dtype])
    flags = dict(NEED_GX=need_gx, NEED_GY=need_gy, WIDE=n > _INT32_SAFE)
    with torch_device_fn.device(x.device):
        if n <= _SMALL_INPUT:
            # A few tiles cannot benefit from tile-geometry tuning, and the autotuner
            # adds per-call key bookkeeping that dominates such a launch.
            _minimum_backward_fixed_kernel[(triton.cdiv(n, _SMALL_BLOCK),)](
                *args, **flags, BLOCK=_SMALL_BLOCK, LH=0, SW=0, num_warps=4
            )
        else:
            minimum_backward_fused_kernel[
                lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
            ](*args, **flags)
    return gx, gy


def _minimum_backward_general(grad, x, y, needs):
    """Broadcast / strided / mixed-dtype path.

    The reduction runs in float32 so a broadcast gradient is summed with the
    accumulation width ATen uses; summing the half-precision values directly
    drifts from the reference by more than the dtype tolerance.
    """
    need_gx, need_gy = needs
    gx = gy = None
    if need_gx:
        selected = minimum_backward_self_kernel(grad, x, y).to(torch.float32)
        gx = _sum_to(selected, x.shape).to(x.dtype)
    if need_gy:
        selected = minimum_backward_other_kernel(grad, x, y).to(torch.float32)
        gy = _sum_to(selected, y.shape).to(y.dtype)
    return gx, gy


class _Minimum(torch.autograd.Function):
    """Autograd wrapper for ``aten::minimum``.

    ATen derives this gradient from a formula instead of a ``*_backward`` op, so
    overriding the CUDA kernel alone would leave the generated
    ``MinimumBackward0`` node (and its several elementwise kernels) in place.
    The FlagGems entry owns the whole differentiable path instead.
    """

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return minimum_kernel(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return minimum_backward(grad_output, x, y, ctx.needs_input_grad)


def minimum(X, Y):
    logger.debug("GEMS MINIMUM")
    assert X.device.type == device and Y.device.type == device
    if torch.is_grad_enabled() and (X.requires_grad or Y.requires_grad):
        return _Minimum.apply(X, Y)
    return minimum_kernel(X, Y)


def minimum_backward(grad_output, x, y, needs_input_grad=(True, True)):
    """Return first-order gradients for the requested minimum operands."""
    if not any(needs_input_grad):
        return None, None
    if _fused_supported(grad_output, x, y):
        return _minimum_backward_fused(grad_output, x, y, needs_input_grad)
    return _minimum_backward_general(grad_output, x, y, needs_input_grad)
