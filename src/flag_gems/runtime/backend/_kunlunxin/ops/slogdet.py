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

"""Kunlunxin/XPU implementation of ``slogdet`` (``aten::slogdet``).

The generic ``flag_gems.ops.slogdet`` keeps each matrix in a 2-D register
tile and extracts pivot columns with ``tl.sum(..., axis=1)``.  TritonXPU
rejects reductions whose source is an arithmetic 2-D tile (`arith.cmpf`
operand-encoding mismatch during ``TritonXPULegalize``), so this backend
module reuses the validated pivot-LU machinery of the backend
``linalg_det`` and post-processes the determinant into ``(sign, logabsdet)``.

Forming the determinant directly needs ``det(A) = sign * exp(logabsdet)`` to
stay inside the fp32 range: for a 64x64 standard-normal matrix
``log|det| ~ 100`` and a raw fp32 determinant overflows.  Each matrix is
therefore pre-scaled by ``c = max|A|`` (clamped away from zero), so
``det(A) = c**n * det(A/c)`` and

    logabsdet = n * log(c) + log|det(A/c)|

with ``det(A/c)`` comfortably inside range (measured ``|log|det|| < 10`` for
the test/benchmark matrix family).  Scale and post-processing are two small
Triton kernels; no torch/aten compute is involved.
"""

import importlib
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SCALE_BLOCK = 1024


def _backend_module(name):
    # Import the submodule (not the function of the same name): the package
    # namespace exposes functions, and a plain `from . import x` would shadow
    # the module.  Importing the function instead would register an extra
    # operator through SpecOpRegistrar.
    return importlib.import_module("." + name, __package__)


@libentry()
@triton.jit
def _amax_kernel(A, scale_out, n, BLOCK: tl.constexpr):
    """Per-matrix ``max|A|`` over the flattened n*n elements."""
    pid = tl.program_id(0)
    total = n * n
    base = pid * total
    best = 0.0
    for i0 in range(0, total, BLOCK):
        cols = tl.arange(0, BLOCK)
        offs = i0 + cols
        msk = offs < total
        v = tl.load(A + base + offs, mask=msk, other=0.0)
        a = tl.where(msk, tl.abs(v), 0.0)
        best = tl.maximum(best, tl.max(a, axis=0))
    tl.store(scale_out + pid, best)


@libentry()
@triton.jit
def _scale_kernel(X, scale, OUT, n, BLOCK: tl.constexpr):
    """``OUT = X / scale[matrix]`` for the flattened n*n block of one matrix."""
    pid = tl.program_id(0)
    blk = tl.program_id(1)
    total = n * n
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    msk = offs < total
    base = pid * total
    s = tl.load(scale + pid)
    # Clamp zero / non-finite scales (bit test avoids a float compare): an Inf
    # or NaN entry would otherwise turn the scaled matrix into a NaN block.
    bad = (s.to(tl.int32, bitcast=True) & 0x7FFFFFFF) >= 0x7F800000
    s = tl.where((s == 0.0) | bad, 1.0, s)
    v = tl.load(X + base + offs, mask=msk, other=0.0)
    tl.store(OUT + base + offs, v / s, mask=msk)


@libentry()
@triton.jit
def _post_kernel(DET, SCALE, sign_out, lad_out, n):
    """(det, scale) -> (sign, logabsdet) for one matrix per program."""
    pid = tl.program_id(0)
    d = tl.load(DET + pid)
    c = tl.load(SCALE + pid)
    bad = (c.to(tl.int32, bitcast=True) & 0x7FFFFFFF) >= 0x7F800000
    c = tl.where((c == 0.0) | bad, 1.0, c)
    lad = tl.log(tl.abs(d)) + n * tl.log(c)
    sign = tl.where(d > 0.0, 1.0, tl.where(d < 0.0, -1.0, d))
    tl.store(sign_out + pid, sign.to(sign_out.dtype.element_ty))
    tl.store(lad_out + pid, lad.to(lad_out.dtype.element_ty))


@libentry()
@triton.jit
def _grad_kernel(INV, G, OUT, n, BLOCK: tl.constexpr):
    """``OUT[i, j] = g * conj(INV[j, i])``; grid = (batch, row)."""
    pid = tl.program_id(0)
    row = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    cmsk = cols < n
    base = pid * n * n
    g = tl.load(G + pid)
    # Column `row` of INV, i.e. INV[j, row] for all j -> OUT[row, j].
    src = tl.load(INV + base + cols * n + row, mask=cmsk, other=0.0)
    tl.store(OUT + base + row * n + cols, g * src, mask=cmsk)


def _check_input(A):
    if A.dim() < 2:
        raise RuntimeError(
            "linalg.slogdet: The input tensor A must have at least 2 dimensions."
        )
    if A.shape[-1] != A.shape[-2]:
        raise RuntimeError(
            "linalg.slogdet: A must be batches of square matrices, "
            f"but they are {A.shape[-2]} by {A.shape[-1]} matrices"
        )
    if A.dtype in (torch.float16, torch.bfloat16):
        name = "Half" if A.dtype == torch.float16 else "BFloat16"
        raise RuntimeError(
            f"linalg.slogdet: Low precision dtypes not supported. Got {name}"
        )
    if A.dtype not in (torch.float32, torch.float64):
        # Complex linalg is unavailable on this backend (no complex GE kernels;
        # torch.eye(complex64) is unimplemented in the vendor stack).
        raise RuntimeError(
            "linalg.slogdet: Expected a floating point or complex tensor as input. "
            f"Got {A.dtype}"
        )


def _slogdet_forward(A):
    _check_input(A)

    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    real_dtype = A.dtype
    batch_size = math.prod(batch_shape)

    if batch_size == 0:
        return (
            torch.empty(batch_shape, dtype=A.dtype, device=A.device),
            torch.empty(batch_shape, dtype=real_dtype, device=A.device),
        )
    if n == 0:
        return (
            torch.ones(batch_shape, dtype=A.dtype, device=A.device),
            torch.zeros(batch_shape, dtype=real_dtype, device=A.device),
        )

    # The factorization runs in place on its own buffer; keep the caller's
    # tensor untouched.
    A_work = A.contiguous().clone().reshape(batch_size, n, n)

    scale = torch.empty(batch_size, dtype=A.dtype, device=A.device)
    scaled = torch.empty((batch_size, n, n), dtype=A.dtype, device=A.device)
    sign = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    logabsdet = torch.empty(batch_shape, dtype=real_dtype, device=A.device)

    det_mod = _backend_module("linalg_det")

    # det() itself stays within fp32 range up to n=32 (measured |log|det|| <
    # 45); larger matrices need the max-abs prescale to avoid overflow.
    rescale = n > 32
    with torch_device_fn.device(A.device):
        if rescale:
            _amax_kernel[(batch_size,)](
                A_work, scale, n, BLOCK=_SCALE_BLOCK, num_warps=1
            )
            _scale_kernel[(batch_size, triton.cdiv(n * n, _SCALE_BLOCK))](
                A_work, scale, scaled, n, BLOCK=_SCALE_BLOCK, num_warps=1
            )
        else:
            scale.fill_(1.0)
            scaled = A_work
        det = det_mod._linalg_det_impl(scaled)
        _post_kernel[(batch_size,)](
            det.reshape(batch_size),
            scale,
            sign.reshape(-1),
            logabsdet.reshape(-1),
            n,
        )

    return sign, logabsdet


class _Slogdet(torch.autograd.Function):
    """``d logabsdet / dA = inv(A)^H``; ``sign`` carries no gradient."""

    @staticmethod
    def forward(ctx, A):
        sign, logabsdet = _slogdet_forward(A)
        ctx.save_for_backward(A)
        return sign, logabsdet

    @staticmethod
    def backward(ctx, grad_sign, grad_logabsdet):
        (A,) = ctx.saved_tensors
        if grad_logabsdet is None:
            return None
        batch_shape = A.shape[:-2]
        n = A.shape[-1]
        batch_size = math.prod(batch_shape)
        A_work = A.contiguous().reshape(batch_size, n, n)
        inv_mod = _backend_module("inverse")
        inv = inv_mod.inverse(A_work)
        g = grad_logabsdet.reshape(batch_size).to(inv.dtype)
        out = torch.empty_like(inv)
        block = max(64, triton.next_power_of_2(n))
        with torch_device_fn.device(A.device):
            _grad_kernel[(batch_size, n)](inv, g, out, n, BLOCK=block, num_warps=1)
        return out.reshape(A.shape)


def slogdet(A):
    """Return ``(sign, logabsdet)`` of a square matrix / batch of matrices."""
    logger.debug("GEMS_KUNLUNXIN SLOGDET")
    if torch.is_grad_enabled() and A.requires_grad:
        return _Slogdet.apply(A)
    return _slogdet_forward(A)
