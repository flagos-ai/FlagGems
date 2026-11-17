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

# Kunlunxin (XPU) implementation of ``matmul_backward``.
#
# Why a backend override at all: the generic ``src/flag_gems/ops/matmul_backward.py``
# imports ``sum_dim`` from the generic ``ops.sum``, whose
# ``sum_dim_kernel_non_inner`` lowers a 2-D tile ``tl.sum(axis=0)`` that this
# backend's TritonXPU cannot legalize ("'triton_xpu.convert_layout' op requires
# the same shape for all operands and results" / uni_sram OutOfResources), so
# every broadcast-reduction shape (e.g. [4,16,32] @ [32,24]) fails to compile.
#
# This implementation mirrors the generic semantics 1:1 but resolves ``sum_dim``
# through the backend package (``.sum``), whose (TLE/KLT row- or fold-)kernels
# are the backend-proven reduction path (same one ``mean_dim`` / ``linear_backward``
# use) and never touch the generic 2-D-tile reduction that cannot lower.
import logging

import torch

from .sum import sum_dim
from .t_copy import t_copy

logger = logging.getLogger(__name__)


def _lookup_matmul_entries():
    """The mm / bmm entries off the flag_gems namespace, resolved at call time.

    Resolving from ``flag_gems`` rather than importing ``ops.mm`` directly is what
    binds the *backend* mm/bmm kernels (SpecOpRegistrar swaps them into the
    namespace): the kunlunxin mm takes SPLIT_K and the general one does not, so a
    direct import would bind the general kernel and then fail with "Keyword
    argument SPLIT_K was specified but unrecognised". Same reasoning as the
    generic ``_lookup_matmul_entries``.
    """
    from flag_gems import bmm, mm

    return mm, bmm


def _matmul_2d_or_batched(a, b):
    """Matrix multiply using the Triton mm/bmm kernels.

    Both ``a`` and ``b`` have their contracting dimension as the last two
    dims (``(..., M, K)`` and ``(..., K, N)``). Leading batch dimensions are
    broadcast against each other, mirroring ATen matmul.
    """
    if a.dim() == 2 and b.dim() == 2:
        mm, _ = _lookup_matmul_entries()
        return mm(a.contiguous(), b.contiguous())

    M, K = a.shape[-2], a.shape[-1]
    K2, N = b.shape[-2], b.shape[-1]
    batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])

    a_exp = a.expand(*batch_shape, M, K).reshape(-1, M, K).contiguous()
    b_exp = b.expand(*batch_shape, K2, N).reshape(-1, K2, N).contiguous()
    _, bmm = _lookup_matmul_entries()
    out = bmm(a_exp, b_exp)
    return out.reshape(*batch_shape, M, N)


def _transpose_operand(x):
    """Contiguous kernel-transposed view of the left/right GEMM operand.

    The 16-bit 2-D case goes through the backend ``t_copy`` SDNN tile-transpose
    (both GM sides stay row-contiguous 2D block transfers; measured 0.283ms vs
    62.7ms for the strided ``contiguous()`` gather on fp16/fp32 4096^2, i.e.
    the copy_slice kernel that was 89.5% of matmul_backward time at 512^3
    fp32).  fp32 keeps the generic lazy transpose view and the mm() call's
    ``.contiguous()``: under the official harness the native strided copy is
    fast (S1 fp32 512 = 0.0744ms with that path), while each extra t_copy
    launch costs ~0.02-0.03ms at small shapes and measurably regressed the
    official fp32 512/1024 cases (0.321->0.218 / 0.654->0.527, 2026-11-16
    post run); the t_copy win for fp16/bf16 is much larger than that launch
    cost, so it stays for the 16-bit dtypes only.  Batched operands keep the
    lazy transpose view; the bmm branch flattens and materialises them itself.
    """
    if x.dim() == 2 and x.dtype in (torch.float16, torch.bfloat16):
        return t_copy(x)
    return x.transpose(-2, -1)


def _sum_to(grad, shape):
    """Reduce ``grad`` (a broadcasted gradient) back to ``shape`` via summation."""
    shape = tuple(shape)
    while grad.dim() > len(shape):
        grad = sum_dim(grad, dim=[0], keepdim=False)
    for i in range(len(shape)):
        if shape[i] == 1 and grad.shape[i] != 1:
            grad = sum_dim(grad, dim=[i], keepdim=True)
    return grad


def matmul_backward(
    grad,
    self,
    other,
    mask,
):
    """Backward pass for ATen matmul (out = self @ other).

    Given ``out = matmul(self, other)`` and the upstream gradient ``grad``,
    returns ``(grad_self, grad_other)``. Each is ``None`` when the
    corresponding entry of ``mask`` is False.
    """
    logger.debug("GEMS_KUNLUNXIN MATMUL_BACKWARD")

    need_self, need_other = bool(mask[0]), bool(mask[1])

    dim_self = self.dim()
    dim_other = other.dim()

    # float16/bfloat16 lose too much precision through the chained matmuls and
    # broadcast reductions; compute in float32 and cast the results back.
    # The upcast only applies to batched / broadcast matmul shapes: for the
    # 2-D / vector cases the backend 16-bit GEMM (fp32-accumulated, fp16
    # output) matches the fp64 CPU reference to within the test tolerance
    # (atol=1e-4*K, rtol=1e-3) exactly like torch's own fp16 GEMM does, so
    # running them natively is both faster and ATen-faithful; the batched
    # reduction (sum over broadcast dims, values rounded again along batch)
    # measurably crosses the tolerance boundary, hence the fp32 upcast there
    # (measured on the failing 21-case matrix, 1116 branch).
    compute_dtype = self.dtype
    upcast = self.dtype in (torch.float16, torch.bfloat16) and (
        self.dim() > 2 or other.dim() > 2
    )
    if upcast:
        compute_dtype = torch.float32

    s = self.to(compute_dtype) if self.dtype != compute_dtype else self
    o = other.to(compute_dtype) if other.dtype != compute_dtype else other
    g = grad.to(compute_dtype) if grad.dtype != compute_dtype else grad

    # Fold 1D operands up to 2D, mirroring how ATen matmul treats vectors:
    # a 1D `self` becomes (1, K) and a 1D `other` becomes (K, 1). The folded
    # dimensions are dropped from the forward output, so they must be inserted
    # back into `grad` before computing the gradients.
    folded_self = dim_self == 1
    folded_other = dim_other == 1

    if folded_self:
        s = s.unsqueeze(-2)
    if folded_other:
        o = o.unsqueeze(-1)

    if folded_other:
        g = g.unsqueeze(-1)
    if folded_self:
        g = g.unsqueeze(-2)

    grad_self = None
    grad_other = None

    if need_self:
        # grad_self = grad @ other^T
        gs = _matmul_2d_or_batched(g, _transpose_operand(o))
        gs = _sum_to(gs, s.shape)
        if folded_self:
            gs = gs.squeeze(-2)
        grad_self = gs.to(self.dtype) if gs.dtype != self.dtype else gs

    if need_other:
        # grad_other = self^T @ grad
        go = _matmul_2d_or_batched(_transpose_operand(s), g)
        go = _sum_to(go, o.shape)
        if folded_other:
            go = go.squeeze(-1)
        grad_other = go.to(other.dtype) if go.dtype != other.dtype else go

    return grad_self, grad_other
