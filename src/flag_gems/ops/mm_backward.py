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

"""First-order gradients of the 2-D matrix product.

``torch.mm`` has no separately registered ATen backward schema: the autograd
engine derives the two gradients from the forward as

.. code-block:: text

    out    = a @ b                        # (M, K) @ (K, N) -> (M, N)
    grad_a = grad_output @ b.T            # (M, N) @ (N, K) -> (M, K)
    grad_b = a.T @ grad_output            # (K, M) @ (M, N) -> (K, N)

so the backward is two GEMMs.  ``mm_backward`` computes exactly those two
products with FlagGems kernels, and ``MmFunction`` attaches it to the public
``mm`` entry so a direct ``flag_gems.mm`` call participates in autograd.

Two launch strategies are used, because the two regimes have opposite
bottlenecks:

* host-bound problems (small ``M*K*N``) are one fused kernel launch that
  computes both gradients, so a backward costs a single dispatch instead of
  two;
* GPU-bound problems reuse the tuned forward kernels through the public ``mm``
  entry, where the per-call dispatch cost is hidden behind the work and the
  backend-specific tile selection (TMA, Stream-K, split-K) stays in charge.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _mm_tile(
    A,
    B,
    C,
    pid,
    rows,
    cols,
    red,
    sa_r,
    sa_k,
    sb_k,
    sb_c,
    sc_r,
    sc_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32X3: tl.constexpr,
):
    """Write one BLOCK_M x BLOCK_N tile of ``C = A @ B``.

    ``A`` is described by ``(sa_r, sa_k)`` and ``B`` by ``(sb_k, sb_c)``, so a
    transposed view of either operand is contracted correctly without a copy.
    """
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(cols, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_r = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_r[:, None] * sa_r + offs_k[None, :] * sa_k
    b_ptrs = B + offs_k[:, None] * sb_k + offs_c[None, :] * sb_c
    r_mask = offs_r < rows
    c_mask = offs_c < cols

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, red, BLOCK_K):
        # offs_k is a fixed arange, so the tail mask must add the block start;
        # otherwise a reduction length that is not a multiple of BLOCK_K would
        # silently drop the last partial block.
        k_mask = (k_start + offs_k) < red
        a = tl.load(a_ptrs, mask=r_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & c_mask[None, :], other=0.0)
        if TF32X3:
            # Same policy as the forward mm kernels: IEEE-accurate FP32 on the
            # tensor cores via the three-pass TF32 decomposition.
            acc = tl.dot(a, b, acc, input_precision="tf32x3")
        else:
            acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * sa_k
        b_ptrs += BLOCK_K * sb_k

    c_ptrs = C + offs_r[:, None] * sc_r + offs_c[None, :] * sc_c
    tl.store(
        c_ptrs,
        acc.to(C.dtype.element_ty),
        mask=r_mask[:, None] & c_mask[None, :],
    )


@triton.jit
def _mm_backward_kernel(
    GRAD_OUT,
    A,
    B,
    GRAD_A,
    GRAD_B,
    M,
    N,
    K,
    go_s0,
    go_s1,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    grad_a_tiles,
    NEED_A: tl.constexpr,
    NEED_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32X3: tl.constexpr,
):
    """Both gradients of ``out = A @ B`` in one launch.

    The first ``grad_a_tiles`` programs write ``grad_a`` and the rest write
    ``grad_b``.  A gradient that was not requested has an empty tile range and
    its branch is compiled out, so the launch always computes exactly the
    requested set.
    """
    pid = tl.program_id(0)
    if NEED_A:
        if pid < grad_a_tiles:
            # grad_a = grad_output @ B.T: contract over N, B is read as (N, K).
            _mm_tile(
                GRAD_OUT,
                B,
                GRAD_A,
                pid,
                M,
                K,
                N,
                go_s0,
                go_s1,
                b_s1,
                b_s0,
                K,
                1,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                TF32X3,
            )
        else:
            if NEED_B:
                # grad_b = A.T @ grad_output: contract over M, A is read as
                # (K, M); output row k is A's second axis, hence stride N/1.
                _mm_tile(
                    A,
                    GRAD_OUT,
                    GRAD_B,
                    pid - grad_a_tiles,
                    K,
                    N,
                    M,
                    a_s1,
                    a_s0,
                    go_s0,
                    go_s1,
                    N,
                    1,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    GROUP_M,
                    TF32X3,
                )
    else:
        if NEED_B:
            _mm_tile(
                A,
                GRAD_OUT,
                GRAD_B,
                pid,
                K,
                N,
                M,
                a_s1,
                a_s0,
                go_s0,
                go_s1,
                N,
                1,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                TF32X3,
            )


# Below this many multiply-accumulates one fused launch beats two tuned
# forward launches, because the forward entry spends ~0.1 ms per call on host
# dispatcher work while a plain Triton launch costs ~0.01 ms.  The limits are
# measured on H800 with the core shapes of benchmark/test_mm.py: 384^3 and
# 1024^3 are host-bound for every dtype, 2048^3 is still host-bound for the
# 16-bit dtypes, and 4096^3 is GPU-bound for all of them.  FP32 crosses over
# earlier because the fused tile performs three TF32 passes per product, which
# only pays off while the launch itself is the bottleneck.  Both kernels are
# correct on either side of the limit, so another backend may move the
# constant without changing semantics.
_HOST_BOUND_LIMIT = {
    torch.float32: 2_000_000_000,
    torch.float16: 20_000_000_000,
    torch.bfloat16: 20_000_000_000,
}

_GROUP_M = 8


def _tile_shape(dtype, rows, cols):
    """Fused-kernel tile, from the measured per-shape sweep on H800.

    Narrow tiles for narrow problems (a 128-wide tile on a 15-row matrix is
    mostly padding), wider tiles once there is enough work to fill the GPU.
    """
    narrow = rows <= 512 and cols <= 512
    if dtype == torch.float32:
        if narrow:
            return 32, 32, 32, 4, 3
        return 128, 64, 32, 4, 3
    if narrow:
        return 64, 64, 64, 4, 3
    if rows * cols <= 4_000_000:
        return 128, 128, 64, 8, 4
    return 128, 256, 64, 8, 3


def _needs_fused_launch(dtype, m, n, k):
    limit = _HOST_BOUND_LIMIT.get(dtype, 20_000_000_000)
    return m * n * k < limit


def _fused_mm_backward(grad_output, a, b, grad_a, grad_b, need_a, need_b):
    m, k = a.shape
    n = b.shape[1]
    dtype = a.dtype
    # grad_a's tile grid is (M, K) and grad_b's is (K, N); both branches share
    # one compile-time configuration, chosen from the wider of the two grids.
    rows = max(m if need_a else 0, k if need_b else 0)
    cols = k if need_a else 0
    if need_b:
        cols = max(cols, n)
    block_m, block_n, block_k, warps, stages = _tile_shape(dtype, rows, cols)
    tiles_a = triton.cdiv(m, block_m) * triton.cdiv(k, block_n) if need_a else 0
    tiles_b = triton.cdiv(k, block_m) * triton.cdiv(n, block_n) if need_b else 0
    if tiles_a + tiles_b == 0:  # pragma: no cover - guarded by mm_backward
        return
    with torch_device_fn.device(a.device):
        _mm_backward_kernel[(tiles_a + tiles_b,)](
            grad_output,
            a,
            b,
            grad_a if need_a else grad_output,
            grad_b if need_b else grad_output,
            m,
            n,
            k,
            grad_output.stride(0),
            grad_output.stride(1),
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            tiles_a,
            NEED_A=need_a,
            NEED_B=need_b,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=_GROUP_M,
            TF32X3=dtype == torch.float32,
            num_warps=warps,
            num_stages=stages,
        )


def _tuned_mm(x, y):
    """One GEMM through the tuned public entry (no autograd node)."""
    from flag_gems import mm as gemm

    return gemm(x.detach(), y.detach())


def mm_backward(grad_output, a, b, needs_input_grad):
    """First-order gradients of ``torch.mm`` for real 2-D operands.

    ``a`` is ``(M, K)``, ``b`` is ``(K, N)`` and ``grad_output`` is ``(M, N)``.
    ``needs_input_grad`` mirrors ``ctx.needs_input_grad``: an entry that is
    ``False`` returns ``None`` instead of launching that GEMM.

    Zero-length dimensions are answered by allocation only: ``M == 0`` or
    ``K == 0`` makes ``grad_a`` empty, and a zero-length contraction makes the
    corresponding gradient an empty or all-zero tensor, exactly as ATen does.
    """
    logger.debug("GEMS MM_BACKWARD")
    if grad_output.is_complex() or a.is_complex() or b.is_complex():
        raise NotImplementedError(
            "FlagGems mm backward does not support complex inputs"
        )
    m, k = a.shape
    n = b.shape[1]
    need_a = bool(needs_input_grad[0])
    need_b = bool(needs_input_grad[1])
    if not need_a and not need_b:
        return None, None

    # A zero-length output dimension is empty and a contraction over no terms
    # is zero; both are answered by allocation, without a kernel launch.
    grad_a = None
    if need_a:
        if m != 0 and k != 0 and n == 0:
            grad_a = torch.zeros((m, k), dtype=a.dtype, device=a.device)
        else:
            grad_a = torch.empty((m, k), dtype=a.dtype, device=a.device)

    grad_b = None
    if need_b:
        if k != 0 and n != 0 and m == 0:
            grad_b = torch.zeros((k, n), dtype=b.dtype, device=b.device)
        else:
            grad_b = torch.empty((k, n), dtype=b.dtype, device=b.device)

    if m == 0 or k == 0 or n == 0:
        return grad_a, grad_b

    if _needs_fused_launch(a.dtype, m, n, k):
        _fused_mm_backward(grad_output, a, b, grad_a, grad_b, need_a, need_b)
    else:
        if need_a:
            grad_a = _tuned_mm(grad_output, b.transpose(0, 1)).to(a.dtype)
        if need_b:
            grad_b = _tuned_mm(a.transpose(0, 1), grad_output).to(b.dtype)
    return grad_a, grad_b


class MmFunction(torch.autograd.Function):
    """First-order autograd node around the generic ``mm`` forward."""

    @staticmethod
    def forward(ctx, a, b):
        from .mm import mm

        ctx.save_for_backward(a, b)
        return mm(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return mm_backward(grad_output, a, b, ctx.needs_input_grad)
