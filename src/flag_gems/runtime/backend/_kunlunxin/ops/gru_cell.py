# Copyright 2026, The FlagOS Contributors.
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
#
# Kunlunxin specialization of ``aten::gru_cell``.
#
# The generic implementation splits the cell into three kernels (input
# projection, hidden projection, gate combine).  Profiling on XPU3 showed the
# elementwise *combine* kernel -- not the GEMMs -- dominating the runtime
# (~90 % at the large shapes): it runs a 2-D tile with ``tl_extra_shim.tanh``
# and per-lane masks, which the backend lowers onto a scalar path (~1.98 ms for
# a (128, 512) cell against ~26 us for ATen).  The two projection kernels are
# fine but were also ~10x slower than torch because their ``tl.dot`` operands
# are fed from masked 2-D loads (``pattern-xpu3-dot-masked-load-miscompile``).
#
# This module keeps three launches but makes every one of them fast:
#
#   1. ``_gru_cell_proj_kernel`` -- rounded ``x @ w^T`` (reused for the input
#      and the hidden projection).  The weight is fed as the *transposed* B
#      operand (``w[n, k]`` contiguous over k), the batch index wraps with
#      ``% M`` and the staging-buffer rows are padded to the block, so the dot
#      is mask-free and in-bounds.  ``% 1`` miscompiles on this backend, so the
#      ``M == 1`` cell falls back to the generic path.
#   2. ``_gru_cell_gate_kernel`` (from ``_gru_cell_gate``) -- the mask-free flat
#      gate chain, i.e. the backend's ``_thnn_fused_gru_cell`` flat gate with
#      the backward workspace stores dropped.  It lives in its own module on
#      purpose: a kernel containing ``tl.sigmoid``/``tt.tanh`` fails a *cold*
#      XPU compile (``KeyError: 'tt.sigmoid'``) when defined in this module once
#      a ``tl.dot`` kernel has been built in the process, while the identical
#      kernel defined/imported from another module compiles cleanly.
#
# Shapes whose hidden size is not a multiple of 16, whose input size is not a
# multiple of 16, a single-row batch, non-contiguous operands, or an empty
# reduction fall back to the generic implementation (correct, edge-only).
#
# Rounding follows ATen: each projection rounds to the input dtype before the
# biases are added in fp32, and only the final store rounds.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._kunlunxin.ops._gru_cell_gate import (
    gru_cell_gate_kernel as _gru_cell_gate_kernel,
)

logger = logging.getLogger(__name__)

_ACC_DTYPE = {
    torch.float16: tl.float32,
    torch.bfloat16: tl.float32,
    torch.float32: tl.float32,
}


def _lowpow2(n):
    """Largest power of two dividing ``n`` (``n > 0``)."""
    return n & (-n)


@triton.jit
def _gru_cell_proj_kernel(
    x_ptr,
    w_ptr,
    c_ptr,
    M,
    N,
    s_xm,
    s_xk,
    s_wn,
    s_wk,
    s_cm,
    K: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Rounded ``x @ w^T`` into ``c`` (``c`` is ``(mp, N)``, ``N = 3*hidden``).

    Reused for both the input- and hidden-projection.  ``w`` is stored
    ``(N, K)`` row-major (contiguous in K) and loaded as the transposed B
    operand.  Batch rows wrap with ``% M`` so the scratch rows of the padded
    staging buffers are discarded rather than masked; every reduction/column
    extent is a block multiple, so no load/store is masked.  ``% 1`` miscompiles
    on this backend, so ``M == 1`` never reaches this kernel (the caller falls
    back to the generic path).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = rm % M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + ram[:, None] * s_xm + rk[None, :] * s_xk)
        w = tl.load(w_ptr + rn[None, :] * s_wn + rk[:, None] * s_wk)
        acc += tl.dot(x, w, out_dtype=ACC)
    tl.store(
        c_ptr + rm[:, None] * s_cm + rn[None, :],
        acc.to(c_ptr.dtype.element_ty),
    )


def _proj_block_k(k):
    if k % 64 == 0:
        return 64
    if k % 32 == 0:
        return 32
    return 16


def _contiguous_2d(t):
    return t.is_contiguous() and t.stride(1) == 1


def _fast_ok(input, hx, w_ih, w_hh, b_ih, b_hh, input_size, hidden_size):
    if input.dtype not in _ACC_DTYPE:
        return False
    # ``rm % M`` (used for the batch wrap) miscompiles at M == 1.
    if input.shape[0] < 2:
        return False
    if hidden_size % 16 != 0:
        return False
    if input_size % 16 != 0 or input_size == 0:
        return False
    for t in (input, hx, w_ih, w_hh):
        if not _contiguous_2d(t):
            return False
    for b in (b_ih, b_hh):
        if b is not None and not b.is_contiguous():
            return False
    return True


def gru_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    """Single-step GRU cell (``aten::gru_cell``)."""
    logger.debug("GEMS GRU_CELL")
    _check_cell_args("gru_cell", input, hx, w_ih, w_hh, b_ih, b_hh, 3)

    batch, input_size = input.shape
    hidden_size = w_hh.shape[1]
    if batch == 0 or hidden_size == 0:
        return torch.empty(
            (batch, hidden_size), dtype=input.dtype, device=input.device
        )

    if not _fast_ok(
        input, hx, w_ih, w_hh, b_ih, b_hh, input_size, hidden_size
    ):
        from flag_gems.ops.gru_cell import gru_cell as _generic

        return _generic(input, hx, w_ih, w_hh, b_ih, b_hh)

    acc = _ACC_DTYPE[input.dtype]
    low_h = _lowpow2(hidden_size)

    block_m = min(128, max(16, triton.next_power_of_2(batch)))
    block_n = min(128, low_h)
    block_k_ih = _proj_block_k(input_size)
    block_k_h = _proj_block_k(hidden_size)

    mp = ((batch + block_m - 1) // block_m) * block_m
    gate_n = 3 * hidden_size
    gi = torch.empty((mp, gate_n), dtype=input.dtype, device=input.device)
    gh = torch.empty((mp, gate_n), dtype=input.dtype, device=input.device)
    out = torch.empty((batch, hidden_size), dtype=input.dtype, device=input.device)

    # The gate kernel always adds both biases; materialize a zero vector for an
    # absent one (bias-free cells are a test-only edge case).
    bi = (
        b_ih
        if b_ih is not None
        else torch.zeros(gate_n, dtype=input.dtype, device=input.device)
    )
    bh = (
        b_hh
        if b_hh is not None
        else torch.zeros(gate_n, dtype=input.dtype, device=input.device)
    )


    grid_ih = (mp // block_m, gate_n // block_n)
    _gru_cell_proj_kernel[grid_ih](
        input,
        w_ih,
        gi,
        batch,
        gate_n,
        input.stride(0),
        input.stride(1),
        w_ih.stride(0),
        w_ih.stride(1),
        gi.stride(0),
        K=input_size,
        ACC=acc,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k_ih,
    )
    grid_h = (mp // block_m, gate_n // block_n)
    _gru_cell_proj_kernel[grid_h](
        hx,
        w_hh,
        gh,
        batch,
        gate_n,
        hx.stride(0),
        hx.stride(1),
        w_hh.stride(0),
        w_hh.stride(1),
        gh.stride(0),
        K=hidden_size,
        ACC=acc,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k_h,
    )

    # GRU gate chain.  ``_gru_cell_gate_kernel`` is the backend's mask-free flat
    # gate (with the backward workspace stores dropped); pick a block that
    # divides ``batch * hidden`` so its maskless addressing stays in bounds.
    n = batch * hidden_size
    flat_block = min(1024, _lowpow2(n))
    _gru_cell_gate_kernel[(n // flat_block,)](
        gi[:batch],
        gh[:batch],
        hx,
        bi,
        bh,
        out,
        hidden_size,
        HAS_BIAS=True,
        IS_FP64=(input.dtype == torch.float64),
        BLOCK_SIZE=flat_block,
    )
    return out


def _check_cell_args(name, input, hx, w_ih, w_hh, b_ih, b_hh, gates):
    """Reproduce the ATen shape/dtype contract for a single RNN cell."""
    if input.dim() < 2:
        raise IndexError(
            "Dimension out of range (expected to be in range of [-1, 0], but got 1)"
        )
    if input.dim() > 2 or input.shape[1] != w_ih.shape[1]:
        raise RuntimeError(
            "input has inconsistent input_size: got %d expected %d"
            % (input.shape[1], w_ih.shape[1])
        )
    if hx.shape[0] != input.shape[0]:
        raise RuntimeError(
            "Input batch size %d doesn't match hidden0 batch size %d"
            % (input.shape[0], hx.shape[0])
        )
    if hx.dim() != 2:
        raise RuntimeError("hidden0 is expected to be 2-D, got %d-D instead" % hx.dim())
    if hx.shape[1] != w_hh.shape[1]:
        raise RuntimeError(
            "hidden0 has inconsistent hidden_size: got %d, expected %d"
            % (hx.shape[1], w_hh.shape[1])
        )
    hidden_size = w_hh.shape[1]
    if w_ih.shape[0] != gates * hidden_size or w_hh.shape[0] != gates * hidden_size:
        raise RuntimeError(
            "%s: expected weight matrices with %d rows, got w_ih with %d and "
            "w_hh with %d" % (name, gates * hidden_size, w_ih.shape[0], w_hh.shape[0])
        )
    for label, bias in (("b_ih", b_ih), ("b_hh", b_hh)):
        if bias is None:
            continue
        if bias.dim() != 1 or bias.shape[0] != gates * hidden_size:
            raise RuntimeError(
                "Expected tensor for argument '%s' to have %d elements; but it "
                "actually has %d elements" % (label, gates * hidden_size, bias.numel())
            )
    for label, tensor in (
        ("hx", hx),
        ("w_ih", w_ih),
        ("w_hh", w_hh),
        ("b_ih", b_ih),
        ("b_hh", b_hh),
    ):
        if tensor is not None and tensor.dtype != input.dtype:
            raise RuntimeError(
                "expected %s and %s to have the same dtype, but got: %s != %s"
                % ("input", label, input.dtype, tensor.dtype)
            )
    if input.is_complex():
        raise RuntimeError('"%s" not implemented for complex dtypes' % name)
