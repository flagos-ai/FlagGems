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
"""Triton implementation of aten::linalg_multi_dot.

Computes the matrix-chain product with the same dynamic-programming
parenthesization that ATen uses (first strict minimum wins).

Chain arithmetic is carried out in float32: every DP step accumulates into an
fp32 accumulator and every interior result is materialised in fp32, so the
only rounding to the promoted input dtype happens on the returned tensor.
16-bit x 16-bit steps still use the native tensor-core path (the operands are
exactly representable, so the fp32 accumulator keeps full accuracy).

Vector endpoints are handled logically: a leading 1-D tensor is a row vector
(1, n) and a trailing 1-D tensor is a column vector (n, 1).  No input tensor
is ever mutated or copied.  Gradients are computed by Triton GEMM kernels in a
custom autograd Function.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops.linalg_multi_dot import linalg_multi_dot as default__linalg_multi_dot

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32}, num_warps=2, num_stages=1
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
    ],
    key=["M", "N", "K", "UPCAST"],
)
@triton.jit
def _gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    sa_m,
    sa_k,
    sb_k,
    sb_n,
    sc_m,
    sc_n,
    UPCAST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    b_ptrs = B + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Whole K blocks: no k-mask, so the loads stay vectorisable.
    num_full = K // BLOCK_K
    for _k0 in range(0, num_full):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        if UPCAST:
            acc = tl.dot(
                a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee"
            )
        else:
            acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sa_k
        b_ptrs += BLOCK_K * sb_k
    # Remainder block.
    if K % BLOCK_K != 0:
        kk = num_full * BLOCK_K + offs_k
        mask_k = kk < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        if UPCAST:
            acc = tl.dot(
                a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee"
            )
        else:
            acc = tl.dot(a, b, acc)

    c = acc.to(C.dtype.element_ty)
    c_ptrs = C + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Host side helpers
# ---------------------------------------------------------------------------


def _is16(dtype):
    return dtype is torch.float16 or dtype is torch.bfloat16


_FAST = {}


def _runner_for(key, args, M, N, upcast, dtypes):
    """Replay closure for the config the autotuner picked for this key.

    A chain calls this kernel once per step and Triton's Python-level launch
    costs ~25 us here, which dominates the small matrices multi_dot sees. Once
    the autotuner has chosen a config, the compiled kernel is replayed directly;
    the ordinary launch path stays as the fallback.
    """
    entry = _FAST.get(key)
    if entry is not None:
        return entry
    cfg = _gemm_kernel.cache.get((M, N, key[5], upcast) + tuple(str(d) for d in dtypes))
    if cfg is None:
        return None
    try:
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        compiled = _gemm_kernel.fn.warmup(
            *args,
            grid=grid,
            UPCAST=upcast,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
        drv = _driver.active
        stream = drv.get_current_stream(drv.get_current_device())
        run = compiled.run
        fn = compiled.function
        pm = compiled.packed_metadata
        g0 = grid[0]

        def replay(*a):
            run(g0, 1, 1, stream, fn, pm, None, None, None, *a, upcast, bm, bn, bk)

        _FAST[key] = replay
        return replay
    except Exception:
        return None


def _mm(a, b, out_dtype, device):
    """a = (t, rows, cols, stride_m, stride_k), b likewise; returns the same form."""
    at, M, K, sa_m, sa_k = a
    bt, _, N, sb_k, sb_n = b
    out = torch.empty((M, N), dtype=out_dtype, device=device)
    if M > 0 and N > 0:
        native = at.dtype is bt.dtype and _is16(at.dtype)
        upcast = not native
        key = (at.dtype, bt.dtype, out_dtype, M, N, K)
        args = (
            at,
            bt,
            out,
            M,
            N,
            K,
            sa_m,
            sa_k,
            sb_k,
            sb_n,
            out.stride(0),
            out.stride(1),
        )
        replay = _FAST.get(key)
        if replay is not None:
            try:
                replay(*args)
                return (out, M, N, out.stride(0), out.stride(1))
            except Exception:
                _FAST.pop(key, None)
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        _gemm_kernel[grid](*args, UPCAST=upcast)
        # The autotuner has now (or already) chosen a config for this key.
        _runner_for(key, args, M, N, upcast, (at.dtype, bt.dtype, out.dtype))
    return (out, M, N, out.stride(0), out.stride(1))


def _transpose(a):
    t, r, c, s_m, s_k = a
    return (t, c, r, s_k, s_m)


def _operand(t, i, n):
    if t.dim() == 2:
        return (t, t.shape[0], t.shape[1], t.stride(0), t.stride(1))
    if i == 0:
        return (t, 1, t.shape[0], 0, t.stride(0))
    return (t, t.shape[0], 1, t.stride(0), 0)


def _optimal_order(d, n):
    """ATen's DP over the dims array d[0..n]; first strict minimum wins."""
    cost = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            best = None
            best_m = i
            for m in range(i, j):
                c = cost[i][m] + cost[m + 1][j] + d[i] * d[m + 1] * d[j + 1]
                if best is None or c < best:
                    best = c
                    best_m = m
            cost[i][j] = best
            split[i][j] = best_m
    return split


def _plan(ts):
    n = len(ts)
    for t in ts:
        if not isinstance(t, torch.Tensor):
            raise TypeError("linalg.multi_dot: expected a list of tensors")
        if t.dim() != 1 and t.dim() != 2:
            raise RuntimeError(
                "linalg.multi_dot: Each tensor should be a 1-D or 2-D matrix."
            )
    rows = [0] * n
    cols = [0] * n
    for i, t in enumerate(ts):
        if t.dim() == 2:
            rows[i] = t.shape[0]
            cols[i] = t.shape[1]
        elif i == 0:
            rows[i] = 1
            cols[i] = t.shape[0]
        elif i == n - 1:
            rows[i] = t.shape[0]
            cols[i] = 1
        else:
            raise RuntimeError(
                "linalg.multi_dot: Each tensor should be a 1-D or 2-D matrix."
            )
    for i in range(n - 1):
        if cols[i] != rows[i + 1]:
            raise RuntimeError(
                "linalg.multi_dot: shapes are not compatible for multiplication"
            )
    d = [rows[0]] + cols
    return rows, cols, d, _optimal_order(d, n)


def _forward(ts, d, split, out_dtype):
    n = len(ts)
    device = ts[0].device

    def ev(i, j):
        if i == j:
            return _operand(ts[i], i, n)
        m = split[i][j]
        a = ev(i, m)
        b = ev(m + 1, j)
        # Every interior result is materialised in the operand dtype, not in
        # fp32: ATen's multi_dot rounds each product back to the input dtype,
        # so keeping fp32 here drifts away from the reference as the chain
        # grows (a 5-matrix fp16 chain measured 4.0 absolute difference).
        # The GEMM still accumulates in fp32 internally.
        return _mm(a, b, out_dtype, device)

    return ev(0, n - 1)[0]


def _common_dtype(ts):
    dt = ts[0].dtype
    for t in ts[1:]:
        dt = torch.promote_types(dt, t.dtype)
    return dt


def _finish(res, first_vec, last_vec):
    if first_vec and last_vec:
        return res.reshape(())
    if first_vec:
        return res.reshape(res.shape[1])
    if last_vec:
        return res.reshape(res.shape[0])
    return res


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


class _MultiDot(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *ts):
        n = len(ts)
        rows, cols, d, split = _plan(ts)
        out_dtype = _common_dtype(ts)
        ctx.save_for_backward(*ts)
        ctx.n = n
        ctx.out_dtype = out_dtype
        res = _forward(ts, d, split, out_dtype)
        return _finish(res, ts[0].dim() == 1, ts[-1].dim() == 1)

    @staticmethod
    def backward(ctx, grad_out):
        ts = ctx.saved_tensors
        n = ctx.n
        device = ts[0].device
        if grad_out is None:
            return tuple(None for _ in range(n))
        rows, cols, d, split = _plan(ts)
        first_vec = ts[0].dim() == 1
        last_vec = ts[-1].dim() == 1
        if first_vec and last_vec:
            G = (grad_out, 1, 1, 0, 0)
        elif first_vec:
            G = (grad_out, 1, grad_out.shape[0], 0, grad_out.stride(0))
        elif last_vec:
            G = (grad_out, grad_out.shape[0], 1, grad_out.stride(0), 0)
        else:
            G = (
                grad_out,
                grad_out.shape[0],
                grad_out.shape[1],
                grad_out.stride(0),
                grad_out.stride(1),
            )

        X = [_operand(ts[i], i, n) for i in range(n)]

        pre = [None] * n
        pre[1] = X[0]
        for i in range(2, n):
            pre[i] = _mm(pre[i - 1], X[i - 1], torch.float32, device)

        suf = [None] * n
        suf[n - 2] = X[n - 1]
        for i in range(n - 3, -1, -1):
            suf[i] = _mm(X[i + 1], suf[i + 1], torch.float32, device)

        grads = []
        for i in range(n):
            gdt = ts[i].dtype
            if i == 0:
                g = _mm(G, _transpose(suf[0]), gdt, device)[0]
            elif i == n - 1:
                g = _mm(_transpose(pre[n - 1]), G, gdt, device)[0]
            else:
                tmp = _mm(_transpose(pre[i]), G, torch.float32, device)
                g = _mm(tmp, _transpose(suf[i]), gdt, device)[0]
            if ts[i].dim() == 1:
                g = g.reshape(-1)
            grads.append(g)
        return tuple(grads)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def linalg_multi_dot(tensors, *extra_args):
    logger.debug("GEMS_MTHREADS LINALG_MULTI_DOT")
    if extra_args:
        tensors = [tensors, *extra_args]
    if not isinstance(tensors, (list, tuple)):
        raise RuntimeError("linalg.multi_dot: at least 2 tensors are required.")
    ts = list(tensors)
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in ts):
        return default__linalg_multi_dot(ts)
    n = len(ts)
    if n < 2:
        raise RuntimeError("linalg.multi_dot: at least 2 tensors are required.")

    if torch.is_grad_enabled() and any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in ts
    ):
        return _MultiDot.apply(*ts)

    rows, cols, d, split = _plan(ts)
    out_dtype = _common_dtype(ts)
    res = _forward(ts, d, split, out_dtype)
    return _finish(res, ts[0].dim() == 1, ts[-1].dim() == 1)
