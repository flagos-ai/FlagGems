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
"""Tuned mask-free GEMM launcher on top of the vendor ``mm`` kernel.

``mm()`` uses fixed 128/256 tiles, which is right for the large square shapes
it was tuned on but leaves the small/medium matrices of ``linear_backward``
and ``linalg_multi_dot`` with only 1-4 programs: measured 4-8x slower than the
native matrix engine on exactly the shapes those two operators are
benchmarked on.

This module keeps the proven ``mm_kernel`` (in-bounds-by-construction
addressing, no load/store masks -- TritonXPU mis-lowers masks that leave an
allocation) and picks the tile from measurements on those shapes:

    M <= 64      -> 32-row tile   (a 128-row tile runs M=64 as one tile)
    M <= 128     -> 128
    M <= 2048    -> 256
    M > 2048     -> 128           (256 rows x 256 cols exceeds the tile SRAM)

    N <= 512     -> 128 cols, else 256 cols
    K <= 512 and K % 256 == 0 -> 256-deep K tile, else 128/64/32 (largest
    divisor of K).  A 256-deep K tile halves the number of dot iterations on
    the short-K shapes (K=256/512) but loses to 128 once K >= 2048.

``nn`` handles ``a @ b``; ``tn`` handles ``a.T @ b`` where ``a`` is stored
(K, M) row-major, which is the layout of ``grad_output_flattened`` in a linear
backward pass -- the vendor ``mm`` would otherwise materialise a transposed
copy of it (a full extra read+write of the gradient, 15-30us).

Both helpers always take the host-padded path, so any block choice stays
correct for shapes that are not tile-aligned: the C buffer is allocated
M/N-padded and sliced, and K is zero-padded by the vendor ``_pad_k``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .mm import (
    _pad_k,
    _restore_matmul_fast_mode,
    _set_matmul_fast_mode,
    get_higher_dtype,
    mm_kernel,
)

logger = logging.getLogger(__name__)

# K values up to this size prefer the 256-deep K tile (measured).
_SHORT_K = 512


def _pick_blocks(M, N, K):
    if M <= 64:
        bm = 32
    elif M <= 128:
        bm = 128
    elif M <= 2048:
        bm = 256
    else:
        bm = 128
    if N >= 128:
        bn = 128 if N <= 512 else 256
    else:
        # Narrow outputs are common in a multi_dot / linear-backward chain.  A
        # 128-wide tile wastes half of it and, worse, makes the result padded,
        # so the caller pays a strided->contiguous copy (~13us on this backend)
        # to hand a plain matrix to the next GEMM.
        bn = 64 if N >= 64 else 32
    if K <= _SHORT_K and K % 256 == 0:
        bk = 256
    elif K % 128 == 0:
        bk = 128
    elif K % 64 == 0:
        bk = 64
    else:
        bk = 32
    nw = 4 if bm <= 64 else 8
    return bm, bn, bk, nw


@triton.jit
def _gemm_tn_kernel(
    at,  # (K, M) row-major source; result is at.T @ b
    b,
    c,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gsize = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsize)
    pid_n = (pid % width) // gsize

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + rk
        # (BK, BM) and (BK, BN): both GM reads stay contiguous along the
        # row-major axis of the source tensors, so nothing is transposed in
        # memory -- only in registers for the dot.
        av = tl.load(at + kk[:, None] * stride_am + rm[None, :] * stride_ak)
        bv = tl.load(b + kk[:, None] * stride_bk + rn[None, :] * stride_bn)
        acc += tl.dot(tl.trans(av), bv, out_dtype=dot_out_dtype, allow_tf32=False)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        c + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc.to(c.dtype.element_ty),
    )


def _empty_out(M, N, bm, bn, dtype, device):
    mp = triton.cdiv(M, bm) * bm
    np_ = triton.cdiv(N, bn) * bn
    return torch.empty((mp, np_), device=device, dtype=dtype), mp, np_


# ---------------------------------------------------------------------------
# Direct compiled-kernel launch
#
# The Triton Python entry point (``libentry`` + ``JITFunction.run``) rebuilds a
# cache key out of every argument on each call and resolves the specialization
# through the ``Heuristics`` wrapper.  Measured on XPU that costs ~30us per
# launch -- twice the wall time of ``torch.mm`` -- and both operators built on
# this module issue 1-5 GEMMs per call, so it dominates their latency.
#
# After the first launch for a shape we therefore remember the compiled kernel
# and call its C launcher directly (``CompiledKernel.__getitem__``), which is
# ~10us and skips all of the Python bookkeeping.  ``_NN_PLANS``/``_TN_PLANS``
# hold everything a repeat call needs -- the bound runner, the output dtype and
# padded extent, and the constant runtime arguments -- so the common path is a
# dict lookup, one ``torch.empty`` and one call.  ``None`` marks a key that only
# has the host-padded (non-direct) path, which permanently uses the regular
# entry point (still correct, just not faster).
#
# A plan is only installed for the mask-free fast case: K already tile-aligned
# and both operands row-major (the ``_pad_k`` path allocates fresh buffers
# whose strides would not match a later call), and, when the caller supplies
# ``out``, M/N tile-aligned too (the kernel stores whole tiles).
# ---------------------------------------------------------------------------
_NN_PLANS = {}
_TN_PLANS = {}
_PLAN_MISS = object()


def _launch_entry(wrapper):
    """The outermost Triton runner under ``libentry``.

    ``mm_kernel`` is ``LibEntry(Heuristics(JITFunction))``; calling ``run`` on
    the ``Heuristics`` layer resolves GROUP_M/SPLIT_K/EVEN_K exactly as a real
    launch would, which the bare ``JITFunction`` cannot do on its own.
    """
    fn = wrapper
    while not isinstance(fn, (triton.runtime.JITFunction, triton.runtime.Heuristics)):
        fn = fn.fn
    return fn


def _install_plan(plans, key, wrapper, grid, c_dtype, mp, np_, args, kwargs):
    """Resolve the compiled kernel for ``key`` and remember its launcher.

    ``run(..., warmup=True)`` computes the very same cache key ``regular()``
    would (real arguments, same ``grid``/constexprs) but returns the
    ``CompiledKernel`` instead of launching it.  Diffing the device cache
    instead would miss whenever an equivalent specialization was already
    compiled -- e.g. the ``out=`` and allocating variants of one GEMM share a
    kernel, and then the second operator's first call finds no new entry and
    would have to fall back to the slow path for good.
    """
    ck = _launch_entry(wrapper).run(*args, grid=grid, warmup=True, **kwargs)
    if hasattr(ck, "result"):
        ck = ck.result()
    plans[key] = (ck[(grid[0], 1, 1)], c_dtype, mp, np_, args[3:])


def _run(runner, a, b, c, tail, device):
    # The launcher resolves the stream of the *current* device, so the guard is
    # only needed when it differs from the tensors' device (~2.5us otherwise).
    if torch_device_fn.current_device() != device.index:
        with torch_device_fn.device(device):
            runner(a, b, c, *tail)
    else:
        runner(a, b, c, *tail)


def nn(a, b, out=None):
    """C = a @ b with the tuned tiles (mask-free, host-padded)."""
    M, K = a.shape
    N = b.shape[1]
    aligned_a = a.data_ptr() % 16 == 0
    aligned_b = b.data_ptr() % 16 == 0
    if out is None:
        key = (
            a.device.index,
            a.dtype,
            b.dtype,
            M,
            N,
            K,
            a.stride(),
            b.stride(),
            aligned_a,
            aligned_b,
        )
    else:
        key = (
            a.device.index,
            a.dtype,
            b.dtype,
            M,
            N,
            K,
            a.stride(),
            b.stride(),
            aligned_a,
            aligned_b,
            out.dtype,
            out.stride(),
            out.data_ptr() % 16 == 0,
        )

    plan = _NN_PLANS.get(key, _PLAN_MISS)
    if plan is _PLAN_MISS or plan is None:
        return _nn_general(a, b, out, key, M, N, K, plan is _PLAN_MISS)

    runner, c_dtype, mp, np_, tail = plan
    fresh = out is None
    if fresh:
        c = torch.empty((mp, np_), device=a.device, dtype=c_dtype)
    else:
        c = out
    _run(runner, a, b, c, tail, a.device)
    if not fresh:
        return out
    if mp == M and np_ == N:
        return c
    return c[:M, :N].contiguous()


def _nn_general(a, b, out, key, M, N, K, resolve):
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    bm, bn, bk, nw = _pick_blocks(M, N, K)

    if out is not None and (M % bm or N % bn):
        # The kernel stores full tiles; keep a caller-supplied ``out`` intact
        # by computing into a padded buffer and copying the exact window.
        tmp = nn(a, b)
        out.copy_(tmp)
        return out

    if out is None:
        c, mp, np_ = _empty_out(M, N, bm, bn, c_dtype, a.device)
    else:
        c = out
        mp, np_ = M, N

    direct = K % bk == 0 and a.stride() == (K, 1) and b.stride() == (N, 1)
    with torch_device_fn.device(a.device):
        if direct:
            # Common case: nothing to pad, skip ``_pad_k``'s per-call checks.
            ap, bp, kp = a, b, K
        else:
            ap, bp, kp = _pad_k(a, b, M, K, N, bk, a.device)
        grid = (triton.cdiv(mp, bm) * triton.cdiv(np_, bn), 1)
        args = (
            ap,
            bp,
            c,
            M,
            N,
            kp,
            ap.stride(0),
            ap.stride(1),
            bp.stride(0),
            bp.stride(1),
            c.stride(0),
            c.stride(1),
        )

        launch_kwargs = dict(
            dot_out_dtype=tl.float32,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=nw,
        )

        def regular():
            mm_kernel[grid](*args, **launch_kwargs)

        # The fast-mode flag is read by the lowering of the bf16 large-K path,
        # so it must be set on every launch, including the direct one.
        saved = _set_matmul_fast_mode(a.dtype, M, N, K)
        try:
            if resolve and direct:
                _install_plan(
                    _NN_PLANS,
                    key,
                    mm_kernel,
                    grid,
                    c_dtype,
                    mp,
                    np_,
                    args,
                    launch_kwargs,
                )
            elif resolve:
                # Padded path: no direct runner to reuse, but remember that so
                # later calls skip the plan lookup.
                _NN_PLANS[key] = None
            regular()
        finally:
            _restore_matmul_fast_mode(saved)

    if out is not None:
        return out
    if mp == M and np_ == N:
        return c
    return c[:M, :N].contiguous()


def tn(at, b, out=None):
    """C = at.T @ b where ``at`` is a (K, M) row-major tensor.

    Returns None when the shape is not tile-aligned along K, M or N (the
    caller then falls back to ``nn`` with a contiguous transpose).
    """
    K, M = at.shape
    N = b.shape[1]
    if (at.stride(0), at.stride(1)) != (M, 1):
        return None
    if (b.stride(0), b.stride(1)) != (N, 1):
        return None
    c_dtype = get_higher_dtype(at.dtype, b.dtype)
    bm, bn, bk, nw = _pick_blocks(M, N, K)
    if K % bk or M % bm or N % bn:
        return None
    if out is not None and out.dtype != c_dtype:
        return None

    key = (
        at.device.index,
        at.dtype,
        b.dtype,
        M,
        N,
        K,
        at.data_ptr() % 16 == 0,
        b.data_ptr() % 16 == 0,
        None if out is None else (out.stride(), out.data_ptr() % 16 == 0),
    )
    plan = _TN_PLANS.get(key, _PLAN_MISS)
    if plan is _PLAN_MISS or plan is None:
        return _tn_general(
            at, b, out, key, M, N, K, c_dtype, bm, bn, bk, nw, plan is _PLAN_MISS
        )

    runner, plan_dtype, _, _, tail = plan
    c = (
        out
        if out is not None
        else torch.empty((M, N), device=at.device, dtype=plan_dtype)
    )
    _run(runner, at, b, c, tail, at.device)
    return c


def _tn_general(at, b, out, key, M, N, K, c_dtype, bm, bn, bk, nw, resolve):
    c = out if out is not None else torch.empty((M, N), device=at.device, dtype=c_dtype)
    with torch_device_fn.device(at.device):
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        args = (
            at,
            b,
            c,
            M,
            N,
            K,
            at.stride(0),
            at.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        )

        launch_kwargs = dict(
            dot_out_dtype=tl.float32,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_M=8,
            num_warps=nw,
        )

        def regular():
            _gemm_tn_kernel[grid](*args, **launch_kwargs)

        saved = _set_matmul_fast_mode(at.dtype, M, N, K)
        try:
            if resolve:
                _install_plan(
                    _TN_PLANS,
                    key,
                    _gemm_tn_kernel,
                    grid,
                    c_dtype,
                    M,
                    N,
                    args,
                    launch_kwargs,
                )
            regular()
        finally:
            _restore_matmul_fast_mode(saved)
    return c
