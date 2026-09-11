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

import collections
import logging
import os
import struct
import threading
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_FUSED_JACOBI_MAX_K = 64
_FUSED_JACOBI_MAX_K_FP64 = 32
_FUSED_JACOBI_MAX_ROWS = 256
# Wider fused tiles (k past the narrow cap) hold both rotation tiles in
# registers at once, so they are only used when the row tile stays small
# enough to avoid register spills.
_FUSED_JACOBI_WIDE_MAX_ROWS = 128
# Hermitian matrices at or above this order use the non-iterative
# tridiagonalization + Sturm-count path instead of Jacobi sweeps. The fp32
# threshold is one higher: at k == 32 the fused fp32 Jacobi kernel is still
# faster than the tridiagonalization path.
_HERM_TRIDIAG_MIN_K_FP64 = 32
_HERM_TRIDIAG_MIN_K_FP32 = 33


def _native_fp64_supported():
    # Unified capability bit (runtime DeviceDetector): False on devices with
    # no usable native FP64. There the float32 Sturm count must run as the
    # pure-FP32 double-single fallback below (no tl.float64 anywhere), and
    # float64 input is rejected at the entry point. Read per call so tests
    # can force the fallback; getattr keeps unknown runtimes on the old path.
    return getattr(runtime_device, "support_fp64", True)


# One-time verdict per device for _blocked_tridiag_ok (see below).  The
# lock makes concurrent first calls run the probe exactly once.
_BLOCKED_TRIDIAG_OK = {}
_BLOCKED_TRIDIAG_LOCK = threading.Lock()


def _blocked_tridiag_probe(device):
    # Run the real blocked path once per device on a dense rank-100 input.
    # Some backends silently miscompile this pipeline even when a small
    # tl.dot probe passes, so any failure selects the slower unblocked path.
    try:
        k = _HERM_TRIDIAG_BLOCKED_MIN_K
        generator = torch.Generator().manual_seed(0)
        # Rank-100 by construction: F @ F.mT / k with F in R^{k x 100} has
        # exactly 100 nonzero eigenvalues; the smallest sits near the
        # Wishart lower edge (~0.4), far above the atol=5e-2 threshold,
        # while the zero eigenspace only sees fp32 rounding noise (~1e-5).
        # Margins are wide on both sides: a working blocked path returns
        # exactly 100.  A single fp64 GEMM (not a full k x k QR) keeps the
        # one-time cold-call cost small.
        factor = torch.randn(k, 100, generator=generator, dtype=torch.float64)
        a = ((factor @ factor.mT) / k).float().to(device)
        tol_dtype = torch.float64 if _native_fp64_supported() else torch.float32
        atol_t = torch.full((1,), 5e-2, dtype=tol_dtype).to(device)
        rtol_t = torch.zeros((1,), dtype=tol_dtype).to(device)
        s = float(a.abs().max().item())
        scale = torch.tensor([s if s > 0 else 1.0], dtype=torch.float32).to(device)
        ws = _herm_tridiag_workspace(device, 1, k, torch.float32, atol_t, rtol_t, True)
        ws["staging"].copy_(a.reshape(1, k, k))
        ws["atol"].copy_(atol_t)
        ws["rtol"].copy_(rtol_t)
        ws["scale"].copy_(scale)
        _herm_tridiag_blocked_run(ws, k, 1, not _native_fp64_supported())
        verdict = bool(ws["rank"][0].item() == 100)
    except Exception:
        verdict = False
    if not verdict:
        logger.warning(
            "matrix_rank: blocked tridiagonalization self-test failed "
            "on %s; using the unblocked path (slower but correct)",
            device,
        )
    return verdict


def _blocked_tridiag_ok(device):
    # Double-checked locking: the probe ends in a synchronizing .item(), so
    # concurrent first calls must not repeat it.
    key = (device.type, device.index)
    verdict = _BLOCKED_TRIDIAG_OK.get(key)
    if verdict is None:
        with _BLOCKED_TRIDIAG_LOCK:
            verdict = _BLOCKED_TRIDIAG_OK.get(key)
            if verdict is None:
                verdict = _blocked_tridiag_probe(device)
                _BLOCKED_TRIDIAG_OK[key] = verdict
    return verdict


# ---------------------------------------------------------------------------
# Optional per-shape graph capture
# ---------------------------------------------------------------------------
# The barrier-free decompositions issue O(k) kernel launches per call, which
# makes host enqueue cost dominate on fast devices.  Where torch exposes
# CUDA-graph capture (device.type == "cuda") the whole launch sequence is
# captured once per (path, shape, dtype, device, stream) and replayed
# afterwards.  This is purely a performance optimization:
# FLAGGEMS_MR_NO_GRAPH=1 or any capture failure falls back to direct
# launches with identical results, and devices without graph support never
# enter this path (no vendor dispatch).
_MR_GRAPHS = collections.OrderedDict()
# Cache budget in WORKSPACE BYTES, not entries: a (1024,1024) fp32 blocked
# graph carries ~10 MB of workspace (padded work matrix + staging + panels),
# so a fixed entry count can pin hundreds of MB across shape/dtype/batch
# combinations.  Least-recently-used eviction by cumulative bytes.
_MR_GRAPH_MAX_BYTES = 512 * 1024 * 1024
_MR_GRAPH_BYTES = 0
_MR_GRAPH_LOCK = threading.Lock()


def _mr_workspace_bytes(ws):
    return sum(
        t.numel() * t.element_size() for t in ws.values() if isinstance(t, torch.Tensor)
    )


def _mr_current_stream_handle(device):
    try:
        return torch.cuda.current_stream(device).cuda_stream
    except Exception:
        return None


def _mr_graph_cached(key, device, make_workspace, copy_in, run, copy_out):
    # key: hashable per-(path, shape, dtype, device) tuple; the current
    # stream is appended here (a capture is only valid on its own stream).
    # make_workspace() allocates the persistent buffers, copy_in(ws) stages
    # the live inputs, run(ws) is the pure launch sequence, copy_out(ws)
    # publishes the result.
    # Graph capture is only a performance optimization. Any capture failure
    # falls back to direct launches, and FLAGGEMS_MR_NO_GRAPH=1 disables it.
    if device.type != "cuda" or os.environ.get("FLAGGEMS_MR_NO_GRAPH") == "1":
        ws = make_workspace()
        copy_in(ws)
        run(ws)
        copy_out(ws)
        return
    full_key = (key, _mr_current_stream_handle(device))
    global _MR_GRAPH_BYTES
    with _MR_GRAPH_LOCK:
        ent = _MR_GRAPHS.pop(full_key, None)
        if ent is not None:
            # LRU: reinsert at the most-recently-used end on every hit.
            _MR_GRAPHS[full_key] = ent
            ws, graph, _ = ent
            copy_in(ws)
            graph.replay()
            copy_out(ws)
            return
        ws = make_workspace()
        # A real run first: it stages the inputs, compiles every kernel
        # in the sequence, and leaves a valid result in the workspace
        # (used directly when capture below fails).
        copy_in(ws)
        run(ws)
        try:
            stream = torch.cuda.Stream(device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                run(ws)  # warmup on a side stream
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(ws)
        except Exception:
            logger.warning(
                "matrix_rank graph capture failed for %s; "
                "falling back to direct launches",
                key,
            )
            copy_out(ws)
            return
        nbytes = _mr_workspace_bytes(ws)
        while _MR_GRAPHS and _MR_GRAPH_BYTES + nbytes > _MR_GRAPH_MAX_BYTES:
            # An evicted graph may still have an in-flight replay (replay
            # is async); dropping the entry frees its workspace.  The LRU
            # victim may live on ANOTHER device, so synchronize the
            # victim's device, not the current one.
            _, (victim_ws, _, victim_bytes) = next(iter(_MR_GRAPHS.items()))
            torch.cuda.synchronize(victim_ws["device_index"])
            _MR_GRAPHS.popitem(last=False)
            _MR_GRAPH_BYTES -= victim_bytes
        _MR_GRAPHS[full_key] = (ws, graph, nbytes)
        _MR_GRAPH_BYTES += nbytes
        copy_out(ws)
        return


def _jacobi_sweeps(k, is_fp64):
    # These are worst-case caps; the kernel may stop earlier when the residual
    # cannot change the rank.
    if is_fp64:
        if k <= 16:
            return 12
        if k <= 32:
            return 16
        if k <= 256:
            return 18
        return 24
    if k <= 16:
        return 8
    if k <= 32:
        return 12
    if k <= 256:
        return 14
    return 18


@libentry()
@triton.jit
def _matrix_rank_zero_kernel(out, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out + offsets, 0, mask=offsets < N)


@libentry()
@triton.jit
def _matrix_rank_safe_scale_kernel(scale):
    # Per-batch scale fixup: zero scale (all-zero matrix) becomes 1 so the
    # in-kernel 1/scale stays finite and zero matrices scale to exact zeros.
    # This is a dedicated kernel instead of torch.clamp_min because the
    # generic clamp op internally casts through fp32: an fp64 floor would
    # flush to zero there, and under use_gems() the dispatch would diverge
    # from the direct-call path (observed: zero fp64 matrix -> NaN ->
    # Sturm sign tests all false -> rank 2k).
    pid = tl.program_id(0)
    value = tl.load(scale + pid)
    tl.store(scale + pid, tl.where(value > 0.0, value, 1.0))


@libentry()
@triton.jit
def _matrix_rank_scale_tol_kernel(ATOL, SCALE, OUT):
    # atol_s = atol / scale inside the graph-captured launch sequence, at
    # the tolerance's own precision (scale is cast UP to atol's dtype, so
    # an fp64 tolerance keeps its fp64 quotient -- the same semantics the
    # pre-scaling torch.div had).  A dedicated kernel instead of torch.div
    # because under use_gems() the generic div would re-dispatch, and an
    # out= variant is not guaranteed everywhere; this is identical on the
    # direct and dispatched paths and capturable in a CUDA graph.
    pid = tl.program_id(0)
    a = tl.load(ATOL + pid)
    s = tl.load(SCALE + pid).to(a.dtype)
    tl.store(OUT + pid, a / s)


@libentry()
@triton.jit
def _matrix_rank_rank1_kernel(
    A,
    ATOL,
    RTOL,
    SCALE,
    OUT,
    ATOL_S: tl.float64,
    RTOL_S: tl.float64,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SCALAR_TOL: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N

    if HERMITIAN:
        values = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
    elif TALL:
        values = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
    else:
        values = tl.load(a_base + rows, mask=row_mask, other=0.0)

    # In-kernel scale normalization: the launcher hands over the per-batch
    # max-abs so small paths pay no extra elementwise kernel.  Tolerances
    # arrive RAW and are scaled here the same way.
    inv_s = 1.0 / tl.load(SCALE + batch)
    values = values * inv_s
    singular_value = tl.sqrt(tl.sum(values * values, axis=0))
    if SCALAR_TOL:
        atol = ATOL_S * inv_s
        rtol = RTOL_S
    else:
        atol = tl.load(ATOL + batch) * inv_s
        rtol = tl.load(RTOL + batch)
    threshold = tl.maximum(atol, rtol * singular_value)
    rank = (singular_value > threshold).to(tl.int64)
    tl.store(OUT + batch, rank)


@libentry()
@triton.jit
def _matrix_rank_rank2_kernel(
    A,
    ATOL,
    RTOL,
    SCALE,
    OUT,
    ATOL_S: tl.float64,
    RTOL_S: tl.float64,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    REL_EPS: tl.constexpr,
    ABS_EPS: tl.constexpr,
    SCALAR_TOL: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N

    if HERMITIAN:
        x = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
        lower_rows = tl.maximum(rows, 1)
        lower_columns = tl.minimum(rows, 1)
        y = tl.load(
            a_base + lower_rows * N + lower_columns,
            mask=row_mask,
            other=0.0,
        )
    elif TALL:
        x = tl.load(a_base + rows * N, mask=row_mask, other=0.0)
        y = tl.load(a_base + rows * N + 1, mask=row_mask, other=0.0)
    else:
        x = tl.load(a_base + rows, mask=row_mask, other=0.0)
        y = tl.load(a_base + N + rows, mask=row_mask, other=0.0)

    # Same in-kernel scale normalization as the rank1 kernel.
    inv_s = 1.0 / tl.load(SCALE + batch)
    x = x * inv_s
    y = y * inv_s

    alpha = tl.sum(x * x, axis=0)
    beta = tl.sum(y * y, axis=0)
    gamma = tl.sum(x * y, axis=0)
    active = tl.abs(gamma) > REL_EPS * tl.sqrt(alpha * beta + ABS_EPS)
    safe_gamma = tl.where(active, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * safe_gamma)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c = 1.0 / tl.sqrt(1.0 + t * t)
    s = t * c
    c = tl.where(active, c, 1.0)
    s = tl.where(active, s, 0.0)

    rotated_x = c * x - s * y
    rotated_y = s * x + c * y
    singular_x = tl.sqrt(tl.sum(rotated_x * rotated_x, axis=0))
    singular_y = tl.sqrt(tl.sum(rotated_y * rotated_y, axis=0))
    max_value = tl.maximum(singular_x, singular_y)

    if SCALAR_TOL:
        atol = ATOL_S * inv_s
        rtol = RTOL_S
    else:
        atol = tl.load(ATOL + batch) * inv_s
        rtol = tl.load(RTOL + batch)
    threshold = tl.maximum(atol, rtol * max_value)
    rank = (singular_x > threshold).to(tl.int32)
    rank += (singular_y > threshold).to(tl.int32)
    tl.store(OUT + batch, rank.to(tl.int64))


@libentry()
@triton.jit
def _matrix_rank_fused_jacobi_kernel(
    A,
    A_WORK,
    ATOL,
    RTOL,
    SCALE,
    OUT,
    ATOL_S: tl.float64,
    RTOL_S: tl.float64,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    HERMITIAN: tl.constexpr,
    IS_FP64: tl.constexpr,
    ROUND: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SWEEPS,
    REL_EPS: tl.constexpr,
    ABS_EPS: tl.constexpr,
    SCALAR_TOL: tl.constexpr,
):
    # One program owns one matrix and runs the whole one-sided cyclic Jacobi
    # iteration. All pairs of a round-robin step are disjoint, so they are
    # processed together as a (BLOCK_P, BLOCK_C) tile instead of one kernel
    # launch (or one scalar loop iteration) per pair.
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N
    work_base = A_WORK + batch * K * ROWS

    column = 0
    inv_s = 1.0 / tl.load(SCALE + batch)
    while column < K:
        if HERMITIAN:
            source_rows = tl.maximum(rows, column)
            source_columns = tl.minimum(rows, column)
            values = tl.load(
                a_base + source_rows * N + source_columns,
                mask=row_mask,
                other=0.0,
            )
        elif TALL:
            values = tl.load(
                a_base + rows * N + column,
                mask=row_mask,
                other=0.0,
            )
        else:
            values = tl.load(
                a_base + column * N + rows,
                mask=row_mask,
                other=0.0,
            )
        # In-kernel scale normalization (same convention as rank1/rank2).
        tl.store(work_base + column * ROWS + rows, values * inv_s, mask=row_mask)
        column += 1
    # The init stores are distributed across all warps of the block; the
    # sweep loop below reads the whole tile with a potentially different
    # lane mapping, so a block barrier is required for visibility.
    tl.debug_barrier()

    pair = tl.arange(0, BLOCK_P)
    ring: tl.constexpr = ROUND - 1
    accumulator_dtype = tl.float64 if IS_FP64 else tl.float32
    singular_indices = tl.arange(0, BLOCK_K)
    if SCALAR_TOL:
        atol = ATOL_S * inv_s
        rtol = RTOL_S
    else:
        atol = tl.load(ATOL + batch) * inv_s
        rtol = tl.load(RTOL + batch)
    sweep = 0
    e2_prev = tl.zeros((), dtype=accumulator_dtype)
    # Rebound by the per-sweep stability check; after the last sweep it
    # holds the column sums of the final work matrix.
    alphas = tl.zeros((BLOCK_K,), dtype=accumulator_dtype)
    keep_sweeping = 1
    while (sweep < SWEEPS) & (keep_sweeping != 0):
        rotations = 0
        e2_local = tl.zeros((), dtype=accumulator_dtype)
        step = 0
        while step < ROUND - 1:
            position_q = ROUND - 1 - pair
            p = tl.where(
                pair == 0,
                0,
                ((pair + ring - step - 1) % ring) + 1,
            )
            q = tl.where(
                position_q == 0,
                0,
                ((position_q + ring - step - 1) % ring) + 1,
            )
            valid_pair = (pair < PAIRS) & (p < K) & (q < K)
            swap = p > q
            ordered_p = tl.where(swap, q, p)
            ordered_q = tl.where(swap, p, q)
            pair_mask = valid_pair[:, None] & row_mask[None, :]

            ap = tl.load(
                work_base + ordered_p[:, None] * ROWS + rows[None, :],
                mask=pair_mask,
                other=0.0,
            )
            aq = tl.load(
                work_base + ordered_q[:, None] * ROWS + rows[None, :],
                mask=pair_mask,
                other=0.0,
            )
            alpha = tl.sum(ap * ap, axis=1)
            beta = tl.sum(aq * aq, axis=1)
            gamma = tl.sum(ap * aq, axis=1)
            e2_local += tl.sum(gamma * gamma, axis=0).to(accumulator_dtype)
            if IS_FP64:
                # Double-single angle chain: native float64 div/sqrt are
                # software sequences on devices with weak FP64 units and
                # dominated the per-step latency. Uses t = sign(diff) *
                # 2*gamma /
                # (|diff| + sqrt(diff^2+4*gamma^2)) to avoid the
                # tau = diff/(2*gamma) overflow for tiny gamma.
                a_h = alpha.to(tl.float32)
                a_l = (alpha - a_h.to(tl.float64)).to(tl.float32)
                b_h = beta.to(tl.float32)
                b_l = (beta - b_h.to(tl.float64)).to(tl.float32)
                g_h = gamma.to(tl.float32)
                g_l = (gamma - g_h.to(tl.float64)).to(tl.float32)
                g2_h, g2_l = _df64_mul_ds(g_h, g_l, g_h, g_l)
                ab_h, ab_l = _df64_mul_ds(a_h, a_l, b_h, b_l)
                eps2 = REL_EPS * REL_EPS
                d_h, d_l = _df64_add(
                    g2_h,
                    g2_l,
                    -ab_h * eps2,
                    -(ab_l * eps2 + ABS_EPS * eps2),
                )
                active = valid_pair & ((d_h > 0.0) | ((d_h == 0.0) & (d_l > 0.0)))
                diff_h, diff_l = _df64_add(b_h, b_l, -a_h, -a_l)
                d2_h, d2_l = _df64_mul_ds(diff_h, diff_l, diff_h, diff_l)
                u_h, u_l = _df64_add(d2_h, d2_l, 4.0 * g2_h, 4.0 * g2_l)
                # A subnormal u means |diff| and |gamma| are both below
                # ~1e-19: the pair is orthogonal far below any relevant
                # tolerance. This also keeps _df64_sqrt_ds away from
                # subnormal inputs, which flush to zero under FTZ.
                active = active & (u_h >= 1.1754944e-38)
                sq_h, sq_l = _df64_sqrt_ds(u_h, u_l)
                ad_h = tl.abs(diff_h)
                ad_l = tl.where(diff_h >= 0.0, diff_l, -diff_l)
                den_h, den_l = _df64_add(ad_h, ad_l, sq_h, sq_l)
                den_zero = den_h == 0.0
                den_h = tl.where(den_zero, 1.0, den_h)
                sign_diff = tl.where(diff_h >= 0.0, 1.0, -1.0)
                t_h, t_l = _df64_div_ds(
                    sign_diff * 2.0 * g_h,
                    sign_diff * 2.0 * g_l,
                    den_h,
                    den_l,
                )
                t_h = tl.where(den_zero, 1.0, t_h)
                t_l = tl.where(den_zero, 0.0, t_l)
                t2_h, t2_l = _df64_mul_ds(t_h, t_l, t_h, t_l)
                v_h, v_l = _df64_add(t2_h, t2_l, 1.0, 0.0)
                sq2_h, sq2_l = _df64_sqrt_ds(v_h, v_l)
                c_h, c_l = _df64_div_ds(
                    tl.zeros_like(v_h) + 1.0, tl.zeros_like(v_h), sq2_h, sq2_l
                )
                s_h, s_l = _df64_mul_ds(t_h, t_l, c_h, c_l)
                c = c_h.to(tl.float64) + c_l.to(tl.float64)
                s = s_h.to(tl.float64) + s_l.to(tl.float64)
            else:
                active = valid_pair & (
                    tl.abs(gamma) > REL_EPS * tl.sqrt(alpha * beta + ABS_EPS)
                )
                safe_gamma = tl.where(active, gamma, 1.0)
                tau = (beta - alpha) / (2.0 * safe_gamma)
                sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
                c = 1.0 / tl.sqrt(1.0 + t * t)
                s = t * c
            rotations += tl.sum(active.to(tl.int32), axis=0)
            c = tl.where(active, c, 1.0)
            s = tl.where(active, s, 0.0)
            tl.store(
                work_base + ordered_p[:, None] * ROWS + rows[None, :],
                c[:, None] * ap - s[:, None] * aq,
                mask=pair_mask,
            )
            tl.store(
                work_base + ordered_q[:, None] * ROWS + rows[None, :],
                s[:, None] * ap + c[:, None] * aq,
                mask=pair_mask,
            )
            # Columns migrate across pair slots (and hence across warps)
            # between steps: the next step's tile loads must not race this
            # step's rotation stores.
            tl.debug_barrier()
            step += 1
        # --- rank-stability check --------------------------------------
        # G = W^T W = diag(alpha) + E with ||E||_F = sqrt(sum gamma^2).
        # Weyl's theorem bounds every singular-value perturbation by
        # ||E||_F, so once ||E||_F stays below half the smallest
        # |alpha_i - tol^2| margin no singular value can cross tol.
        check_tile = tl.load(
            work_base + singular_indices[:, None] * ROWS + rows[None, :],
            mask=(singular_indices < K)[:, None] & row_mask[None, :],
            other=0.0,
        )
        # The next sweep's first rotation stores must not overtake this
        # read in a laggard warp.
        tl.debug_barrier()
        alphas = tl.sum(check_tile * check_tile, axis=1).to(accumulator_dtype)
        maxa = tl.max(alphas, axis=0)
        tol = tl.maximum(atol, rtol * tl.sqrt(maxa))
        tol2 = tol * tol
        margin = tl.min(
            tl.where(
                singular_indices < K,
                tl.abs(alphas - tol2),
                tl.full((BLOCK_K,), float("inf"), dtype=accumulator_dtype),
            ),
            axis=0,
        )
        # Two sufficient stop conditions: the Weyl bound proves every
        # singular value is separated from the threshold, or the residual
        # stalled at the arithmetic noise floor (equivalent to the classic
        # "no more rotations" Jacobi exit).
        stall_floor = 64.0 * REL_EPS * maxa
        stable = (e2_local <= 0.25 * margin * margin) | (
            (sweep > 0)
            & (e2_local >= 0.8 * e2_prev)
            & (e2_local <= stall_floor * stall_floor)
        )
        e2_prev = e2_local
        keep_sweeping = ((rotations != 0) & (stable == 0)).to(tl.int32)
        sweep += 1

    # The last stability check already read the final work matrix, so its
    # column sums give the singular values directly.
    singular_values = tl.sqrt(alphas)

    max_value = tl.max(singular_values, axis=0)
    threshold = tl.maximum(atol, rtol * max_value)
    rank = tl.sum(
        ((singular_values > threshold) & (singular_indices < K)).to(tl.int32),
        axis=0,
    )
    tl.store(OUT + batch, rank.to(tl.int64))


@triton.jit
def _df64_add(h1, l1, h2, l2):
    # Error-free addition of two double-single numbers (Knuth TwoSum on the
    # hi parts, lo parts gathered afterwards, then one renormalization).
    s = h1 + h2
    z = s - h1
    e = (h1 - (s - z)) + (h2 - z)
    lo = l1 + l2 + e
    h = s + lo
    e2 = lo - (h - s)
    return h, e2


@triton.jit
def _df64_mul_ds(a_h, a_l, b_h, b_l):
    # Double-single product: TwoProd on the hi parts plus the cross terms.
    p = a_h * b_h
    e = tl.fma(a_h, b_h, -p) + a_h * b_l + a_l * b_h
    h = p + e
    lo = e - (h - p)
    return h, lo


@triton.jit
def _df64_div_ds(a_h, a_l, b_h, b_l):
    # Double-single division: fp32 quotient plus one df64 correction step.
    q1 = a_h / b_h
    p = q1 * b_h
    pe = tl.fma(q1, b_h, -p)
    r_h, r_l = _df64_add(a_h, a_l, -p, -(pe + q1 * b_l))
    q2 = r_h / b_h
    h = q1 + q2
    lo = q2 - (h - q1)
    return h, lo


@triton.jit
def _df64_sqrt_ds(a_h, a_l):
    # Double-single square root: fp32 root plus one Newton/df64 correction.
    x = tl.sqrt(a_h)
    p = x * x
    pe = tl.fma(x, x, -p)
    r_h, r_l = _df64_add(a_h, a_l, -p, -pe)
    corr = r_h / (2.0 * x)
    h = x + corr
    lo = corr - (h - x)
    not_positive = a_h <= 0.0
    h = tl.where(not_positive, 0.0, h)
    lo = tl.where(not_positive, 0.0, lo)
    return h, lo


# ===========================================================================
# Tridiagonal spectral counting helpers
# ===========================================================================


@libentry()
@triton.jit
def _matrix_rank_gk_init_kernel(
    D,
    E,
    GD,
    GE,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Golub-Kahan tridiagonal of order 2K for the bidiagonal (D, E): zero
    # diagonal, off-diagonal [d0, e0, d1, e1, ..., d_{K-1}]. Its eigenvalues
    # are exactly +/- sigma_i of the bidiagonal matrix.
    batch = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    tl.store(
        GD + batch * 2 * K + idx,
        tl.zeros((BLOCK,), dtype=GD.dtype.element_ty),
        mask=idx < 2 * K,
    )
    jj = idx // 2
    even = (idx % 2) == 0
    dv = tl.load(D + batch * K + jj, mask=even & (idx < 2 * K - 1), other=0.0)
    ev = tl.load(E + batch * K + jj, mask=(~even) & (idx < 2 * K - 1), other=0.0)
    tl.store(
        GE + batch * 2 * K + idx,
        tl.where(even, dv, ev),
        mask=idx < 2 * K - 1,
    )


@triton.jit
def _sturm_count_less(D, E2H, E2L, base, K: tl.constexpr, x, STRICT: tl.constexpr):
    # Number of eigenvalues of the tridiagonal T = diag(d) + diag(e, +/-1)
    # below the threshold x, via the qd recurrence. The zero-pivot guard
    # picks the tie convention (LAPACK DLANEG): with a tiny NEGATIVE
    # replacement the count is #{lambda <= x}; with a tiny POSITIVE one
    # (STRICT=True) it is exactly #{lambda < x}. torch's hermitian semantics
    # rank = #{lambda > tol} + #{lambda < -tol} are strict on both sides:
    # the positive side K - #{lambda <= tol} is already strict, while the
    # negative side must use the STRICT form so that an eigenvalue exactly
    # equal to -tol is not counted. The recurrence runs in
    # double-single arithmetic: native fp64 division is a slow software
    # sequence on this target and would dominate the O(k) chain.
    if STRICT:
        zero_pivot: tl.constexpr = 1.1754944e-38
    else:
        zero_pivot: tl.constexpr = -1.1754944e-38
    xh = x.to(tl.float32)
    xl = (x - xh.to(tl.float64)).to(tl.float32)
    d0 = tl.load(D + base)
    dh = d0.to(tl.float32)
    dl = (d0 - dh.to(tl.float64)).to(tl.float32)
    qh, ql = _df64_add(dh, dl, -xh, -xl)
    zero_q = (qh == 0.0) & (ql == 0.0)
    qh = tl.where(zero_q, zero_pivot, qh)
    ql = tl.where(zero_q, 0.0, ql)
    neg = tl.where(qh < 0.0, 1, 0)
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        dh = di.to(tl.float32)
        dl = (di - dh.to(tl.float64)).to(tl.float32)
        th, t_l = _df64_add(dh, dl, -xh, -xl)
        e2h = tl.load(E2H + base + i - 1)
        e2l = tl.load(E2L + base + i - 1)
        rh, rl = _df64_div_ds(e2h, e2l, qh, ql)
        qh, ql = _df64_add(th, t_l, -rh, -rl)
        zero_q = (qh == 0.0) & (ql == 0.0)
        qh = tl.where(zero_q, zero_pivot, qh)
        ql = tl.where(zero_q, 0.0, ql)
        neg += tl.where(qh < 0.0, 1, 0)
        i += 1
    return neg


@libentry()
@triton.jit
def _matrix_rank_sturm_rank_kernel(
    D,
    E,
    ATOL,
    RTOL,
    OUT,
    E2H,
    E2L,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    GK: tl.constexpr,
):
    # Rank of a symmetric matrix from its tridiagonal form:
    #   rank = #{lambda > tol} + #{lambda < -tol},
    #   tol  = max(atol, rtol * sigma_max),
    # where sigma_max = max |lambda| comes from Gershgorin bounds, refined
    # by bisection on the Sturm count only when the rank actually depends
    # on the refinement (otherwise the cheap two-sided evaluation is
    # already exact).
    batch = tl.program_id(0)
    idx = tl.arange(0, BLOCK_K)
    base = batch * K

    d = tl.load(D + base + idx, mask=idx < K, other=0.0).to(tl.float64)
    e_cur = tl.load(E + base + idx, mask=idx < K - 1, other=0.0).to(tl.float64)
    e_prev = tl.load(
        E + base + idx - 1,
        mask=(idx >= 1) & (idx < K),
        other=0.0,
    )
    gershgorin = tl.abs(d) + tl.abs(e_cur) + tl.abs(e_prev)
    hi = tl.max(gershgorin, axis=0)
    dmax = tl.max(d, axis=0)
    dmin = tl.min(d, axis=0)

    # Precompute e^2 in double-single form (shared by every count).
    eh = e_cur.to(tl.float32)
    el = (e_cur - eh.to(tl.float64)).to(tl.float32)
    e2h, e2l = _df64_mul_ds(eh, el, eh, el)
    tl.store(E2H + base + idx, e2h, mask=idx < K - 1)
    tl.store(E2L + base + idx, e2l, mask=idx < K - 1)

    atol = tl.load(ATOL + batch).to(tl.float64)
    rtol = tl.load(RTOL + batch).to(tl.float64)

    if hi == 0.0:
        # The tridiagonal (and hence the matrix) is exactly zero.
        tl.store(OUT + batch, tl.zeros((), dtype=tl.int64))
    else:
        if GK:
            # Zero diagonal (Golub-Kahan form): the largest |e| is a lower
            # bound of sigma_max by 2x2 interlacing.
            sigma_lo = tl.max(tl.abs(e_cur), axis=0)
        else:
            sigma_lo = tl.maximum(tl.abs(dmax), tl.abs(dmin))
        tol_lo = tl.maximum(atol, rtol * sigma_lo)
        tol_hi = tl.maximum(atol, rtol * hi)
        cnt_lo = _sturm_count_less(D, E2H, E2L, base, K, tol_lo, STRICT=False)
        cnt_hi = _sturm_count_less(D, E2H, E2L, base, K, tol_hi, STRICT=False)
        if GK:
            # Eigenvalues come in +/- sigma pairs, so the positive side
            # alone gives #{sigma > tol} without parity issues.
            rank_lo = K - cnt_lo
            rank_hi = K - cnt_hi
        else:
            # Negative side must be strict: #{lambda < -tol}, not <=.
            rank_lo = (K - cnt_lo) + _sturm_count_less(
                D, E2H, E2L, base, K, -tol_lo, STRICT=True
            )
            rank_hi = (K - cnt_hi) + _sturm_count_less(
                D, E2H, E2L, base, K, -tol_hi, STRICT=True
            )
        rank = rank_lo
        if rank_lo != rank_hi:
            # The rank depends on sigma_max: refine it by bisection.
            # lambda_max in [dmax, hi_pad] (count < K ... count == K).
            lo = dmax
            hi_p = hi * (1.0 + 1e-9) + 1e-292
            it = 0
            while it < BISECT_ITERS:
                mid = 0.5 * (lo + hi_p)
                cnt = _sturm_count_less(D, E2H, E2L, base, K, mid, STRICT=False)
                if cnt >= K:
                    hi_p = mid
                else:
                    lo = mid
                it += 1
            lmax = 0.5 * (lo + hi_p)
            if GK:
                sigma_max = lmax
            else:
                # lambda_min in [-hi_pad, dmin] (count == 0 ... count > 0).
                lo = -(hi * (1.0 + 1e-9) + 1e-292)
                hi_p = dmin
                it = 0
                while it < BISECT_ITERS:
                    mid = 0.5 * (lo + hi_p)
                    cnt = _sturm_count_less(D, E2H, E2L, base, K, mid, STRICT=False)
                    if cnt > 0:
                        hi_p = mid
                    else:
                        lo = mid
                    it += 1
                lmin = 0.5 * (lo + hi_p)
                sigma_max = tl.maximum(tl.abs(lmax), tl.abs(lmin))
            tol = tl.maximum(atol, rtol * sigma_max)
            cnt = _sturm_count_less(D, E2H, E2L, base, K, tol, STRICT=False)
            if GK:
                rank = K - cnt
            else:
                rank = (K - cnt) + _sturm_count_less(
                    D, E2H, E2L, base, K, -tol, STRICT=True
                )
        tl.store(OUT + batch, rank.to(tl.int64))


# ===========================================================================
# Pure-FP32 Sturm fallback for devices without native FP64
# (_native_fp64_supported() == False, float32 input only).  The
# factorization paths are already FP32-only for float32 input; historically
# only the Sturm count promoted to tl.float64.  Here every intermediate is
# FP32: a bracket stage with plain-FP32 qd recurrences bounds sigma_max and
# the tolerance, and the decisive count runs behind a kernel boundary in
# FP32 hi/lo double-single arithmetic (the _df64_* primitives above are pure
# FP32).  The bracket's rank is never the answer: the final kernel always
# recomputes the count at the handed-off tolerance, so the ~sqrt(eps) noise
# floor of the bracket's squared/FP32 arithmetic cannot affect the result.
# ===========================================================================


@triton.jit
def _sturm32_count_less(D, E, base, K, x):
    # Plain-FP32 qd count #{lambda <= x} (DLANEG tie convention: a zero
    # pivot is replaced by a tiny negative value).  Bracket stage only.
    q = tl.load(D + base) - x
    q = tl.where(q == 0.0, -1.1754944e-38, q)
    neg = (q < 0.0).to(tl.int32)
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        ei = tl.load(E + base + i - 1)
        q = (di - x) - ei * ei / q
        q = tl.where(q == 0.0, -1.1754944e-38, q)
        neg += (q < 0.0).to(tl.int32)
        i += 1
    return neg


@triton.jit
def _sturm32_count_posneg2(D, E, base, K, xp, xn):
    # Hermitian tolerance-bracket pair in ONE pass of the qd recurrence:
    # chain 1 counts #{lambda <= xp} (zero pivot -> tiny NEGATIVE), chain 2
    # counts #{lambda < xn} STRICTLY (zero pivot -> tiny POSITIVE, the
    # mirrored tie convention).  With xp = tol, xn = -tol the pair yields
    # #{|lambda| > tol} with exact tie semantics; pairing halves the serial
    # K-step chain.
    d0 = tl.load(D + base)
    q1 = d0 - xp
    q2 = d0 - xn
    q1 = tl.where(q1 == 0.0, -1.1754944e-38, q1)
    q2 = tl.where(q2 == 0.0, 1.1754944e-38, q2)
    neg1 = (q1 < 0.0).to(tl.int32)
    neg2 = (q2 < 0.0).to(tl.int32)
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        ei = tl.load(E + base + i - 1)
        e2 = ei * ei
        q1 = (di - xp) - e2 / q1
        q2 = (di - xn) - e2 / q2
        q1 = tl.where(q1 == 0.0, -1.1754944e-38, q1)
        q2 = tl.where(q2 == 0.0, 1.1754944e-38, q2)
        neg1 += (q1 < 0.0).to(tl.int32)
        neg2 += (q2 < 0.0).to(tl.int32)
        i += 1
    return neg1, neg2


@libentry()
@triton.jit
def _matrix_rank_sturm32_tridiag_bracket_kernel(
    D,
    E,
    ATOL,
    RTOL,
    OUT,
    TOL2,
    K,
    BLOCK: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
):
    # Hermitian bracket on the RAW tridiagonal (d, e), pure FP32.  Counts
    # #{|lambda| > max(atol, rtol*|lambda|max)} = (K - cle(tol)) + clt(-tol)
    # with cle/clt from _sturm32_count_posneg2.  |lambda|max is bracketed by
    # [max|d_i|, max Gershgorin radius] (max_i |d_ii| <= |lambda|max by
    # Rayleigh) and refined by bisection on f(x) = #{|lambda| > x} only when
    # the two bounds give different ranks.  TOL2 hands the (LINEAR, not
    # squared) tolerance to the decisive double-single kernel.
    batch = tl.program_id(0)
    kidx = tl.arange(0, BLOCK)
    base = batch * K
    kmask = kidx < K
    d = tl.load(D + base + kidx, mask=kmask, other=0.0)
    ee = tl.load(E + base + kidx, mask=kidx < K - 1, other=0.0)
    ee_prev = tl.load(E + base + kidx - 1, mask=(kidx >= 1) & kmask, other=0.0)
    gershgorin = tl.abs(d) + tl.abs(ee) + tl.abs(ee_prev)
    hi = tl.max(gershgorin, axis=0)
    dmax = tl.max(tl.abs(d), axis=0)
    atol = tl.load(ATOL + batch)
    rtol = tl.load(RTOL + batch)
    if hi == 0.0:
        # Zero matrix. TOL2 must still be written: the final kernel runs
        # unconditionally and reads it.
        tl.store(OUT + batch, tl.zeros((), dtype=tl.int64))
        tl.store(TOL2 + batch, 0.0)
    else:
        tol_lo = tl.maximum(atol, rtol * dmax)
        tol_hi = tl.maximum(atol, rtol * hi)
        c_lo, m_lo = _sturm32_count_posneg2(D, E, base, K, tol_lo, -tol_lo)
        c_hi, m_hi = _sturm32_count_posneg2(D, E, base, K, tol_hi, -tol_hi)
        rank_lo = K - c_lo + m_lo
        rank_hi = K - c_hi + m_hi
        tol2 = tol_lo
        if rank_lo != rank_hi:
            # Bisect |lambda|max on [dmax, hi]: f(x) = #{|lambda| > x} is
            # monotone; |lambda|max = sup{x : f(x) > 0}.  The pad must be an
            # FP32-representable ULP-scale bump: 2*eps_fp32 lifts the bound
            # by 2 ULP (1 + 1e-9 rounds back to 1.0 in FP32 and would pad
            # nothing), which keeps hi_p strictly above the rounded
            # Gershgorin bound and hence above |lambda|max.
            lo = dmax
            hi_p = hi * (1.0 + 2.3841858e-07)
            it = 0
            while it < BISECT_ITERS:
                mid = 0.5 * (lo + hi_p)
                c_m, m_m = _sturm32_count_posneg2(D, E, base, K, mid, -mid)
                if K - c_m + m_m >= 1:
                    lo = mid
                else:
                    hi_p = mid
                it += 1
            lmax = 0.5 * (lo + hi_p)
            tol2 = tl.maximum(atol, rtol * lmax)
        tl.store(OUT + batch, rank_lo.to(tl.int64))
        tl.store(TOL2 + batch, tol2)


@libentry()
@triton.jit
def _matrix_rank_sturm32_tridiag_final_kernel(D, E, TOL2, OUT, K):
    # Decisive hermitian count in double-single arithmetic from the RAW
    # tridiagonal: rank = #{lambda > tol} + #{lambda < -tol}, two lockstep
    # qd chains q_i = d_i - x - e_{i-1}^2 / q_{i-1}.  The positive chain at
    # x = tol uses the DLANEG guard (zero pivot -> tiny negative), so
    # K - neg1 IS the strict #{lambda > tol}; the negative chain at x = -tol
    # uses the MIRRORED guard (zero pivot -> tiny positive), so neg2 IS the
    # strict #{lambda < -tol} -- exact for every tol, including tol == 0,
    # where it yields #{lambda < 0}.  The e^2 terms are TwoProd'd so the
    # recurrence keeps ~eps^2 relative precision.
    # Requires enable_fp_fusion=False (fma contraction breaks the
    # TwoSum/TwoProd error-free transforms).
    batch = tl.program_id(0)
    tol = tl.load(TOL2 + batch)
    xn = -tol
    base = batch * K
    d0 = tl.load(D + base)
    q1h, q1l = _df64_add(d0, 0.0, -tol, 0.0)
    q2h, q2l = _df64_add(d0, 0.0, -xn, 0.0)
    zero1 = (q1h == 0.0) & (q1l == 0.0)
    q1h = tl.where(zero1, -1.1754944e-38, q1h)
    q1l = tl.where(zero1, 0.0, q1l)
    zero2 = (q2h == 0.0) & (q2l == 0.0)
    q2h = tl.where(zero2, 1.1754944e-38, q2h)
    q2l = tl.where(zero2, 0.0, q2l)
    neg1 = ((q1h < 0.0) | ((q1h == 0.0) & (q1l < 0.0))).to(tl.int32)
    neg2 = ((q2h < 0.0) | ((q2h == 0.0) & (q2l < 0.0))).to(tl.int32)
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        ei = tl.load(E + base + i - 1)
        eih, eil = _df64_mul_ds(ei, 0.0, ei, 0.0)
        qd1h, qd1l = _df64_div_ds(eih, eil, q1h, q1l)
        s1h, s1l = _df64_add(di, 0.0, -tol, 0.0)
        q1h, q1l = _df64_add(s1h, s1l, -qd1h, -qd1l)
        qd2h, qd2l = _df64_div_ds(eih, eil, q2h, q2l)
        s2h, s2l = _df64_add(di, 0.0, -xn, 0.0)
        q2h, q2l = _df64_add(s2h, s2l, -qd2h, -qd2l)
        zero1 = (q1h == 0.0) & (q1l == 0.0)
        q1h = tl.where(zero1, -1.1754944e-38, q1h)
        q1l = tl.where(zero1, 0.0, q1l)
        zero2 = (q2h == 0.0) & (q2l == 0.0)
        q2h = tl.where(zero2, 1.1754944e-38, q2h)
        q2l = tl.where(zero2, 0.0, q2l)
        neg1 += ((q1h < 0.0) | ((q1h == 0.0) & (q1l < 0.0))).to(tl.int32)
        neg2 += ((q2h < 0.0) | ((q2h == 0.0) & (q2l < 0.0))).to(tl.int32)
        i += 1
    rank = (K - neg1) + neg2
    tl.store(OUT + batch, rank.to(tl.int64))


@libentry()
@triton.jit
def _matrix_rank_bidiag32_to_tridiag_kernel(D, E, DD, EE, K, BLOCK: tl.constexpr):
    # Construct the B^T B tridiagonal (dd_i = d_i^2 + e_{i-1}^2,
    # ee_i = d_i * e_i) from the raw bidiagonal d/e, in its OWN launch so
    # the Sturm bracket only READS global memory.  The FP32 tridiagonal has
    # an absolute element rounding of ~eps*sigma_max^2, which swamps any
    # squared singular value below ~(sqrt(eps)*sigma_max)^2 -- i.e.
    # everything near the default tolerance (k*eps*sigma_max).  It therefore
    # feeds only the bracket; the decisive kernel works from the RAW d/e
    # (relative precision ~K*eps regardless of conditioning) and squares
    # them inside the double-single arithmetic.
    batch = tl.program_id(0)
    kidx = tl.arange(0, BLOCK)
    base = batch * K
    kmask = kidx < K
    d = tl.load(D + base + kidx, mask=kmask, other=0.0)
    e_cur = tl.load(E + base + kidx, mask=kidx < K - 1, other=0.0)
    e_prev = tl.load(E + base + kidx - 1, mask=(kidx >= 1) & kmask, other=0.0)
    tl.store(DD + base + kidx, d * d + e_prev * e_prev, mask=kmask)
    tl.store(EE + base + kidx, d * e_cur, mask=kmask)


@libentry()
@triton.jit
def _matrix_rank_sturm32_bidiag_bracket_kernel(
    D,
    E,
    ATOL,
    RTOL,
    OUT,
    TOL2,
    K,
    BLOCK: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
):
    # Bidiagonal bracket on the B^T B tridiagonal (dd, ee) written by
    # _matrix_rank_bidiag32_to_tridiag_kernel, pure FP32.  Counts
    # #{sigma > max(atol, rtol*sigma_max)} with sigma_max bracketed by
    # sqrt(max dd) / sqrt(Gershgorin) and refined by bisection only when the
    # two bounds give different ranks.
    batch = tl.program_id(0)
    kidx = tl.arange(0, BLOCK)
    base = batch * K
    kmask = kidx < K
    dd = tl.load(D + base + kidx, mask=kmask, other=0.0)
    ee = tl.load(E + base + kidx, mask=kidx < K - 1, other=0.0)
    ee_prev = tl.load(E + base + kidx - 1, mask=(kidx >= 1) & kmask, other=0.0)
    gershgorin = tl.abs(dd) + tl.abs(ee) + tl.abs(ee_prev)
    hi = tl.max(gershgorin, axis=0)
    dmax = tl.max(dd, axis=0)
    atol = tl.load(ATOL + batch)
    rtol = tl.load(RTOL + batch)
    if hi == 0.0:
        # Zero matrix: rank 0. TOL2 must still be written: the final
        # double-single kernel runs unconditionally and reads it.
        tl.store(OUT + batch, tl.zeros((), dtype=tl.int64))
        tl.store(TOL2 + batch, 0.0)
    else:
        sigma_lo = tl.sqrt(tl.maximum(dmax, 0.0))
        sigma_hi = tl.sqrt(hi)
        tol_lo = tl.maximum(atol, rtol * sigma_lo)
        tol_hi = tl.maximum(atol, rtol * sigma_hi)
        cnt_lo = _sturm32_count_less(D, E, base, K, tol_lo * tol_lo)
        cnt_hi = _sturm32_count_less(D, E, base, K, tol_hi * tol_hi)
        rank_lo = K - cnt_lo
        rank_hi = K - cnt_hi
        rank = rank_lo
        tol2 = tol_lo * tol_lo
        if rank_lo != rank_hi:
            # Same FP32-representable ULP pad as the hermitian bracket:
            # 2*eps_fp32 (2 ULP), since 1 + 1e-9 rounds back to 1.0 in FP32.
            lo = tl.maximum(dmax, 0.0)
            hi_p = hi * (1.0 + 2.3841858e-07)
            it = 0
            while it < BISECT_ITERS:
                mid = 0.5 * (lo + hi_p)
                cnt = _sturm32_count_less(D, E, base, K, mid)
                if cnt >= K:
                    hi_p = mid
                else:
                    lo = mid
                it += 1
            lmax = 0.5 * (lo + hi_p)
            sigma_max = tl.sqrt(lmax)
            tol2 = tl.maximum(atol, rtol * sigma_max)
            tol2 = tol2 * tol2
        tl.store(OUT + batch, rank.to(tl.int64))
        tl.store(TOL2 + batch, tol2)


@libentry()
@triton.jit
def _matrix_rank_sturm32_bidiag_final_kernel(D, E, TOL2, OUT, K):
    # The decisive count, in double-single arithmetic, from the RAW
    # bidiagonal d/e (NOT the FP32 dd/ee tridiagonal, whose squaring rounds
    # each element absolutely by ~eps*sigma_max^2 -- a sqrt(eps) noise floor
    # in the sigma domain).  Squaring the FP32 d/e inside the double-single
    # arithmetic keeps every intermediate at ~eps^2 relative precision.  The
    # qd recurrence for B^T B: dd_i = d_i^2 + e_{i-1}^2, ee_i = d_i*e_i,
    # q_i = dd_i - tol2 - ee_{i-1}^2 / q_{i-1}.
    # Requires enable_fp_fusion=False (fma contraction breaks the
    # TwoSum/TwoProd error-free transforms).
    batch = tl.program_id(0)
    tol2 = tl.load(TOL2 + batch)
    base = batch * K
    d0 = tl.load(D + base)
    d0h, d0l = _df64_mul_ds(d0, 0.0, d0, 0.0)
    q_h, q_l = _df64_add(d0h, d0l, -tol2, 0.0)
    zero = (q_h == 0.0) & (q_l == 0.0)
    q_h = tl.where(zero, -1.1754944e-38, q_h)
    q_l = tl.where(zero, 0.0, q_l)
    neg = ((q_h < 0.0) | ((q_h == 0.0) & (q_l < 0.0))).to(tl.int32)
    dprev = d0
    i = 1
    while i < K:
        di = tl.load(D + base + i)
        ei = tl.load(E + base + i - 1)
        dih, dil = _df64_mul_ds(di, 0.0, di, 0.0)
        eih, eil = _df64_mul_ds(ei, 0.0, ei, 0.0)
        dd_h, dd_l = _df64_add(dih, dil, eih, eil)
        ph, pl = _df64_mul_ds(dprev, 0.0, ei, 0.0)
        ee_h, ee_l = _df64_mul_ds(ph, pl, ph, pl)
        qd_h, qd_l = _df64_div_ds(ee_h, ee_l, q_h, q_l)
        s_h, s_l = _df64_add(dd_h, dd_l, -tol2, 0.0)
        q_h, q_l = _df64_add(s_h, s_l, -qd_h, -qd_l)
        zero = (q_h == 0.0) & (q_l == 0.0)
        q_h = tl.where(zero, -1.1754944e-38, q_h)
        q_l = tl.where(zero, 0.0, q_l)
        neg += ((q_h < 0.0) | ((q_h == 0.0) & (q_l < 0.0))).to(tl.int32)
        dprev = di
        i += 1
    rank_b = (K - neg).to(tl.int64)
    tl.store(OUT + batch, rank_b)


def _expand_tolerance(value, batch_shape, input, name):
    # Tolerances keep their own precision instead of being rounded down to
    # the input dtype: a float64 (or high-precision Python float) tolerance
    # sitting next to a singular value of a float32 matrix must decide the
    # rank at ITS precision, matching torch.  On devices without native
    # FP64 the comparison precision is FP32 anyway (ds32 Sturm tail), so
    # float32 tolerances lose nothing there.
    tol_dtype = torch.float64 if _native_fp64_supported() else torch.float32
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise RuntimeError(
                f"torch.linalg.matrix_rank: {name} tensor of complex type is not "
                f"supported. Got {value.dtype}"
            )
        if value.device != input.device:
            raise RuntimeError(
                f"torch.linalg.matrix_rank: Expected {name} and input tensors to "
                f"be on the same device, but got {name} on {value.device} and "
                f"input on {input.device}"
            )
        try:
            value = value.expand(batch_shape)
        except RuntimeError as error:
            raise RuntimeError(
                f"torch.linalg.matrix_rank: {name} with shape {tuple(value.shape)} "
                f"is not broadcastable to batch shape {tuple(batch_shape)}"
            ) from error
        return value.to(dtype=tol_dtype).contiguous()

    raise TypeError(f"torch.linalg.matrix_rank: {name} must be a float or Tensor")


def _scalar_tolerance(value, name):
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"torch.linalg.matrix_rank: {name} must be a float or Tensor"
        ) from error


def _is_exact_float32(value):
    try:
        rounded = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error):
        return False
    return rounded == value


def _materialize_tolerance(value, batch_shape, input):
    # Scalar -> device tensor, for the graph-captured large-matrix paths
    # (replay must re-read tolerances from staged buffers) and for the
    # mixed scalar/tensor case on small paths.
    if isinstance(value, torch.Tensor):
        return value
    tol_dtype = torch.float64 if _native_fp64_supported() else torch.float32
    return torch.full(batch_shape, value, dtype=tol_dtype, device=input.device)


def _prepare_tolerances(input, atol, rtol):
    # Each return is either a Python float (scalar fast path -- no device
    # tensor is materialized) or a device tensor expanded to the batch
    # shape.  Only genuine tensor tolerances take the tensor path.
    batch_shape = input.shape[:-2]
    atol_is_set = atol is not None
    if atol is None:
        atol_val = 0.0
    elif isinstance(atol, torch.Tensor):
        atol_val = _expand_tolerance(atol, batch_shape, input, "atol")
    else:
        atol_val = _scalar_tolerance(atol, "atol")

    if isinstance(rtol, torch.Tensor):
        rtol_val = _expand_tolerance(rtol, batch_shape, input, "rtol")
    elif rtol is not None:
        rtol_val = _scalar_tolerance(rtol, "rtol")
    else:
        default_rtol = max(input.shape[-2:]) * torch.finfo(input.dtype).eps
        if not atol_is_set:
            rtol_val = default_rtol
        elif isinstance(atol_val, torch.Tensor):
            rtol_val = torch.where(
                atol_val > 0,
                torch.zeros_like(atol_val),
                torch.full_like(atol_val, default_rtol),
            )
        else:
            rtol_val = 0.0 if atol_val > 0 else default_rtol
    if isinstance(rtol_val, torch.Tensor):
        rtol_val = rtol_val.contiguous()
    return atol_val, rtol_val


def _check_input(input, hermitian):
    if input.ndim < 2:
        raise RuntimeError(
            "torch.linalg.matrix_rank: input must have at least 2 dimensions"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_matrix_rank currently supports float32 and float64 "
            f"real inputs only; got {input.dtype}"
        )
    if input.dtype == torch.float64 and not _native_fp64_supported():
        # Fail fast, before any shape dispatch: every float64 path below
        # computes in native FP64 (or fp64-backed double-single splits).
        raise NotImplementedError(
            "FlagGems linalg_matrix_rank: float64 input requires native FP64 "
            "support, which this device does not provide"
        )
    if hermitian and input.shape[-2] != input.shape[-1]:
        raise RuntimeError(
            "torch.linalg.matrix_rank: A must be batches of square matrices when "
            "hermitian=True"
        )


def _empty_matrix_rank(input, output_shape):
    out = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    output_size = out.numel()
    if output_size:
        block_size = min(256, triton.next_power_of_2(output_size))
        with torch_device_fn.device(input.device):
            _matrix_rank_zero_kernel[(triton.cdiv(output_size, block_size),)](
                out,
                N=output_size,
                BLOCK_SIZE=block_size,
            )
    return out


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_pad_init_kernel(A, W, SCALE, K, RS, WPITCH):
    # Barrier-free tridiagonalization init: copy the SYMMETRIZED input
    # (lower triangle read, mirrored) into the padded W work matrix,
    # dividing by the per-batch scale in-kernel (the staged input is
    # UNSCALED; scaling here avoids an O(K^2) temporary outside the
    # graph).  W is (KP, RS) with KP == RS == cdiv(K,64)*64 + 64 and
    # arrives zero-initialized, so the step/mat/apply kernels can read the
    # padding without masks; this kernel only writes the K x K corner.
    # A is the (batch, K, K) contiguous input.
    b = tl.program_id(0)
    c0 = tl.program_id(1) * 64
    lc = tl.arange(0, 64)
    a_base = A + b * K * K
    wbase = W + b * WPITCH
    s = tl.load(SCALE + b)
    for rb in tl.range(0, (K + 63) // 64):
        lr = rb * 64 + tl.arange(0, 64)
        cc = c0 + lc
        mask = (lr < K)[:, None] & (cc < K)[None, :]
        # Lower triangle only: read A[max(r,c), min(r,c)].
        src_r = tl.maximum(lr[:, None], cc[None, :])
        src_c = tl.minimum(lr[:, None], cc[None, :])
        at = tl.load(a_base + src_r * K + src_c, mask=mask, other=0.0)
        tl.store(wbase + lr[:, None] * RS + cc[None, :], at / s, mask=mask)


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_step_kernel(
    W, V, D, E, TAU, ACC, CSCA, J, K, RS, WPITCH, APITCH
):
    # Barrier-free hermitian tridiagonalization, step J: column J below the
    # diagonal defines the reflector (support rows > J), D[J] = W[J,J]
    # (untouched by this step's reflection), E[J] = +/-sigma.  Single
    # program per matrix: reads row J of W (== column J by symmetry),
    # stores the provisional reflector into V row J with the scalar pivot
    # overwrite, and zeroes the ACC/CSCA accumulation slots the mat
    # kernel's atomics add into.  Cross-program ordering comes from kernel
    # launch boundaries only -- no grid barrier.
    pid = tl.program_id(0)
    wbase = W + pid * WPITCH
    lr = tl.arange(0, 64)
    dtype = W.dtype.element_ty
    dj = tl.zeros((), dtype=dtype)
    ssq = tl.zeros((), dtype=dtype)
    x0 = tl.zeros((), dtype=dtype)
    for rb in tl.range(J // 64, (K + 63) // 64):
        r0 = rb * 64
        chf = tl.load(wbase + J * RS + r0 + lr)
        dj += tl.sum(chf * ((r0 + lr > J - 1) & (r0 + lr < J + 1)).to(dtype), axis=0)
        ch = chf * ((r0 + lr) > J).to(dtype)
        tl.store(V + pid * RS + r0 + lr, ch)
        ssq += tl.sum(ch * ch, axis=0)
        x0 += tl.sum(ch * ((r0 + lr > J) & (r0 + lr < J + 2)).to(dtype), axis=0)
    sigma = tl.sqrt(ssq)
    alpha = tl.where(x0 >= 0.0, -sigma, sigma)
    vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
    tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
    tl.store(V + pid * RS + (J + 1), x0 - alpha)
    tl.store(D + pid * K + J, dj)
    tl.store(E + pid * K + J, alpha)
    tl.store(TAU + pid * K + J, tau)
    tl.store(CSCA + pid, tl.zeros((), dtype=dtype))
    for cb in tl.range((J + 1) // 64, APITCH // 64):
        tl.store(ACC + pid * APITCH + cb * 64 + lr, tl.zeros((64,), dtype=dtype))


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_mat_kernel(W, V, ACC, CSCA, J, RS, WPITCH, APITCH, NRT):
    # omega = W_trailing @ v (symmetric matvec), multi-program over
    # (col tile, row strip) with atomic accumulation.  NRT counts the
    # K-trimmed trailing tiles (cdiv(K-1-J, 64)): with c0/r0 = J+1+tile*64
    # the last tile reaches at most K+62 < RS, so no row-pitch straddle
    # and no masks (pad rows/cols are zeros by construction and excluded
    # entirely).  Each program also accumulates its slice of v^T omega
    # into CSCA (for the rank-2 correction coefficient in apply).
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NRT
    rt = flat % NRT
    c0 = J + 1 + ct * 64
    r0 = J + 1 + rt * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    wbase = W + pid * WPITCH
    tile = tl.load(wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :])
    vc = tl.load(V + pid * RS + c0 + lc)
    part = tl.sum(tl.trans(tile) * vc[None, :], axis=1)
    tl.atomic_add(ACC + pid * APITCH + r0 + lr, part)
    vp = tl.load(V + pid * RS + r0 + lr)
    tl.atomic_add(CSCA + pid, tl.sum(part * vp, axis=0))


_HERM_TRIDIAG_BLOCKED_MIN_K = 768
_HERM_TRIDIAG_PANEL = 32


_HERM_TRIDIAG_SCRATCH_LINE = 32  # one 128B cache line per scratch entry


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_pcol_kernel(
    W,
    V,
    PV,
    PW,
    ACC,
    CSCA,
    PSCR,
    J,
    P,
    K,
    RS,
    WPITCH,
    APITCH,
    PVP,
    SPITCH,
    NB_P: tl.constexpr,
    SL: tl.constexpr,
):
    # Blocked WY tridiagonalization (LAPACK DSYTRD), column J = panel slot
    # P, phase A -- MULTI-PROGRAM over 64-row chunks: apply the deferred
    # panel update to column J,
    #   a_J -= Vp @ Wp[J, :] + Wp @ Vp[J, :]        (panel columns q < P)
    # store the (pre-pivot) reflector into row J of V, and atomically
    # accumulate the per-chunk partials the next kernels need:
    # sigma^2 / dj / x0 scalars and the w1 = Wp^T v, w2 = Vp^T v panel
    # vectors (each scratch entry padded to its own 128B line -- thousands
    # of atomics into one shared line serialize in the L2).  Every program
    # also zeroes the ACC chunk it owns (first program: CSCA) for the GEMV
    # kernel's atomics.  Cross-program ordering comes from kernel launch
    # boundaries only -- no grid barrier.
    #
    # Column J is read as ROW J: the padded work matrix is symmetrized
    # (pad_init mirrors the lower triangle) and rank-2k keeps the trailing
    # block symmetric, so row J == column J exactly and the read is fully
    # coalesced.  The updated column is NEVER written back to W: later
    # columns read their own rows, and the panel region of W is dead once
    # factored.  (A strided column access with stride RS measured ~30
    # us/column at k=1024 -- it dominated the whole path.)
    #
    # Panel slots q >= P hold stale FINITE values (zero-initialized once at
    # workspace creation) and are masked with tl.where on unmasked loads: a
    # runtime mask on the loads themselves makes Triton emit serialized
    # predicated updates, and Inf/NaN * 0 would poison the sums.
    pid = tl.program_id(0)
    rb = tl.program_id(1) + J // 64
    wrow = W + pid * WPITCH + J * RS
    vrow = V + pid * RS
    pvbase = PV + pid * PVP
    pwbase = PW + pid * PVP
    lr = tl.arange(0, 64)
    qidx = tl.arange(0, NB_P)
    dtype = W.dtype.element_ty
    tl.store(ACC + pid * APITCH + rb * 64 + lr, tl.zeros((64,), dtype=dtype))
    if tl.program_id(1) == 0:
        tl.store(CSCA + pid, tl.zeros((), dtype=dtype))
    if rb < (K + 63) // 64:
        slot = PSCR + pid * SPITCH + (J % 2) * (3 + 2 * NB_P) * SL
        r = rb * 64 + lr
        col = tl.load(wrow + r)
        if P > 0:
            qm = qidx < P
            v_row_j = tl.where(qm, tl.load(pvbase + J * NB_P + qidx), 0.0)
            w_row_j = tl.where(qm, tl.load(pwbase + J * NB_P + qidx), 0.0)
            v_own = tl.load(pvbase + r[:, None] * NB_P + qidx[None, :])
            w_own = tl.load(pwbase + r[:, None] * NB_P + qidx[None, :])
            col -= tl.sum(v_own * w_row_j[None, :], axis=1) + tl.sum(
                w_own * v_row_j[None, :], axis=1
            )
            # w1/w2 partials on the pre-pivot reflector; the pivot entry
            # discrepancy (-alpha * panel[J+1, q]) is corrected in pfin.
            ch = tl.where(r > J, col, 0.0)
            tl.atomic_add(
                slot + (3 + qidx) * SL,
                tl.sum(w_own * ch[:, None], axis=0),
                sem="relaxed",
            )
            tl.atomic_add(
                slot + (3 + NB_P + qidx) * SL,
                tl.sum(v_own * ch[:, None], axis=0),
                sem="relaxed",
            )
        # Store the reflector zeroed below J + 1; the pivot is adjusted on
        # load by pmat/pfin (which recompute alpha from the partials).
        tl.store(vrow + r, tl.where(r > J, col, 0.0))
        ch = tl.where(r > J, col, 0.0)
        tl.atomic_add(slot, tl.sum(ch * ch, axis=0), sem="relaxed")
        tl.atomic_add(
            slot + SL, tl.sum(tl.where(r == J, col, 0.0), axis=0), sem="relaxed"
        )
        tl.atomic_add(
            slot + 2 * SL,
            tl.sum(tl.where(r == J + 1, col, 0.0), axis=0),
            sem="relaxed",
        )


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_pmat_kernel(
    W,
    V,
    D,
    E,
    ACC,
    CSCA,
    PSCR,
    J,
    K,
    RS,
    WPITCH,
    APITCH,
    SPITCH,
    NRT,
    NB_P: tl.constexpr,
    SL: tl.constexpr,
):
    # Blocked WY column J, phase B: omega = W_trailing @ v (symmetric
    # matvec, multi-program with atomic accumulation into ACC, plus the
    # v^T omega partial into CSCA).  Every program redundantly recomputes
    # the Householder scalars from pcol's partials (three scalar loads) and
    # applies the pivot overwrite v[J+1] = x0 - alpha on load; program
    # flat == 0 additionally stores D[J] / E[J].  Tiles are K-trimmed
    # trailing blocks (c0/r0 = J+1+tile*64 reach at most K+62 < RS), so no
    # masks -- pad rows/cols are zeros by construction.
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NRT
    rt = flat % NRT
    c0 = J + 1 + ct * 64
    r0 = J + 1 + rt * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    slot = PSCR + pid * SPITCH + (J % 2) * (3 + 2 * NB_P) * SL
    sig = tl.load(slot)
    x0 = tl.load(slot + 2 * SL)
    sigma = tl.sqrt(sig)
    alpha = tl.where(x0 >= 0.0, -sigma, sigma)
    wbase = W + pid * WPITCH
    vrow = V + pid * RS
    tile = tl.load(wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :])
    vc = tl.load(vrow + c0 + lc)
    vc = tl.where(c0 + lc == J + 1, x0 - alpha, vc)
    part = tl.sum(tl.trans(tile) * vc[None, :], axis=1)
    tl.atomic_add(ACC + pid * APITCH + r0 + lr, part)
    vp = tl.load(vrow + r0 + lr)
    vp = tl.where(r0 + lr == J + 1, x0 - alpha, vp)
    tl.atomic_add(CSCA + pid, tl.sum(part * vp, axis=0))
    if flat == 0:
        tl.store(D + pid * K + J, tl.load(slot + SL))
        tl.store(E + pid * K + J, alpha)


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_pfin_kernel(
    V,
    PV,
    PW,
    ACC,
    CSCA,
    PSCR,
    J,
    P,
    K,
    RS,
    APITCH,
    PVP,
    SPITCH,
    NB_P: tl.constexpr,
    SL: tl.constexpr,
):
    # Blocked WY column J = panel slot P, phase C -- MULTI-PROGRAM over
    # 64-row chunks: finish the corrected w and store the panel columns.
    # Every program redundantly recomputes the Householder scalars and the
    # rank-2 coefficient from the scratch partials (the identity
    # v^T A_p v = v^T S v - 2 (w1 . w2) needs no second global reduction;
    # v^T S v is the CSCA partial the GEMV kernel accumulated):
    #   w = tau * (omega - Vp w1 - Wp w2) - (tau^2/2)(v^T A_p v) * v
    # The tau^2 factor is evaluated as tau * (tau * dot), never tau*tau:
    # strongly deflated spectra drive tau to ~1e20 and the square overflows
    # fp32 even though the final coefficient is O(1).  pcol accumulated
    # w1/w2 with the PRE-pivot value x0 at row J+1, so both get the
    # -alpha * panel[J+1, q] correction here.  Program 0 additionally
    # zeroes the NEXT column's scratch slot (wrapping to slot 0 after the
    # last column) so the next call / graph replay finds zeros.
    pid = tl.program_id(0)
    rb = tl.program_id(1) + (J + 1) // 64
    pvbase = PV + pid * PVP
    pwbase = PW + pid * PVP
    vrow = V + pid * RS
    lr = tl.arange(0, 64)
    qidx = tl.arange(0, NB_P)
    dtype = V.dtype.element_ty
    slot = PSCR + pid * SPITCH + (J % 2) * (3 + 2 * NB_P) * SL
    sig = tl.load(slot)
    x0 = tl.load(slot + 2 * SL)
    sigma = tl.sqrt(sig)
    alpha = tl.where(x0 >= 0.0, -sigma, sigma)
    vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
    tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
    qm = qidx < P
    w1 = tl.load(slot + (3 + qidx) * SL)
    w2 = tl.load(slot + (3 + NB_P + qidx) * SL)
    w1 -= alpha * tl.load(pwbase + (J + 1) * NB_P + qidx)
    w2 -= alpha * tl.load(pvbase + (J + 1) * NB_P + qidx)
    w1 = tl.where(qm, w1, 0.0)
    w2 = tl.where(qm, w2, 0.0)
    dot = tl.load(CSCA + pid) - 2.0 * tl.sum(w1 * w2, axis=0)
    coef = (-0.5 * tau) * (tau * dot)
    r = rb * 64 + lr
    v = tl.load(vrow + r)
    v = tl.where(r == J + 1, x0 - alpha, v)
    w_raw = tl.load(ACC + pid * APITCH + r)
    if P > 0:
        v_own = tl.load(pvbase + r[:, None] * NB_P + qidx[None, :])
        w_own = tl.load(pwbase + r[:, None] * NB_P + qidx[None, :])
        w_raw -= tl.sum(v_own * w1[None, :], axis=1) + tl.sum(
            w_own * w2[None, :], axis=1
        )
    w = tau * w_raw + coef * v
    # Pad rows keep their zero-init; stale entries at rows <= J from the
    # previous panel's slot P are finite and are masked by q < p at every
    # read site.
    tl.store(pwbase + r * NB_P + P, w, mask=(r > J) & (r < K))
    tl.store(pvbase + r * NB_P + P, v, mask=(r > J) & (r < K))
    if tl.program_id(1) == 0:
        jn = tl.where(J + 1 > K - 2, 0, J + 1)
        slot_n = PSCR + pid * SPITCH + (jn % 2) * (3 + 2 * NB_P) * SL
        eidx = tl.arange(0, 4 * NB_P)
        tl.store(
            slot_n + eidx * SL,
            tl.zeros((4 * NB_P,), dtype=dtype),
            mask=eidx < 3 + 2 * NB_P,
        )


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_rank2k_kernel(
    W, PV, PW, T0, PE, RS, WPITCH, PVP, NT, NB_P: tl.constexpr
):
    # Trailing symmetric rank-2k update S -= Vp Wp^T + Wp Vp^T over
    # rows/cols >= T0, one program per (col tile, row strip) of the
    # K-trimmed trailing block: T0 + 63 <= K + 62 < RS (padded) so tiles
    # never straddle the pitch and need no masks, and the zero pad rows of
    # PV/PW make the update exactly zero in the padding.  Only the
    # rt >= ct half of the tiles does any work: the update of the mirror
    # tile (c, r) is exactly tl.trans(upd), so it is written back from the
    # same program, halving the tl.dot work while keeping both triangles
    # of W consistent for pmat's full-block reads.
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NT
    rt = flat % NT
    if rt < ct:
        return
    r0 = T0 + rt * 64
    c0 = T0 + ct * 64
    lr = tl.arange(0, 64)
    qidx = tl.arange(0, NB_P)
    qm = qidx < PE
    pvbase = PV + pid * PVP
    pwbase = PW + pid * PVP
    r = r0 + lr
    c = c0 + lr
    v_r = tl.where(
        qm[None, :], tl.load(pvbase + r[:, None] * NB_P + qidx[None, :]), 0.0
    )
    w_r = tl.where(
        qm[None, :], tl.load(pwbase + r[:, None] * NB_P + qidx[None, :]), 0.0
    )
    v_c = tl.where(
        qm[None, :], tl.load(pvbase + c[:, None] * NB_P + qidx[None, :]), 0.0
    )
    w_c = tl.where(
        qm[None, :], tl.load(pwbase + c[:, None] * NB_P + qidx[None, :]), 0.0
    )
    if tl.constexpr(W.dtype.element_ty == tl.float64):
        # fp64 tl.dot is miscompiled on some vendor backends (wrong results
        # for every block shape / num_warps variant measured), so
        # accumulate the rank-2k update as per-column outer products.  The
        # v_r/w_r/v_c/w_c tiles above are dead in this branch and get
        # eliminated.
        upd = tl.zeros((64, 64), dtype=W.dtype.element_ty)
        for q in tl.range(0, PE):
            v_rq = tl.load(pvbase + r * NB_P + q)
            w_rq = tl.load(pwbase + r * NB_P + q)
            v_cq = tl.load(pvbase + c * NB_P + q)
            w_cq = tl.load(pwbase + c * NB_P + q)
            upd += v_rq[:, None] * w_cq[None, :] + w_rq[:, None] * v_cq[None, :]
    else:
        upd = tl.dot(v_r, tl.trans(w_c), input_precision="ieee") + tl.dot(
            w_r, tl.trans(v_c), input_precision="ieee"
        )
    # upd[i, j] belongs to (row r_i, col c_j); the pointer tile must use the
    # same orientation or the update lands transposed and breaks symmetry.
    ptrs = W + pid * WPITCH + r[:, None] * RS + c[None, :]
    tile = tl.load(ptrs)
    tl.store(ptrs, tile - upd)
    if rt > ct:
        mptrs = W + pid * WPITCH + c[:, None] * RS + r[None, :]
        mtile = tl.load(mptrs)
        tl.store(mptrs, mtile - tl.trans(upd))


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_last_diag_kernel(W, D, K, RS, WPITCH):
    # The blocked loop factors columns 0 .. K-2; D[K-1] is whatever the
    # last rank-2k update left at W[K-1, K-1].
    pid = tl.program_id(0)
    tl.store(
        D + pid * K + (K - 1),
        tl.load(W + pid * WPITCH + (K - 1) * RS + (K - 1)),
    )


@libentry()
@triton.jit
def _matrix_rank_herm_tridiag_apply_kernel(
    W, V, TAU, ACC, CSCA, J, K, RS, WPITCH, APITCH, NRT
):
    # Trailing symmetric rank-2 update W -= v w^T + w v^T with
    # w = tau*omega - (tau^2/2)(v^T omega) v (LAPACK DSYTD2 form), one
    # program per (col tile, row strip) of the K-trimmed trailing block
    # (tiles never reach RS, so no masks).
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NRT
    rt = flat % NRT
    c0 = J + 1 + ct * 64
    r0 = J + 1 + rt * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    wbase = W + pid * WPITCH
    tau = tl.load(TAU + pid * K + J)
    cs = tl.load(CSCA + pid)
    # coef = (tau^2/2)(v^T omega): evaluate as tau*(tau*cs), NOT tau*tau*cs.
    # When the trailing subdiagonal deflates (sigma ~ 1e-10 on clustered or
    # strongly rank-deficient spectra), tau = 2/vnorm2 ~ 1e20 and tau*tau
    # overflows fp32 (2e40 > 3.4e38) even though the final coefficient is
    # O(1).  The regrouped form never squares tau.
    coef = 0.5 * tau * (tau * cs)
    om_r = tl.load(ACC + pid * APITCH + r0 + lr)
    v_r = tl.load(V + pid * RS + r0 + lr)
    om_c = tl.load(ACC + pid * APITCH + c0 + lc)
    v_c = tl.load(V + pid * RS + c0 + lc)
    w_r = tau * om_r - coef * v_r
    w_c = tau * om_c - coef * v_c
    ptrs = wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :]
    tile = tl.load(ptrs)
    tile = tile - tl.reshape(v_c, (64, 1)) * tl.reshape(w_r, (1, 64))
    tile = tile - tl.reshape(w_c, (64, 1)) * tl.reshape(v_r, (1, 64))
    tl.store(ptrs, tile)


def _herm_tridiag_workspace(
    device, batch_count, k, work_dtype, atol_tensor, rtol_tensor, blocked
):
    # All buffers the barrier-free tridiagonalization touches, allocated up
    # front so the launch sequence is a pure function of the workspace and
    # can be graph-captured (capture requires stable buffer addresses).
    # Panel buffers (pv/pw/pscr) are allocated ONLY for the blocked path --
    # fp64 and k < _HERM_TRIDIAG_BLOCKED_MIN_K never take it.
    kp = triton.cdiv(k, 64) * 64 + 64
    rs = kp  # hermitian input is square: rows == k
    nb_p = _HERM_TRIDIAG_PANEL
    slot = (3 + 2 * nb_p) * _HERM_TRIDIAG_SCRATCH_LINE
    ws = {
        "device_index": device.index,
        "kp": kp,
        "rs": rs,
        "wpitch": kp * rs,
        "apitch": kp + 64,
        "pvp": kp * nb_p,
        "spitch": 2 * slot,
        "nb_p": nb_p,
        # Zero-init (not empty): the mat/apply kernels read the padding
        # without masks and rely on it staying zero.
        "w_buf": torch.zeros((batch_count, kp, rs), dtype=work_dtype, device=device),
        # Only the CURRENT column's reflector is alive at any time, so a
        # single row per batch suffices (kernels zero the prefix below J+1
        # explicitly when they store the reflector).  Zero-init: the prefix
        # of column 0's row must read as zeros.
        "v_buf": torch.zeros((batch_count, rs), dtype=work_dtype, device=device),
        # Panel reflector/correction buffers (blocked path only).
        # Zero-init: stale slots q >= p must be finite because the kernels
        # multiply unmasked tile loads against tl.where-zeroed values.
        "pv": (
            torch.zeros((batch_count, kp, nb_p), dtype=work_dtype, device=device)
            if blocked
            else None
        ),
        "pw": (
            torch.zeros((batch_count, kp, nb_p), dtype=work_dtype, device=device)
            if blocked
            else None
        ),
        # Reduction scratch for the blocked path: TWO rotating slots (column
        # parity), entries padded to 128B lines so concurrent atomics do not
        # serialize in the L2.  Zero-init: pcol accumulates with atomics and
        # pfin re-zeroes the OTHER slot (next column's) after use, with the
        # last column re-zeroing slot 0, so every call / graph replay starts
        # from zeros.  Two slots are sufficient because the zeroing targets
        # the slot parity nobody in the current launch is reading.
        "pscr": (
            torch.zeros((batch_count, 2 * slot), dtype=work_dtype, device=device)
            if blocked
            else None
        ),
        "diag": torch.empty((batch_count, k), dtype=work_dtype, device=device),
        "offdiag": torch.empty((batch_count, k), dtype=work_dtype, device=device),
        "taul": torch.empty((batch_count, k), dtype=work_dtype, device=device),
        "acc": torch.zeros((batch_count, kp + 64), dtype=work_dtype, device=device),
        "csca": torch.zeros((batch_count,), dtype=work_dtype, device=device),
        "e2_hi": torch.empty((batch_count, k), dtype=torch.float32, device=device),
        "e2_lo": torch.empty((batch_count, k), dtype=torch.float32, device=device),
        "tol2": torch.empty((batch_count,), dtype=torch.float32, device=device),
        "staging": torch.empty((batch_count, k, k), dtype=work_dtype, device=device),
        "atol": torch.empty((batch_count,), dtype=atol_tensor.dtype, device=device),
        "rtol": torch.empty((batch_count,), dtype=rtol_tensor.dtype, device=device),
        # Staged UNSCALED per-batch scale; the pad-init kernel divides the
        # matrix by it in-kernel and _matrix_rank_scale_tol_kernel divides
        # atol by it into atol_s (both inside the captured sequence).
        "scale": torch.empty((batch_count,), dtype=work_dtype, device=device),
        "atol_s": torch.empty(
            (batch_count,),
            dtype=torch.promote_types(atol_tensor.dtype, work_dtype),
            device=device,
        ),
        "rank": torch.empty((batch_count,), dtype=torch.int64, device=device),
    }
    return ws


def _herm_tridiag_run(ws, k, batch_count, ds32):
    # The launch sequence of the barrier-free one-sided Householder
    # tridiagonalization: three kernels per column (reflector step /
    # symmetric matvec / rank-2 apply) plus the Sturm tail.  Kernel launch
    # boundaries are the ONLY cross-program ordering -- no software grid
    # barrier, so the launch never depends on SM count or block
    # co-residency.  Pure function of the workspace so it can be
    # graph-captured.
    rs = ws["rs"]
    wpitch, apitch = ws["wpitch"], ws["apitch"]
    w_buf, v_buf = ws["w_buf"], ws["v_buf"]
    diag, offdiag, taul = ws["diag"], ws["offdiag"], ws["taul"]
    acc, csca = ws["acc"], ws["csca"]
    _matrix_rank_scale_tol_kernel[(batch_count,)](
        ws["atol"], ws["scale"], ws["atol_s"], num_warps=1
    )
    _matrix_rank_herm_tridiag_pad_init_kernel[(batch_count, triton.cdiv(k, 64))](
        ws["staging"],
        w_buf,
        ws["scale"],
        K=k,
        RS=rs,
        WPITCH=wpitch,
        num_warps=4,
    )
    for j in range(k):
        _matrix_rank_herm_tridiag_step_kernel[(batch_count,)](
            w_buf,
            v_buf,
            diag,
            offdiag,
            taul,
            acc,
            csca,
            j,
            k,
            rs,
            wpitch,
            apitch,
            num_warps=4,
            num_stages=1,
        )
        if j + 1 < k:
            nrt = triton.cdiv(k - 1 - j, 64)
            _matrix_rank_herm_tridiag_mat_kernel[(batch_count, nrt * nrt)](
                w_buf,
                v_buf,
                acc,
                csca,
                j,
                rs,
                wpitch,
                apitch,
                nrt,
                num_warps=4,
                num_stages=1,
            )
            _matrix_rank_herm_tridiag_apply_kernel[(batch_count, nrt * nrt)](
                w_buf,
                v_buf,
                taul,
                acc,
                csca,
                j,
                k,
                rs,
                wpitch,
                apitch,
                nrt,
                num_warps=4,
                num_stages=1,
            )
    _herm_tridiag_sturm_tail(ws, k, batch_count, ds32)


def _herm_tridiag_blocked_run(ws, k, batch_count, ds32):
    # Blocked WY variant for large matrices: columns are factored in panels
    # of NB_P.  Inside a panel the trailing matrix is NOT updated; each
    # column step is three launches (deferred column update + reflector /
    # symmetric GEMV against the STALE trailing block / corrected-w finish
    # storing the panel V/W columns), and the panel's reflection is applied
    # afterwards as ONE symmetric rank-2k update built with tl.dot.  This
    # replaces the unblocked path's per-column trailing rank-2 read+write
    # (BLAS2, bandwidth-bound) with a per-panel BLAS3 pass.  Kernel launch
    # boundaries remain the ONLY cross-program ordering.  Pure function of
    # the workspace so it can be graph-captured.
    rs = ws["rs"]
    wpitch, apitch, pvp = ws["wpitch"], ws["apitch"], ws["pvp"]
    nb_p = ws["nb_p"]
    sl = _HERM_TRIDIAG_SCRATCH_LINE
    spitch = ws["spitch"]
    w_buf, v_buf = ws["w_buf"], ws["v_buf"]
    pv, pw, pscr = ws["pv"], ws["pw"], ws["pscr"]
    diag, offdiag = ws["diag"], ws["offdiag"]
    acc, csca = ws["acc"], ws["csca"]
    _matrix_rank_scale_tol_kernel[(batch_count,)](
        ws["atol"], ws["scale"], ws["atol_s"], num_warps=1
    )
    _matrix_rank_herm_tridiag_pad_init_kernel[(batch_count, triton.cdiv(k, 64))](
        ws["staging"],
        w_buf,
        ws["scale"],
        K=k,
        RS=rs,
        WPITCH=wpitch,
        num_warps=4,
    )
    j0 = 0
    while j0 < k - 1:
        pe = min(nb_p, k - 1 - j0)
        for p in range(pe):
            j = j0 + p
            # pcol covers the ACC-zero range (APITCH // 64 chunks from
            # J // 64); programs past the real row count only zero ACC.
            nc = apitch // 64 - j // 64
            _matrix_rank_herm_tridiag_pcol_kernel[(batch_count, nc)](
                w_buf,
                v_buf,
                pv,
                pw,
                acc,
                csca,
                pscr,
                j,
                p,
                k,
                rs,
                wpitch,
                apitch,
                pvp,
                spitch,
                NB_P=nb_p,
                SL=sl,
                num_warps=4,
                num_stages=1,
            )
            nrt = triton.cdiv(k - 1 - j, 64)
            _matrix_rank_herm_tridiag_pmat_kernel[(batch_count, nrt * nrt)](
                w_buf,
                v_buf,
                diag,
                offdiag,
                acc,
                csca,
                pscr,
                j,
                k,
                rs,
                wpitch,
                apitch,
                spitch,
                nrt,
                NB_P=nb_p,
                SL=sl,
                num_warps=8,
                num_stages=1,
            )
            nc2 = triton.cdiv(k, 64) - (j + 1) // 64
            _matrix_rank_herm_tridiag_pfin_kernel[(batch_count, nc2)](
                v_buf,
                pv,
                pw,
                acc,
                csca,
                pscr,
                j,
                p,
                k,
                rs,
                apitch,
                pvp,
                spitch,
                NB_P=nb_p,
                SL=sl,
                num_warps=4,
                num_stages=1,
            )
        t0 = j0 + pe
        if t0 < k:
            nt = triton.cdiv(k - t0, 64)
            _matrix_rank_herm_tridiag_rank2k_kernel[(batch_count, nt * nt)](
                w_buf,
                pv,
                pw,
                t0,
                pe,
                rs,
                wpitch,
                pvp,
                nt,
                NB_P=nb_p,
                num_warps=4,
                num_stages=1,
            )
        j0 += pe
    _matrix_rank_herm_tridiag_last_diag_kernel[(batch_count,)](
        w_buf, diag, k, rs, wpitch
    )
    _herm_tridiag_sturm_tail(ws, k, batch_count, ds32)


def _herm_tridiag_sturm_tail(ws, k, batch_count, ds32):
    diag, offdiag = ws["diag"], ws["offdiag"]
    if ds32:
        _matrix_rank_sturm32_tridiag_bracket_kernel[(batch_count,)](
            diag,
            offdiag,
            ws["atol_s"],
            ws["rtol"],
            ws["rank"],
            ws["tol2"],
            k,
            BLOCK=triton.next_power_of_2(k),
            BISECT_ITERS=32,
            num_warps=1,
        )
        _matrix_rank_sturm32_tridiag_final_kernel[(batch_count,)](
            diag,
            offdiag,
            ws["tol2"],
            ws["rank"],
            k,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return
    # Bisection refines sigma_max inside the Gershgorin bracket; 32
    # iterations give ~1e-10 relative convergence, enough for fp32 data
    # (24-bit mantissa) but not for fp64 thresholds -- use 64 there.
    bisect_iters = 64 if ws["diag"].dtype == torch.float64 else 32
    _matrix_rank_sturm_rank_kernel[(batch_count,)](
        diag,
        offdiag,
        ws["atol_s"],
        ws["rtol"],
        ws["rank"],
        ws["e2_hi"],
        ws["e2_lo"],
        K=k,
        BLOCK_K=triton.next_power_of_2(k),
        BISECT_ITERS=bisect_iters,
        num_warps=1,
        GK=False,
        enable_fp_fusion=False,
    )


def _launch_herm_tridiag_rank(
    matrix, atol_tensor, rtol_tensor, scale, out, k, batch_count, input
):
    # Non-iterative hermitian path: symmetrize into a PADDED work matrix,
    # tridiagonalize with per-column barrier-free Householder steps, then
    # count eigenvalues outside [-tol, tol] with Sturm sequences.  The O(k)
    # launch sequence is graph-captured per shape where the device supports
    # it (see _mr_graph_cached); otherwise it runs as direct launches.
    device = input.device
    work_dtype = input.dtype
    # Devices without native FP64 take the pure-FP32 double-single Sturm
    # tail; float64 input never reaches here on them (entry fail-fast).
    ds32 = work_dtype == torch.float32 and not _native_fp64_supported()
    # Large fp32 matrices take the blocked WY panel factorization (BLAS3
    # trailing updates).  Smaller sizes stay on the per-column unblocked
    # path (measured crossover between 512 and 1024), and fp64 stays
    # unblocked too: with no fast fp64 tl.dot on current hardware the
    # rank-2k falls back to an outer-product loop and the panel algebra
    # costs more than it saves (measured: blocked fp64 loses at every
    # size).  The blocked path additionally requires passing a one-time
    # known-answer self-test on this backend (see _blocked_tridiag_ok):
    # a backend miscompile anywhere in the panel pipeline produces
    # silently wrong ranks, not errors.
    blocked = (
        k >= _HERM_TRIDIAG_BLOCKED_MIN_K
        and work_dtype == torch.float32
        and _blocked_tridiag_ok(device)
    )

    def copy_in(ws):
        ws["staging"].copy_(matrix)
        ws["atol"].copy_(atol_tensor)
        ws["rtol"].copy_(rtol_tensor)
        ws["scale"].copy_(scale)

    def copy_out(ws):
        out.copy_(ws["rank"].reshape(out.shape))

    _mr_graph_cached(
        # ds32 is part of the key: the same shape must not reuse a graph
        # captured with the native-FP64 Sturm tail after the capability is
        # switched off (e.g. by tests monkeypatching support_fp64).
        # blocked is in the key for the same reason (probe monkeypatching
        # in tests; in production the verdict is fixed per device).
        ("herm_tridiag", k, batch_count, work_dtype, ds32, blocked, device.index),
        device,
        lambda: _herm_tridiag_workspace(
            device, batch_count, k, work_dtype, atol_tensor, rtol_tensor, blocked
        ),
        copy_in,
        lambda ws: (
            _herm_tridiag_blocked_run(ws, k, batch_count, ds32)
            if blocked
            else _herm_tridiag_run(ws, k, batch_count, ds32)
        ),
        copy_out,
    )


@libentry()
@triton.jit
def _matrix_rank_bidiag_bf_init_kernel(
    A, W, SCALE, M, N, K, ROWS, RS, WPITCH, TALL: tl.constexpr
):
    # Barrier-free bidiagonalization init: copy the input into the padded
    # tall work matrix W (KP x RS, zero-initialized on allocation),
    # dividing by the per-batch scale in-kernel (the staged input is
    # UNSCALED; scaling here avoids an O(M*N) temporary outside the
    # graph): W[c, r] = A[r, c] for tall input (M >= N), W[c, r] = A[c, r]
    # for wide input (i.e. the tall orientation of A).  Only the K x ROWS
    # corner is written; the padding stays zero so the step/mat/apply
    # kernels can read tiles without masks.
    b = tl.program_id(0)
    c0 = tl.program_id(1) * 64
    lc = tl.arange(0, 64)
    a_base = A + b * M * N
    wbase = W + b * WPITCH
    s = tl.load(SCALE + b)
    for rb in tl.range(0, (RS - 64) // 64):
        rr = rb * 64 + tl.arange(0, 64)
        cc = c0 + lc
        mask = (rr < ROWS)[:, None] & (cc < K)[None, :]
        if TALL:
            at = tl.load(a_base + rr[:, None] * N + cc[None, :], mask=mask, other=0.0)
        else:
            at = tl.load(a_base + cc[None, :] * N + rr[:, None], mask=mask, other=0.0)
        tl.store(wbase + cc[None, :] * RS + rr[:, None], at / s, mask=mask)


@libentry()
@triton.jit
def _matrix_rank_bidiag_lstep_kernel(W, V, D, TAU, ACC, J, K, RS, WPITCH, APITCH):
    # Barrier-free Golub-Kahan bidiagonalization, left reflector step J:
    # column J on rows >= J defines the reflector; D[J] = +/-sigma.  Single
    # program per matrix: stores the (masked) reflector vector
    # provisionally while accumulating the norm, then overwrites the pivot
    # element with x0 - alpha (stores only, no read-back), and zeroes the
    # ACC slots the left matvec's atomics add into.
    pid = tl.program_id(0)
    wbase = W + pid * WPITCH
    lr = tl.arange(0, 64)
    dtype = W.dtype.element_ty
    ssq = tl.zeros((), dtype=dtype)
    x0 = tl.zeros((), dtype=dtype)
    for rb in tl.range(J // 64, RS // 64):
        r0 = rb * 64
        ch = tl.load(wbase + J * RS + r0 + lr)
        ch = ch * ((r0 + lr) >= J).to(dtype)
        tl.store(V + pid * RS + r0 + lr, ch)
        ssq += tl.sum(ch * ch, axis=0)
        x0 += tl.sum(ch * ((r0 + lr > J - 1) & (r0 + lr < J + 1)).to(dtype), axis=0)
    sigma = tl.sqrt(ssq)
    alpha = tl.where(x0 >= 0.0, -sigma, sigma)
    vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
    tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
    tl.store(V + pid * RS + J, x0 - alpha)
    tl.store(D + pid * K + J, alpha)
    tl.store(TAU + pid * K + J, tau)
    for cb in tl.range(0, APITCH // 64):
        tl.store(ACC + pid * APITCH + cb * 64 + lr, tl.zeros((64,), dtype=dtype))


@libentry()
@triton.jit
def _matrix_rank_bidiag_lmat_kernel(W, V, ACC, J, RS, WPITCH, APITCH, NRC, RB0):
    # Left matvec omega[c] = sum_r W[c, r] * v[r] over trailing columns
    # c > J, multi-program over (col tile, row strip) with atomic
    # accumulation.  Row tiles below RB0 = J // 64 are skipped: v is zero
    # above row J, so they contribute nothing to the reduction.
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NRC
    rc = flat % NRC
    c0 = J + 1 + ct * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    r0 = (RB0 + rc) * 64
    wbase = W + pid * WPITCH
    tile = tl.load(wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :])
    v2p = tl.load(V + pid * RS + r0 + lr)
    part = tl.sum(tile * v2p[None, :], axis=1)
    tl.atomic_add(ACC + pid * APITCH + c0 + lc, part)


@libentry()
@triton.jit
def _matrix_rank_bidiag_lapply_kernel(
    W, V, TAU, ACC, J, K, RS, WPITCH, APITCH, NRC, RB0
):
    # Left trailing rank-1 update W[c, r] -= tau * omega[c] * v[r].
    # Row tiles below RB0 are skipped: v is zero there, so the update
    # would be a no-op (and the load+store is pure waste).
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NRC
    rc = flat % NRC
    c0 = J + 1 + ct * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    r0 = (RB0 + rc) * 64
    wbase = W + pid * WPITCH
    tau = tl.load(TAU + pid * K + J)
    w = tl.load(ACC + pid * APITCH + c0 + lc) * tau
    v2p = tl.load(V + pid * RS + r0 + lr)
    tile = tl.load(wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :])
    tile = tile - tl.reshape(w, (64, 1)) * tl.reshape(v2p, (1, 64))
    tl.store(wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :], tile)


@libentry()
@triton.jit
def _matrix_rank_bidiag_rstep_kernel(W, U, E, TAU, ACC, J, K, RS, WPITCH, APITCH):
    # Right reflector step J: row J on columns > J defines the reflector;
    # E[J] = +/-sigma.  Columns <= J contribute nothing (u has support on
    # c > J), so the pass starts at the (J+1)'s tile.
    pid = tl.program_id(0)
    wbase = W + pid * WPITCH
    lc = tl.arange(0, 64)
    dtype = W.dtype.element_ty
    ssq = tl.zeros((), dtype=dtype)
    x0 = tl.zeros((), dtype=dtype)
    for cb in tl.range((J + 1) // 64, (K + 63) // 64):
        c0 = cb * 64
        ch = tl.load(wbase + (c0 + lc) * RS + J, mask=(c0 + lc) < K, other=0.0)
        ch = ch * ((c0 + lc) > J).to(dtype)
        tl.store(U + pid * K + c0 + lc, ch, mask=(c0 + lc) < K)
        ssq += tl.sum(ch * ch, axis=0)
        x0 += tl.sum(ch * ((c0 + lc > J) & (c0 + lc < J + 2)).to(dtype), axis=0)
    sigma = tl.sqrt(ssq)
    alpha = tl.where(x0 >= 0.0, -sigma, sigma)
    vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
    tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
    tl.store(U + pid * K + (J + 1), x0 - alpha)
    tl.store(E + pid * K + J, alpha)
    tl.store(TAU + pid * K + J, tau)
    for cb in tl.range(0, APITCH // 64):
        tl.store(ACC + pid * APITCH + cb * 64 + lc, tl.zeros((64,), dtype=dtype))


@libentry()
@triton.jit
def _matrix_rank_bidiag_rmat_kernel(W, U, ACC, J, K, RS, WPITCH, APITCH, NCC, CC0):
    # Right matvec omega[r] = sum_c W[c, r] * u[c] over trailing rows
    # r > J, multi-program over (row tile, col strip) with atomic
    # accumulation.  Column tiles below CC0 = (J+1) // 64 are skipped: u
    # is zero on columns <= J.  The J+1-aligned row grid is not
    # 64-aligned, so the last tile straddles the RS row pitch; mask it
    # (the wrapped-around address is the NEXT column's row 0 -- an
    # out-of-bounds read/write otherwise).
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    rt = flat // NCC
    cc = flat % NCC
    r0 = J + 1 + rt * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    c0 = (CC0 + cc) * 64
    wbase = W + pid * WPITCH
    rmask = (r0 + lr) < RS
    tile = tl.load(
        wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :],
        mask=rmask[None, :],
        other=0.0,
    )
    up = tl.load(U + pid * K + c0 + lc, mask=(c0 + lc) < K, other=0.0)
    part = tl.sum(tl.trans(tile) * up[None, :], axis=1)
    tl.atomic_add(ACC + pid * APITCH + r0 + lr, part)


@libentry()
@triton.jit
def _matrix_rank_bidiag_rapply_kernel(W, U, TAU, ACC, J, K, RS, WPITCH, APITCH, NTR):
    # Right trailing rank-1 update W[c, r] -= u[c] * (tau * omega[r]),
    # one program per (col tile, row strip).  Same RS-straddle mask as
    # the right matvec.
    pid = tl.program_id(0)
    flat = tl.program_id(1)
    ct = flat // NTR
    rc = flat % NTR
    c0 = J + 1 + ct * 64
    r0 = J + 1 + rc * 64
    lc = tl.arange(0, 64)
    lr = tl.arange(0, 64)
    wbase = W + pid * WPITCH
    tau = tl.load(TAU + pid * K + J)
    rmask = (r0 + lr) < RS
    wu = tl.load(ACC + pid * APITCH + r0 + lr, mask=rmask, other=0.0) * tau
    up = tl.load(U + pid * K + c0 + lc, mask=(c0 + lc) < K, other=0.0)
    tile = tl.load(
        wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :],
        mask=rmask[None, :],
        other=0.0,
    )
    tile = tile - tl.reshape(up, (64, 1)) * tl.reshape(wu, (1, 64))
    tl.store(
        wbase + (c0 + lc)[:, None] * RS + (r0 + lr)[None, :],
        tile,
        mask=rmask[None, :],
    )


def _bidiag_workspace(
    device, batch_count, m, n, k, rows, work_dtype, atol_tensor, rtol_tensor, ds32
):
    # All buffers the barrier-free bidiagonalization touches, allocated up
    # front so the launch sequence is a pure function of the workspace and
    # can be graph-captured (capture requires stable buffer addresses).
    kp = triton.cdiv(k, 64) * 64 + 64  # one slack tile: tile accesses are
    rs = triton.cdiv(rows, 64) * 64 + 64  # mostly unmasked and must not
    # run off the allocation
    ws = {
        "device_index": device.index,
        "kp": kp,
        "rs": rs,
        "wpitch": kp * rs,
        "w_buf": torch.zeros((batch_count, kp, rs), dtype=work_dtype, device=device),
        # Only the CURRENT step's left/right reflector is alive at any
        # time, so one row per batch suffices for each (the step kernels
        # zero the prefix explicitly when storing the reflector).
        "v_buf": torch.zeros((batch_count, rs), dtype=work_dtype, device=device),
        "u_buf": torch.zeros((batch_count, k), dtype=work_dtype, device=device),
        "diag": torch.empty((batch_count, k), dtype=work_dtype, device=device),
        "offdiag": torch.empty((batch_count, k), dtype=work_dtype, device=device),
        "taul": torch.empty((batch_count, k), dtype=work_dtype, device=device),
        "taur": torch.empty((batch_count, k), dtype=work_dtype, device=device),
        # One extra tile of slack on the right accumulation: the right
        # matvec's atomic index r0 + 63 can reach rs (pad-row tiles are
        # updated, never read).
        "acc": torch.zeros((batch_count, kp + 64), dtype=work_dtype, device=device),
        "uacc": torch.zeros((batch_count, rs + 64), dtype=work_dtype, device=device),
        "staging": torch.empty((batch_count, m, n), dtype=work_dtype, device=device),
        "atol": torch.empty((batch_count,), dtype=atol_tensor.dtype, device=device),
        "rtol": torch.empty((batch_count,), dtype=rtol_tensor.dtype, device=device),
        # Staged UNSCALED per-batch scale; the init kernel divides the
        # matrix by it in-kernel and _matrix_rank_scale_tol_kernel divides
        # atol by it into atol_s (both inside the captured sequence).
        "scale": torch.empty((batch_count,), dtype=work_dtype, device=device),
        "atol_s": torch.empty(
            (batch_count,),
            dtype=torch.promote_types(atol_tensor.dtype, work_dtype),
            device=device,
        ),
        "rank": torch.empty((batch_count,), dtype=torch.int64, device=device),
    }
    if ds32:
        # B^T B tridiagonal (size k, fp32) built from the raw bidiagonal for
        # the pure-FP32 bracketing pass; the decisive count re-reads the raw
        # bidiagonal in double-single arithmetic.
        ws["gk_dd"] = torch.empty((batch_count, k), dtype=torch.float32, device=device)
        ws["gk_ee"] = torch.empty((batch_count, k), dtype=torch.float32, device=device)
        ws["tol2"] = torch.empty((batch_count,), dtype=torch.float32, device=device)
    else:
        ws["gk_diag"] = torch.empty(
            (batch_count, 2 * k), dtype=torch.float64, device=device
        )
        # Keep the per-batch stride at 2K (one slack entry): the Sturm kernel
        # indexes the off-diagonal with stride K == 2k.
        ws["gk_off"] = torch.empty(
            (batch_count, 2 * k), dtype=work_dtype, device=device
        )
        ws["e2_hi"] = torch.empty(
            (batch_count, 2 * k), dtype=torch.float32, device=device
        )
        ws["e2_lo"] = torch.empty(
            (batch_count, 2 * k), dtype=torch.float32, device=device
        )
    return ws


def _bidiag_run(ws, m, n, k, rows, batch_count, ds32):
    # The launch sequence of the barrier-free Golub-Kahan
    # bidiagonalization: six kernels per column (left reflector / matvec /
    # apply, then right reflector / matvec / apply) plus the Sturm tail.
    # Kernel launch boundaries are the ONLY cross-program ordering -- no
    # software grid barrier, so the launch never depends on SM count or
    # block co-residency.  Pure function of the workspace so it can be
    # graph-captured.
    kp, rs = ws["kp"], ws["rs"]
    wpitch = ws["wpitch"]
    w_buf, v_buf, u_buf = ws["w_buf"], ws["v_buf"], ws["u_buf"]
    diag, offdiag = ws["diag"], ws["offdiag"]
    taul, taur = ws["taul"], ws["taur"]
    acc, uacc = ws["acc"], ws["uacc"]
    _matrix_rank_scale_tol_kernel[(batch_count,)](
        ws["atol"], ws["scale"], ws["atol_s"], num_warps=1
    )
    _matrix_rank_bidiag_bf_init_kernel[(batch_count, triton.cdiv(k, 64))](
        ws["staging"],
        w_buf,
        ws["scale"],
        m,
        n,
        k,
        rows,
        rs,
        wpitch,
        TALL=m >= n,
        num_warps=4,
        num_stages=1,
    )
    nrc_full = (rs - 64) // 64
    ncc_full = (kp - 64) // 64
    for j in range(k):
        ntl = triton.cdiv(k - 1 - j, 64)
        # Trim the matvec/apply grids to the trailing block: v/u are zero
        # above/left of j, so those tiles are pure waste (halves the
        # average per-step work of the four big kernels).
        rb0 = j // 64
        nrct = nrc_full - rb0
        _matrix_rank_bidiag_lstep_kernel[(batch_count,)](
            w_buf,
            v_buf,
            diag,
            taul,
            acc,
            j,
            k,
            rs,
            wpitch,
            kp,
            num_warps=4,
            num_stages=1,
        )
        if ntl > 0:
            _matrix_rank_bidiag_lmat_kernel[(batch_count, ntl * nrct)](
                w_buf,
                v_buf,
                acc,
                j,
                rs,
                wpitch,
                kp,
                nrct,
                rb0,
                num_warps=4,
                num_stages=1,
            )
            _matrix_rank_bidiag_lapply_kernel[(batch_count, ntl * nrct)](
                w_buf,
                v_buf,
                taul,
                acc,
                j,
                k,
                rs,
                wpitch,
                kp,
                nrct,
                rb0,
                num_warps=4,
                num_stages=1,
            )
        if j + 1 < k:
            ntr = triton.cdiv(rs - 1 - j, 64)
            cc0 = (j + 1) // 64
            ncct = ncc_full - cc0
            _matrix_rank_bidiag_rstep_kernel[(batch_count,)](
                w_buf,
                u_buf,
                offdiag,
                taur,
                uacc,
                j,
                k,
                rs,
                wpitch,
                rs,
                num_warps=4,
                num_stages=1,
            )
            _matrix_rank_bidiag_rmat_kernel[(batch_count, ntr * ncct)](
                w_buf,
                u_buf,
                uacc,
                j,
                k,
                rs,
                wpitch,
                rs,
                ncct,
                cc0,
                num_warps=4,
                num_stages=1,
            )
            _matrix_rank_bidiag_rapply_kernel[(batch_count, ntl * ntr)](
                w_buf,
                u_buf,
                taur,
                uacc,
                j,
                k,
                rs,
                wpitch,
                rs,
                ntr,
                num_warps=4,
                num_stages=1,
            )
    if ds32:
        _matrix_rank_bidiag32_to_tridiag_kernel[(batch_count,)](
            diag,
            offdiag,
            ws["gk_dd"],
            ws["gk_ee"],
            k,
            BLOCK=triton.next_power_of_2(k),
            num_warps=1,
        )
        _matrix_rank_sturm32_bidiag_bracket_kernel[(batch_count,)](
            ws["gk_dd"],
            ws["gk_ee"],
            ws["atol_s"],
            ws["rtol"],
            ws["rank"],
            ws["tol2"],
            k,
            BLOCK=triton.next_power_of_2(k),
            BISECT_ITERS=32,
            num_warps=1,
        )
        _matrix_rank_sturm32_bidiag_final_kernel[(batch_count,)](
            diag,
            offdiag,
            ws["tol2"],
            ws["rank"],
            k,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return
    _matrix_rank_gk_init_kernel[(batch_count,)](
        diag,
        offdiag,
        ws["gk_diag"],
        ws["gk_off"],
        K=k,
        BLOCK=triton.next_power_of_2(2 * k),
        num_warps=4,
    )
    # See the herm tail: fp64 thresholds need more bisection iterations
    # than fp32 to converge the sigma_max bracket to mantissa precision.
    bisect_iters = 64 if ws["gk_diag"].dtype == torch.float64 else 32
    _matrix_rank_sturm_rank_kernel[(batch_count,)](
        ws["gk_diag"],
        ws["gk_off"],
        ws["atol_s"],
        ws["rtol"],
        ws["rank"],
        ws["e2_hi"],
        ws["e2_lo"],
        K=2 * k,
        BLOCK_K=triton.next_power_of_2(2 * k),
        BISECT_ITERS=bisect_iters,
        num_warps=1,
        GK=True,
        enable_fp_fusion=False,
    )


def _launch_bidiag_rank(
    matrix, atol_tensor, rtol_tensor, scale, out, m, n, k, rows, batch_count, input
):
    # Non-hermitian path past the fused-Jacobi size limits: copy into a
    # PADDED tall work matrix, reduce to bidiagonal form with per-column
    # barrier-free Golub-Kahan steps, then count singular values above the
    # tolerance with Sturm sequences.  The O(k) launch sequence is
    # graph-captured per shape where the device supports it (see
    # _mr_graph_cached); otherwise it runs as direct launches.
    device = input.device
    work_dtype = input.dtype
    # Devices without native FP64 take the pure-FP32 double-single Sturm
    # tail; float64 input never reaches here on them (entry fail-fast).
    ds32 = work_dtype == torch.float32 and not _native_fp64_supported()

    def copy_in(ws):
        ws["staging"].copy_(matrix)
        ws["atol"].copy_(atol_tensor)
        ws["rtol"].copy_(rtol_tensor)
        ws["scale"].copy_(scale)

    def copy_out(ws):
        out.copy_(ws["rank"].reshape(out.shape))

    _mr_graph_cached(
        # ds32 is part of the key, same reason as the herm path above.
        ("bidiag", m, n, batch_count, work_dtype, ds32, device.index),
        device,
        lambda: _bidiag_workspace(
            device,
            batch_count,
            m,
            n,
            k,
            rows,
            work_dtype,
            atol_tensor,
            rtol_tensor,
            ds32,
        ),
        copy_in,
        lambda ws: _bidiag_run(ws, m, n, k, rows, batch_count, ds32),
        copy_out,
    )


def _launch_matrix_rank(
    input,
    atol,
    rtol,
    hermitian,
):
    # atol/rtol are tolerance descriptors: Python floats on the scalar fast
    # path, device tensors (batch shape) otherwise.
    output_shape = input.shape[:-2]
    m, n = input.shape[-2:]
    k = min(m, n)
    rows = max(m, n)
    is_fp64 = input.dtype == torch.float64
    herm_tridiag = hermitian and k >= (
        _HERM_TRIDIAG_MIN_K_FP64 if is_fp64 else _HERM_TRIDIAG_MIN_K_FP32
    )
    # Single-program fused Jacobi eligibility (no grid synchronization by
    # construction).  Everything beyond this goes to the barrier-free
    # decompositions (hermitian tridiagonalization / bidiagonalization),
    # whose per-column kernels synchronize at launch boundaries.
    fused_eligible = rows <= _FUSED_JACOBI_MAX_ROWS and (
        (
            is_fp64
            and k <= _FUSED_JACOBI_MAX_K_FP64
            and (k <= 16 or rows <= _FUSED_JACOBI_WIDE_MAX_ROWS)
        )
        or (
            not is_fp64
            and k <= _FUSED_JACOBI_MAX_K
            and (k <= 32 or rows <= _FUSED_JACOBI_WIDE_MAX_ROWS)
        )
    )
    use_bidiag = (not hermitian) and not fused_eligible

    batch_count = input.numel() // (m * n)
    matrix = input.contiguous().reshape(batch_count, m, n)
    # Per-batch scale normalization.  The Householder algebra squares the
    # matrix scale (w = A·v is O(sigma^2)), so fp32 inputs beyond ~1e19
    # overflow and below ~1e-19 underflow even if every norm computation
    # were internally scaled -- scaling must happen to the matrix itself.
    # Rank is invariant: max(atol, rtol*sigma_max)/s ==
    # max(atol/s, rtol*(sigma_max/s)), so shrinking atol by the same
    # factor preserves the exact threshold semantics (rtol is relative
    # and needs no change).  For hermitian input the scale ignores the
    # strict upper triangle, which torch never reads (it may hold
    # garbage).
    visible = torch.tril(matrix) if hermitian else matrix
    scale = visible.abs().amax(dim=(-2, -1))
    # Zero scale -> 1 with a dedicated kernel (NOT torch.clamp_min: the
    # generic clamp casts through fp32 under use_gems() dispatch, flushing
    # any fp64 floor to zero; a zero scale then turns 0/0 into NaN and the
    # Sturm sign counts collapse).  One kernel, identical on the direct
    # and dispatched paths.
    with torch_device_fn.device(input.device):
        _matrix_rank_safe_scale_kernel[(batch_count,)](scale, num_warps=1)
    # small_path must mirror the dispatch order below exactly: hermitian
    # matrices eligible for BOTH fused Jacobi and tridiagonalization take
    # the tridiag branch (use_bidiag and fused are mutually exclusive, so
    # no guard is needed there).
    small_path = (k <= 2) or (fused_eligible and not herm_tridiag)
    # Scalar tolerances are passed as tl.float64 kernel arguments; on
    # devices without native FP64 the f64 kernel code is off-limits even
    # under a constexpr guard that never fires, so fall back to the pointer
    # path there. Some Triton backends still lower scalar arguments to fp32,
    # so a value that is not exactly representable in fp32 also uses an fp64
    # pointer instead.
    scalar_tol = (
        not isinstance(atol, torch.Tensor)
        and not isinstance(rtol, torch.Tensor)
        and _native_fp64_supported()
        and _is_exact_float32(atol)
        and _is_exact_float32(rtol)
    )
    if small_path:
        # Small single-kernel paths take RAW tolerances and the scale, and
        # do both the matrix scaling and the atol/scale division in-kernel
        # -- no torch.full / div elementwise launches.  Scalars go straight
        # to kernel arguments; tensors are read through pointers.
        if scalar_tol:
            atol_ptr = rtol_ptr = scale  # dummies, unused under SCALAR_TOL
            atol_s, rtol_s = float(atol), float(rtol)
        else:
            atol_ptr = _materialize_tolerance(atol, output_shape, input)
            rtol_ptr = _materialize_tolerance(rtol, output_shape, input)
            atol_s = rtol_s = 0.0
    else:
        # Large graph-captured paths keep the staged-tensor convention
        # (replay must re-read inputs from workspace buffers).  The matrix
        # and atol are staged UNSCALED and divided by the staged scale
        # INSIDE the captured sequence (pad/bidiag init kernels and
        # _matrix_rank_scale_tol_kernel), so no O(k^2) scaled temporary is
        # materialized outside the graph.  The workspace buffers are flat
        # (batch_count,), so flatten the staged metadata the same way the
        # matrix was flattened: a (2, 3, 65, 65) input arrives with
        # tolerances shaped (2, 3), and Tensor.copy_ does not reshape
        # equal-numel tensors.
        atol_tensor = _materialize_tolerance(atol, output_shape, input)
        rtol_tensor = _materialize_tolerance(rtol, output_shape, input)
        atol_tensor = atol_tensor.reshape(batch_count)
        rtol_tensor = rtol_tensor.reshape(batch_count)
    out = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    block_r = triton.next_power_of_2(rows)
    relative_epsilon = 1.0e-15 if is_fp64 else 1.0e-7
    absolute_epsilon = 1.0e-300 if is_fp64 else 1.0e-30
    num_warps = 1 if block_r <= 64 else 4

    with torch_device_fn.device(input.device):
        if k == 1:
            _matrix_rank_rank1_kernel[(batch_count,)](
                matrix,
                atol_ptr,
                rtol_ptr,
                scale,
                out,
                atol_s,
                rtol_s,
                M=m,
                N=n,
                ROWS=rows,
                TALL=m >= n,
                HERMITIAN=hermitian,
                BLOCK_R=block_r,
                SCALAR_TOL=scalar_tol,
                num_warps=num_warps,
            )
        elif k == 2:
            _matrix_rank_rank2_kernel[(batch_count,)](
                matrix,
                atol_ptr,
                rtol_ptr,
                scale,
                out,
                atol_s,
                rtol_s,
                M=m,
                N=n,
                ROWS=rows,
                TALL=m >= n,
                HERMITIAN=hermitian,
                BLOCK_R=block_r,
                REL_EPS=relative_epsilon,
                ABS_EPS=absolute_epsilon,
                SCALAR_TOL=scalar_tol,
                num_warps=num_warps,
            )
        elif herm_tridiag:
            _launch_herm_tridiag_rank(
                matrix, atol_tensor, rtol_tensor, scale, out, k, batch_count, input
            )
        elif use_bidiag:
            _launch_bidiag_rank(
                matrix,
                atol_tensor,
                rtol_tensor,
                scale,
                out,
                m,
                n,
                k,
                rows,
                batch_count,
                input,
            )
        else:
            # fused_eligible: small hermitian matrices and small non-hermitian
            # matrices both land here (single-program fused Jacobi, no grid
            # barrier by construction).
            work = torch.empty(
                (batch_count, k, rows),
                dtype=input.dtype,
                device=input.device,
            )
            round_size = k if k % 2 == 0 else k + 1
            pairs = round_size // 2
            block_p = triton.next_power_of_2(pairs)
            block_k = triton.next_power_of_2(k)
            block_c = min(256, block_r)
            sweeps = _jacobi_sweeps(k, is_fp64)
            fused_warps = 8 if block_p * block_c >= 8192 else 4
            _matrix_rank_fused_jacobi_kernel[(batch_count,)](
                matrix,
                work,
                atol_ptr,
                rtol_ptr,
                scale,
                out,
                atol_s,
                rtol_s,
                M=m,
                N=n,
                K=k,
                ROWS=rows,
                TALL=m >= n,
                HERMITIAN=hermitian,
                IS_FP64=is_fp64,
                ROUND=round_size,
                PAIRS=pairs,
                BLOCK_R=block_r,
                BLOCK_P=block_p,
                BLOCK_K=block_k,
                SWEEPS=sweeps,
                REL_EPS=relative_epsilon,
                ABS_EPS=absolute_epsilon,
                SCALAR_TOL=scalar_tol,
                num_warps=fused_warps,
                enable_fp_fusion=not is_fp64,
            )
    return out


def _needs_negative_tolerance_fixup(atol, rtol):
    # tol = max(atol, rtol*sigma_max) < 0 is reachable only where BOTH the
    # effective atol and rtol are negative; since the defaults are >= 0 both
    # must be given explicitly. Pure host-side check when both are scalars.
    if atol is None or rtol is None:
        return False
    if isinstance(atol, torch.Tensor) or isinstance(rtol, torch.Tensor):
        # Cannot inspect values without a device sync; take the async path.
        return True
    return float(atol) < 0.0 and float(rtol) < 0.0


def _correct_negative_tolerance_rank(input, result, atol, rtol, hermitian):
    # torch does not clamp the tolerance at zero. Where tol < 0 (both atol
    # and rtol negative), every singular value (>= 0) exceeds tol, so a
    # nonzero matrix has full rank k, while a zero matrix still has rank 0
    # (rtol*0 == 0 lifts tol back to max(atol, 0) == 0). The counting
    # kernels assume tol >= 0 -- the hermitian Sturm path splits
    # #{|lambda| > tol} into #{lambda > tol} + #{lambda < -tol}, which double
    # counts the overlap for tol < 0, and the Golub-Kahan path counts in the
    # sigma^2 domain where tol gets squared -- so fix the result up here.
    # Fully asynchronous: no Python branching on device data.
    # hermitian reads only the lower triangle: strict-upper garbage is
    # invisible to torch, so the "nonzero" test must ignore it too.
    visible = torch.tril(input) if hermitian else input
    # abs().amax() instead of (amax > 0) | (amin < 0): the generic amin op
    # has no float64 kernel, and one reduction is cheaper than two.
    nonzero = visible.abs().amax(dim=(-2, -1)) > 0
    k = min(input.shape[-2:])
    if isinstance(atol, torch.Tensor) or isinstance(rtol, torch.Tensor):
        # At least one tolerance is a device tensor; a Python float on the
        # other side broadcasts into the predicate.
        neg_pair = (atol < 0) & (rtol < 0)
        return torch.where(neg_pair & nonzero, k, result)
    # Both tolerances are scalars: the caller already verified on the host
    # that both are negative, so the predicate is uniformly true.
    return torch.where(nonzero, k, result)


def linalg_matrix_rank(input, *, atol=None, rtol=None, hermitian=False):
    """Computes numerical matrix rank using shape-specialized Triton kernels."""
    logger.debug("GEMS LINALG_MATRIX_RANK")
    _check_input(input, hermitian)

    output_shape = input.shape[:-2]
    # Validate tolerances BEFORE the empty-input return: native torch runs
    # its same-device / non-complex tolerance checks first, so an empty
    # matrix must still reject an invalid tensor tolerance.
    atol_val, rtol_val = _prepare_tolerances(input, atol, rtol)
    if input.numel() == 0:
        return _empty_matrix_rank(input, output_shape)
    result = _launch_matrix_rank(
        input,
        atol_val,
        rtol_val,
        hermitian,
    )
    if _needs_negative_tolerance_fixup(atol, rtol):
        result = _correct_negative_tolerance_rank(
            input, result, atol_val, rtol_val, hermitian
        )
    return result


def linalg_matrix_rank_tol(input, tol, hermitian=False):
    """NumPy-compatible legacy overload where tol is an absolute tolerance."""
    return linalg_matrix_rank(input, atol=tol, rtol=0.0, hermitian=hermitian)


def _copy_rank_to_out(input, result, out):
    if out is None:
        raise TypeError("torch.linalg.matrix_rank: out must be a Tensor")
    if out.device != input.device:
        raise RuntimeError(
            "torch.linalg.matrix_rank: Expected result and input tensors to be on "
            f"the same device, but got result on {out.device} and input on "
            f"{input.device}"
        )
    if not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError(
            "torch.linalg.matrix_rank: Expected result to be safely castable from "
            f"Long dtype, but got result with dtype {out.dtype}"
        )

    if out.numel() != 0 and out.shape != result.shape:
        warnings.warn(
            "An output with one or more elements was resized because it had shape "
            f"{tuple(out.shape)}, which does not match the required output shape "
            f"{tuple(result.shape)}.",
            UserWarning,
            stacklevel=3,
        )
    out.resize_(result.shape)
    out.copy_(result)
    return out


def linalg_matrix_rank_out(input, *, atol=None, rtol=None, hermitian=False, out=None):
    result = linalg_matrix_rank(input, atol=atol, rtol=rtol, hermitian=hermitian)
    return _copy_rank_to_out(input, result, out)


def linalg_matrix_rank_tol_out(input, tol, hermitian=False, *, out=None):
    result = linalg_matrix_rank_tol(input, tol, hermitian)
    return _copy_rank_to_out(input, result, out)
