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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

_CORES = 32
_FIXED_COPY_US = 0.40
_CORE_BW = 87e9  # bytes/s per AI core, streaming GM->L1
_CUBE_RATE = 5.5e12  # FLOP/s per AI core (cube, fp32 accumulate)
_MEM_CAP = 176000  # bytes of on-chip tile budget for one program
_PROG_US = 0.15  # per-program dispatch/prologue cost (us)
_MIN_DOT = 16  # tl.dot needs >= 16 on every axis
_GEMV_FLOOR_BYTES = 8 * 1024 * 1024
# On-chip budgets, probed on this target with a compile/execute feasibility test
# of the same tl.dot body (triton-ascend 3.6.0.post2, Ascend950DT_9582):
#   * bf16 256x256x128 -> MLIRCompilationError "cc overflow, requires 4194304
#     bits while 2097152 bits available": the dot accumulator plane is
#     double-buffered against a 256 KB budget, so one fp32 plane must stay at or
#     below 128 KB (bm*bn <= 32768).
#   * bf16 256x128x128 and 128x256x128 -> compiled and matched the reference
#     exactly, including the single-k-chunk M=65536, N=128, K=128 shape.
#   * fp32 (two accumulator planes) 128x128x128 -> ok; 256x128x128 and
#     256x256x128 -> "ub overflow", so the planes together must fit 253952 B.
_L1_CAP = 262144  # bytes, double-buffered GM->L1 staging for one program
_CC_PLANE_CAP = 131072  # bytes, one fp32 dot-accumulator plane (cc is x2)
_UB_ACC_CAP = 253952  # bytes, all accumulator planes resident in UB


@triton.autotune(
    configs=[
        # --- DEFER=0 controls (bound the downside of the narrower list) --------
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 512, "DEFER": 0}, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 1024, "DEFER": 0}, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 2048, "DEFER": 0}, num_stages=3),
        # --- deferred fp32 tile accumulation: burst x width ladder --------------
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 512, "DEFER": 1}, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 1024, "DEFER": 1}, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 1024, "DEFER": 1}, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 1024, "DEFER": 1}, num_stages=4),
        triton.Config({"BLOCK_N": 8, "BLOCK_K": 2048, "DEFER": 1}, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 2048, "DEFER": 1}, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256, "DEFER": 1}, num_stages=2),
        # --- wider-BLOCK_N DEFER=0 tiles, restored ----------------------------
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 512, "DEFER": 0}, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256, "DEFER": 0}, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256, "DEFER": 0}, num_stages=3),
        # --- additive probes ---------------------------------------------------
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "DEFER": 1}, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "DEFER": 1}, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 2048, "DEFER": 0}, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 2048, "DEFER": 1}, num_stages=4),
    ],
    key=["N", "KC"],
)
@triton.jit
def _gemv_kernel(
    X,
    W,
    B,
    Y,
    N,
    KC: tl.constexpr,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    DEFER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # KC is K as a compile-time constant and, for a contiguous [N, K] weight, it
    # is also the row stride (host-guarded by w.stride(0) == K). This makes the
    # K-loop trip count, the loop bound and every weight-row base offset
    # compile-time quantities instead of runtime scalar arithmetic.
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N

    x_ptrs = X + offs_k
    w_ptrs = W + offs_n[:, None] * KC + offs_k[None, :]

    if DEFER:
        # Accumulate the whole [BLOCK_N, BLOCK_K] product tile in fp32 with a
        # fused multiply-add and reduce across K exactly once at the end: the
        # horizontal (cross-lane) reduction then costs BLOCK_N*BLOCK_K adds per
        # program instead of one per K iteration. Same fp32 arithmetic and the
        # same single bf16 store, only the summation order inside fp32 differs.
        acc2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for _ in range(0, KC // BLOCK_K):
            xv = tl.load(x_ptrs).to(tl.float32)
            wv = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
            acc2 += wv * xv[None, :]
            x_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        acc = tl.sum(acc2, axis=1)
    else:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for _ in range(0, KC // BLOCK_K):
            xv = tl.load(x_ptrs).to(tl.float32)
            wv = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(wv * xv[None, :], axis=1)
            x_ptrs += BLOCK_K
            w_ptrs += BLOCK_K

    if HAS_BIAS:
        acc += tl.load(B + offs_n, mask=n_mask, other=0.0).to(tl.float32)

    y_ptrs = Y + offs_n * stride_yn
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=n_mask)


@triton.jit
def _linear_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    COMPENSATE: tl.constexpr,
    BROADCAST_X: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    comp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if BROADCAST_X:
            # M == 1: only one input row exists, so load it once as a vector and
            # broadcast. Every lane of the A tile then holds that row, which
            # makes the single stored output row bit-identical (verified on the
            # target, max diff 0.0 vs F.linear) while dropping the 16-row masked
            # X copy - up to 12% of a decode shape's GM->L1 traffic.
            if EVEN_K:
                x_row = tl.load(X + (k0 + offs_k) * stride_xk)
            else:
                x_row = tl.load(
                    X + (k0 + offs_k) * stride_xk, mask=(k0 + offs_k) < K, other=0.0
                )
            a = tl.broadcast_to(x_row[None, :], (BLOCK_M, BLOCK_K))
        elif EVEN_M:
            a = tl.load(x_ptrs)
        elif EVEN_K:
            a = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        else:
            a = tl.load(
                x_ptrs,
                mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K),
                other=0.0,
            )
        if EVEN_N:
            if EVEN_K:
                w = tl.load(w_ptrs)
            else:
                w = tl.load(w_ptrs, mask=(k0 + offs_k)[None, :] < K, other=0.0)
        elif EVEN_K:
            w = tl.load(w_ptrs, mask=offs_n[:, None] < N, other=0.0)
        else:
            w = tl.load(
                w_ptrs,
                mask=(offs_n[:, None] < N) & ((k0 + offs_k)[None, :] < K),
                other=0.0,
            )
        if COMPENSATE:
            s = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
            y = acc + (s - comp)
            comp = (y - acc) - (s - comp)
            acc = y
        else:
            acc = tl.dot(a, tl.trans(w), acc, out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        if EVEN_N:
            b = tl.load(B + offs_n)
        else:
            b = tl.load(B + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]

    y_ptrs = Y + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    if EVEN_M and EVEN_N:
        tl.store(y_ptrs, acc.to(Y.dtype.element_ty))
    elif EVEN_N:
        tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=offs_m[:, None] < M)
    else:
        tl.store(
            y_ptrs,
            acc.to(Y.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


@triton.jit
def _splitk_kernel(
    X,
    W,
    P,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ps,
    stride_pm,
    stride_pn,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BROADCAST_X: tl.constexpr,
    COMPENSATE: tl.constexpr,
    ATOMIC: tl.constexpr,
):
    """Same cube body as _linear_kernel, restricted to one contiguous K slice.

    The purpose is occupancy, not byte reduction: when the output plane is so
    small that (M/BLOCK_M)*(N/BLOCK_N) cannot fill the 32 cube cores, the K axis
    is the only remaining source of parallelism. Each program owns a disjoint
    [k_begin, k_end) range and writes an fp32 partial plane; _reduce_kernel sums
    the SPLIT planes in a fixed order, so the result does not depend on how the
    hardware schedules them.
    """
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    num_tiles = tl.cdiv(M, BLOCK_M) * num_n
    pid_s = pid // num_tiles
    pid_t = pid % num_tiles
    pid_m = pid_t // num_n
    pid_n = pid_t % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    chunk = (K // BLOCK_K) // SPLIT
    k_begin = pid_s * chunk * BLOCK_K
    k_end = k_begin + chunk * BLOCK_K

    x_ptrs = (
        X
        + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        + k_begin * stride_xk
    )
    w_ptrs = (
        W
        + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        + k_begin * stride_wk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    comp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        if BROADCAST_X:
            a = tl.broadcast_to(
                tl.load(X + (k0 + offs_k) * stride_xk)[None, :], (BLOCK_M, BLOCK_K)
            )
        elif EVEN_M:
            a = tl.load(x_ptrs)
        else:
            a = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        if EVEN_N:
            w = tl.load(w_ptrs)
        else:
            w = tl.load(w_ptrs, mask=offs_n[:, None] < N, other=0.0)
        if COMPENSATE:
            s = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
            y = acc + (s - comp)
            comp = (y - acc) - (s - comp)
            acc = y
        else:
            acc = tl.dot(a, tl.trans(w), acc, out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    p_ptrs = (
        P
        + pid_s * stride_ps
        + (offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn)
    )
    if ATOMIC:
        # fp32 output: accumulate straight into the result plane, which the host
        # zeroed (the neutral identity for this sum), so the split launch needs no
        # second reduce kernel. Probed on this target: tl.atomic_add on fp32
        # global compiles, runs, matches F.linear at rel_err 8.7e-7 and does not
        # accumulate across calls. The summation order inside fp32 is then
        # hardware-scheduled rather than fixed.
        tl.atomic_add(p_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    elif EVEN_M and EVEN_N:
        tl.store(p_ptrs, acc)
    else:
        tl.store(p_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _reduce_kernel(
    P,
    Y,
    MN,
    stride_ps,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(P + offs, mask=mask, other=0.0)
    for s in range(1, SPLIT):
        acc += tl.load(P + s * stride_ps + offs, mask=mask, other=0.0)
    tl.store(Y + offs, acc.to(Y.dtype.element_ty), mask=mask)


def _pow2_ceil(x):
    p = _MIN_DOT
    while p < x and p < 256:
        p *= 2
    return p


def _plan(M, N, K, esize):
    """Pick (BLOCK_M, BLOCK_N, BLOCK_K) minimizing a wave-quantised cost model."""
    bm_max = _pow2_ceil(M)
    bn_max = _pow2_ceil(N)
    bks = (1024, 512, 256, 128, 64, 32)
    acc_tiles = 2 if esize >= 4 else 1  # fp32 path also keeps a compensation tile
    best = None
    for bk in bks:
        if K % bk:
            continue
        fixed_us = (K // bk) * _FIXED_COPY_US
        for bm in (16, 32, 64, 128, 256):
            if bm > bm_max:
                continue
            for bn in (16, 32, 64, 128, 256):
                if bn > bn_max:
                    continue
                blocks = -(-M // bm) * -(-N // bn)
                # On-chip footprint. The original budget charged the L1 staging
                # tile and the accumulators against one 176 KB pool, which capped
                # BLOCK_M at 128 for every reduced-precision shape. The probe
                # shows the pools are separate on this SoC, so a wider tile is
                # admitted *only* when it still fills the machine (>= 32
                # programs): a wide tile on a tiny grid trades away the
                # parallelism an already-starved row needs, and row 257x128x128
                # measured 0.98 with its narrow 17-program tile.
                l1_bytes = ((bm if M > 1 else 1) + bn) * bk * esize
                acc_bytes = acc_tiles * bm * bn * 4
                if l1_bytes + acc_bytes > _MEM_CAP:
                    if not (
                        blocks >= _CORES
                        and 2 * l1_bytes <= _L1_CAP
                        and bm * bn * 4 <= _CC_PLANE_CAP
                        and acc_bytes <= _UB_ACC_CAP
                    ):
                        continue
                waves = -(-blocks // _CORES)
                mem_us = (
                    fixed_us + ((bm if M > 1 else 1) + bn) * K * esize * 1e6 / _CORE_BW
                )
                cube_us = 2.0 * bm * bn * K * 1e6 / _CUBE_RATE
                per_block = mem_us if mem_us > cube_us else cube_us
                # per-program dispatch/prologue cost (~0.15 us per program,
                # measured as aic_scalar_time/Block Num). It overlaps with the
                # data pipes on long kernels, so it is a floor, not an adder, and
                # it is paid per program *resident on a core*: charging it per
                # block over-penalises large grids and, for M=65536 shapes whose
                # grid cannot fall below 256, made it the whole objective and
                # collapsed BLOCK_K to 32.
                pred = waves * per_block
                dispatch_us = min(blocks, _CORES) * _PROG_US
                if dispatch_us > pred:
                    pred = dispatch_us
                if best is None or pred < best[0]:
                    best = (pred, bm, bn, bk)
    if best is None:
        return _MIN_DOT, _MIN_DOT, 16
    return best[1], best[2], best[3]


_PLAN_CACHE = {}
_REDUCE_BLOCK = 2048


def _split_choice(M, N, K, block_m, block_n, block_k, esize, has_bias):
    """How many K slices to use, or 0 to keep the single-launch kernel.

    Only the occupancy-starved cube rows qualify. Measured with msprof on this
    target, 8x6144x128 fp32 runs 8 programs (cube utilization 23.3%) and its
    resident cores are MAC-bound at 0.826 with the MTE2 pipe at 0.358, while
    1x4096x256 fp32 runs 16 - in both the machine is idle, so a disjoint K slice
    per program is the only way to widen the grid. Rows that already fill 32
    cores are excluded: 1024x160x6144 measures exactly 32 programs, and the
    archived medium-plane split-K lowered the suite twice for cutting bytes on
    an already-full grid. This is the other case recorded there as the reopen
    condition - a starved grid gaining programs.

    The K >= 4096 and N*K*esize >= 2 MiB floors keep the second launch off the
    rows that cannot amortise it. Round 4 measured that boundary directly: on
    8x6144x160 bf16 (N*K*esize = 1.97 MB, ~8.3 us) the split bought nothing and
    the extra launch pushed the row from 0.927 to 0.716, while 8x6144x128 fp32
    (3.1 MB, ~15.5 us) and 1x4096x256 fp32 (4.2 MB, ~13.1 us) both improved. The
    2-4 us shapes (1x128x128 measures 2.36 us) are already at or above the
    reference and must never pay the extra launch.
    """
    if has_bias or K < 4096 or N * K * esize < (1 << 21):
        return 0
    grid = -(-M // block_m) * -(-N // block_n)
    if grid >= _CORES:
        return 0
    nk = K // block_k
    for s in (2, 3, 4, 6, 8, 12):
        if nk % s:
            continue
        if grid * s >= _CORES:
            return s
    return 0


def _launch(input, weight, bias=None):
    x = input
    w = weight
    shape = x.shape
    if x.dim() == 2:
        x2 = x
    else:
        x2 = x.reshape(-1, w.shape[-1])
    M = x2.shape[0]
    K = x2.shape[1]
    N = w.shape[0]
    esize = x.element_size()

    key = (M, N, K, esize)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _plan(M, N, K, esize)
        _PLAN_CACHE[key] = plan
    block_m, block_n, block_k = plan

    has_bias = bias is not None
    # fp32 operands are accumulated with a compensated (Kahan) update: the cube
    # products are only exact up to fp32 rounding, so the chunk-level summation
    # dominates the f32 error; the bf16 path keeps the plain cube accumulator
    # (its products are exact and that reproduces the reference bit-for-bit).
    compensate = (x.dtype == torch.float32) and (K >= 512)

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # M == 1 is a pure weight-stream: one activation row against N*K elements of
    # weight. Measured with msprof on this target, the cube launch saturates the
    # GM->L1 pipe (MTE2 duty 0.99 at 32 programs) and still lands 0.55-0.65 of
    # the reference latency on the multi-hundred-MB streams, because the L1
    # fractal fill is the per-byte bottleneck there.
    #
    # The 8 MiB floor keeps the small-stream M==1 rows on the cube launch: a
    # ~1 MiB stream cannot fill the device from the vector cores, and that
    # dispatch decision was measured to roughly halve the N=256/K=2048 row.
    use_gemv = (
        M == 1
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x2.stride(1) == 1
        and w.stride(0) == K
        and w.stride(1) == 1
        and (K % 2048) == 0
        and N * K * w.element_size() > _GEMV_FLOOR_BYTES
    )
    split = _split_choice(M, N, K, block_m, block_n, block_k, esize, has_bias)
    # The atomic fusion of the fp32 reduce (round 5) measured slower than the
    # separate reduce launch: 8x6144x128 fp32 went 0.01201 -> 0.01363 ms and
    # 0.697 -> 0.614 of the reference. The likely cause is that a kernel
    # containing atomics loses the compiler's AutoBlockify pass, which costs more
    # on a 12 us row than the ~2.4 us launch it saves. Keep the reduce kernel.
    atomic_split = False
    if atomic_split:
        # zeroed fp32 plane: the neutral identity for the atomic accumulation
        out = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    if use_gemv:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
        _gemv_kernel[grid](
            x2,
            w,
            x2 if not has_bias else bias,
            out,
            N,
            K,
            out.stride(1),
            HAS_BIAS=has_bias,
        )
    elif atomic_split:
        num_tiles = (-(-M // block_m)) * (-(-N // block_n))
        _splitk_kernel[(split * num_tiles,)](
            x2,
            w,
            out,
            M,
            N,
            K,
            x2.stride(0),
            x2.stride(1),
            w.stride(0),
            w.stride(1),
            0,
            out.stride(0),
            out.stride(1),
            SPLIT=split,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            EVEN_M=(M % block_m) == 0,
            EVEN_N=(N % block_n) == 0,
            BROADCAST_X=(M == 1),
            COMPENSATE=compensate,
            ATOMIC=True,
        )
    elif split:
        num_tiles = (-(-M // block_m)) * (-(-N // block_n))
        part = torch.empty((split * M * N,), dtype=torch.float32, device=x.device)
        _splitk_kernel[(split * num_tiles,)](
            x2,
            w,
            part,
            M,
            N,
            K,
            x2.stride(0),
            x2.stride(1),
            w.stride(0),
            w.stride(1),
            M * N,
            N,
            1,
            SPLIT=split,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            EVEN_M=(M % block_m) == 0,
            EVEN_N=(N % block_n) == 0,
            BROADCAST_X=(M == 1),
            COMPENSATE=compensate,
            ATOMIC=False,
        )
        _reduce_kernel[(-(-M * N // _REDUCE_BLOCK),)](
            part,
            out,
            M * N,
            M * N,
            SPLIT=split,
            BLOCK=_REDUCE_BLOCK,
        )
    else:
        grid = (-(-M // block_m) * -(-N // block_n),)
        _linear_kernel[grid](
            x2,
            w,
            x2 if not has_bias else bias,
            out,
            M,
            N,
            K,
            x2.stride(0),
            x2.stride(1),
            w.stride(0),
            w.stride(1),
            out.stride(0),
            out.stride(1),
            HAS_BIAS=has_bias,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            EVEN_M=(M % block_m) == 0,
            EVEN_N=(N % block_n) == 0,
            EVEN_K=(K % block_k) == 0,
            COMPENSATE=compensate,
            BROADCAST_X=(M == 1),
        )

    if x.dim() != 2:
        out = out.view(*shape[:-1], N)
    return out


def linear(input, weight, bias=None):
    """A5 dense small-M BF16 projections; other layouts use the existing backend."""
    if (
        input.device.type == "npu"
        and weight.device == input.device
        and input.ndim == 2
        and weight.ndim == 2
        and 1 <= input.shape[0] <= 8
        and 0 < weight.shape[0] <= 128
        and input.shape[1] == weight.shape[1]
        and 0 < input.shape[1] <= 6144
        and input.is_contiguous()
        and weight.is_contiguous()
        and input.dtype == weight.dtype == torch.bfloat16
        and (
            bias is None
            or (
                bias.shape == (weight.shape[0],)
                and bias.is_contiguous()
                and bias.dtype == input.dtype
                and bias.device == input.device
            )
        )
    ):
        with torch_device_fn.device(input.device):
            return _launch(input, weight, bias)
    from flag_gems.ops.linear import linear as generic_linear
    from flag_gems.runtime.backend import get_customized_ops

    previous = dict(get_customized_ops("ascend")).get("linear", generic_linear)
    return previous(input, weight, bias)
