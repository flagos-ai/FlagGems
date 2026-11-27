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

"""chain_matmul in Triton.

Computes the product of a chain of 2-D matrices using an optimal (minimum-flop)
parenthesization, mirroring torch.linalg.multi_dot / aten::chain_matmul
semantics. The host wrapper only runs the matrix-chain DP (shape metadata);
every GEMM node comes from this backend, so each DP node rounds back to the
input dtype the way a native-dtype reference does rather than an
internally-widened one.

Node kernel (metadata only, measured on GCU400): ``_chain_gemm_kernel``. It
keeps the K loop at runtime and only the tile plus the stride pattern as
constexpr, so a chain compiles once per tile configuration instead of once per
distinct node K. The tile comes from ``_chain_config``: a per-shape table
fitted by an on-device node sweep and bounded by the 448 KiB DSM half line and
the 32x128 per-warp floor, with a mask-free load/store path whenever the node
is tile-divisible.
The alternatives were measured and rejected:

* the lean kernel of ``linalg_multi_dot`` unrolls K through ``K_BLOCKS`` as a
  constexpr, so it recompiles for every distinct node K: 2.4s of first-touch
  compile for one (256,1024)x(1024,256) fp16 node, 13.2s for a four-matrix
  fp16 chain;
* the tuned ``mm`` autotune space is worse (48-57s of first-touch compile on a
  wide-K node) and never returned on a seven-matrix chain (>20 min inside
  gcu-compiler-compile);
* a fully static K unroll (``tl.static_range`` over a constexpr trip count)
  measured identical to the runtime loop on every node shape, so it is not
  worth recompiling per node K.

No chain node runs a native GEMM: a node the kernel cannot take by dtype or
size falls through to the generic Triton GEMM of ``linalg_multi_dot``.

The ``out`` variant writes the result straight into the caller's buffer
(a constexpr-strided Triton store). Only the autograd path, which needs its own
result tensor, pays an extra copy into ``out``.

Interior nodes of a longer chain take their buffer from a small recycled pool
(``_chain_scratch``) instead of ``torch.empty``: on this backend a per-node
allocation costs ~16us of allocator churn and was the largest single term of a
three-matrix chain. The last node always writes to the caller's ``out`` or to a
freshly allocated tensor, so the returned result never aliases the pool.

Launch path (the other half of a chain's cost). A node kernel takes 6-14us, but
on this backend one launch used to cost far more than that on the host side
(measured per launch, host-bound, fp16): ``@libentry()`` + JIT binding + the
device guard + stride reads added up to ~59us, of which ~23us was the libentry
wrapper alone and ~6us the ``torch_device_fn.device`` guard. Two dependent nodes
therefore starved the device: a six-matrix fp16 chain took 0.266ms against a
native 0.044ms even though its node kernels summed to 0.050ms. So a node is
launched through a cached pre-compiled runner (``CompiledKernel``) keyed by the
full node signature - shape, strides, tile, dtype and pointer alignment - which
skips the per-call specialization and cache-key rebuild; the device guard is
skipped when the current device already matches. The JIT path stays as the
fallback, and the cache key covers every constexpr slot, so a compiled binary is
never reused for a different signature.
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.linalg_multi_dot import (
    _matrix_multiply,
)

logger = logging.getLogger(__name__)

# Dtypes the chain node kernel takes. bfloat16 is included: the node kernel
# accumulates in fp32, and routing bf16 anywhere else would either hit the
# tuned ``mm`` (48s of first-touch compile) or the generic multi_dot GEMM.
_CHAIN_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Upper bound on any node dimension. Past this the caller falls through to the
# generic Triton GEMM of ``linalg_multi_dot`` - still Triton, never native.
_CHAIN_MAX_DIM = 4096
# Node tile table measured on GCU400 (2026-11-25 per-node sweep, no-WS). The
# best tile is node-specific; _chain_config keys it by shape. Every entry keeps
# the per-warp slice at 32x128 or above and the DSM under the 448 KiB
# double-program line: (BLOCK_M + BLOCK_N) * BLOCK_K * itemsize.
_CHAIN_WORK_LARGE = 1 << 30
_CHAIN_DSM_HALF_BYTES = 458752
_CHAIN_DSM_FULL_BYTES = 917504
_CHAIN_MIN_BLOCK_K = 32
# Warp specialization always runs a two-stage pipeline. Deeper stages were
# measured on every chain node and are 8-20x slower, so a WS node may not pick
# another value.
_CHAIN_WS_STAGES = 2


@triton.jit
def _copy(
    A,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    AM: tl.constexpr,
    AN: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
):
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    m = x // N
    n = x % N
    v = tl.load(A + m * AM + n * AN, x < M * N, 0)
    tl.store(C + m * CM + n * CN, v, x < M * N)


def _copy2d(a, c):
    m, n = a.shape
    if m * n:
        _copy[(triton.cdiv(m * n, 256),)](a, c, m, n, *a.stride(), *c.stride())
    return c


def _split(p):
    n = len(p) - 1
    cost = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]
    for length in range(1, n):
        for i in range(n - length):
            j = i + length
            best = None
            for k in range(i, j):
                v = cost[i][k] + cost[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if best is None or v < best:
                    best = v
                    s[i][j] = k
            cost[i][j] = best
    return s


@lru_cache(maxsize=128)
def _schedule(p):
    # Only immutable integer metadata is retained, never tensors or values.
    s = _split(p)
    order = []

    def build(i, j):
        if i != j:
            k = s[i][j]
            build(i, k)
            build(k + 1, j)
            order.append((i, j, k))

    build(0, len(p) - 2)
    return tuple(order)


@triton.jit
def _chain_gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    FULL_EVEN: tl.constexpr,
    USE_WS: tl.constexpr,
):
    """Chain-node GEMM: one tile configuration, runtime K loop.

    The K loop must stay a runtime loop. ``linalg_multi_dot``'s lean kernel
    unrolls it through ``K_BLOCKS`` as a constexpr, so every distinct K node of
    a chain triggers a fresh compile (2.4s for a (256,1024)x(1024,256) fp16
    node, 5.3s for the fp32 one). A static unroll here measured identical to
    this runtime loop, so only the tile and the stride pattern are constexpr
    and a chain compiles once per tile.

    ``EVEN_K`` / ``FULL_EVEN`` are node metadata computed on the host: they
    strip the K mask (and then every other mask) when the node is
    tile-divisible, which is the common case for chain nodes. The K index is
    rebuilt from the loop variable rather than accumulated so the masked and
    unmasked forms share one body.
    """
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, K, BLOCK_K, warp_specialize=USE_WS):
        if FULL_EVEN:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        elif EVEN_K:
            a = tl.load(a_ptrs, rm[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, rn[None, :] < N, other=0.0)
        else:
            mask_k = (k_start + rk) < K
            a = tl.load(a_ptrs, (rm[:, None] < M) & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask_k[:, None] & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    out_val = acc.to(C.dtype.element_ty)
    if FULL_EVEN:
        tl.store(c_ptrs, out_val)
    else:
        tl.store(c_ptrs, out_val, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _chain_node_ok(left, right, out):
    """Metadata-only eligibility of a chain node for the chain GEMM kernel."""
    if left.dtype not in _CHAIN_DTYPES or right.dtype is not left.dtype:
        return False
    if left.device != right.device or left.device.type == "cpu":
        return False
    M, K = left.shape
    N = right.shape[1]
    if M < 1 or N < 1 or K < 1:
        return False
    if M > _CHAIN_MAX_DIM or N > _CHAIN_MAX_DIM or K > _CHAIN_MAX_DIM:
        return False
    if out is not None:
        if tuple(out.shape) != (M, N) or out.dtype is not left.dtype:
            return False
    return True


def _chain_use_ws(M, N, K, block_m, block_k, itemsize):
    """Per-node warp-specialization decision.

    Swept on 2026-11-25 over every chain node (WS tiles 128x256 / 256x256 with
    a full 896 KiB DSM budget): every node was 1.7-2.9x slower than its no-WS
    tile -- the K loop is only 1-8 trips once BLOCK_K is full-K, so a
    producer/consumer split has nothing to overlap. WS is therefore off; it is
    pinned to ``_CHAIN_WS_STAGES`` (2) so a node can never ask for another
    stage count, and the kernel keeps the ``USE_WS``/``tl.range`` hook should a
    future shape or compiler change make it pay off.
    """
    return False


def _chain_config(M, N, K, itemsize):
    """Tile and launch configuration for one chain node (metadata only).

    Fitted on GCU400 (2026-11-25 per-node sweeps over every node of the
    benchmark chains, both 16- and 32-bit):

    * narrow deep-K nodes (BLOCK_N <= 128, K >= 512) land on very few programs,
      so one kernel slot owns the whole DSM and the K loop runs as one full-K
      dot: 13-27% faster per node than a 512-wide K loop at the same tile.
    * N <= 64 nodes take a 64x64 tile instead of a 128-wide N tile: measured
      tie-or-better on every node tried and 5-20% faster on the deep-K ones
      (256x64x1024 -9%/-17%, 512x64x2048 -20%, 64x32x2048 -14%), because the
      narrower N tile doubles the program count on the N axis and halves the
      per-program B tile. It also keeps an fp32 N == 1 node off the tile that
      fails to link on topscc 4.1.11 (BLOCK_N=128 there emits an unresolved
      ICALL to a device helper).
    * every other node is best at BLOCK_M chosen so the tile grid lands near
      16 programs of 128-wide N tiles, BLOCK_M = pow2(M * ceil(N/128) / 16)
      clamped to [32, 256], BLOCK_N = 128. One warp suffices at BLOCK_M = 32
      (the 32x128 per-warp floor); bigger tiles use two.
    * BLOCK_K is full K except that a 256-row tile with K >= 256 prefers 128,
      and the huge-work nodes keep the measured 256x128 tile with a 128 (32-bit)
      or 256 (16-bit) K step.

    The DSM budget follows the slot rule: a grid of at most 24 programs can
    use the full 896 KiB, otherwise each program must fit the 448 KiB line so
    two of them can coexist.
    """
    k_pow2 = max(_CHAIN_MIN_BLOCK_K, triton.next_power_of_2(K))
    narrow = min(M, N)
    if N <= 64:
        block_m, block_n, block_k = 64, 64, k_pow2
        grid = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
        budget = _CHAIN_DSM_FULL_BYTES if grid <= 24 else _CHAIN_DSM_HALF_BYTES
        while (block_m + block_n) * block_k * itemsize > budget and (
            block_k > _CHAIN_MIN_BLOCK_K
        ):
            block_k //= 2
        return block_m, block_n, block_k, 2, _CHAIN_WS_STAGES, False
    if K >= 512 and 32 <= narrow <= 128:
        grid = triton.cdiv(M, 32) * triton.cdiv(N, 128)
        budget = _CHAIN_DSM_FULL_BYTES if grid <= 24 else _CHAIN_DSM_HALF_BYTES
        block_k = k_pow2
        while (32 + 128) * block_k * itemsize > budget and (
            block_k > _CHAIN_MIN_BLOCK_K
        ):
            block_k //= 2
        return 32, 128, block_k, 1, _CHAIN_WS_STAGES, False

    if M * N * K >= _CHAIN_WORK_LARGE:
        # The 1024^3-class node is the only one that reaches this branch. For
        # 32-bit it was re-swept on 2026-11-26 over the tiles the earlier pass
        # could not afford (grid > 24 had been capped at 448 KiB): 128x256x128
        # is reproducibly 0.6% faster than 256x128x128 (51.26 vs 51.59 us,
        # interleaved A/B, 5 rounds, spread < 0.3 us), while for 16-bit the
        # 256x128x256 tile stays 2.3% ahead of it.
        if itemsize >= 4:
            block_m, block_n, block_k = 128, 256, 128
        else:
            block_m, block_n, block_k = 256, 128, 256
    else:
        n_tiles = triton.cdiv(N, 128)
        block_m = triton.next_power_of_2(max(1, M * n_tiles // 16))
        block_m = min(256, max(32, block_m))
        block_n = 128
        if K >= 512:
            block_k = k_pow2
        elif block_m >= 256 and K >= 256:
            block_k = 128
        else:
            block_k = k_pow2
    num_warps = 1 if block_m == 32 else 2

    grid = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    budget = _CHAIN_DSM_FULL_BYTES if grid <= 24 else _CHAIN_DSM_HALF_BYTES
    # No warp specialization, so num_stages does not add a DSM buffer.
    while (block_m + block_n) * block_k * itemsize > budget:
        if block_k > _CHAIN_MIN_BLOCK_K:
            block_k //= 2
        elif block_n > 128:
            block_n //= 2
        elif block_m > 64:
            block_m //= 2
        else:
            break
    if block_m <= 32:
        num_warps = 1
    use_ws = _chain_use_ws(M, N, K, block_m, block_k, itemsize)
    return block_m, block_n, block_k, num_warps, _CHAIN_WS_STAGES, use_ws


_MISS = object()
# Node signature -> (cached CompiledKernel runner, positional tail, grid3), or
# None when the runner could not be built and the JIT path must be used instead.
_CHAIN_LAUNCH = {}
# Device index this module last activated, so a repeated guard can be skipped.
_ACTIVE_DEVICE = -1


def _build_node_launcher(left, right, out, key, tail, grid3, num_warps, num_stages):
    """Compile one node without launching and cache its runner.

    ``warmup`` runs the full specialization/compile path once, so the returned
    ``CompiledKernel`` already matches this node's tile and stride pattern. Every
    constexpr slot is part of ``key``, so the runner is never reused across
    different signatures.
    """
    try:
        compiled = _chain_gemm_kernel.warmup(
            left,
            right,
            out,
            *tail,
            grid=grid3,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        runner = compiled[grid3]
    except Exception:  # noqa: B902 - any failure just means "use the JIT"
        runner = None
    _CHAIN_LAUNCH[key] = runner
    return runner


def _node_launch_spec(
    left, right, out, block_m, block_n, block_k, num_warps, num_stages, use_ws
):
    """Cached (runner, tail, grid3) for one node, or None to use the JIT path.

    Every constexpr slot (tile, strides, dtype, pointer alignment, warp
    specialization) is part of the key, so a runner is never reused for a
    different signature.
    """
    M, K = left.shape
    N = right.shape[1]
    grid3 = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), 1, 1)
    stride_am, stride_ak = left.stride(0), left.stride(1)
    stride_bk, stride_bn = right.stride(0), right.stride(1)
    stride_cm, stride_cn = out.stride(0), out.stride(1)
    # Pointer divisibility is part of Triton's specialization, so it is part of
    # the key as well; one OR folds the three checks into a single test.
    aligned = (left.data_ptr() | right.data_ptr() | out.data_ptr()) % 16 == 0
    key = (
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        use_ws,
        left.dtype,
        out.dtype,
        aligned,
    )
    entry = _CHAIN_LAUNCH.get(key, _MISS)
    if entry is _MISS:
        tail = (
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            block_m,
            block_n,
            block_k,
            (K % block_k == 0),
            (K % block_k == 0 and M % block_m == 0 and N % block_n == 0),
            use_ws,
        )
        runner = _build_node_launcher(
            left, right, out, key, tail, grid3, num_warps, num_stages
        )
        entry = None if runner is None else (runner, tail, grid3)
        _CHAIN_LAUNCH[key] = entry
    return entry


def _chain_gemm(left, right, out=None):
    global _ACTIVE_DEVICE
    M, K = left.shape
    N = right.shape[1]
    block_m, block_n, block_k, num_warps, num_stages, use_ws = _chain_config(
        M, N, K, left.dtype.itemsize
    )
    if out is None:
        out = torch.empty((M, N), dtype=left.dtype, device=left.device)
    entry = _node_launch_spec(
        left, right, out, block_m, block_n, block_k, num_warps, num_stages, use_ws
    )
    if entry is not None:
        runner, tail, _ = entry
        index = left.device.index or 0
        # The guard costs ~6us per launch; skip it only when the current device
        # provably already is the target one.
        if index == _ACTIVE_DEVICE and torch_device_fn.current_device() == index:
            runner(left, right, out, *tail)
            return out
        with torch_device_fn.device(left.device):
            runner(left, right, out, *tail)
        _ACTIVE_DEVICE = index
        return out
    _chain_gemm_kernel[(triton.cdiv(M, block_m) * triton.cdiv(N, block_n), 1, 1)](
        left,
        right,
        out,
        M,
        N,
        K,
        left.stride(0),
        left.stride(1),
        right.stride(0),
        right.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN_K=(K % block_k == 0),
        FULL_EVEN=(K % block_k == 0 and M % block_m == 0 and N % block_n == 0),
        USE_WS=use_ws,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _gemm(left, right, out=None):
    """One GEMM node of the chain, always a Triton kernel.

    ``_chain_gemm`` covers every fp16/bf16/fp32 node; an out-of-range shape
    falls through to the generic Triton GEMM of ``linalg_multi_dot``. Neither
    path calls a native GEMM, and neither touches the tuned ``mm`` autotune
    space (48-57s of first-touch compile on a wide-K chain node).
    """
    if _chain_node_ok(left, right, out):
        return _chain_gemm(left, right, out)
    return _matrix_multiply(left, right, out=out, allow_lean=False)


def _tree(mats):
    p = tuple(a.shape[0] for a in mats) + (mats[-1].shape[1],)
    order = _schedule(p)
    t = {(i, i): a for i, a in enumerate(mats)}
    for i, j, k in order:
        t[i, j] = _gemm(t[i, k], t[k + 1, j])
    return t[0, len(mats) - 1], t, order


# Recycled intermediates for the interior nodes of a chain. A fresh
# ``torch.empty`` per node costs ~16us of allocator churn on this backend: the
# buffer is written by one node and freed right after the next node reads it,
# which measured 34.7us for a node pair that takes 18.0us with a preallocated
# buffer (three-matrix chain: 47.8us vs 19.7us of pure kernel time). Only
# strictly internal intermediates go through here, and the key carries a slot
# index that resets per call, so two nodes that need the same shape never
# alias. The last node always lands in the caller's ``out`` or in a fresh
# tensor, so the returned result is never a recycled buffer.
_CHAIN_SCRATCH = {}


def _chain_scratch(shape, dtype, device, slot):
    key = (shape, dtype, device, slot)
    buf = _CHAIN_SCRATCH.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _CHAIN_SCRATCH[key] = buf
    return buf


def _matrix_key(a):
    return (a.shape[0], a.shape[1], a.stride(0), a.stride(1), a.data_ptr() & 15)


def _plan_key(mats, out):
    return (
        mats[0].dtype,
        mats[0].device,
        tuple(_matrix_key(a) for a in mats),
        None if out is None else (_matrix_key(out), out.dtype),
    )


# Chain signature -> (steps, interior buffers, final shape), or None when the
# chain cannot use the pre-bound path. A step is (runner, tail, a, b, out) with
# value indices into [*inputs, *interior buffers, final out]. A hit turns the
# whole call into one dict lookup plus the bare launches: the schedule, tile
# table, node keys and stream/device bookkeeping all happened at build time.
_CHAIN_PLAN = {}
_CHAIN_PLAN_MAX = 512


def _build_plan(mats, out):
    n = len(mats)
    dtype, device = mats[0].dtype, mats[0].device
    p = tuple(a.shape[0] for a in mats) + (mats[-1].shape[1],)
    order = _schedule(p)
    last = len(order) - 1
    vals = list(mats)
    shapes = [tuple(a.shape) for a in mats]
    refs = {(i, i): i for i in range(n)}
    slots = {}
    steps = []
    for step, (i, j, k) in enumerate(order):
        ai, bi = refs[i, k], refs[k + 1, j]
        left, right = vals[ai], vals[bi]
        M, N, K = shapes[ai][0], shapes[bi][1], shapes[ai][1]
        if step == last:
            oi = len(vals)
            node_out = (
                out
                if out is not None
                else torch.empty((M, N), dtype=dtype, device=device)
            )
        else:
            shape = (M, N)
            slot = slots.get(shape, 0)
            slots[shape] = slot + 1
            node_out = _chain_scratch(shape, dtype, device, slot)
            vals.append(node_out)
            shapes.append(shape)
            oi = len(vals) - 1
        if not _chain_node_ok(left, right, node_out):
            return None
        block_m, block_n, block_k, num_warps, num_stages, use_ws = _chain_config(
            M, N, K, dtype.itemsize
        )
        entry = _node_launch_spec(
            left,
            right,
            node_out,
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
            use_ws,
        )
        if entry is None:
            return None
        runner, tail, _ = entry
        refs[i, j] = oi
        steps.append((runner, tail, ai, bi, oi))
    return (tuple(steps), tuple(vals[n:]), (mats[0].shape[0], mats[-1].shape[1]))


_TAKES_STREAM = None


def _runner_takes_stream(runner):
    global _TAKES_STREAM
    if _TAKES_STREAM is None:
        try:
            import inspect

            _TAKES_STREAM = "stream" in inspect.signature(runner).parameters
        except (TypeError, ValueError):
            _TAKES_STREAM = False
    return _TAKES_STREAM


def _current_stream(index):
    # One lookup per chain call instead of one per launch; the runner otherwise
    # resolves the current device and stream itself on every call.
    try:
        return triton.runtime.driver.active.get_current_stream(index)
    except Exception:  # noqa: B902 - fall back to the runner's own lookup
        return None


def _run_plan(plan, mats, out):
    global _ACTIVE_DEVICE
    steps, bufs, final_shape = plan
    if out is None:
        # Allocate before the first launch: when the chain's first node is
        # short, an allocation sitting between two dependent launches leaves
        # the device idle, and a freshly allocated buffer is always 16-byte
        # aligned, which is what the pre-compiled runners were keyed on.
        out = torch.empty(final_shape, dtype=mats[0].dtype, device=mats[0].device)
        if out.data_ptr() & 15:
            return None
    vals = list(mats)
    vals.extend(bufs)
    vals.append(out)

    def fire(stream):
        use_stream = stream is not None and _runner_takes_stream(steps[0][0])
        for runner, tail, ai, bi, oi in steps:
            if use_stream:
                runner(vals[ai], vals[bi], vals[oi], *tail, stream=stream)
            else:
                runner(vals[ai], vals[bi], vals[oi], *tail)
        return vals[-1]

    index = mats[0].device.index or 0
    if index == _ACTIVE_DEVICE and torch_device_fn.current_device() == index:
        return fire(_current_stream(index))
    with torch_device_fn.device(mats[0].device):
        result = fire(_current_stream(index))
    _ACTIVE_DEVICE = index
    return result


def _forward_slow(mats, out=None):
    if len(mats) == 2:
        return _gemm(mats[0], mats[1], out=out)
    p = tuple(a.shape[0] for a in mats) + (mats[-1].shape[1],)
    order = _schedule(p)
    t = {(i, i): a for i, a in enumerate(mats)}
    last = len(order) - 1
    slots = {}
    for step, (i, j, k) in enumerate(order):
        a, b = t[i, k], t[k + 1, j]
        if step == last:
            t[i, j] = _gemm(a, b, out=out)
            continue
        shape = (a.shape[0], b.shape[1])
        key = (shape, a.dtype)
        slot = slots.get(key, 0)
        slots[key] = slot + 1
        t[i, j] = _gemm(a, b, out=_chain_scratch(shape, a.dtype, a.device, slot))
    return t[0, len(mats) - 1]


def _forward(mats, out=None):
    if len(mats) == 1:
        return _copy2d(mats[0], out if out is not None else torch.empty_like(mats[0]))
    key = _plan_key(mats, out)
    plan = _CHAIN_PLAN.get(key, _MISS)
    if plan is _MISS:
        if len(_CHAIN_PLAN) >= _CHAIN_PLAN_MAX:
            _CHAIN_PLAN.clear()
        plan = _build_plan(mats, out)
        _CHAIN_PLAN[key] = plan
    if plan is not None:
        result = _run_plan(plan, mats, out)
        if result is not None:
            return result
    return _forward_slow(mats, out)


class _Chain(torch.autograd.Function):
    @staticmethod
    def forward(*mats):
        return _forward(mats)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad):
        mats = ctx.saved_tensors
        n = len(mats)
        if n == 1:
            return (grad,)
        # Same node dispatch as the forward.
        _, t, order = _tree(mats)
        grads = {(0, n - 1): grad}
        for i, j, k in reversed(order):
            g = grads.pop((i, j))
            grads[i, k] = _gemm(g, t[k + 1, j].t())
            grads[k + 1, j] = _gemm(t[i, k].t(), g)
        return tuple(grads[i, i] for i in range(n))


def run(matrices, *args, out=None):
    logger.debug("GEMS_ENFLAME CHAIN_MATMUL")
    if isinstance(matrices, torch.Tensor):
        mats = [matrices, *args]
    elif isinstance(matrices, (list, tuple)):
        if args:
            raise TypeError("chain_matmul(): unexpected extra arguments")
        mats = list(matrices)
    else:
        raise TypeError("chain_matmul(): expected a list of tensors")
    if not mats:
        raise RuntimeError("chain_matmul(): Expected one or more matrices")
    if any(not isinstance(a, torch.Tensor) for a in mats):
        raise TypeError("expected tensors")
    dtype, device = mats[0].dtype, mats[0].device
    for i, a in enumerate(mats):
        if a.ndim != 2:
            raise RuntimeError(f"Tensor dimension is {a.ndim}, expected 2 instead.")
        if a.dtype != dtype or a.device != device:
            raise RuntimeError(
                "multi_dot(): all tensors must have the same dtype and device"
            )
        if not dtype.is_floating_point:
            raise RuntimeError("chain_matmul(): unsupported dtype")
        if i and mats[i - 1].shape[1] != a.shape[0]:
            raise RuntimeError("multi_dot(): tensors cannot be multiplied")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("out must be a Tensor")
        if out.dtype != dtype or out.device != device:
            raise RuntimeError("multi_dot(): out dtype or device mismatch")
        if tuple(out.shape) != (mats[0].shape[0], mats[-1].shape[1]):
            raise RuntimeError("multi_dot(): out shape mismatch")
    result = (
        _Chain.apply(*mats)
        if torch.is_grad_enabled() and any(a.requires_grad for a in mats)
        else _forward(mats, out)
    )
    if out is None:
        return result
    if result is not out:
        _copy2d(result, out)
    return out


def chain_matmul_out(matrices, *args, out):
    """aten::chain_matmul.out - same product, written into the caller's buffer."""
    logger.debug("GEMS_ENFLAME CHAIN_MATMUL_OUT")
    return run(matrices, *args, out=out)
