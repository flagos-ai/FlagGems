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
distinct node K. The tile comes from ``_chain_config``: a per-dtype-class
workload bucket table fitted by a measured sweep (see the constants below),
with an L2-swizzled program order and mask-free loads/stores whenever the node
is tile-divisible. The alternatives were measured and rejected:

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
Whole-chain native delegation survives only as a dtype escape hatch.

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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.linalg_multi_dot import (
    _DEVICE_KEYSET,
    _matrix_multiply,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native-kernel capture (import time), same pattern as the local bmm / mm /
# linalg_multi_dot vendor ops. Only the whole-chain capture is kept, and only
# as a dtype escape hatch: no chain node ever runs a native GEMM here.
# ---------------------------------------------------------------------------
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
_NATIVE_CHAIN = None
_NATIVE_CHAIN_OUT = None
if _CAN_CAPTURE_NATIVE_KERNEL:
    try:
        _NATIVE_CHAIN = torch.library.get_kernel(
            torch.ops.aten.chain_matmul.default, runtime.device.dispatch_key
        )
        _NATIVE_CHAIN_OUT = torch.library.get_kernel(
            torch.ops.aten.chain_matmul.out, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_CHAIN = None
        _NATIVE_CHAIN_OUT = None

# Dtypes the chain node kernel takes. bfloat16 is included: the node kernel
# accumulates in fp32, and routing bf16 anywhere else would either hit the
# tuned ``mm`` (48s of first-touch compile) or a native composite.
_CHAIN_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Upper bound on any node dimension. Past this the caller falls through to the
# generic Triton GEMM of ``linalg_multi_dot`` - still Triton, never native.
_CHAIN_MAX_DIM = 4096
# Node tile table. Each entry is
# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages) and was fitted by
# an on-device sweep of the benchmark chain nodes (measured against the native
# GEMM at the same node, median of 100 reps):
#   16-bit  512x512x512      (128,128,512,w2,s1) 1.92x   (old 64/64 tile: 0.37x)
#   16-bit  1024x512x256     (128,128,512,w2,s1) 1.91x
#   16-bit  512x1024x256     (128,128,512,w2,s1) 1.71x
#   16-bit  1024^3           (256,128,256,w2,s2) 0.96x   (old tile: 0.26x)
#   32-bit  512^3            (128,128,512,w4,s1) 1.75x   (old tile: 0.53x)
#   32-bit  512x1024x256     (64,128,256,w2,s2)  1.84x
#   32-bit  1024^3           (256,128,128,w2,s2) 0.79x   (old tile: 0.41x)
# At 1024^3 the native primo++ pipeline saturates the device and no swept tile
# reached parity (best 0.96x 16-bit / 0.79x fp32); below that line the Triton
# tile wins. Tiles whose BLOCK_N or BLOCK_K exceeded the DSM budget, and the
# (128,256,*) / (256,256,*) families that measurably mis-compute on this
# backend, are excluded.
_CHAIN_CFG_16_SMALL = (128, 128, 512, 8, 2, 1)
_CHAIN_CFG_16_LARGE = (256, 128, 256, 8, 2, 2)
_CHAIN_CFG_32_SMALL = (128, 128, 512, 8, 4, 1)
_CHAIN_CFG_32_LARGE = (256, 128, 128, 8, 2, 2)
_CHAIN_CFG_TINY = (64, 128, 256, 8, 2, 2)
# Workload (M*N*K) where the native GEMM saturates the device: 1024^3 is
# 1.07e9, and every node above it measured the wide-BLOCK_M tile best.
_CHAIN_WORK_LARGE = 1 << 30
# Below this the node is launch-bound and a narrower tile spreads better.
_CHAIN_WORK_TINY = 1 << 27
_CHAIN_MIN_BLOCK_K = 32
# Per-program DSM budget of the matrix unit. (BLOCK_M + BLOCK_N) * BLOCK_K *
# itemsize * num_stages must stay under it or the kernel does not launch.
_CHAIN_DSM_BUDGET_BYTES = 917504


def _should_delegate_to_native(mats, native_kernel):
    """Metadata-only dtype escape hatch for the whole chain.

    Every chain length stays on the Triton node kernel; only a dtype the node
    kernel cannot take hands the chain to the native composite.
    """
    if native_kernel is None:
        return False
    if not mats:
        return False
    first = mats[0]
    if first.device.type == "cpu":
        return False
    return first.dtype not in _CHAIN_DTYPES


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
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    FULL_EVEN: tl.constexpr,
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
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # L2-friendly program order: sweep GROUP_M row tiles per column stripe.
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        if FULL_EVEN:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        elif EVEN_K:
            a = tl.load(a_ptrs, rm[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, rn[None, :] < N, other=0.0)
        else:
            mask_k = (k_start + rk) < K
            a = tl.load(
                a_ptrs, (rm[:, None] < M) & mask_k[None, :], other=0.0
            )
            b = tl.load(
                b_ptrs, mask_k[:, None] & (rn[None, :] < N), other=0.0
            )
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


def _chain_config(M, N, K, itemsize):
    """Tile and launch configuration for one chain node (metadata only)."""
    work = M * N * K
    if itemsize >= 4:
        if work >= _CHAIN_WORK_LARGE:
            config = _CHAIN_CFG_32_LARGE
        elif work >= _CHAIN_WORK_TINY:
            config = _CHAIN_CFG_32_SMALL
        else:
            config = _CHAIN_CFG_TINY
    else:
        if work >= _CHAIN_WORK_LARGE:
            config = _CHAIN_CFG_16_LARGE
        elif work >= _CHAIN_WORK_TINY:
            config = _CHAIN_CFG_16_SMALL
        else:
            config = _CHAIN_CFG_TINY
    block_m, block_n, block_k, group_m, num_warps, num_stages = config
    # ``tl.dot`` always computes the whole tile, so a tile wider than the node
    # is pure waste: a 32x32x2048 node measured 0.34x of native with a
    # (64,128,256) tile and 1.1x once the tile shrank to the node (16 is the
    # smallest legal ``tl.dot`` edge).
    block_m = max(16, min(block_m, triton.next_power_of_2(M)))
    block_n = max(16, min(block_n, triton.next_power_of_2(N)))
    k_pow2 = max(_CHAIN_MIN_BLOCK_K, triton.next_power_of_2(K))
    if k_pow2 < block_k:
        # A contraction shorter than the tile wastes the whole K stage.
        block_k = k_pow2
    # Every tier must still fit the DSM budget of the matrix unit.
    while (
        block_k > _CHAIN_MIN_BLOCK_K
        and (block_m + block_n) * block_k * itemsize * num_stages
        > _CHAIN_DSM_BUDGET_BYTES
    ):
        block_k //= 2
    return block_m, block_n, block_k, group_m, num_warps, num_stages


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
            left, right, out, *tail, grid=grid3, num_warps=num_warps,
            num_stages=num_stages,
        )
        runner = compiled[grid3]
    except Exception:  # noqa: B902 - any failure just means "use the JIT"
        runner = None
    _CHAIN_LAUNCH[key] = runner
    return runner


def _chain_gemm(left, right, out=None):
    global _ACTIVE_DEVICE
    M, K = left.shape
    N = right.shape[1]
    block_m, block_n, block_k, group_m, num_warps, num_stages = _chain_config(
        M, N, K, left.dtype.itemsize
    )
    if out is None:
        out = torch.empty((M, N), dtype=left.dtype, device=left.device)
    tile_count = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    grid3 = (tile_count, 1, 1)
    stride_am, stride_ak = left.stride(0), left.stride(1)
    stride_bk = right.stride(0)
    stride_bn = right.stride(1)
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
        group_m,
        num_warps,
        num_stages,
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
            group_m,
            (K % block_k == 0),
            (K % block_k == 0 and M % block_m == 0 and N % block_n == 0),
        )
        runner = _build_node_launcher(
            left, right, out, key, tail, grid3, num_warps, num_stages
        )
        entry = None if runner is None else (runner, tail, grid3)
        _CHAIN_LAUNCH[key] = entry
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
    _chain_gemm_kernel[grid3](
        left,
        right,
        out,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        EVEN_K=(K % block_k == 0),
        FULL_EVEN=(K % block_k == 0 and M % block_m == 0 and N % block_n == 0),
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


def _forward(mats, out=None):
    native_kernel = _NATIVE_CHAIN if out is None else _NATIVE_CHAIN_OUT
    if _should_delegate_to_native(mats, native_kernel):
        if out is not None:
            return native_kernel.call_boxed(_DEVICE_KEYSET, list(mats), out=out)
        return native_kernel.call_boxed(_DEVICE_KEYSET, list(mats))
    if len(mats) == 1:
        return _copy2d(mats[0], out if out is not None else torch.empty_like(mats[0]))
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
        t[i, j] = _gemm(
            a, b, out=_chain_scratch(shape, a.dtype, a.device, slot)
        )
    return t[0, len(mats) - 1]


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
