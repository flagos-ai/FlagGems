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

# mm is rolled back wholesale to the pre-optimization version per the fp16-first principle (fp16 was optimal).
# Deterministic large tiles (BK=512) consistently degrade fp16 at 4096^2: 3 benchmark rounds measured
# fp16 mean 1.0268x -> 0.9638x, fp16 4096^2 gems 0.998/0.968 ms -> 1.298/1.232 ms;
# while bf16 with the same config improved (0.7565x -> 0.8803x). The overall mean gain came at fp16's expense,
# which is unacceptable under the "fp16 must not regress" hard constraint, so the whole implementation is rolled back.
# Retained optimization conclusions are in the report and optimizer_agent_gcu400.md items 44/45/46.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    strategy=["log", "log", "log"],
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.jit
def mm_kernel(
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
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    K_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    # Unroll the K loop: gcu400 miscompiles a loop-carried fp32 tl.dot
    # accumulator inside a dynamic range loop (see mm_heur_k_blocks).
    for k in tl.static_range(K_BLOCKS):
        if EVEN_K:
            mask_a = (rm < M)[:, None]
            mask_b = (rn < N)[None, :]
            a = tl.load(A, mask_a, other=0)
            b = tl.load(B, mask_b, other=0)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            mask_a = (rm < M)[:, None] & (rk < k_remaining)[None, :]
            mask_b = (rk < k_remaining)[:, None] & (rn < N)[None, :]
            a = tl.load(A, mask=mask_a, other=0)
            b = tl.load(B, mask=mask_b, other=0)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a
    assert a in _ordered_datatypes
    assert b in _ordered_datatypes
    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


# ---------------------------------------------------------------------------
# Native-kernel delegation (capture + call_boxed), same pattern as the local
# bmm / router_gemm / linalg_multi_dot ops. The kernels are captured at import
# time - before any use_gems() registration replaces the PrivateUse1 entry -
# so call_boxed runs the original device kernel bit-identically with zero
# dispatch overhead (bmm precedent: verified 1.000-1.006x vs plain torch.mm).
# ---------------------------------------------------------------------------
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
_NATIVE_MM = None
_NATIVE_MM_OUT = None
if _CAN_CAPTURE_NATIVE_KERNEL:
    try:
        _NATIVE_MM = torch.library.get_kernel(
            torch.ops.aten.mm.default, runtime.device.dispatch_key
        )
        _NATIVE_MM_OUT = torch.library.get_kernel(
            torch.ops.aten.mm.out, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_MM = None
        _NATIVE_MM_OUT = None

_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)
# Dispatch line: total work M*N*K >= 2^29 routes to the native kernel.
# Measured anchor points on both sides of the line (this campaign, kernel
# mode do_bench median, raw-kernel launches):
#   384^3  Triton 1.59/1.75x (fp16/fp32)   512^3  Triton 1.70/1.72x
#   768^3  Triton 1.14/1.13/0.92x (16/16/32-bit)
#   1024^3 native wins  (Triton 0.94/0.93/0.70x)
#   2048^3 native wins  (Triton 0.69x)     4096^3 native wins (0.68-0.76x)
_NATIVE_MM_WORK_THRESHOLD = 1 << 29
# The M*N*K dispatch line is lower for 4-byte dtypes (fp32): an fp32 Triton tile is
# limited by DSM capacity, and around 704^3 (M*N*K ~ 3.5e8) Triton measures only
# 0.84x of the native kernel, while 384^3/512^3 are still clearly ahead
# (2.01x/1.36x), so the 4-byte dispatch line sits between the two.
_NATIVE_MM_WORK_THRESHOLD_4BYTE = 1 << 28
# Long-K skinny GEMMs (K >= 2^15): static-unroll safety + native C++ wins.
_MM_LONG_K_THRESHOLD = 1 << 15


def _native_mm_work_threshold(dtype):
    """Return the native-kernel M*N*K dispatch line for the element width of the input dtype."""
    return (
        _NATIVE_MM_WORK_THRESHOLD_4BYTE
        if dtype.itemsize >= 4
        else _NATIVE_MM_WORK_THRESHOLD
    )


_INT32_ELEMENT_LIMIT = 2**31

_MM_GEMM_CONFIGS = [
    # shared gcu400 "mm" configs (large-shape safety net, all num_warps=4)
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": 8},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    # num_warps=2 winners for small/medium GEMMs (router_gemm / tensordot /
    # linear precedent; DSM <= 917504B for every 16-bit entry)
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": 8},
        num_warps=2,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=2,
        num_stages=1,
    ),
    # linear-session winners (gcu400 NT-GEMM w2 sweet spot; fp32 medium
    # winner; small-MN/K-heavy candidates)
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 512, "GROUP_M": 16},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=2,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    # this-campaign candidates (symmetric twin of the proven (256,128,256,w2)
    # winner; wide fp32 tile for the saturated non-delegated segment)
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_M": 16},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
]


def _mm_heur_even_k(args):
    return args["K"] % args["BLOCK_K"] == 0


def _mm_heur_full_even(args):
    return (
        args["K"] % args["BLOCK_K"] == 0
        and args["M"] % args["BLOCK_M"] == 0
        and args["N"] % args["BLOCK_N"] == 0
    )


def _mm_heur_k_blocks(args):
    # Unrolled K-loop trip count: gcu400 miscompiles a loop-carried fp32
    # tl.dot accumulator inside a dynamic range loop (see mm_heur_k_blocks
    # in heuristics_config_utils), so the fast kernel keeps the static
    # unroll of the legacy kernel.
    return triton.cdiv(args["K"], args["BLOCK_K"])


def _mm_config_prune(configs, named_args, **kwargs):
    # Cap the static-unroll trip count: a tiny-BK tile on a huge K would
    # explode the unrolled code size (see _mm_heur_k_blocks). Large-BK
    # entries always survive; if even they exceed the cap (K > 64 * 512)
    # the host has already routed the call to the native kernel, and as a
    # last resort keep the largest-BK entries so the space is never empty.
    K = named_args["K"]
    kept = [cfg for cfg in configs if triton.cdiv(K, cfg.kwargs["BLOCK_K"]) <= 64]
    if not kept:
        max_bk = max(cfg.kwargs["BLOCK_K"] for cfg in configs)
        kept = [cfg for cfg in configs if cfg.kwargs["BLOCK_K"] == max_bk]
    return kept


@libentry()
@libtuner(
    configs=_MM_GEMM_CONFIGS,
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    prune_configs_by={"early_config_prune": _mm_config_prune},
    warmup=5,
    rep=10,
)
@triton.heuristics(
    {
        "EVEN_K": _mm_heur_even_k,
        "FULL_EVEN": _mm_heur_full_even,
        "K_BLOCKS": _mm_heur_k_blocks,
    }
)
@triton.jit
def mm_gemm_kernel(
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
    K_BLOCKS: tl.constexpr,
):
    # Tiled matmul C = A @ B with A (M, K), B (K, N); int32 addressing only
    # (the host guarantees every linear offset stays below 2**31).
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Unroll the K loop: gcu400 miscompiles a loop-carried fp32 tl.dot
    # accumulator inside a dynamic range loop (see _mm_heur_k_blocks).
    for k in tl.static_range(K_BLOCKS):
        if FULL_EVEN:
            a = tl.load(A)
            b = tl.load(B)
        elif EVEN_K:
            mask_a = (rm < M)[:, None]
            mask_b = (rn < N)[None, :]
            a = tl.load(A, mask_a, other=0)
            b = tl.load(B, mask_b, other=0)
        else:
            k_remaining = K - k * BLOCK_K
            mask_a = (rm < M)[:, None] & (rk < k_remaining)[None, :]
            mask_b = (rk < k_remaining)[:, None] & (rn < N)[None, :]
            a = tl.load(A, mask=mask_a, other=0)
            b = tl.load(B, mask=mask_b, other=0)
        if a.dtype != b.dtype:
            # Mixed input dtypes: promote both operands to the output dtype
            # before the dot (same contract as the generic mm kernel).
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)

    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    if FULL_EVEN:
        tl.store(C, acc)
    else:
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C, acc, mask=mask)


def _launch_legacy_mm(a, b, c, M, K, N):
    # int32-unsafe segment (a linear offset >= 2**31): stay on the legacy
    # kernel, whose behavior is byte-for-byte the pre-campaign one.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        dot_out_dtype=tl.float32,
        GROUP_M=8,
    )


def mm(a, b):
    logger.debug("GEMS_ENFLAME MM")
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # Saturated segment (large workload) and long-K skinny segment both go
    # to the captured native kernel - bit-identical to the torch production
    # path, and the native kernel takes arbitrary strides directly (no
    # .contiguous() copy). K >= 2^15 also disarms the static-unroll trip
    # count (see _mm_config_prune): skinny long-K GEMMs are launch-bound,
    # where the native C++ path is fastest anyway.
    if (
        _NATIVE_MM is not None
        and (
            M * N * K >= _native_mm_work_threshold(a.dtype) or K >= _MM_LONG_K_THRESHOLD
        )
        and a.device.type == runtime.device.name
    ):
        return _NATIVE_MM.call_boxed(_DEVICE_KEYSET, a, b)
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    if K == 0:
        # Empty contraction: the result is all zeros (ATen semantics); a
        # zero-iteration device loop clobbers the carried accumulator on
        # GCU400, so fill host-side.
        return torch.zeros((M, N), device=a.device, dtype=c_dtype)
    c = torch.empty((M, N), device=a.device, dtype=c_dtype)
    if M == 0 or N == 0:
        # Empty result: nothing to compute, no kernel launch.
        return c
    if (
        M * K >= _INT32_ELEMENT_LIMIT
        or K * N >= _INT32_ELEMENT_LIMIT
        or M * N >= _INT32_ELEMENT_LIMIT
    ):
        _launch_legacy_mm(a, b, c, M, K, N)
        return c
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    mm_gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_ENFLAME MM_OUT")
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    # Saturated segment: delegate to the captured native out= kernel. The
    # strict guard (dtype / contiguity / shape) keeps call_boxed on the
    # well-formed fast case; anything else falls through to the Triton
    # kernel, which writes row-strided outs (stride(0) != 1) correctly
    # through constexpr strides. NOTE: a non-unit *innermost* store stride
    # (stride(1) != 1) is a GCU400 2D-tile store limitation that silently
    # mis-writes on the legacy mm_kernel as well - the captured native
    # backend kernel cannot take such an out= either (cuBLAS-style ldc
    # constraint), so that pathological segment keeps its pre-campaign
    # behavior (documented in the report; torch production handles it via
    # a dispatcher-level temp+copy).
    if (
        _NATIVE_MM_OUT is not None
        and (
            M * N * K >= _native_mm_work_threshold(a.dtype) or K >= _MM_LONG_K_THRESHOLD
        )
        and a.device.type == runtime.device.name
        and out.dtype == c_dtype
        and out.is_contiguous()
        and tuple(out.shape) == (M, N)
    ):
        return _NATIVE_MM_OUT.call_boxed(_DEVICE_KEYSET, a, b, out=out)
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    c = out
    if K == 0:
        # Empty contraction: zeros per ATen semantics (see mm above).
        if tuple(out.shape) == (M, N):
            out.zero_()
        else:
            out.copy_(torch.zeros((M, N), device=a.device, dtype=c_dtype))
        return c
    if (
        M * K >= _INT32_ELEMENT_LIMIT
        or K * N >= _INT32_ELEMENT_LIMIT
        or M * N >= _INT32_ELEMENT_LIMIT
    ):
        _launch_legacy_mm(a, b, c, M, K, N)
        return c
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    mm_gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


# ---------------------------------------------------------------------------
# router_gemm: bf16 x bf16 -> fp32 GEMM for the MoE router gate.
#
# Measured on L600 (bench kernel mode, do_bench median w1000/r100, core
# shapes 384^3 / 1024^3 / 2048^3 / 4096^3, baseline = torch.mm(x, w.t())
# .to(float32)):
#   - the old host materialized `weight.t().contiguous()` on every call: an
#     extra copy kernel plus one more dispatch gap (4.2us at 384^3 on a
#     12us baseline). The kernel below takes B strides as constexpr, so the
#     (N, K) weight is passed directly in NT layout - no copy at all.
#   - the shared gcu400 "mm" tune configs are all num_warps=4; the probe
#     sweep found num_warps=2 winners for small/medium GEMMs (same blind
#     spot as tensordot): (256,128,256,w2,s2) 1024^3 28.1->21.2us (1.19x),
#     (128,128,512,w2,s2) 384^3 8.0->7.4us (1.64x). The 6 shared configs
#     stay in the list as the large-shape safety net.
#   - FULL_EVEN strips every load/store mask when M/N/K are tile-divisible
#     (bmm precedent: a runtime branch degrades tl.dot to scalar code).
#   - SPLIT_K stays pinned to 1: tl.atomic_add split-K reduction measured
#     30-240x slower on GCU400 (perf_playbook veto).
#   - once the GEMM saturates the device the native primo++ pipeline is
#     structurally ~1.3-1.8x faster than any swept Triton config (2048^3
#     0.77x / 4096^3 0.66x kernel-only, GROUP_M/stages/dot-acc variants all
#     within noise), so workloads >= 2^33 delegate to the captured aten::mm +
#     aten::_to_copy kernels via call_boxed - bit-identical to the torch
#     production path with ~zero dispatch overhead (1.002x/0.979x measured).
#     Below the line the fp32-accumulated Triton kernel wins (384^3 1.5x+,
#     1024^3 1.19x). The delegated segment stores bf16-rounded logits into
#     an fp32 tensor, matching the plain torch router path; everything below
#     the line keeps full fp32 accumulation.
# ---------------------------------------------------------------------------

_NATIVE_MM = None
_NATIVE_TO_COPY = None
if hasattr(torch.library, "get_kernel"):
    try:
        _NATIVE_MM = torch.library.get_kernel(
            torch.ops.aten.mm.default, runtime.device.dispatch_key
        )
        _NATIVE_TO_COPY = torch.library.get_kernel(
            torch.ops.aten._to_copy.default, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_MM = None
        _NATIVE_TO_COPY = None

_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)
_NATIVE_WORK_THRESHOLD = 1 << 33

_ROUTER_GEMM_CONFIGS = [
    # shared gcu400 "mm" configs (large-shape safety net, all num_warps=4)
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2
    ),
    # num_warps=2 winners (DSM <= 917504B for every entry)
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256}, num_warps=2, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256}, num_warps=2, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512}, num_warps=2, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512}, num_warps=2, num_stages=1
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 512}, num_warps=2, num_stages=2
    ),
    # Narrow-N configs: the router gate has N = num_experts in 8..128, so the
    # 128/256-wide column tiles above waste most of the dot lanes for N <= 32.
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 16, "BLOCK_K": 256}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 256}, num_warps=4, num_stages=2
    ),
    # Narrow-M configs: the router gate's M (num_tokens) is often 1..64 while
    # K is 4096..7168, so every tile above wastes (BLOCK_M - M) rows and the
    # K loop dominates. Measured on gcu400 (do_bench median, fp16/bf16 agree):
    #   M=1  / N=8  / K=4096  (16,16,1024,w2) 0.0129ms vs 0.2093ms for the old
    #                         pruned single survivor (32,32,32,w4)  -> 16x
    #   M=64 / N=8  / K=4096  (16,16,1024,w2) 0.0129ms vs 0.0514ms      -> 4.0x
    #   M=1024/N=8  / K=4096  (64,32,512,w2)  0.0244ms vs 0.0531ms      -> 2.2x
    #   M=64 / N=64 / K=7168  (16,16,1024,w2) 0.0251ms vs 0.0527ms      -> 2.1x
    # BLOCK_M=16 is the key: the previous list bottomed out at 32, and the
    # prune heuristic could only ever keep (32,32,32,w4) for these shapes.
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 1024}, num_warps=2, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 512}, num_warps=2, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 512}, num_warps=2, num_stages=2
    ),
]


def _router_gemm_prune_configs(configs, named_args, **kwargs):
    """Drop tiles that are far wider/taller than the problem.

    Keeps the autotune compile set proportional to the narrow router shapes
    (N is 8..128), which both cuts compile time and removes configs that can
    never win.
    """
    mat_m = named_args.get("M")
    mat_n = named_args.get("N")
    if not isinstance(mat_m, int) or not isinstance(mat_n, int):
        return configs
    max_bn = max(32, 2 * mat_n)
    max_bm = max(64, 4 * mat_m)
    kept = [
        cfg
        for cfg in configs
        if cfg.kwargs["BLOCK_N"] <= max_bn and cfg.kwargs["BLOCK_M"] <= max_bm
    ]
    return kept or configs[:1]


def _router_gemm_heur_even_k(args):
    return args["K"] % args["BLOCK_K"] == 0


def _router_gemm_heur_full_even(args):
    return (
        args["K"] % args["BLOCK_K"] == 0
        and args["M"] % args["BLOCK_M"] == 0
        and args["N"] % args["BLOCK_N"] == 0
    )


@libentry()
@libtuner(
    configs=_ROUTER_GEMM_CONFIGS,
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
    prune_configs_by={"early_config_prune": _router_gemm_prune_configs},
)
@triton.heuristics(
    {
        "EVEN_K": _router_gemm_heur_even_k,
        "FULL_EVEN": _router_gemm_heur_full_even,
    }
)
@triton.jit
def router_gemm_kernel(
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
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    FULL_EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if FULL_EVEN:
            a = tl.load(A)
            b = tl.load(B)
        elif EVEN_K:
            mask_a = (rm < M)[:, None]
            mask_b = (rn < N)[None, :]
            a = tl.load(A, mask_a, other=0)
            b = tl.load(B, mask_b, other=0)
        else:
            k_remaining = K - k * BLOCK_K
            mask_a = (rm < M)[:, None] & (rk < k_remaining)[None, :]
            mask_b = (rk < k_remaining)[:, None] & (rn < N)[None, :]
            a = tl.load(A, mask=mask_a, other=0)
            b = tl.load(B, mask=mask_b, other=0)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    if FULL_EVEN:
        tl.store(C, acc)
    else:
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C, acc, mask=mask)


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM for MoE router gate. weight shape: (N, K).

    Small/medium workloads run the fp32-accumulated Triton kernel (weight is
    consumed in NT layout straight from its (N, K) storage - no transpose
    copy); workloads >= 2^33 FLOPs delegate to the captured native aten::mm +
    aten::_to_copy pair, which is bit-identical to the plain torch router
    path (bf16-rounded logits stored in fp32).
    """
    logger.debug("GEMS_ENFLAME ROUTER_GEMM")
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    if min(M, N, K) == 0:
        return torch.zeros((M, N), device=x.device, dtype=torch.float32)
    if (
        M * N * K >= _NATIVE_WORK_THRESHOLD
        and _NATIVE_MM is not None
        and _NATIVE_TO_COPY is not None
        and x.device.type == runtime.device.name
    ):
        bf16 = _NATIVE_MM.call_boxed(_DEVICE_KEYSET, x, weight.t())
        return _NATIVE_TO_COPY.call_boxed(_DEVICE_KEYSET, bf16, dtype=torch.float32)
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    router_gemm_kernel[grid](
        x,
        weight,
        c,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(1),
        weight.stride(0),
        c.stride(0),
        c.stride(1),
        dot_out_dtype=tl.float32,
        GROUP_M=8,
    )
    return c
