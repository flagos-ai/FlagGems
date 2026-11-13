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

# mm 按 fp16 优先原则整体回退到优化前版本（fp16 最优）。
# 确定性大 tile（BK=512）在 fp16 4096² 上稳定劣化：3 轮 benchmark 实测
# fp16 均值 1.0268x -> 0.9638x，fp16 4096² gems 0.998/0.968 ms -> 1.298/1.232 ms；
# 而 bf16 同配置反而变好（0.7565x -> 0.8803x）。整体均值的提升来自牺牲 fp16，
# 按「fp16 不降」硬约束不可接受，因此回退整份实现。
# 保留的优化结论见报告与 optimizer_agent_gcu400.md 条目 44/45/46。

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
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
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


def mm(a, b):
    logger.debug("GEMS_ENFLAME MM")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=a.device, dtype=c_dtype)
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
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_ENFLAME MM_OUT")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c = out
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
]


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
