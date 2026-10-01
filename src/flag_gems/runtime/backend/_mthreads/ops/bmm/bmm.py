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

import logging

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems import runtime
from flag_gems.ops.bmm import bmm_out as default_bmm_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

from . import core, explicit, persistent, smallm, special, split, spmc

logger = logging.getLogger(__name__)

_C0_SHAPE = (16384, 7168, 1024)
_C1_SHAPE = (16384, 2112, 7168)
_C2_SHAPE = (448, 7168, 256)
_C3_SHAPE = (14429, 7168, 1024)
_C4_SHAPE = (14429, 2112, 7168)
_SMALL_M_SHAPE = (4, 15, 160, 1024)
_CORE_SMALL_SHAPE = (2, 384, 384, 384)
_CORE_PERSISTENT_SHAPES = {
    (2, 4096, 4096, 4096),
    (16, 1024, 1024, 1024),
    (16, 2048, 2048, 2048),
    (16, 4096, 4096, 4096),
}
_CORE_FP32_SHAPES = {
    (2, 384, 384, 384),
    (2, 4096, 4096, 4096),
    (16, 1024, 1024, 1024),
    (16, 2048, 2048, 2048),
    (16, 4096, 4096, 4096),
}


def is_supported_sqmma_layout(tensor):
    return tensor.is_contiguous() or (
        tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
    )


def is_sqmma_compatible(a, b, N, K):
    return (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and is_supported_sqmma_layout(a)
        and is_supported_sqmma_layout(b)
        and N % 8 == 0
        and K % 8 == 0
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
)
@triton.heuristics(runtime.get_heuristic_config("bmm"))
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    # batch offsets
    pid_b = ext.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    if IS_FP64:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float64)
    else:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask_a)
        b = tl.load(b_ptrs, mask_b)

        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)


def bmm_fma_out(A, B, out):
    logger.debug("GEMS_MTHREADS BMM_FMA")
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid_fn](A, B, out, M, N, K, IS_FP64=A.dtype == torch.float64)
    return out


def bmm_sqmma_descriptor_pre_hook(nargs):
    nargs["a_desc"].block_shape = [nargs["TILE_M"], nargs["TILE_K"]]
    nargs["b_desc"].block_shape = [nargs["TILE_K"], nargs["TILE_N"]]
    nargs["c_desc"].block_shape = [nargs["TILE_M"], nargs["TILE_N"]]


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {
                "TILE_M": 128,
                "TILE_N": 128,
                "TILE_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "TILE_M": 128,
                "TILE_N": 64,
                "TILE_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "TILE_M": 64,
                "TILE_N": 128,
                "TILE_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "TILE_M": 64,
                "TILE_N": 64,
                "TILE_K": 64,
                "GROUP_M": 4,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "TILE_M": 128,
                "TILE_N": 128,
                "TILE_K": 128,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "TILE_M": 128,
                "TILE_N": 128,
                "TILE_K": 256,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
    ],
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=5,
    flagtune_op_name="bmm",
    flagtune_pre_hook=bmm_sqmma_descriptor_pre_hook,
)
@triton.jit
def bmm_sqmma_kernel(
    a_desc,
    b_desc,
    c_desc,
    batch,
    M,
    N,
    K,
    stride_am,
    stride_bk,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_index = tl.program_id(axis=1)
    grid_m = tl.cdiv(M, TILE_M)
    grid_n = tl.cdiv(N, TILE_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    offs_am = (pid_m * TILE_M + batch_index * M).to(tl.int32)
    offs_bn = (pid_n * TILE_N).to(tl.int32)
    offs_ak = 0
    offs_ak = offs_ak.to(tl.int32)
    offs_bk = (batch_index * K).to(tl.int32)
    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_ak])
        b = tl.load_tensor_descriptor(b_desc, [offs_bk, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_ak += TILE_K
        offs_bk += TILE_K
    tl.store_tensor_descriptor(c_desc, [offs_am, offs_bn], accumulator.to(c_desc.dtype))


def bmm_sqmma_out(A, B, out, batch, M, N, K):
    desc_a = TensorDescriptor.from_tensor(A.reshape(batch * M, K), [1, 1])
    desc_b = TensorDescriptor.from_tensor(B.reshape(batch * K, N), [1, 1])
    desc_c = TensorDescriptor.from_tensor(out.reshape(batch * M, N), [1, 1])
    grid = lambda META: (
        triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),
        batch,
        1,
    )
    bmm_sqmma_kernel[grid](
        desc_a,
        desc_b,
        desc_c,
        batch,
        M,
        N,
        K,
        A.stride(1),
        B.stride(1),
    )
    return out


def bmm_sqmma(A, B, elem_type, batch, M, N, K):
    """Compatibility wrapper retained for the existing baddbmm SQMMA path."""

    c_type = elem_type if elem_type != torch.bfloat16 else torch.float16
    out = torch.empty((batch, M, N), dtype=torch.float16, device=A.device).to(c_type)
    return bmm_sqmma_out(A, B, out, batch, M, N, K)


def _normalized_shape(A, B):
    return A.shape[1], B.shape[2], A.shape[2]


def _validate_bmm_inputs(A, B):
    assert A.ndim == B.ndim == 3, "bmm expects rank-3 tensors"
    assert A.shape[0] == B.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    assert A.dtype == B.dtype, "Dtype mismatch"
    assert A.device == B.device, "Device mismatch"


def _validate_bmm_out(A, B, out):
    _validate_bmm_inputs(A, B)
    assert out.ndim == 3, "bmm expects a rank-3 output tensor"
    assert tuple(out.shape) == (
        A.shape[0],
        A.shape[1],
        B.shape[2],
    ), "Output shape mismatch"
    assert A.dtype == out.dtype, "Dtype mismatch"
    assert A.device == out.device, "Device mismatch"


def _contiguous_bf16_path_eligible(A, B, out, shape):
    return (
        A.shape[0] == B.shape[0] == 1
        and _normalized_shape(A, B) == shape
        and A.dtype == torch.bfloat16
        and A.is_contiguous()
        and B.is_contiguous()
        and out.is_contiguous()
    )


def _small_m_path_eligible(A, B, out):
    if (
        (A.shape[0], A.shape[1], B.shape[2], A.shape[2]) != _SMALL_M_SHAPE
        or A.dtype not in (torch.float16, torch.bfloat16)
        or not A.is_contiguous()
        or not out.is_contiguous()
    ):
        return False
    return B.is_contiguous() or tuple(B.stride()) == (160 * 1024, 1, 1024)


def _core_persistent_path_eligible(A, B, out):
    return (
        (A.shape[0], A.shape[1], B.shape[2], A.shape[2]) in _CORE_PERSISTENT_SHAPES
        and A.dtype in (torch.float16, torch.bfloat16)
        and A.dtype == B.dtype == out.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and out.is_contiguous()
    )


def _core_fp32_path_eligible(A, B, out):
    return (
        (A.shape[0], A.shape[1], B.shape[2], A.shape[2]) in _CORE_FP32_SHAPES
        and A.dtype == B.dtype == out.dtype == torch.float32
        and A.is_contiguous()
        and B.is_contiguous()
        and out.is_contiguous()
    )


def _core_small_path_eligible(A, B, out):
    return (
        (A.shape[0], A.shape[1], B.shape[2], A.shape[2]) == _CORE_SMALL_SHAPE
        and A.dtype in (torch.float16, torch.bfloat16)
        and A.dtype == B.dtype == out.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and out.is_contiguous()
    )


def _launch_special_path(A, B, out):
    path = special.dispatch_path(A, B)
    if path is None or not special.output_eligible(out, A, B):
        return False
    with torch_device_fn.device(A.device):
        special.launch(path, A, B, out)
    return True


def _general_bmm_out(A, B, out):
    batch, M, K = A.shape
    N = B.shape[2]
    if not out.is_contiguous():
        return default_bmm_out(A, B, out)
    if is_sqmma_compatible(A, B, N, K) and M >= 128:
        return bmm_sqmma_out(A, B, out, batch, M, N, K)
    return bmm_fma_out(A, B, out)


def _bmm_out_impl(A, B, out):
    if _core_small_path_eligible(A, B, out):
        return core.bmm_core_small_out(A, B, out)
    if _core_fp32_path_eligible(A, B, out):
        return core.bmm_core_fp32_out(A, B, out)
    if _core_persistent_path_eligible(A, B, out):
        return core.bmm_core_persistent_out(A, B, out)
    if _small_m_path_eligible(A, B, out):
        return smallm.bmm_smallm_out(A, B, out)
    if _launch_special_path(A, B, out):
        return out
    if _contiguous_bf16_path_eligible(A, B, out, _C0_SHAPE):
        persistent.bmm_persistent_out(
            A[0],
            B[0],
            out[0],
            group_m=2,
            unroll_k=2,
        )
        return out
    if _contiguous_bf16_path_eligible(A, B, out, _C1_SHAPE):
        persistent.bmm_persistent_out(
            A[0],
            B[0],
            out[0],
            group_m=4,
            unroll_k=1,
        )
        return out
    if _contiguous_bf16_path_eligible(A, B, out, _C2_SHAPE):
        return explicit.bmm_explicit_out(A, B, out)
    if _contiguous_bf16_path_eligible(A, B, out, _C3_SHAPE):
        split.bmm_split_out(A[0], B[0], out[0])
        return out
    if _contiguous_bf16_path_eligible(A, B, out, _C4_SHAPE):
        spmc.bmm_spmc_out(A[0], B[0], out[0])
        return out
    return _general_bmm_out(A, B, out)


def bmm(A, B):
    logger.debug("GEMS_MTHREADS BMM")
    _validate_bmm_inputs(A, B)
    out = torch.empty(
        (A.shape[0], A.shape[1], B.shape[2]), dtype=A.dtype, device=A.device
    )
    return _bmm_out_impl(A, B, out)


def bmm_out(A, B, out):
    logger.debug("GEMS_MTHREADS BMM_OUT")
    _validate_bmm_out(A, B, out)
    return _bmm_out_impl(A, B, out)
