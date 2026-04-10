# SPDX-License-Identifier: Apache-2.0
# QC-GEM: Quantized Computing GEM library for FlagGems
# Main GEMM kernels for quantized matrix multiplication

import torch, math, copy
from torch import Tensor
import triton
import triton.language as tl

from .dtypes import is_mx_dtype
from .config import AUTOTUNE
from .utils import *


KEYS = ['M_CLOSEST', 'N', 'K', 'group_size', 'elements_per_sample', 'type_id', 'a_sizeof', 'b_sizeof']
MATMUL_TYPE = "GEMM"


def kernel_config_pruner(configs, nargs, **kwargs):
    from .core import QCGEM_TRITON_CONFIG_CACHE

    m = nargs['M']
    n = nargs['N']
    k = nargs['K']
    g = nargs['group_size']
    e = nargs['elements_per_sample']
    t = nargs['type_id']
    a_sizeof = nargs['a_sizeof']
    b_sizeof = nargs['b_sizeof']

    if MATMUL_TYPE in QCGEM_TRITON_CONFIG_CACHE:
        signature = str(tuple([get_closest_m(m), n, k, g, e, t]))
        if signature in QCGEM_TRITON_CONFIG_CACHE[MATMUL_TYPE]:
            config = copy.deepcopy(QCGEM_TRITON_CONFIG_CACHE[MATMUL_TYPE][signature])
            num_stages = config.pop('num_stages')
            num_warps = config.pop('num_warps')
            num_ctas = config.pop('num_ctas')

            config.pop('num_buffers_warp_spec', None)
            config.pop('num_consumer_groups', None)
            config.pop('reg_dec_producer', None)
            config.pop('reg_inc_consumer', None)
            config["NUM_STAGES"] = num_stages

            yield triton.Config(config, num_stages=num_stages, num_warps=num_warps)
            return

    gpu_shared_memory = get_gpu_shared_memory()
    load_scales_as_block = kwargs['load_scales_as_block']
    used = set()

    if k < 32:
        return

    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = config.kwargs['BLOCK_SIZE_M']
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])

        A_load_order = config.kwargs['A_load_order']
        num_stages = config.num_stages
        num_warps = config.num_warps

        if m <= 16:
            block_size_m = 16
        elif m <= 32:
            block_size_m = min(max(block_size_m, 16), 32)
        elif m <= 64:
            block_size_m = min(max(block_size_m, 32), 64)
        elif m <= 128:
            block_size_m = min(max(block_size_m, 64), 128)
        elif m <= 256:
            block_size_m = min(max(block_size_m, 64), 256)
        else:
            block_size_m = min(max(block_size_m, 64), 256)

        if load_scales_as_block:
            num_stages = max(num_stages, 2)
            if e > 1:
                block_size_k = max(block_size_k, 64)
            else:
                block_size_k = max(block_size_k, 32)
        else:
            block_size_k = min(block_size_k, g)

        block_size_k = next_power_of_2(block_size_k)
        block_size_n = next_power_of_2(block_size_n)

        if not IS_HIP:
            if e > 1 and not load_scales_as_block:
                num_stages = min(num_stages, 4)
            if e == 1 and num_stages == 1:
                continue

        while num_stages > 0 and not load_scales_as_block:
            shared_mem = (block_size_m * block_size_k * a_sizeof + block_size_k * block_size_n * b_sizeof)
            if e > 1:
                shared_mem += block_size_k * block_size_n * a_sizeof
            shared_mem *= num_stages
            if int(shared_mem) <= gpu_shared_memory:
                break
            num_stages -= 1

        if num_stages == 0:
            continue

        if load_scales_as_block:
            block_size_k = min(block_size_k, 256)

        if block_size_k < 32 or block_size_m < 1 or block_size_n < 32:
            continue

        key = (block_size_m, block_size_n, block_size_k, group_size_m, A_load_order, num_stages, num_warps)

        new_config = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "GROUP_SIZE_M": group_size_m,
            "A_load_order": A_load_order,
            "NUM_STAGES": num_stages,
        }

        if IS_HIP:
            new_config['waves_per_eu'] = config.kwargs.get('waves_per_eu', 0)
            new_config['matrix_instr_nonkdim'] = config.kwargs.get('matrix_instr_nonkdim', 16)
            key = key + (new_config['waves_per_eu'], new_config['matrix_instr_nonkdim'])

        if key in used:
            continue

        used.add(key)
        yield triton.Config(new_config, num_stages=num_stages, num_warps=num_warps)

    if not used:
        block_size_m = 16 if m <= 16 else min(32, m) if m <= 32 else min(64, m)
        block_size_k = min(k, 256) if load_scales_as_block else min(k, g)
        block_size_k = next_power_of_2(block_size_k)
        block_size_k = max(block_size_k, 32)
        while block_size_k > 32 and not is_divisible(k, block_size_k):
            block_size_k //= 2
        block_size_k = max(block_size_k, 32)
        block_size_n = min(n, next_power_of_2(64))
        block_size_n = max(block_size_n, 32)
        num_stages = 2
        for _ in range(4):
            shared_mem = (block_size_m * block_size_k * a_sizeof + block_size_k * block_size_n * b_sizeof)
            if e > 1:
                shared_mem += block_size_k * block_size_n * a_sizeof
            shared_mem *= num_stages
            if int(shared_mem) <= gpu_shared_memory:
                break
            num_stages -= 1
        if num_stages < 1:
            num_stages = 1
        new_config = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "GROUP_SIZE_M": 8,
            "A_load_order": 0,
            "NUM_STAGES": num_stages,
        }
        if IS_HIP:
            new_config["waves_per_eu"] = 0
            new_config["matrix_instr_nonkdim"] = 16
        yield triton.Config(new_config, num_stages=num_stages, num_warps=4)


def get_max_autotune_config_nvidia():
    stages = [1, 4, 5] if gpu_has_more_shared_memory() else [1, 2, 4]
    configs = []
    for A in [0, 2]:
        for w in [4, 8]:
            for s in stages:
                for M in [16, 32, 64, 128, 256]:
                    for N in [32, 64, 128, 256, 512]:
                        for K in [32, 64, 128, 256, 512]:
                            configs.append(
                                triton.Config(
                                    {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K, "GROUP_SIZE_M": 8, "A_load_order": A},
                                    num_warps=w, num_stages=s,
                                )
                            )
    return configs


def get_fast_autotune_config_nvidia():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'A_load_order': 2}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'A_load_order': 2}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'A_load_order': 2}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8, 'A_load_order': 0}, num_warps=4, num_stages=5))

    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'A_load_order': 0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'A_load_order': 0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'A_load_order': 0}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8, 'A_load_order': 2}, num_warps=8, num_stages=4))

    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'A_load_order': 2}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'A_load_order': 0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'A_load_order': 2}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8, 'A_load_order': 0}, num_warps=4, num_stages=4))

    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'A_load_order': 2}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'A_load_order': 0}, num_warps=8, num_stages=4))

    configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'A_load_order': 2}, num_warps=8, num_stages=3))
    return configs


def get_default_config_nvidia():
    return [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'A_load_order': 0, 'NUM_STAGES': 4}, num_warps=4, num_stages=4)]


if IS_HIP:
    get_max_autotune_config = get_max_autotune_config_nvidia
    get_fast_autotune_config = get_fast_autotune_config_nvidia
    get_default_config = get_default_config_nvidia
else:
    get_max_autotune_config = get_max_autotune_config_nvidia
    get_fast_autotune_config = get_fast_autotune_config_nvidia
    get_default_config = get_default_config_nvidia


AUTOTUNE_SETTING = AUTOTUNE.GEMM
if AUTOTUNE_SETTING == 'max':
    get_autotune_config = get_max_autotune_config
elif AUTOTUNE_SETTING == 'fast':
    get_autotune_config = get_fast_autotune_config
else:
    get_autotune_config = get_default_config


@triton.autotune(
    configs=get_autotune_config(),
    key=KEYS,
    prune_configs_by={'early_config_prune': kernel_config_pruner},
    use_cuda_graph=AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.jit
def gemm_INT_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N, K, M_CLOSEST,
    W_nbits: tl.constexpr,
    group_size: tl.constexpr,
    unpack_mask: tl.constexpr,
    elements_per_sample: tl.constexpr,
    type_id: tl.constexpr,
    a_sizeof: tl.constexpr,
    b_sizeof: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_a_m, stride_meta_a_g,
    stride_meta_g, stride_meta_n,
    load_scales_as_block,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr,
    A_load_order: tl.constexpr,
    data_contiguous: tl.constexpr,
    meta_evict_policy: tl.constexpr = '',
    a_evict: tl.constexpr = '',
    b_evict: tl.constexpr = '',
):
    pid = tl.program_id(axis=0)

    if elements_per_sample > 1:
        pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
    else:
        pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if data_contiguous:
        offs_bn = offs_n
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_ak = offs_k
    offs_bk = offs_k

    b_ptrs = b_ptr + ((offs_bk[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn)
    q_shift = ((offs_bk % elements_per_sample) * W_nbits).to(tl.int32)[:, None]

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    a_mask = ((offs_am[:, None] < M) & (offs_ak[None, :] < K)).to(tl.int1)

    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs = zeros_ptr + offs_bn[None, :] * stride_meta_n

    stride_mul: tl.constexpr = BLOCK_SIZE_K / group_size
    BLOCK_SIZE_K_P: tl.constexpr = (BLOCK_SIZE_K // elements_per_sample)

    if zero_is_scalar:
        zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(num_pid_k):
        if A_load_order == 0:
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

        b = tl.load(b_ptrs, eviction_policy=b_evict)

        if A_load_order == 1:
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

        if W_group_mode > 0:
            k_m = (k * stride_mul).to(tl.int32)

        if W_group_mode >= 2:
            scales = tl.load(scales_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy)
        else:
            scales = None

        if W_group_mode == 1 or W_group_mode >= 3:
            if zero_is_scalar:
                zeros = zero_scalar
            else:
                zeros = tl.load(zeros_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy)
        else:
            zeros = None

        if A_load_order == 2:
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

        b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

        if A_load_order == 3:
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

        acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K_P * stride_bk

    if channel_scale_mode == 1:
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc = acc.to(meta_dtype) * scales_b[None, :]

    if channel_scale_mode == 2:
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=meta_dtype)
        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if channel_scale_mode == 3:
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    acc = acc.to(output_dtype)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.autotune(
    configs=get_autotune_config(),
    key=KEYS,
    prune_configs_by={'early_config_prune': kernel_config_pruner},
    use_cuda_graph=AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.jit
def gemm_MX_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N, K, M_CLOSEST,
    W_nbits: tl.constexpr,
    group_size: tl.constexpr,
    unpack_mask: tl.constexpr,
    elements_per_sample: tl.constexpr,
    type_id: tl.constexpr,
    a_sizeof: tl.constexpr,
    b_sizeof: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_a_m: tl.constexpr, stride_meta_a_g: tl.constexpr,
    stride_meta_n: tl.constexpr, stride_meta_g: tl.constexpr,
    load_scales_as_block,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr,
    A_load_order: tl.constexpr,
    data_contiguous: tl.constexpr,
    meta_evict_policy: tl.constexpr = '',
    a_evict: tl.constexpr = '',
    b_evict: tl.constexpr = '',
    meta_scale_norm: tl.constexpr = (0.05 ** 2),
):
    pid = tl.program_id(axis=0)
    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    a_ptr_dtype: tl.constexpr = a_ptr.dtype.element_ty
    if a_ptr_dtype == tl.float16:
        a_dtype: tl.constexpr = "fp16"
        elements_per_sample_a: tl.constexpr = 1
    if a_ptr_dtype == tl.bfloat16:
        a_dtype: tl.constexpr = "bf16"
        elements_per_sample_a: tl.constexpr = 1
    if a_ptr_dtype == tl.float8e4nv:
        a_dtype: tl.constexpr = "e4m3"
        elements_per_sample_a: tl.constexpr = 1
    if a_ptr_dtype == tl.uint8:
        a_dtype: tl.constexpr = "e2m1"
        elements_per_sample_a: tl.constexpr = 2

    if elements_per_sample == 1:
        b_dtype: tl.constexpr = "e4m3"
    if elements_per_sample == 2:
        b_dtype: tl.constexpr = "e2m1"

    BLOCK_SIZE_K_A: tl.constexpr = BLOCK_SIZE_K // elements_per_sample_a
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ak = tl.arange(0, BLOCK_SIZE_K_A)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    a_mask = ((offs_am[:, None] < M) & (offs_ak[None, :] < K // elements_per_sample_a)).to(tl.int1)

    BLOCK_SIZE_K_B: tl.constexpr = BLOCK_SIZE_K // elements_per_sample
    offs_bk = tl.arange(0, BLOCK_SIZE_K_B)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    stride_mul: tl.constexpr = BLOCK_SIZE_K / group_size
    BLOCK_SIZE_K_S: tl.constexpr = BLOCK_SIZE_K // group_size
    offs_k_scales = tl.arange(0, BLOCK_SIZE_K_S)
    offs_n_b_scales = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    scales_b_ptrs = scales_ptr + offs_n_b_scales[:, None] * stride_meta_n + offs_k_scales[None, :] * stride_meta_g

    if channel_scale_mode == 4:
        scales_a_ptrs = scales_a_ptr + offs_am[:, None] * stride_meta_a_m + offs_k_scales[None, :] * stride_meta_a_g

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in tl.range(num_pid_k, num_stages=NUM_STAGES):
        a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        b = tl.load(b_ptrs, eviction_policy=b_evict)

        k_m = k * BLOCK_SIZE_K_S
        scales_b = tl.load(scales_b_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy)

        if channel_scale_mode == 4:
            scales_a = tl.load(scales_a_ptrs + k_m * stride_meta_a_g, eviction_policy=meta_evict_policy)
        else:
            scales_a = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K_S), value=127, dtype=tl.uint8)

        acc = tl.dot_scaled(a, scales_a, a_dtype, b, scales_b, b_dtype, acc)

        a_ptrs += BLOCK_SIZE_K_A * stride_ak
        b_ptrs += BLOCK_SIZE_K_B * stride_bk

    if group_size == 16:
        acc *= meta_scale_norm

    if channel_scale_mode == 2:
        dtype: tl.constexpr = c_ptr.dtype.element_ty
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=dtype)
        acc = acc.to(dtype) * (scales_a[:, None] * scales_b[None, :])

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = ((offs_cm[:, None] < M) & (offs_cn[None, :] < N)).to(tl.int1)
    tl.store(c_ptrs, acc, mask=mask)


def gemm_forward(x, W_q, scales, zeros, scales_x,
                 W_nbits, group_size, unpack_mask, elements_per_sample,
                 input_dtype, output_dtype, acc_dtype, meta_dtype,
                 channel_scale_mode, W_group_mode, data_contiguous, type_id):

    M, K, N = x.shape[0], W_q.shape[0] * elements_per_sample, W_q.shape[1]
    M_CLOSEST = get_closest_m(M)

    output = torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype])

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    if scales_x is not None:
        stride_meta_a_m, stride_meta_a_g = scales_x.stride(0), scales_x.stride(1)
    else:
        stride_meta_a_m, stride_meta_a_g = None, None

    if is_mx_dtype(input_dtype):
        gemm_kernel = gemm_MX_kernel
        load_scales_as_block = True
    else:
        gemm_kernel = gemm_INT_kernel
        load_scales_as_block = False

    gemm_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K, M_CLOSEST,
        W_nbits, group_size, unpack_mask, elements_per_sample,
        type_id, x.dtype.itemsize, W_q.dtype.itemsize,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        stride_meta_a_m, stride_meta_a_g,
        scales.stride(0), scales.stride(1),
        load_scales_as_block=load_scales_as_block,
        input_dtype=DTYPE_TO_TRITON[input_dtype],
        output_dtype=TORCH_DTYPE_TO_TRITON[output.dtype],
        acc_dtype=DTYPE_TO_TRITON[acc_dtype],
        meta_dtype=DTYPE_TO_TRITON[meta_dtype],
        channel_scale_mode=channel_scale_mode,
        W_group_mode=W_group_mode,
        zero_is_scalar=zeros.numel() == 1,
        data_contiguous=data_contiguous,
    )

    return output


class gemm:
    kernel = [gemm_INT_kernel, gemm_MX_kernel]
    forward = gemm_forward
    matmul_type = MATMUL_TYPE


__all__ = ["gemm"]
