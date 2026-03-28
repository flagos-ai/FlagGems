# SPDX-License-Identifier: Apache-2.0
# QC-GEM: GEMM Split-K Persistent kernels for FlagGems
# Ported from GemLite by Dr. Hicham Badri @Mobius Labs GmbH

import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl

from .dtypes import is_mx_dtype, DType, DTYPE_TO_TORCH, TORCH_DTYPE_TO_TRITON, DTYPE_TO_TRITON
from .config import AUTOTUNE
from .utils import (
    dequantize, init_to_zero, next_power_of_2, is_hip, is_divisible, swizzle_tile, swizzle_tile_persistent,
    get_gpu_shared_memory, gpu_has_more_shared_memory, get_closest_m,
    gpu_supports_bfloat16_atomicadd,
)


KEYS = ['M_CLOSEST', 'N', 'K', 'group_size', 'elements_per_sample', 'type_id', 'a_sizeof', 'b_sizeof']
MATMUL_TYPE = "GEMM_SPLITK_PERSISTENT"
NATIVE_ATOMIC = gpu_supports_bfloat16_atomicadd()


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

            yield triton.Config(config,
                num_stages=num_stages,
                num_warps=num_warps,
                pre_hook=init_to_zero("c_ptr") if (config['SPLIT_K'] > 1) else None,
            )
            return

    gpu_shared_memory = get_gpu_shared_memory()
    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = config.kwargs['BLOCK_SIZE_M']
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])
        split_k = config.kwargs['SPLIT_K']

        A_load_order = config.kwargs['A_load_order']
        num_stages = config.num_stages
        num_warps = config.num_warps

        if m <= 16:
            block_size_m = 16
        elif m <= 32:
            block_size_m = min(max(block_size_m, 16), 32)
        elif m <= 64:
            block_size_m = min(max(block_size_m, 32), 64)
        elif m > 64:
            block_size_m = 64

        if m >= 32:
            split_k = min(split_k, 8)

        block_size_k = min(block_size_k, g)
        block_size_k = next_power_of_2(block_size_k)
        block_size_n = next_power_of_2(block_size_n)

        while split_k > 1 and not is_divisible(k, block_size_k * split_k):
            split_k //= 2

        if not is_hip():
            if e > 1:
                num_stages = min(num_stages, 4)
            if e == 1 and num_stages == 1:
                continue

        while num_stages > 0:
            shared_mem = (block_size_m * block_size_k * a_sizeof + block_size_k * block_size_n * b_sizeof)
            if e > 1:
                shared_mem += block_size_k * block_size_n * a_sizeof
            shared_mem *= num_stages
            if int(shared_mem) <= gpu_shared_memory:
                break
            num_stages -= 1

        if num_stages == 0:
            continue

        if block_size_k < 16 or block_size_m < 1 or block_size_n < 1:
            continue

        key = (block_size_m, block_size_n, block_size_k, group_size_m, split_k, A_load_order, num_stages, num_warps)

        new_config = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "GROUP_SIZE_M": group_size_m,
            "SPLIT_K": split_k,
            "A_load_order": A_load_order,
        }

        if is_hip():
            new_config['waves_per_eu'] = config.kwargs.get('waves_per_eu', 0)
            new_config['matrix_instr_nonkdim'] = config.kwargs.get('matrix_instr_nonkdim', 16)
            key = key + (new_config['waves_per_eu'], new_config['matrix_instr_nonkdim'])

        if key in used:
            continue

        used.add(key)
        yield triton.Config(new_config,
            num_stages=num_stages,
            num_warps=num_warps,
            pre_hook=init_to_zero("c_ptr") if split_k > 1 else None,
        )

    if not used:
        block_size_m = 16 if m <= 16 else min(32, m) if m <= 32 else min(64, m)
        block_size_k = min(k, g)
        block_size_k = next_power_of_2(block_size_k)
        block_size_k = max(block_size_k, 16)
        while block_size_k > 16 and not is_divisible(k, block_size_k):
            block_size_k //= 2
        block_size_k = max(block_size_k, 16)
        block_size_n = min(n, next_power_of_2(64))
        block_size_n = max(block_size_n, 16)
        split_k = 1
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
            "SPLIT_K": split_k,
            "A_load_order": 0,
        }
        if is_hip():
            new_config["waves_per_eu"] = 0
            new_config["matrix_instr_nonkdim"] = 16
        yield triton.Config(new_config,
            num_stages=num_stages,
            num_warps=4,
            pre_hook=None,
        )


def get_max_autotune_config_nvidia():
    stages = [1, 2, 4, 5] if gpu_has_more_shared_memory() else [1, 2, 4]
    configs = []
    for A in [0, 2]:
        for w in [4, 8]:
            for s in stages:
                for M in [16, 32, 64]:
                    for N in [32, 64, 128, 256, 512]:
                        for K in [32, 64, 128, 256, 512]:
                            for split_k in [1, 2, 4, 8, 16]:
                                configs.append(
                                    triton.Config(
                                        {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K,
                                        "SPLIT_K": split_k, "GROUP_SIZE_M": 8, "A_load_order": A},
                                        num_warps=w, num_stages=s,
                                    )
                                )
    return configs


def get_fast_autotune_config_nvidia():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':64,  'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':256, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':512, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32,  'SPLIT_K':8, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':512, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'SPLIT_K':8, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'SPLIT_K':2, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':256, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':512, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':512, 'BLOCK_SIZE_K':32,  'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=4))
    return configs


def get_default_config_nvidia():
    return [triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=2)]


def get_max_autotune_config_amd():
    configs = []
    for A in [0]:
        for w in [4, 8]:
            for s in [1, 2]:
                for v in [0, 2, 4]:
                    for M in [16, 32, 64]:
                        for N in [32, 64, 128, 256, 512]:
                            for K in [32, 64, 128, 256, 512]:
                                for split_k in [1, 2, 4, 8, 16]:
                                    configs.append(
                                        triton.Config(
                                            {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K,
                                            "SPLIT_K": split_k, "GROUP_SIZE_M": 8, "A_load_order": A, 'waves_per_eu': v},
                                            num_warps=w, num_stages=s,
                                        )
                                    )
    return configs


def get_fast_autotune_config_amd():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':0}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':0}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':4, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':512, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'SPLIT_K':2, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':2, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':4, 'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':8, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K':8, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':512, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':4 ,'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':8 ,'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':2, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':0}, num_warps=8, num_stages=1))
    return configs


def get_default_config_amd():
    return [triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=2)]


if is_hip():
    get_max_autotune_config = get_max_autotune_config_amd
    get_fast_autotune_config = get_fast_autotune_config_amd
    get_default_config = get_default_config_amd
else:
    get_max_autotune_config = get_max_autotune_config_nvidia
    get_fast_autotune_config = get_fast_autotune_config_nvidia
    get_default_config = get_default_config_nvidia


AUTOTUNE_SETTING = AUTOTUNE.GEMM_SPLITK
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
def gemm_splitK_persistent_kernel(
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
    stride_meta_g, stride_meta_n,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
    A_load_order: tl.constexpr,
    data_contiguous: tl.constexpr,
    NUM_SMS: tl.constexpr,
    meta_evict_policy: tl.constexpr = '',
    atomic_mode: tl.constexpr = 'relaxed',
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
):
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    num_tiles = num_pid_m * num_pid_n

    num_sms = NUM_SMS

    tile_id_c = pid - num_sms
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in range(pid, num_tiles, num_sms):
        pid_m, pid_n = swizzle_tile_persistent(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        if data_contiguous:
            offs_bn = offs_n
        else:
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

        offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)

        scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
        zeros_ptrs = zeros_ptr + offs_bn[None, :] * stride_meta_n
        stride_mul: tl.constexpr = BLOCK_SIZE_K / group_size
        if zero_is_scalar:
            zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')

        a_ptrs = a_ptr + offs_am[:, None] * stride_am
        b_ptrs = b_ptr + offs_bn[None, :] * stride_bn
        offs_k_base = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, num_pid_k):
            offs_ak = offs_k_base + k * (BLOCK_SIZE_K * SPLIT_K)
            offs_bk = offs_ak // elements_per_sample
            a_mask = ((offs_am[:, None] < M) & (offs_ak[None, :] < K)).to(tl.int1)

            if A_load_order == 0:
                a = tl.load(a_ptrs + offs_ak[None, :] * stride_ak, mask=a_mask, other=0., eviction_policy=a_evict)

            b = tl.load(b_ptrs + offs_bk[:, None] * stride_bk, eviction_policy=b_evict)

            if A_load_order == 1:
                a = tl.load(a_ptrs + offs_ak[None, :] * stride_ak, mask=a_mask, other=0., eviction_policy=a_evict)

            if W_group_mode > 0:
                k_m = ((k * SPLIT_K + pid_k) * stride_mul).to(tl.int32)

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
                a = tl.load(a_ptrs + offs_ak[None, :] * stride_ak, mask=a_mask, other=0., eviction_policy=a_evict)

            if elements_per_sample > 1:
                q_shift = ((offs_ak % elements_per_sample) * W_nbits).to(tl.int32)[:, None]
                b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

            acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype)

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

        tile_id_c += num_sms
        pid_m, pid_n = swizzle_tile_persistent(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        if SPLIT_K > 1:
            tl.atomic_add(c_ptrs, acc, mask=mask, sem=atomic_mode)
        else:
            tl.store(c_ptrs, acc, mask=mask)


def gemm_splitK_persistent_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                   W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int,
                                   input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype: int,
                                   channel_scale_mode: int, W_group_mode: int, data_contiguous: bool, type_id: int,
                                   ) -> Tensor:

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]

    M_CLOSEST = get_closest_m(M)

    native_atomic = (output_dtype in [DType.FP16.value, DType.FP32.value]) or NATIVE_ATOMIC
    output = torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype] if native_atomic else torch.float32)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N'])), META['SPLIT_K'])

    gemm_splitK_persistent_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K, M_CLOSEST,
        W_nbits, group_size, unpack_mask, elements_per_sample, type_id,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        input_dtype=DTYPE_TO_TRITON[input_dtype],
        output_dtype=TORCH_DTYPE_TO_TRITON[output.dtype],
        acc_dtype=DTYPE_TO_TRITON[acc_dtype],
        meta_dtype=DTYPE_TO_TRITON[meta_dtype],
        channel_scale_mode=channel_scale_mode,
        W_group_mode=W_group_mode,
        zero_is_scalar=zeros.numel() == 1,
        data_contiguous=data_contiguous,
        NUM_SMS=NUM_SMS,
    )

    if not native_atomic:
        output = output.to(DTYPE_TO_TORCH[output_dtype])

    return output


class gemm_splitK_persistent:
    kernel = gemm_splitK_persistent_kernel
    forward = gemm_splitK_persistent_forward
    matmul_type = MATMUL_TYPE


__all__ = ["gemm_splitK_persistent"]
