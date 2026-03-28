# SPDX-License-Identifier: Apache-2.0
# QC-GEM: GEMV (General Matrix-Vector) kernels for FlagGems
# Ported from GemLite by Dr. Hicham Badri @Mobius Labs GmbH

import torch, math, copy
from torch import Tensor
import triton
import triton.language as tl

from .dtypes import is_mx_dtype, DType, DTYPE_TO_TORCH, TORCH_DTYPE_TO_TRITON, DTYPE_TO_TRITON
from .config import AUTOTUNE, KERNEL
from .utils import (
    dequantize, init_to_zero, next_power_of_2, is_hip,
    gpu_supports_bfloat16_atomicadd,
)


KEYS = ['M', 'N', 'K', 'group_size', 'elements_per_sample', 'type_id']
MATMUL_TYPE = "GEMV"
NATIVE_ATOMIC = gpu_supports_bfloat16_atomicadd()

# FP4 mapping for MXFP4/NVFP4 dequantization
_fp4_values_cpu = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=torch.int8,
)
fp4_mapping = [_fp4_values_cpu.cuda(device=i) for i in range(torch.cuda.device_count())]


def kernel_config_pruner(configs, nargs, **kwargs):
    global KEYS
    from .core import QCGEM_TRITON_CONFIG_CACHE

    m = nargs['M']
    n = nargs['N']
    k = nargs['K']
    g = nargs['group_size']
    e = nargs['elements_per_sample']

    pre_hook = init_to_zero("c_ptr") if nargs.get('use_prehook', False) else None

    if MATMUL_TYPE in QCGEM_TRITON_CONFIG_CACHE:
        signature = str(tuple([nargs[i] for i in KEYS]))
        if signature in QCGEM_TRITON_CONFIG_CACHE[MATMUL_TYPE]:
            config = copy.deepcopy(QCGEM_TRITON_CONFIG_CACHE[MATMUL_TYPE][signature])
            num_stages = config.pop('num_stages')
            num_warps = config.pop('num_warps')
            config.pop('num_ctas', None)
            config.pop('num_buffers_warp_spec', None)
            config.pop('num_consumer_groups', None)
            config.pop('reg_dec_producer', None)
            config.pop('reg_inc_consumer', None)
            configs['NUM_STAGES'] = num_stages
            yield triton.Config(config, num_stages=num_stages, num_warps=num_warps, pre_hook=pre_hook)
            return

    used = set()
    for config in configs:
        block_size_m = 1
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])
        block_size_k = min(g, block_size_k)
        block_size_k = next_power_of_2(block_size_k)
        block_size_n = next_power_of_2(block_size_n)

        if is_hip():
            if block_size_n * block_size_k >= 65536:
                continue

        if block_size_k < e:
            continue
        if block_size_k < 16:
            continue

        A_load_order = config.kwargs['A_load_order']
        dot_prod_mode = config.kwargs['dot_prod_mode']
        num_stages = config.num_stages
        num_warps = config.num_warps

        key = (block_size_m, block_size_n, block_size_k, A_load_order, dot_prod_mode, num_stages, num_warps)

        new_config = {
            'BLOCK_SIZE_M': block_size_m,
            'BLOCK_SIZE_N': block_size_n,
            'BLOCK_SIZE_K': block_size_k,
            'A_load_order': A_load_order,
            'dot_prod_mode': dot_prod_mode,
            'NUM_STAGES': num_stages,
        }

        if is_hip():
            new_config['waves_per_eu'] = config.kwargs.get('waves_per_eu', 0)
            key = key + (new_config['waves_per_eu'],)

        if key in used:
            continue

        used.add(key)
        yield triton.Config(new_config, num_stages=num_stages, num_warps=num_warps, pre_hook=pre_hook)

    if not used:
        block_size_k = min(g, 128)
        block_size_k = next_power_of_2(block_size_k)
        block_size_k = max(block_size_k, e, 16)
        block_size_n = min(n, next_power_of_2(64))
        block_size_n = max(block_size_n, 16)
        new_config = {'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': block_size_n, 'BLOCK_SIZE_K': block_size_k,
                      'A_load_order': 0, 'dot_prod_mode': 0}
        if is_hip():
            new_config['waves_per_eu'] = 0
        yield triton.Config(new_config, num_stages=2, num_warps=2, pre_hook=pre_hook)


def get_max_autotune_config_nvidia():
    configs = []
    for A in [0]:
        for D in [0]:
            for w in [1, 2, 4]:
                for s in [1, 2]:
                    for N in [32, 64, 128, 256, 512]:
                        for K in [8, 16, 32, 64, 128]:
                            configs.append(
                                triton.Config(
                                    {'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': N, 'BLOCK_SIZE_K': K,
                                     'A_load_order': A, 'dot_prod_mode': D},
                                    num_stages=s, num_warps=w,
                                )
                            )
    return configs


def get_fast_autotune_config_nvidia():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 16,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 16,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=1, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=2, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=2, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64,  'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=2, num_stages=1))
    return configs


def get_default_config_nvidia():
    return [triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'A_load_order': 0,
                            'dot_prod_mode': 0, 'NUM_STAGES': 1}, num_warps=1, num_stages=1)]


def get_max_autotune_config_amd():
    configs = []
    for A in [0]:
        for D in [0]:
            for w in [1, 2, 4]:
                for s in [1, 2]:
                    for v in [0, 2, 4]:
                        for N in [32, 64, 128, 256, 512, 1024]:
                            for K in [8, 16, 32, 64, 128]:
                                configs.append(
                                    triton.Config(
                                        {'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': N, 'BLOCK_SIZE_K': K,
                                         'A_load_order': A, 'dot_prod_mode': D, 'waves_per_eu': v},
                                        num_stages=s, num_warps=w,
                                    )
                                )
    return configs


def get_fast_autotune_config_amd():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 8,  'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 16, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 2}, num_warps=1, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 16, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 2}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 0}, num_warps=1, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 4}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 0}, num_warps=1, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 2}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 2}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 2}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 0}, num_warps=1, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 2}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 8,  'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 32, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 0}, num_warps=2, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_K': 32, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 0}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_K': 32, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 4}, num_warps=4, num_stages=1))
    return configs


def get_default_config_amd():
    return [triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'A_load_order': 0,
                            'dot_prod_mode': 0, 'NUM_STAGES': 1}, num_warps=1, num_stages=1)]


if is_hip():
    get_max_autotune_config = get_max_autotune_config_amd
    get_fast_autotune_config = get_fast_autotune_config_amd
    get_default_config = get_default_config_amd
else:
    get_max_autotune_config = get_max_autotune_config_nvidia
    get_fast_autotune_config = get_fast_autotune_config_nvidia
    get_default_config = get_default_config_nvidia

AUTOTUNE_SETTING = AUTOTUNE.GEMV
if AUTOTUNE_SETTING == 'max':
    get_autotune_config = get_max_autotune_config
elif AUTOTUNE_SETTING == 'fast':
    get_autotune_config = get_fast_autotune_config
else:
    get_autotune_config = get_default_config


KERNEL_CACHE = {}


@triton.autotune(
    configs=get_autotune_config(),
    key=KEYS,
    restore_value=['a_ptr', 'b_ptr', 'c_ptr'],
    prune_configs_by={'early_config_prune': kernel_config_pruner},
    use_cuda_graph=AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.jit
def gemv_INT_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    mapping_ptr,
    M, N, K,
    W_nbits: tl.constexpr,
    group_size: tl.constexpr,
    unpack_mask: tl.constexpr,
    elements_per_sample: tl.constexpr,
    type_id: tl.constexpr,
    use_prehook: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_a_m, stride_meta_a_g,
    stride_meta_g, stride_meta_n,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    A_load_order: tl.constexpr, NUM_STAGES: tl.constexpr,
    dot_prod_mode: tl.constexpr,
    data_contiguous: tl.constexpr,
    dump_b_val: tl.constexpr = 0,
    meta_evict_policy: tl.constexpr = '',
    atomic_mode: tl.constexpr = 'relaxed',
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
    join_version: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_m, pid_n = pid % M, pid // M

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if data_contiguous:
        offs_bn = offs_n
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_ak = offs_k
    offs_bk = offs_k

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

    if join_version:
        BLOCK_SIZE_K_E = BLOCK_SIZE_K // elements_per_sample
        offs_bk = pid_k * BLOCK_SIZE_K_E + tl.arange(0, BLOCK_SIZE_K_E)
        b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    else:
        b_ptrs = b_ptr + (offs_bk[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn

    if A_load_order == 0:
        a = tl.load(a_ptrs, eviction_policy=a_evict)

    b = tl.load(b_ptrs, eviction_policy=b_evict)

    if A_load_order == 1:
        a = tl.load(a_ptrs, eviction_policy=a_evict)

    if W_group_mode > 0:
        k_m = (pid_k * (BLOCK_SIZE_K / group_size)).to(tl.int32)

    if W_group_mode >= 2:
        scales = tl.load(scales_ptr + k_m * stride_meta_g + offs_bn[None, :] * stride_meta_n, eviction_policy=meta_evict_policy)
    else:
        scales = None

    if W_group_mode == 1 or W_group_mode >= 3:
        if zero_is_scalar:
            zeros = tl.load(zeros_ptr, eviction_policy=a_evict)
        else:
            zeros = tl.load(zeros_ptr + k_m * stride_meta_g + offs_bn[None, :] * stride_meta_n, eviction_policy=meta_evict_policy)
    else:
        zeros = None

    if A_load_order == 2:
        a = tl.load(a_ptrs, eviction_policy=a_evict)

    if join_version:
        if elements_per_sample == 2:
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K, BLOCK_SIZE_N), can_reorder=False)
        if elements_per_sample == 8:
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K // 4, BLOCK_SIZE_N), can_reorder=False)
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K // 2, BLOCK_SIZE_N), can_reorder=False)
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K, BLOCK_SIZE_N), can_reorder=False)

    q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]
    b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

    if A_load_order == 3:
        a = tl.load(a_ptrs, eviction_policy=a_evict)

    if dump_b_val > 0:
        b = b.to(tl.float32) * dump_b_val

    if dot_prod_mode == 0:
        acc = tl.sum(a.reshape((BLOCK_SIZE_K, 1), can_reorder=False).to(acc_dtype) * b.to(acc_dtype), axis=0, keep_dims=True)
    if dot_prod_mode == 1:
        acc = tl.sum(a.reshape((BLOCK_SIZE_K, 1), can_reorder=False) * b.to(input_dtype), axis=0, keep_dims=True)

    if dump_b_val > 0:
        acc /= dump_b_val

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

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc, sem=atomic_mode)


@triton.autotune(
    configs=get_autotune_config(),
    key=KEYS,
    restore_value=['a_ptr', 'b_ptr', 'c_ptr'],
    prune_configs_by={'early_config_prune': kernel_config_pruner},
    use_cuda_graph=AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.jit
def gemv_MX_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    mapping_ptr,
    M, N, K,
    W_nbits: tl.constexpr,
    group_size: tl.constexpr,
    unpack_mask: tl.constexpr,
    elements_per_sample: tl.constexpr,
    type_id: tl.constexpr,
    use_prehook: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_a_m, stride_meta_a_g,
    stride_meta_g, stride_meta_n,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    A_load_order: tl.constexpr, NUM_STAGES: tl.constexpr,
    dot_prod_mode: tl.constexpr,
    data_contiguous: tl.constexpr,
    dump_b_val: tl.constexpr = 0,
    meta_evict_policy: tl.constexpr = 'evict_first',
    atomic_mode: tl.constexpr = 'relaxed',
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
    join_version: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_m, pid_n = pid % M, pid // M

    a_ptr_dtype = a_ptr.dtype.element_ty
    if a_ptr_dtype == tl.float16:
        elements_per_sample_a = 1
    if a_ptr_dtype == tl.bfloat16:
        elements_per_sample_a = 1
    if a_ptr_dtype == tl.float8e4nv:
        elements_per_sample_a = 1
    if a_ptr_dtype == tl.uint8:
        elements_per_sample_a = 2

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if data_contiguous:
        offs_bn = offs_n
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_ak = offs_k // elements_per_sample_a
    offs_bk = offs_k // elements_per_sample

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    a_mask = ((offs_am[:, None] < M) & (offs_ak[None, :] < (K // elements_per_sample_a))).to(tl.int1)

    if W_nbits == 4:
        mapping = tl.load(mapping_ptr + tl.arange(0, 16), eviction_policy='evict_last')[None, :].broadcast_to((BLOCK_SIZE_K, 16))

    if A_load_order == 0:
        a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

    b = tl.load(b_ptrs, eviction_policy=b_evict)

    if A_load_order == 1:
        a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

    k_m = (pid_k * (BLOCK_SIZE_K / group_size)).to(tl.int32)
    scales_b_ptrs = scales_ptr + k_m * stride_meta_g + offs_bn[None, :] * stride_meta_n
    scales_b = tl.load(scales_b_ptrs, eviction_policy=meta_evict_policy)
    if scales_ptr.dtype.element_ty == tl.uint8:
        scales_b = (tl.exp2(scales_b.to(tl.float32) - 127) * 0.50)
    scales_b = scales_b.to(acc_dtype)

    if channel_scale_mode == 4:
        scales_a_ptrs = scales_a_ptr + k_m * stride_meta_a_g + offs_am[None, :] * stride_meta_a_m
        scales_a = tl.load(scales_a_ptrs, eviction_policy=meta_evict_policy)
        if scales_a_ptr.dtype.element_ty == tl.uint8:
            scales_a = (tl.exp2(scales_a.to(tl.float32) - 127) * 0.50)
        scales_a = scales_a.to(acc_dtype)

    if channel_scale_mode == 2:
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        if scales_a_ptr.dtype.element_ty == tl.uint8:
            scales_a = (tl.exp2(scales_a.to(tl.float32) - 127) * 0.50)
        scales_a = scales_a.to(acc_dtype)

    if A_load_order == 2:
        a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

    a = a.reshape((BLOCK_SIZE_K, 1), can_reorder=False)
    if elements_per_sample_a == 2:
        q_shift = ((offs_k % 2) * 4).to(tl.int32)[:, None]
        a = (a >> q_shift) & 15
        a = tl.gather(mapping, a, axis=1)

    a = a.to(acc_dtype)
    if channel_scale_mode == 2:
        a = a * scales_a
    if channel_scale_mode == 4:
        a = a * scales_a

    if elements_per_sample == 2:
        q_shift = ((offs_k % 2) * 4).to(tl.int32)[:, None]
        b = (b >> q_shift) & 15
        b = tl.gather(mapping, b, axis=1)

    b = b.to(acc_dtype) * scales_b

    if dot_prod_mode == 0:
        acc = tl.sum(a.to(acc_dtype) * b.to(acc_dtype), axis=0, keep_dims=True)
    if dot_prod_mode == 1:
        acc = tl.sum(a * b.to(a.dtype), axis=0, keep_dims=True)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc, sem=atomic_mode)


def gemv_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                 W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int,
                 input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype: int,
                 channel_scale_mode: int, W_group_mode: int, data_contiguous: bool, type_id: int,
                 ) -> Tensor:

    global KERNEL_CACHE

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]

    native_atomic = (output_dtype in [DType.FP16.value, DType.FP32.value]) or NATIVE_ATOMIC
    kernel_output_dtype = DTYPE_TO_TORCH[output_dtype] if native_atomic else torch.float32

    if KERNEL.ENABLE_CACHING and M == 1:
        if (M, N) not in KERNEL_CACHE:
            KERNEL_CACHE[(M, N)] = {
                "data": torch.empty((KERNEL.CACHE_SIZE, M, N), device=W_q.device, dtype=kernel_output_dtype),
                "ptr": 0,
            }
        entry = KERNEL_CACHE[(M, N)]
        if entry["ptr"] % KERNEL.CACHE_SIZE == 0:
            entry["data"].zero_()
            entry["ptr"] = 0
        output = entry["data"][entry["ptr"] % KERNEL.CACHE_SIZE]
        entry["ptr"] += 1
        use_prehook = False
    else:
        output = torch.zeros((M, N), device=W_q.device, dtype=kernel_output_dtype)
        use_prehook = False

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
        triton.cdiv(K, meta['BLOCK_SIZE_K'])
    )

    device_index = W_q.device.index

    if scales_x is not None:
        stride_meta_a_m, stride_meta_a_g = scales_x.stride(0), scales_x.stride(1)
    else:
        stride_meta_a_m, stride_meta_a_g = None, None
        channel_scale_mode = 0

    dtype = DTYPE_TO_TRITON[input_dtype]
    if dtype in [tl.float16, tl.bfloat16, tl.float32]:
        _acc_dtype = dtype
    else:
        _acc_dtype = DTYPE_TO_TRITON[acc_dtype]

    if is_mx_dtype(input_dtype):
        gemv_kernel = gemv_MX_kernel
        scales = scales.T
        if scales_x is not None:
            scales_x = scales_x.T
    else:
        gemv_kernel = gemv_INT_kernel

    gemv_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        fp4_mapping[device_index],
        M, N, K,
        W_nbits, group_size, unpack_mask, elements_per_sample, type_id, use_prehook,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        stride_meta_a_m, stride_meta_a_g,
        scales.stride(0), scales.stride(1),
        input_dtype=DTYPE_TO_TRITON[input_dtype],
        output_dtype=TORCH_DTYPE_TO_TRITON[output.dtype],
        acc_dtype=_acc_dtype,
        meta_dtype=DTYPE_TO_TRITON[meta_dtype],
        channel_scale_mode=channel_scale_mode,
        W_group_mode=W_group_mode,
        zero_is_scalar=zeros.numel() == 1,
        data_contiguous=data_contiguous,
        dump_b_val=0.001 if (W_group_mode in [0, 1] and _acc_dtype == DType.FP16.value and W_nbits == 8) else 0,
    )

    if not native_atomic:
        output = output.to(DTYPE_TO_TORCH[output_dtype])

    return output


class gemv:
    kernel = [gemv_INT_kernel, gemv_MX_kernel]
    forward = gemv_forward
    matmul_type = MATMUL_TYPE


__all__ = ["gemv"]
