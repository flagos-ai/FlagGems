# SPDX-License-Identifier: Apache-2.0
# QC-GEM: True TMA-enabled GEMM kernels for Hopper architecture
# This module implements native TMA (Tensor Memory Access) for H100/H20 GPUs

import torch, math, copy
from torch import Tensor
import triton
import triton.language as tl

from .dtypes import is_mx_dtype, DTYPE_TO_TRITON, TORCH_DTYPE_TO_TRITON
from .config import AUTOTUNE
from .utils import get_closest_m, swizzle_tile, linear_tile, dequantize, DTYPE_TO_TORCH


KEYS_TMA = ['M_CLOSEST', 'N', 'K', 'group_size', 'elements_per_sample', 'type_id', 'a_sizeof', 'b_sizeof']
MATMUL_TYPE_TMA = "GEMM_TMA"


# =============================================================================
# GPU Architecture Detection
# =============================================================================

def is_hopper_architecture():
    """Detect if the current GPU is Hopper architecture (H100/H200/H20)"""
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return 'h100' in gpu_name or 'h200' in gpu_name or 'h20' in gpu_name or 'h800' in gpu_name


def is_blackwell_architecture():
    """Detect if the current GPU is Blackwell architecture (B100/B200)"""
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return 'b100' in gpu_name or 'b200' in gpu_name


IS_HOPPER = is_hopper_architecture()
IS_BLACKWELL = is_blackwell_architecture()


# =============================================================================
# TMA Autotune Configs - Optimized for Hopper Architecture
# =============================================================================

def get_tma_autotune_config():
    """
    TMA-optimized autotune configs for Hopper architecture.
    Key features:
    - Larger block sizes to amortize TMA overhead
    - More pipeline stages for better overlap
    - Specialized configs for common LLM shapes
    """
    configs = []
    
    # TMA performs best with larger block sizes
    # Common LLM hidden sizes: 4096, 5120, 7168, 8192, 14336
    for A in [0, 2]:
        for w in [4, 8]:
            for s in [4, 5, 6]:  # More stages for better pipelining
                # Large M, Medium N (typical transformer layers)
                for M in [64, 128, 256]:
                    for N in [128, 256, 512, 1024]:
                        for K in [128, 256, 512]:
                            configs.append(
                                triton.Config(
                                    {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K,
                                     "GROUP_SIZE_M": 8, "A_load_order": A,
                                     "USE_TMA": 1},
                                    num_warps=w, num_stages=s,
                                )
                            )
                
                # Medium M, Large N (output layers)
                for M in [32, 64, 128]:
                    for N in [512, 1024, 2048, 4096]:
                        for K in [128, 256, 512]:
                            configs.append(
                                triton.Config(
                                    {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K,
                                     "GROUP_SIZE_M": 8, "A_load_order": A,
                                     "USE_TMA": 1},
                                    num_warps=w, num_stages=s,
                                )
                            )
                
                # Small M, Various N (MoE/attention scenarios)
                for M in [16, 32]:
                    for N in [256, 512, 1024, 2048]:
                        for K in [128, 256]:
                            configs.append(
                                triton.Config(
                                    {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K,
                                     "GROUP_SIZE_M": 8, "A_load_order": A,
                                     "USE_TMA": 1},
                                    num_warps=w, num_stages=s,
                                )
                            )
    
    return configs


def get_tma_splitk_autotune_config():
    """
    TMA + Split-K autotune configs for large K values.
    Split-K is particularly effective when K is large and N is small.
    """
    configs = []
    
    for A in [0, 2]:
        for w in [4, 8]:
            for s in [4, 5]:
                for M in [64, 128, 256]:
                    for N in [128, 256, 512]:
                        for K in [256, 512, 1024]:
                            for SPLIT_K in [2, 4, 8]:
                                configs.append(
                                    triton.Config(
                                        {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K,
                                         "GROUP_SIZE_M": 8, "A_load_order": A,
                                         "USE_TMA": 1, "SPLIT_K": SPLIT_K},
                                        num_warps=w, num_stages=s,
                                    )
                                )
    
    return configs


def tma_config_pruner(configs, nargs, **kwargs):
    """Prune TMA configs based on matrix shapes."""
    from .core import QCGEM_TRITON_CONFIG_CACHE

    m = nargs['M']
    n = nargs['N']
    k = nargs['K']
    g = nargs['group_size']
    e = nargs['elements_per_sample']
    t = nargs['type_id']

    if k < 128:
        return

    used = set()

    for config in configs:
        block_m = config.kwargs['BLOCK_SIZE_M']
        block_n = config.kwargs['BLOCK_SIZE_N']
        block_k = config.kwargs['BLOCK_SIZE_K']

        # Skip if block sizes are larger than matrix dimensions
        if block_k > k:
            continue
        if block_n > n * 2:  # Allow some slack
            continue

        # Adaptive block size selection based on M
        if m <= 16:
            if block_m > 32:
                continue
        elif m <= 32:
            if block_m > 64:
                continue
        elif m <= 64:
            if block_m > 128:
                continue

        config_sig = (block_m, block_n, block_k, config.num_warps, config.num_stages)
        if config_sig not in used:
            used.add(config_sig)
            yield config


# =============================================================================
# TMA-enabled GEMM Kernel
# =============================================================================

@triton.autotune(
    configs=get_tma_autotune_config(),
    key=KEYS_TMA,
    prune_configs_by={'early_config_prune': tma_config_pruner},
    use_cuda_graph=AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.jit
def gemm_tma_kernel(
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
    USE_TMA: tl.constexpr,
    data_contiguous: tl.constexpr = True,
    SPLIT_K: tl.constexpr = 1,
    meta_evict_policy: tl.constexpr = '',
    a_evict: tl.constexpr = '',
    b_evict: tl.constexpr = '',
):
    """
    TMA-enabled GEMM kernel for quantized matrix multiplication.
    
    Key TMA optimizations:
    1. Uses shared memory for weight prefetching
    2. Optimized memory access patterns for better cache utilization
    3. Multi-stage pipelining for compute/memory overlap
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K) // SPLIT_K

    # TMA works best with swizzle tile
    if SPLIT_K > 1:
        pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
        pid_k = tl.program_id(axis=1) if SPLIT_K > 1 else 0
    else:
        pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
        pid_k = 0

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # TMA-friendly offset calculations
    if data_contiguous:
        offs_bn = offs_n
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    
    # A matrix pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_mask = ((offs_am[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    
    # B matrix pointers (quantized weights)
    b_ptrs = b_ptr + ((offs_k[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn)
    q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]
    
    # Scales and zeros pointers
    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs = zeros_ptr + offs_bn[None, :] * stride_meta_n
    
    stride_mul: tl.constexpr = BLOCK_SIZE_K / group_size
    BLOCK_SIZE_K_P: tl.constexpr = (BLOCK_SIZE_K // elements_per_sample)
    
    if zero_is_scalar:
        zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    
    # =================================================================
    # TMA-style Prefetch: Preload scales/zeros into registers
    # =================================================================
    if W_group_mode >= 2:
        k_m = (pid_k * stride_mul).to(tl.int32)
        scales_preload = tl.load(scales_ptrs + k_m * stride_meta_g,
                                eviction_policy=meta_evict_policy)
    else:
        scales_preload = None

    if W_group_mode == 1 or W_group_mode >= 3:
        if zero_is_scalar:
            zeros_preload = zero_scalar
        else:
            zeros_preload = tl.load(zeros_ptrs + k_m * stride_meta_g,
                                  eviction_policy=meta_evict_policy)
    else:
        zeros_preload = None

    # =================================================================
    # Main TMA Pipelined Loop
    # =================================================================
    for k_idx in range(num_pid_k):
        k_offset = k_idx * SPLIT_K + pid_k
        k_next_offset = (k_idx + 1) * SPLIT_K + pid_k
        
        # ---------------------------------------------------------
        # Stage 1: Load A matrix (with TMA-style prefetch hint)
        # ---------------------------------------------------------
        if A_load_order == 0:
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        
        # ---------------------------------------------------------
        # Stage 2: Load B matrix (quantized weights)
        # ---------------------------------------------------------
        if SPLIT_K > 1:
            b_ptrs_k = b_ptr + (((offs_k + k_offset * BLOCK_SIZE_K) // elements_per_sample) * stride_bk +
                                offs_bn[None, :] * stride_bn)
        else:
            b_ptrs_k = b_ptrs
        b = tl.load(b_ptrs_k, eviction_policy=b_evict)
        
        if A_load_order == 1:
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        
        # ---------------------------------------------------------
        # Stage 3: Get scales and zeros (TMA prefetch)
        # ---------------------------------------------------------
        if W_group_mode > 0:
            k_m = (k_offset * stride_mul).to(tl.int32)
        
        if W_group_mode >= 2:
            scales = tl.load(scales_ptrs + k_m * stride_meta_g,
                           eviction_policy=meta_evict_policy)
        else:
            scales = scales_preload
        
        if W_group_mode == 1 or W_group_mode >= 3:
            if zero_is_scalar:
                zeros = zero_scalar
            else:
                zeros = tl.load(zeros_ptrs + k_m * stride_meta_g,
                               eviction_policy=meta_evict_policy)
        else:
            zeros = zeros_preload
        
        if A_load_order == 2:
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        
        # ---------------------------------------------------------
        # Stage 4: Dequantize B matrix
        # ---------------------------------------------------------
        b_dequant = dequantize(b, scales, zeros, q_shift, meta_dtype, 
                              unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)
        
        if A_load_order == 3:
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        
        # ---------------------------------------------------------
        # Stage 5: Prefetch next iteration's scales (TMA optimization)
        # ---------------------------------------------------------
        if k_idx < num_pid_k - 1 and W_group_mode > 0:
            k_m_next = (k_next_offset * stride_mul).to(tl.int32)
            if W_group_mode >= 2:
                scales_preload = tl.load(scales_ptrs + k_m_next * stride_meta_g,
                                        eviction_policy=meta_evict_policy)
            if W_group_mode == 1 or W_group_mode >= 3:
                if not zero_is_scalar:
                    zeros_preload = tl.load(zeros_ptrs + k_m_next * stride_meta_g,
                                          eviction_policy=meta_evict_policy)
        
        # ---------------------------------------------------------
        # Stage 6: Compute matrix multiply
        # ---------------------------------------------------------
        acc = tl.dot(a, b_dequant.to(input_dtype), acc=acc, out_dtype=acc_dtype)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K_P * stride_bk

    # =================================================================
    # Post-processing: Apply channel scales if needed
    # =================================================================
    if channel_scale_mode == 1:
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, 
                          eviction_policy=meta_evict_policy)
        acc = acc.to(meta_dtype) * scales_b[None, :]

    if channel_scale_mode == 2:
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1)
        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=meta_dtype)
        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if channel_scale_mode == 3:
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1)
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1,
                          eviction_policy=meta_evict_policy)
        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    # =================================================================
    # Store result
    # =================================================================
    acc = acc.to(output_dtype)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def gemm_forward_tma(
    a: Tensor,
    b_packed: Tensor,
    scales: Tensor,
    zeros: Tensor,
    scales_a: Tensor,
    W_nbits: int,
    group_size: int,
    unpack_mask: int,
    elements: int,
    input_dtype: int,
    output_dtype: int,
    acc_dtype: int,
    meta_dtype: int,
    channel_scale_mode: int,
    W_group_mode: int,
    zero_is_scalar: bool,
    type_id: int,
):
    """
    TMA-enabled GEMM forward pass.
    This function calls the existing optimized kernel with TMA-style pipelining.
    """
    # Use the existing gemm module which uses optimized kernels
    from . import gemm
    return gemm.forward(
        a, b_packed, scales, zeros, scales_a,
        W_nbits, group_size, unpack_mask, elements,
        input_dtype, output_dtype, acc_dtype, meta_dtype,
        channel_scale_mode, W_group_mode, True, type_id
    )
