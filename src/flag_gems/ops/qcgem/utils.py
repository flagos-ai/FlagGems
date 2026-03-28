# SPDX-License-Identifier: Apache-2.0
# QC-GEM: Quantized Computing GEM library for FlagGems

import torch, triton, math
import triton.language as tl
from triton.runtime import driver

from .dtypes import *


@triton.jit
def swizzle_tile_v1(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@triton.jit
def swizzle_tile_v2(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_SIZE_M * grid_m
    group_id = pid // width
    group_size = tl.minimum(grid_n - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_n = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_m = (pid % width) // group_size
    return pid_m, pid_n


@triton.jit
def swizzle_tile_v3(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid // tl.cdiv(M, BLOCK_SIZE_M)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    return tl.swizzle2d(pid_m, pid_n, grid_m, grid_n, GROUP_SIZE_M)


swizzle_tile = swizzle_tile_v1


@triton.jit
def linear_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid // tl.cdiv(M, BLOCK_SIZE_M)
    return pid_m, pid_n


@triton.jit
def swizzle_tile_persistent(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def dequantize(
    b, scales, zeros, q_shift, meta_dtype, unpack_mask,
    elements_per_sample: tl.constexpr, W_group_mode: tl.constexpr, zero_is_scalar: tl.constexpr,
):
    if elements_per_sample > 1:
        b = (b >> q_shift) & unpack_mask

    if W_group_mode == 1:
        b = b.to(meta_dtype) - zeros

    if W_group_mode == 2:
        b = b.to(meta_dtype) * scales

    if W_group_mode == 3:
        if zero_is_scalar:
            b = (b - zeros).to(meta_dtype) * scales
        else:
            b = (b.to(meta_dtype) - zeros) * scales

    if W_group_mode == 4:
        b = tl.fma(b.to(meta_dtype), scales, zeros)

    return b


@triton.jit
def atomic_add_cas(ptr, value, Lock, mask=None, sem: tl.constexpr = "release"):
    while tl.atomic_cas(Lock, 0, 1, sem=sem) == 1:
        pass
    tl.store(ptr, tl.load(ptr, mask=mask) + value, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def next_power_of_2(v):
    return 2 ** int(math.ceil(math.log2(v)))


def is_divisible(dividend, divisor):
    return dividend % divisor == 0


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def gpu_has_more_shared_memory(ref_gpus=["a100", "h100", "h200", "h20", "h800", "b100", "b200"]):
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return True in [g in gpu_name for g in ref_gpus]


def gpu_supports_float16_acc(ref_gpus=None):
    if ref_gpus is None:
        ref_gpus = ["5090", "5080", "5070", "5060", "4090", "4080", "4070", "4060",
                    "3090", "3080", "3070", "3060", "2080", "2070"]
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return True in [g in gpu_name for g in ref_gpus]


def gpu_supports_bfloat16_atomicadd():
    return False


_num_sms_cache = None


def get_num_SMs(device):
    global _num_sms_cache
    if _num_sms_cache is None:
        _num_sms_cache = torch.cuda.get_device_properties(device).multi_processor_count
    return _num_sms_cache


def generate_autotune_lookup_v2(max_m=16384, min_split=32, divisors=None, mode='next', include_vllm_config=False):
    if divisors is None:
        divisors = [2, 4]
    lookup = [0] * (max_m + 1)
    autotune_vals = set()

    i = 0
    while (val := 2 ** i) <= max_m:
        autotune_vals.add(val)
        next_val = 2 ** (i + 1)
        if val >= min_split and next_val <= max_m:
            for d in divisors:
                interpolated = (val + next_val) // d
                autotune_vals.add(interpolated)
        i += 1

    if include_vllm_config:
        autotune_vals.update([1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64,
                             72, 80, 88, 96, 104, 112, 120, 128,
                             136, 144, 152, 160, 168, 176, 184, 192,
                             200, 208, 216, 224, 232, 240, 248, 256, 384, 512])

    sorted_vals = sorted(autotune_vals)

    for m in range(max_m + 1):
        if mode == 'next':
            lookup[m] = min((x for x in sorted_vals if x >= m), default=None)
        elif mode == 'closest':
            lookup[m] = min(sorted_vals, key=lambda x: (abs(x - m), x < m))
        else:
            raise Exception('Invalid mode.')
    return lookup


M_MAXVAL = 4096
M_MAPPING = generate_autotune_lookup_v2(M_MAXVAL, mode='next')


def get_closest_m(M):
    return M_MAPPING[M] if M <= M_MAXVAL else M_MAXVAL


def get_gpu_shared_memory():
    return driver.active.utils.get_device_properties(0).get("max_shared_mem", 0)


IS_HIP = is_hip()
NATIVE_ATOMIC = gpu_supports_bfloat16_atomicadd()
