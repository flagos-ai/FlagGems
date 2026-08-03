import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_w8a16_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    group_ids = cols // GROUP_SIZE
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w_scale = tl.load(w_scale_ptr + group_ids, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w * w_scale
    tl.store(out_ptr + pid * N + cols, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_w8a16_grouped_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    groups = tl.arange(0, NUM_GROUPS)
    cols = tl.arange(0, GROUP_SIZE)
    offsets = groups[:, None] * GROUP_SIZE + cols[None, :]
    mask = offsets < N

    x = tl.load(in_ptr + pid * N + offsets, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w_scale = tl.load(w_scale_ptr + groups).to(tl.float32)[:, None]
    y = x * rrms * w * w_scale
    tl.store(out_ptr + pid * N + offsets, y, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rms_norm_loop"),
    key=["N"],
)
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_w8a16_loop_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x

    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x

    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)

    prev_multiple = prev_multiple_of(N, TILE_N)

    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        group_ids = n_offsets // GROUP_SIZE
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + group_ids, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w * w_scale
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(tl.float32)
        group_ids = n_offsets // GROUP_SIZE
        w = tl.load(w_ptr + n_offsets).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + group_ids).to(tl.float32)
        y = x * rrms * w * w_scale
        tl.store(out_ptr + pid * N + n_offsets, y)


def rms_norm_fp8_w8a16(
    x, normalized_shape, weight_fp8, weight_scale, eps=1e-5, group_size=128
):
    logger.debug("GEMS RMS_NORM FP8 W8A16 FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    with torch_device_fn.device(x.device):
        if N == 4096 and M >= 512:
            rms_norm_fp8_w8a16_grouped_kernel[M,](
                y,
                x,
                weight_fp8,
                weight_scale,
                N,
                eps,
                GROUP_SIZE=group_size,
                NUM_GROUPS=N // group_size,
                num_warps=8,
            )
        elif N == 4096 and M >= 128:
            rms_norm_fp8_w8a16_grouped_kernel[M,](
                y,
                x,
                weight_fp8,
                weight_scale,
                N,
                eps,
                GROUP_SIZE=group_size,
                NUM_GROUPS=N // group_size,
                num_warps=4,
            )
        elif N == 8192 and M >= 64:
            rms_norm_fp8_w8a16_grouped_kernel[M,](
                y,
                x,
                weight_fp8,
                weight_scale,
                N,
                eps,
                GROUP_SIZE=group_size,
                NUM_GROUPS=N // group_size,
                num_warps=4,
            )
        elif N == 16384:
            rms_norm_fp8_w8a16_grouped_kernel[M,](
                y,
                x,
                weight_fp8,
                weight_scale,
                N,
                eps,
                GROUP_SIZE=group_size,
                NUM_GROUPS=N // group_size,
                num_warps=4,
            )
        elif N > 16384:
            rms_norm_fp8_w8a16_loop_kernel[M,](
                y, x, weight_fp8, weight_scale, N, eps, GROUP_SIZE=group_size
            )
        else:
            BLOCK_SIZE = triton.next_power_of_2(N)
            rms_norm_fp8_w8a16_kernel[M,](
                y, x, weight_fp8, weight_scale, N, eps, BLOCK_SIZE, group_size
            )
    return y
