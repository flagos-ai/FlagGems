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

import flag_gems
from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# GCU400 tile 扫描实测（2026-11-13，event per-launch 中位数，fp16 大 shape）：
# 带宽随 BLOCK_M*BLOCK_DIFF 总 tile 元素数单调上升，128K 元素（DSM 768KB < 896KB
# 硬限）达 2.05 TB/s（32K tile 仅 1.42 TB/s）。最优形态为 BLOCK_DIFF 贴满行宽、
# BLOCK_M 补足 tile 预算：长行 (N>=4096) 选 32x4096，中行 (N~1024) 选 128x1024，
# 短行 (N~256) 选 512x256。num_warps 不敏感（<1.5%）。DSM = tile*itemsize*3，
# fp32/int64 下 128K tile 超 896KB 会被 OutOfResources 捕获跳过，64K 档兜底；
# 32x16 兜底极短行。
_GCU400_DIFF_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_DIFF": 16}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_DIFF": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_DIFF": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 512, "BLOCK_DIFF": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_DIFF": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_DIFF": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_DIFF": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_DIFF": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_DIFF": 2048}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_DIFF": 2048}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_DIFF": 2048}, num_warps=4),
    triton.Config({"BLOCK_M": 8, "BLOCK_DIFF": 4096}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_DIFF": 4096}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_DIFF": 4096}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_DIFF": 4096}, num_warps=8),
]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("diff_1d"),
    key=["N"],
)
@triton.jit
def diff_kernel_1d(in_ptr, out_ptr, N, BLOCK_DIFF: tl.constexpr):
    pid = tle.program_id(0)

    in_offsets = pid * BLOCK_DIFF + tl.arange(0, BLOCK_DIFF)
    mask_in = in_offsets < N - 1
    in_block = tl.load(in_ptr + in_offsets, mask_in)
    next_block = tl.load(in_ptr + in_offsets + 1, mask_in)
    tl.store(out_ptr + in_offsets, next_block - in_block, mask_in)


@libentry()
@libtuner(
    configs=_GCU400_DIFF_CONFIGS,
    key=["M", "N"],
)
@triton.jit
def diff_kernel_2d(
    in_ptr,
    out_ptr,
    M,
    N,
    M_STRIDE_IN,
    M_STRIDE_OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_DIFF: tl.constexpr,
):
    pid_M = tle.program_id(0)
    pid_diff = tle.program_id(1)

    M_offsets = pid_M * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_M = M_offsets < M

    in_offsets_diff = pid_diff * BLOCK_DIFF + tl.arange(0, BLOCK_DIFF)
    mask_in_diff = in_offsets_diff < N - 1

    in_offsets = M_offsets[:, None] * M_STRIDE_IN + in_offsets_diff
    out_offsets = M_offsets[:, None] * M_STRIDE_OUT + in_offsets_diff
    mask_in = mask_M[:, None] & mask_in_diff

    in_block = tl.load(in_ptr + in_offsets, mask_in)
    next_block = tl.load(in_ptr + in_offsets + 1, mask_in)
    tl.store(out_ptr + out_offsets, next_block - in_block, mask_in)


def diff(input, n=1, dim=-1, prepend=None, append=None) -> torch.Tensor:
    logger.debug("GEMS DIFF")

    if prepend is not None:
        input = flag_gems.cat([prepend, input], dim=dim)
    if append is not None:
        input = flag_gems.cat([input, append], dim=dim)

    if n <= 0:
        return input

    shape = list(input.shape)
    dim = dim % input.ndim
    reduce_len = shape[dim]

    if n >= reduce_len:
        empty_tensor = torch.tensor([], dtype=input.dtype, device=input.device)
        return torch.reshape(empty_tensor, shape[:dim] + [0] + shape[(dim + 1) :])

    input = dim_compress(input, dim)
    N = reduce_len
    M = input.numel() // N

    is_1d = len(shape) == 1

    def _launch(src, dst, in_stride_m, out_stride_m, n_bound):
        if is_1d:
            grid = lambda meta: (triton.cdiv(n_bound - 1, meta["BLOCK_DIFF"]),)
            with torch_device_fn.device(src.device):
                diff_kernel_1d[grid](src, dst, n_bound)
        else:
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(n_bound - 1, meta["BLOCK_DIFF"]),
            )
            with torch_device_fn.device(src.device):
                diff_kernel_2d[grid](src, dst, M, n_bound, in_stride_m, out_stride_m)

    # Allocate the final output at its exact post-diff size [..., N-n] so
    # that the last kernel writes directly into it. This eliminates the
    # tail `output[..., :N-n].contiguous()` copy.
    out_shape = list(input.shape)
    out_shape[-1] = N - n
    output = torch.empty(out_shape, device=input.device, dtype=input.dtype)

    if n == 1:
        _launch(input, output, N, N - 1, N)
        return torch.moveaxis(output, -1, dim)

    # n >= 2: ping-pong between two scratch buffers (sized N-1 and N-2 for the diff dim),
    # writing the last iteration directly into `output` (size N-n).
    # Avoid _copy().
    scratch_a_shape = list(input.shape)
    scratch_a_shape[-1] = N - 1
    scratch_a = torch.empty(scratch_a_shape, device=input.device, dtype=input.dtype)
    if n >= 3:
        scratch_b_shape = list(input.shape)
        scratch_b_shape[-1] = N - 2
        scratch_b = torch.empty(scratch_b_shape, device=input.device, dtype=input.dtype)

    # iter 0: input -> scratch_a
    _launch(input, scratch_a, N, N - 1, N)
    src, src_stride = scratch_a, N - 1

    # iter 1 to (n - 1)
    for k in range(1, n):
        if k == n - 1:
            dst, dst_stride = output, N - n
        elif k % 2 == 1:
            dst, dst_stride = scratch_b, N - 2
        else:
            dst, dst_stride = scratch_a, N - 1
        _launch(src, dst, src_stride, dst_stride, N - k)
        src, src_stride = dst, dst_stride

    return torch.moveaxis(output, -1, dim)
