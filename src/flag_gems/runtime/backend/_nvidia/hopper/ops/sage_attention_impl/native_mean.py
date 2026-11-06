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

"""Pinned ATen CUDA MeanOps order for NHD FP16/BF16 D128.

Reimplements the installed Torch 2.11 Reduce.cuh configuration and arithmetic;
does not call torch.mean or a native operator. Runtime device limits determine
CTA count. Two kernels replace the native last-CTA semaphore, retaining its
fixed numerical combination order. Reference source hashes are in the report.
"""

import torch
import triton
import triton.language as tl


def _ceil(a, b):
    return (a + b - 1) // b


def _floor_power_two(n):
    return 1 << max(0, n.bit_length() - 1)


def native_geometry(key):
    b, n, h, d = key.shape
    if key.device.type != "cuda" or key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Native-order mean requires CUDA FP16/BF16")
    if d != 128 or key.stride(-1) != 1 or min(b, n, h) < 1:
        raise ValueError("Requires nonempty NHD D128, contiguous last dimension")
    # Larger TensorIterator inputs use sub-iterators, a distinct native order.
    largest_offset = sum(
        (size - 1) * stride for size, stride in zip(key.shape, key.stride())
    )
    if key.numel() >= 2**31 or largest_offset * key.element_size() >= 2**31:
        raise NotImplementedError("Native TensorIterator 32-bit indexing scope only")
    vec = 4
    alignment = [key.data_ptr() // key.element_size()]
    alignment += [key.stride(i) for i in range(3) if key.shape[i] > 1]
    for value in alignment:
        while value % vec:
            vec //= 2
    outputs = b * h * d
    max_threads = 512 // vec
    xbound = min(_floor_power_two(outputs // vec), max_threads)
    ybound = min(_floor_power_two(n), max_threads)
    width = min(xbound, 32)
    height = min(ybound, max_threads // width)
    width = min(xbound, max_threads // height)
    threads = width * height
    split_y = n >= min(height * 16, 256)
    step_input = height if split_y else 1
    step_output = width if split_y else width * height
    grid_x = _ceil(outputs // vec, step_output)
    props = torch.cuda.get_device_properties(key.device)
    target_grid = props.multi_processor_count * (
        props.max_threads_per_multi_processor // threads
    )
    values_per_thread = _ceil(n, step_input)
    ctas = 1
    if split_y and values_per_thread >= 256 and grid_x <= target_grid:
        ctas = max(
            min(_ceil(target_grid, grid_x), _ceil(values_per_thread, 16)),
            _ceil(values_per_thread, 256),
        )
    return dict(
        output_vector=vec,
        block_width=width,
        block_height=height,
        native_grid_x=grid_x,
        ctas=ctas,
        reduce_y=height if split_y else 1,
        multiprocessors=props.multi_processor_count,
        max_threads_per_mp=props.max_threads_per_multi_processor,
    )


@triton.jit
def _ordered_y_sum(x, Y: tl.constexpr, C: tl.constexpr):
    # Explicit descending offsets reproduce block_y_reduce, independent of
    # compiler reduction-layout selection. Do not substitute an arbitrary sum.
    if Y >= 16:
        a, b = tl.split(tl.permute(tl.reshape(x, (2, 8, C)), (1, 2, 0)))
        x = a + b
    if Y >= 8:
        a, b = tl.split(tl.permute(tl.reshape(x, (2, 4, C)), (1, 2, 0)))
        x = a + b
    if Y >= 4:
        a, b = tl.split(tl.permute(tl.reshape(x, (2, 2, C)), (1, 2, 0)))
        x = a + b
    if Y >= 2:
        a, b = tl.split(tl.permute(tl.reshape(x, (2, 1, C)), (1, 2, 0)))
        x = a + b
    return tl.reshape(x, (C,))


@triton.jit
def _native_partial(
    X,
    P,
    M,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOTAL: tl.constexpr,
    SZ: tl.constexpr,
    SN: tl.constexpr,
    SH: tl.constexpr,
    CTAS: tl.constexpr,
    Y: tl.constexpr,
    C: tl.constexpr,
):
    column = tl.program_id(0) * C + tl.arange(0, C)
    cta = tl.program_id(1)
    y = tl.arange(0, Y)
    offset = (column // (H * D)) * SZ + ((column // D) % H) * SH + column % D
    stride: tl.constexpr = CTAS * Y
    start = cta * Y + y
    a0 = tl.zeros((Y, C), tl.float32)
    a1 = tl.zeros((Y, C), tl.float32)
    a2 = tl.zeros((Y, C), tl.float32)
    a3 = tl.zeros((Y, C), tl.float32)
    for block in range(tl.cdiv(N, 4 * stride)):
        n = start + block * 4 * stride
        x0 = tl.load(
            X + n[:, None] * SN + offset[None, :],
            (n[:, None] < N) & (column[None, :] < TOTAL),
            0,
        ).to(tl.float32)
        x1 = tl.load(
            X + (n[:, None] + stride) * SN + offset[None, :],
            (n[:, None] + stride < N) & (column[None, :] < TOTAL),
            0,
        ).to(tl.float32)
        x2 = tl.load(
            X + (n[:, None] + 2 * stride) * SN + offset[None, :],
            (n[:, None] + 2 * stride < N) & (column[None, :] < TOTAL),
            0,
        ).to(tl.float32)
        x3 = tl.load(
            X + (n[:, None] + 3 * stride) * SN + offset[None, :],
            (n[:, None] + 3 * stride < N) & (column[None, :] < TOTAL),
            0,
        ).to(tl.float32)
        a0 = a0 + x0
        a1 = a1 + x1
        a2 = a2 + x2
        a3 = a3 + x3
    combined = ((a0 + a1) + a2) + a3
    partial = _ordered_y_sum(combined, Y, C)
    if CTAS == 1:
        mean = partial * tl.full((), 1.0 / N, tl.float32)
        tl.store(M + column, mean, column < TOTAL)
    else:
        tl.store(P + cta * TOTAL + column, partial, column < TOTAL)


@triton.jit
def _native_finish(
    P,
    M,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    CTAS: tl.constexpr,
    Y: tl.constexpr,
    C: tl.constexpr,
):
    column = tl.program_id(0) * C + tl.arange(0, C)
    y = tl.arange(0, Y)
    acc = tl.zeros((Y, C), tl.float32)
    for block in range(tl.cdiv(CTAS, Y)):
        cta = block * Y + y
        value = tl.load(
            P + cta[:, None] * TOTAL + column[None, :],
            (cta[:, None] < CTAS) & (column[None, :] < TOTAL),
            0,
        )
        acc = acc + value
    total = _ordered_y_sum(acc, Y, C)
    mean = total * tl.full((), 1.0 / N, tl.float32)
    tl.store(M + column, mean, column < TOTAL)


def native_k_mean(key, *, out=None, return_kernels=False):
    config = native_geometry(key)
    b, n, h, d = key.shape
    total = b * h * d
    if out is None:
        out = torch.empty((b, 1, h, d), device=key.device, dtype=key.dtype)
    elif (
        out.shape != (b, 1, h, d)
        or not out.is_contiguous()
        or out.device != key.device
        or out.dtype != key.dtype
    ):
        raise ValueError("out must be matching contiguous [B,1,H,D]")
    ctas, y = config["ctas"], config["reduce_y"]
    partial = (
        torch.empty((ctas, total), device=key.device, dtype=torch.float32)
        if ctas > 1
        else out
    )
    kernels = [
        _native_partial[(triton.cdiv(total, 128), ctas)](
            key,
            partial,
            out,
            n,
            h,
            d,
            total,
            *key.stride()[:3],
            ctas,
            y,
            128,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    ]
    if ctas > 1:
        kernels.append(
            _native_finish[(triton.cdiv(total, 128),)](
                partial,
                out,
                n,
                total,
                ctas,
                y,
                128,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
        )
    return (out, kernels, config) if return_kernels else out
