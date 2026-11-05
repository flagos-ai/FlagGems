# Copyright 2026 XCoreSigma Contributors
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

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flag_gems.runtime import torch_device_fn

FALLBACK_AIC_CORES = 24
WEIGHT_BLOCK_N = 256
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _get_aic_core_num(device_index):
    try:
        cores = torch_device_fn.get_device_limit(device_index)["cube_core_num"]
        if isinstance(cores, int) and not isinstance(cores, bool) and cores > 0:
            return cores
    except (AttributeError, KeyError, TypeError, RuntimeError):
        pass
    return FALLBACK_AIC_CORES


@triton.jit
def _block_indexer(
    tile_idx_in_gemm,
    num_m_tiles,
    num_n_tiles: tl.constexpr,
    BLOCK_THRESHOLD: tl.constexpr,
):
    threshold_n: tl.constexpr = BLOCK_THRESHOLD * num_n_tiles
    full_panel_tasks = num_m_tiles * num_n_tiles // threshold_n * threshold_n
    panel_m = (
        BLOCK_THRESHOLD
        if tile_idx_in_gemm < full_panel_tasks
        else num_m_tiles % BLOCK_THRESHOLD
    )
    square_tasks = panel_m * BLOCK_THRESHOLD
    local_tile = tile_idx_in_gemm % threshold_n % square_tasks
    tile_m = local_tile % panel_m + tile_idx_in_gemm // threshold_n * BLOCK_THRESHOLD

    full_panel_columns = panel_m * num_n_tiles // square_tasks * square_tasks
    panel_n = (
        BLOCK_THRESHOLD
        if tile_idx_in_gemm % threshold_n < full_panel_columns
        else num_n_tiles % BLOCK_THRESHOLD
    )
    x, y = panel_m, panel_n
    while y != 0:
        x, y = y, x % y
    lcm = panel_m * panel_n // x
    tile_n = (
        local_tile + local_tile // lcm
    ) % panel_n + tile_idx_in_gemm % threshold_n // square_tasks * BLOCK_THRESHOLD
    return tile_m, tile_n


@triton.jit
def _grouped_matmul_nd_kernel(
    A,
    B,
    C,
    group_list,
    GROUPS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 256,
    BLOCK_K: tl.constexpr = 256,
    BLOCK_THRESHOLD: tl.constexpr = 8,
):
    total_cores = tl.num_programs(0)
    core_idx = tl.program_id(0)
    num_n_tiles: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    num_k_tiles: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    last_count = 0
    group_start = 0
    group_end = 0

    for group_idx in range(GROUPS):
        current_group_value = tl.load(group_list + group_idx).to(tl.int32)
        group_m = (
            current_group_value - group_end
            if GROUP_LIST_TYPE == 0
            else current_group_value
        )
        group_end = group_start + group_m
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        current_count = last_count + num_m_tiles * num_n_tiles
        current_block = core_idx if core_idx >= last_count else core_idx + total_cores

        for block_idx in range(current_block, current_count, total_cores):
            tile_m, tile_n = _block_indexer(
                block_idx - last_count,
                num_m_tiles,
                num_n_tiles,
                BLOCK_THRESHOLD,
            )
            offsets_m = group_start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offsets_k = tl.arange(0, BLOCK_K)
            # Widen before multiplying: A, B and C can exceed 2^31 elements.
            offsets_m = offsets_m.to(tl.int64)
            offsets_k = offsets_k.to(tl.int64)
            group_base = group_idx.to(tl.int64) * K * N
            a_base = A + offsets_m[:, None] * K + offsets_k[None, :]
            b_base = B + group_base + offsets_k[:, None] * N + offsets_n[None, :]
            mask_m = offsets_m < group_end
            mask_n = offsets_n < N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_idx in tl.range(num_k_tiles):
                rotated_k = ((k_idx + tile_m) % num_k_tiles).to(tl.int64)
                mask_k = offsets_k < K - rotated_k * BLOCK_K
                a = tl.load(
                    a_base + rotated_k * BLOCK_K,
                    mask=mask_m[:, None] and mask_k[None, :],
                    other=0.0,
                )
                b = tl.load(
                    b_base + rotated_k * BLOCK_K * N,
                    mask=mask_k[:, None] and mask_n[None, :],
                    other=0.0,
                )
                al.compile_hint(a, "dot_pad_only_k")
                al.compile_hint(b, "dot_pad_only_k")
                accumulator = tl.dot(a, b, accumulator)

            output = C + offsets_m[:, None] * N + offsets_n[None, :]
            tl.store(
                output,
                accumulator.to(C.dtype.element_ty),
                mask=mask_m[:, None] and mask_n[None, :],
            )

        last_count = current_count % total_cores
        group_start = group_end


@triton.jit
def _grouped_matmul_panel_kernel(
    A,
    B,
    C,
    group_list,
    GROUPS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    PANEL_M: tl.constexpr = 8,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 256,
    BLOCK_K: tl.constexpr = 256,
):
    total_cores = tl.num_programs(0)
    core_idx = tl.program_id(0)
    num_n_tiles: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    num_k_tiles: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    last_count = 0
    group_start = 0
    group_end = 0

    for group_idx in range(GROUPS):
        current_group_value = tl.load(group_list + group_idx).to(tl.int32)
        group_m = (
            current_group_value - group_end
            if GROUP_LIST_TYPE == 0
            else current_group_value
        )
        group_end = group_start + group_m
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        current_count = last_count + num_m_tiles * num_n_tiles
        current_block = core_idx if core_idx >= last_count else core_idx + total_cores

        for block_idx in range(current_block, current_count, total_cores):
            local_tile = block_idx - last_count
            panel_tasks: tl.constexpr = PANEL_M * num_n_tiles
            panel_idx = local_tile // panel_tasks
            panel_start_m = panel_idx * PANEL_M
            remaining_m = num_m_tiles - panel_start_m
            panel_m = PANEL_M if remaining_m >= PANEL_M else remaining_m
            tile_in_panel = local_tile % panel_tasks
            tile_m = panel_start_m + tile_in_panel % panel_m
            tile_n = tile_in_panel // panel_m

            offsets_m = group_start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offsets_k = tl.arange(0, BLOCK_K)
            # Widen before multiplying: A, B and C can exceed 2^31 elements.
            offsets_m = offsets_m.to(tl.int64)
            offsets_k = offsets_k.to(tl.int64)
            group_base = group_idx.to(tl.int64) * K * N
            a_base = A + offsets_m[:, None] * K + offsets_k[None, :]
            b_base = B + group_base + offsets_k[:, None] * N + offsets_n[None, :]
            mask_m = offsets_m < group_end
            mask_n = offsets_n < N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_idx in tl.range(num_k_tiles):
                rotated_k = ((k_idx + tile_m) % num_k_tiles).to(tl.int64)
                mask_k = offsets_k < K - rotated_k * BLOCK_K
                a = tl.load(
                    a_base + rotated_k * BLOCK_K,
                    mask=mask_m[:, None] and mask_k[None, :],
                    other=0.0,
                )
                b = tl.load(
                    b_base + rotated_k * BLOCK_K * N,
                    mask=mask_k[:, None] and mask_n[None, :],
                    other=0.0,
                )
                al.compile_hint(a, "dot_pad_only_k")
                al.compile_hint(b, "dot_pad_only_k")
                accumulator = tl.dot(a, b, accumulator)

            output = C + offsets_m[:, None] * N + offsets_n[None, :]
            tl.store(
                output,
                accumulator.to(C.dtype.element_ty),
                mask=mask_m[:, None] and mask_n[None, :],
            )

        last_count = current_count % total_cores
        group_start = group_end


def _validate_common(
    x: torch.Tensor,
    group_list: torch.Tensor,
    groups: int,
    group_list_type: int,
) -> None:
    if group_list_type not in (0, 1):
        raise ValueError("group_list_type must be 0 or 1")
    if x.ndim != 2:
        raise ValueError("x must have shape [M, K]")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError("grouped_matmul supports float16 and bfloat16 inputs")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if group_list.ndim != 1 or group_list.numel() != groups:
        raise ValueError("group_list must be a 1D tensor with one value per group")
    if group_list.dtype not in (torch.int32, torch.int64):
        raise TypeError("group_list must have dtype int32 or int64")
    if group_list.device != x.device:
        raise ValueError("x and group_list must be on the same device")
    if groups <= 0:
        raise ValueError("at least one group is required")


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 0,
) -> torch.Tensor:
    """Compute ``Y[group] = X[group] @ Weight[group]`` on Ascend.

    ``x`` has shape ``[M,K]``, ``weight`` has shape ``[G,K,N]`` and
    ``group_list`` stores cumulative ends (type 0) or group sizes (type 1).
    """
    if weight.ndim != 3:
        raise ValueError("weight must have shape [G, K, N]")
    groups, weight_k, n = weight.shape
    _validate_common(x, group_list, groups, group_list_type)
    m, k = x.shape
    if k <= 0 or n <= 0:
        raise ValueError("K and N must be positive")
    if weight_k != k:
        raise ValueError("x and weight must have the same K dimension")
    if weight.dtype != x.dtype:
        raise TypeError("x and weight must have the same dtype")
    if weight.device != x.device:
        raise ValueError("x and weight must be on the same device")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")

    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    average_group_m = (m + groups - 1) // groups
    use_weight_major_panel = average_group_m >= 128 and (k, n) == (1024, 4096)

    with torch_device_fn.device(x.device):
        num_cores = _get_aic_core_num(x.device.index)
        if use_weight_major_panel:
            _grouped_matmul_panel_kernel[(num_cores,)](
                x,
                weight,
                output,
                group_list,
                GROUPS=groups,
                N=n,
                K=k,
                GROUP_LIST_TYPE=group_list_type,
                PANEL_M=8,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        else:
            num_n_tiles = (n + WEIGHT_BLOCK_N - 1) // WEIGHT_BLOCK_N
            block_threshold = (
                24
                if group_list_type == 0 and num_n_tiles >= 16 and average_group_m >= 128
                else 8
            )
            _grouped_matmul_nd_kernel[(num_cores,)](
                x,
                weight,
                output,
                group_list,
                GROUPS=groups,
                N=n,
                K=k,
                GROUP_LIST_TYPE=group_list_type,
                BLOCK_THRESHOLD=block_threshold,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
    return output


__all__ = ["grouped_matmul"]
