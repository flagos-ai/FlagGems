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

import functools
import logging
import threading
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.device_info import get_sm_count

logger = logging.getLogger(__name__)


@triton.jit
def _grouped_tile_id(
    tile_id,
    m,
    n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    num_m_tiles = tl.cdiv(m, BLOCK_M)
    num_n_tiles = tl.cdiv(n, BLOCK_N)
    tiles_per_group = GROUP_M * num_n_tiles
    group_id = tile_id // tiles_per_group
    first_m_tile = group_id * GROUP_M
    group_m = tl.minimum(num_m_tiles - first_m_tile, GROUP_M)
    tile_in_group = tile_id % tiles_per_group
    tile_m = first_m_tile + tile_in_group % group_m
    tile_n = tile_in_group // group_m
    return tile_m, tile_n


@libentry()
@triton.jit
def group_gemm_w8a8_fp8_kernel(
    A,
    B,
    A_SCALE,
    B_SCALE,
    C,
    OFFS,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_offs: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    SCALE_BLOCK_M: tl.constexpr,
    SCALE_BLOCK_N: tl.constexpr,
    A_SCALE_PER_ROW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_GROUP_CHUNKS: tl.constexpr,
    USE_INT32_SCHEDULER: tl.constexpr,
    USE_INT32_ADDRESSES: tl.constexpr,
    SWAP_AB: tl.constexpr,
    K_ALIGNED: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    chunk_id = tl.program_id(0)
    if USE_INT32_SCHEDULER:
        tile_id = tl.program_id(1).to(tl.int32)
        grid_size = tl.num_programs(1).to(tl.int32)
        problem_start = tl.full((), 0, tl.int32)
    else:
        tile_id = tl.program_id(1).to(tl.int64)
        grid_size = tl.num_programs(1).to(tl.int64)
        problem_start = tl.full((), 0, tl.int64)

    # Interleave groups across a few CTA cohorts so short-K kernels do not make
    # every CTA scan the complete offsets array.
    for group_slot in tl.range(tl.cdiv(NUM_GROUPS, NUM_GROUP_CHUNKS)):
        group_id = chunk_id + group_slot * NUM_GROUP_CHUNKS
        valid_group = group_id < NUM_GROUPS
        if USE_INT32_ADDRESSES:
            address_group_id = group_id
        else:
            address_group_id = group_id.to(tl.int64)
        if USE_INT32_SCHEDULER:
            group_start = tl.load(
                OFFS + (address_group_id - 1) * stride_offs,
                mask=valid_group & (group_id > 0),
                other=0,
            ).to(tl.int32)
            group_end = tl.load(
                OFFS + address_group_id * stride_offs,
                mask=valid_group,
                other=group_start,
            ).to(tl.int32)
        else:
            group_start = tl.load(
                OFFS + (address_group_id - 1) * stride_offs,
                mask=valid_group & (group_id > 0),
                other=0,
            ).to(tl.int64)
            group_end = tl.load(
                OFFS + address_group_id * stride_offs,
                mask=valid_group,
                other=group_start,
            ).to(tl.int64)
        group_m = group_end - group_start
        num_tiles = tl.cdiv(group_m, BLOCK_M) * tl.cdiv(N, BLOCK_N)
        problem_end = problem_start + num_tiles

        if tile_id >= problem_start and tile_id < problem_end:
            loop_count = (problem_end - tile_id + grid_size - 1) // grid_size
            for _ in tl.range(loop_count):
                local_tile = tile_id - problem_start
                tile_m, tile_n = _grouped_tile_id(
                    local_tile, group_m, N, BLOCK_M, BLOCK_N, GROUP_M
                )
                m_offset = group_start + tile_m * BLOCK_M
                n_offset = tile_n * BLOCK_N
                rows = m_offset + tl.arange(0, BLOCK_M)
                cols = n_offset + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                if SWAP_AB:
                    accumulator_nm = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
                else:
                    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k_block in range(0, tl.cdiv(K, BLOCK_K)):
                    k_offset = k_block * BLOCK_K
                    ks = k_offset + offs_k
                    if USE_INT32_ADDRESSES:
                        a_ptrs = A + rows[:, None] * stride_am + ks[None, :] * stride_ak
                        b_ptrs = (
                            B
                            + group_id * stride_bg
                            + ks[:, None] * stride_bk
                            + cols[None, :] * stride_bn
                        )
                    else:
                        address_rows = rows.to(tl.int64)
                        address_cols = cols.to(tl.int64)
                        address_ks = ks.to(tl.int64)
                        a_ptrs = (
                            A
                            + address_rows[:, None] * stride_am
                            + address_ks[None, :] * stride_ak
                        )
                        b_ptrs = (
                            B
                            + address_group_id * stride_bg
                            + address_ks[:, None] * stride_bk
                            + address_cols[None, :] * stride_bn
                        )
                    if K_ALIGNED:
                        a = tl.load(
                            a_ptrs,
                            mask=(rows[:, None] < group_end) & (rows[:, None] < M),
                            other=0.0,
                        )
                    else:
                        a = tl.load(
                            a_ptrs,
                            mask=(rows[:, None] < group_end)
                            & (rows[:, None] < M)
                            & (ks[None, :] < K),
                            other=0.0,
                        )
                    if K_ALIGNED and N_ALIGNED:
                        b = tl.load(b_ptrs)
                    elif K_ALIGNED:
                        b = tl.load(b_ptrs, mask=cols[None, :] < N, other=0.0)
                    else:
                        b = tl.load(
                            b_ptrs,
                            mask=(ks[:, None] < K) & (cols[None, :] < N),
                            other=0.0,
                        )

                    if A_SCALE_PER_ROW:
                        a_scale_rows = rows
                    else:
                        a_scale_rows = rows // SCALE_BLOCK_M
                    if not USE_INT32_ADDRESSES:
                        a_scale_rows = a_scale_rows.to(tl.int64)
                    scale_k = k_block
                    a_scale = tl.load(
                        A_SCALE + a_scale_rows * stride_asm + scale_k * stride_ask,
                        mask=(rows < group_end) & (rows < M),
                        other=0.0,
                    ).to(tl.float32)
                    scale_n = n_offset // SCALE_BLOCK_N
                    if USE_INT32_ADDRESSES:
                        b_scale_ptr = (
                            B_SCALE
                            + group_id * stride_bsg
                            + scale_k * stride_bsk
                            + scale_n * stride_bsn
                        )
                    else:
                        b_scale_ptr = (
                            B_SCALE
                            + address_group_id * stride_bsg
                            + scale_k * stride_bsk
                            + scale_n * stride_bsn
                        )
                    b_scale = tl.load(b_scale_ptr).to(tl.float32)
                    if SWAP_AB:
                        combined_scale_nm = b_scale * a_scale[None, :]
                        accumulator_nm += (
                            tl.dot(tl.trans(b), tl.trans(a), allow_tf32=False)
                            * combined_scale_nm
                        )
                    else:
                        combined_scale = a_scale * b_scale
                        accumulator += (
                            tl.dot(a, b, allow_tf32=False) * combined_scale[:, None]
                        )

                if SWAP_AB:
                    accumulator = tl.trans(accumulator_nm)
                if USE_INT32_ADDRESSES:
                    c_ptrs = C + rows[:, None] * stride_cm + cols[None, :] * stride_cn
                else:
                    c_ptrs = (
                        C
                        + rows.to(tl.int64)[:, None] * stride_cm
                        + cols.to(tl.int64)[None, :] * stride_cn
                    )
                c_mask = (
                    (rows[:, None] < group_end)
                    & (rows[:, None] < M)
                    & (cols[None, :] < N)
                )
                tl.store(c_ptrs, accumulator, mask=c_mask)
                tile_id += grid_size

        problem_start = problem_end


_FLOAT8_DTYPES = tuple(
    getattr(torch, name)
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
    )
    if hasattr(torch, name)
)
_INT32_MAX = torch.iinfo(torch.int32).max


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@functools.lru_cache(maxsize=128)
def _int32_addressing_is_safe(
    m,
    n,
    k,
    num_groups,
    block_m,
    block_n,
    block_k,
    scale_block_m,
    scale_block_n,
    a_scale_per_row,
    a_strides,
    b_strides,
    a_scale_strides,
    b_scale_strides,
    out_strides,
    offs_stride,
):
    # Masked tail lanes still form addresses, and a group-local M tile can
    # extend BLOCK_M - 1 rows past the final logical row.
    max_row = m + block_m - 2
    max_col = ((n + block_n - 1) // block_n) * block_n - 1
    max_k = max(((k + block_k - 1) // block_k) * block_k - 1, 0)
    max_group = num_groups - 1
    max_scale_k = max((k + block_k - 1) // block_k - 1, 0)
    max_scale_n = (n - 1) // scale_block_n
    max_scale_row = max_row if a_scale_per_row else max_row // scale_block_m

    if max(max_row, max_col, max_k, max_group) > _INT32_MAX:
        return False

    def offset_fits(coordinates, strides):
        return (
            sum(
                coordinate * abs(stride)
                for coordinate, stride in zip(coordinates, strides)
            )
            <= _INT32_MAX
        )

    return all(
        (
            offset_fits((max_row, max_k), a_strides),
            offset_fits((max_group, max_k, max_col), b_strides),
            offset_fits((max_scale_row, max_scale_k), a_scale_strides),
            offset_fits((max_group, max_scale_k, max_scale_n), b_scale_strides),
            offset_fits((max_row, max_col), out_strides),
            max_group * abs(offs_stride) <= _INT32_MAX,
        )
    )


@functools.lru_cache
def _get_group_gemm_w8a8_fp8_config(
    m: int,
    n: int,
    k: int,
    num_groups: int,
    block_n: int,
    block_k: int,
    is_h20: bool,
):
    average_m = triton.cdiv(m, num_groups)
    if is_h20 and average_m <= 8 and k > 256:
        block_m, group_m = 8, 1
    elif average_m <= 16:
        block_m, group_m = 16, 1
    elif is_h20 and average_m <= 32:
        block_m, group_m = 32, 8
    else:
        block_m, group_m = 64, 8

    if is_h20 and 32 < average_m < 64 and k % block_k != 0 and n % 64 != 0:
        tile_n = min(block_n, 32)
    elif average_m > 16:
        tile_n = min(block_n, 64)
    elif n <= 64:
        tile_n = min(block_n, 64)
    elif average_m <= 32 and n < 1024 and k > 128:
        tile_n = min(block_n, 64)
    else:
        tile_n = min(block_n, 128)
    tile_n = min(tile_n, max(16, triton.next_power_of_2(n)))
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": tile_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "num_warps": 4,
        "num_stages": 2 if is_h20 and k <= 256 else 3,
    }


@functools.lru_cache
def _is_h20(device: torch.device) -> bool:
    return "H20" in torch_device_fn.get_device_name(device).split()


@functools.lru_cache
def _get_sm_count_for_device(device: torch.device) -> int:
    # device_info.get_sm_count() is cached globally, so resolve properties in
    # the tensor's context to keep heterogeneous multi-GPU dispatch correct.
    with torch_device_fn.device(device):
        properties = torch_device_fn.get_device_properties(
            torch_device_fn.current_device()
        )
    sm_count = getattr(properties, "multi_processor_count", None) or getattr(
        properties, "multiProcessorCount", None
    )
    return sm_count if sm_count is not None else get_sm_count()


_compiled_kernel_cache = {}
_compiled_kernel_cache_lock = threading.Lock()
_COMPILED_KERNEL_CACHE_LIMIT = 128
_CACHE_MISS = object()
_FAST_PATH_UNAVAILABLE = object()


def _store_compiled_kernel(cache_key, compiled):
    # This helper is only called while holding _compiled_kernel_cache_lock.
    if len(_compiled_kernel_cache) >= _COMPILED_KERNEL_CACHE_LIMIT:
        _compiled_kernel_cache.clear()
    _compiled_kernel_cache[cache_key] = compiled


def _launch_group_gemm_w8a8_fp8(
    grid, kernel_args, cache_key, num_warps: int, num_stages: int
):
    # Bypass repeated JIT binder/cache-key work after the first H20 launch.
    compiled = _compiled_kernel_cache.get(cache_key, _CACHE_MISS)
    if compiled is not _CACHE_MISS and compiled is not _FAST_PATH_UNAVAILABLE:
        compiled[(grid[0], grid[1], 1)](*kernel_args)
        return

    if compiled is _CACHE_MISS:
        # Triton's compilation and first CompiledKernel handle initialization
        # are not thread-safe. Cache hits remain lock-free.
        with _compiled_kernel_cache_lock:
            compiled = _compiled_kernel_cache.get(cache_key, _CACHE_MISS)
            if compiled is _CACHE_MISS:
                jit_fn = group_gemm_w8a8_fp8_kernel.fn
                if not hasattr(jit_fn, "warmup"):
                    compiled = _FAST_PATH_UNAVAILABLE
                    _store_compiled_kernel(cache_key, compiled)
                else:
                    compiled = jit_fn.warmup(
                        *kernel_args,
                        grid=grid,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    # Initialize the handle and submit the first real launch
                    # before publishing the entry to concurrent callers.
                    compiled[(grid[0], grid[1], 1)](*kernel_args)
                    _store_compiled_kernel(cache_key, compiled)
                    return

    if compiled is _FAST_PATH_UNAVAILABLE:
        # LibEntry serializes its own first compilation on older Triton builds.
        group_gemm_w8a8_fp8_kernel[grid](
            *kernel_args, num_warps=num_warps, num_stages=num_stages
        )
    else:
        # Another thread populated the cache while this caller waited.
        compiled[(grid[0], grid[1], 1)](*kernel_args)


def group_gemm_w8a8_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    offs: torch.Tensor,
    block_size: Sequence[int] = (128, 128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Grouped W8A8 FP8 GEMM with block-wise dequantization.

    ``A`` has shape ``[M, K]``, ``B`` has shape ``[G, K, N]``, and ``offs``
    contains the nondecreasing cumulative row count for every group and must
    end at ``M``. ``B_scale`` has shape
    ``[G, ceil(K / block_k), ceil(N / block_n)]``. ``A_scale`` can use either
    symmetric blocks, ``[ceil(M / block_m), ceil(K / block_k)]``, or per-row
    activation blocks, ``[M, ceil(K / block_k)]``. Both common ``B`` strides
    are supported; on H20, a logical ``[G, K, N]`` transpose view backed by a
    contiguous ``[G, N, K]`` tensor gives the highest throughput.
    """
    logger.debug("GEMS GROUP GEMM W8A8 FP8")
    if A.ndim != 2 or B.ndim != 3:
        raise RuntimeError("A must be 2D and B must be 3D")
    a_dtype = A.dtype
    a_device = A.device
    if a_dtype != B.dtype or a_dtype not in _FLOAT8_DTYPES:
        raise RuntimeError("A and B must have the same FP8 dtype")
    if a_device != B.device:
        raise RuntimeError("A and B must be on the same device")
    if len(block_size) != 3:
        raise RuntimeError("block_size must contain (block_m, block_n, block_k)")

    block_m, block_n, block_k = (int(value) for value in block_size)
    if not (
        _is_power_of_two(block_m)
        and _is_power_of_two(block_n)
        and _is_power_of_two(block_k)
    ):
        raise RuntimeError("all block sizes must be positive powers of two")
    if min(block_m, block_n) < 16 or not 32 <= block_k <= 256:
        raise RuntimeError(
            "block_m and block_n must be at least 16; block_k must be in [32, 256]"
        )

    M, K = A.shape
    num_groups, BK, N = B.shape
    if num_groups == 0:
        raise RuntimeError("B must contain at least one group")
    if K != BK:
        raise RuntimeError(f"K dimension mismatch: {K} and {BK}")
    if offs.ndim != 1 or offs.dtype != torch.int32 or offs.numel() != num_groups:
        raise RuntimeError("offs must be a 1D int32 tensor with one entry per group")
    if offs.device != a_device:
        raise RuntimeError("offs must be on the same device as A")
    num_k_blocks = (K + block_k - 1) // block_k
    num_n_blocks = (N + block_n - 1) // block_n
    block_a_shape = ((M + block_m - 1) // block_m, num_k_blocks)
    row_a_shape = (M, num_k_blocks)
    if A_scale.shape == row_a_shape:
        a_scale_per_row = True
    elif A_scale.shape == block_a_shape:
        a_scale_per_row = False
    else:
        raise RuntimeError(f"A_scale must have shape {block_a_shape} or {row_a_shape}")
    expected_b_scale_shape = (num_groups, num_k_blocks, num_n_blocks)
    if B_scale.shape != expected_b_scale_shape:
        raise RuntimeError(f"B_scale must have shape {expected_b_scale_shape}")
    if A_scale.dtype != torch.float32 or B_scale.dtype != torch.float32:
        raise RuntimeError("A_scale and B_scale must have dtype torch.float32")
    if A_scale.device != a_device or B_scale.device != a_device:
        raise RuntimeError("scales must be on the same device as A")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError("out_dtype must be bfloat16, float16, or float32")

    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=a_device)
    elif out.shape != (M, N) or out.dtype != out_dtype or out.device != a_device:
        raise RuntimeError("out must match the expected shape, dtype, and device")
    if M == 0 or N == 0:
        return out

    is_h20 = _is_h20(a_device)
    config = _get_group_gemm_w8a8_fp8_config(
        M, N, K, num_groups, block_n, block_k, is_h20
    )
    average_m = (M + num_groups - 1) // num_groups
    if block_n % config["BLOCK_N"] != 0:
        raise RuntimeError("block_n must be divisible by the selected BLOCK_N")

    if is_h20 and config["BLOCK_M"] == 8 and N <= 512:
        grid_multiplier = 2
    else:
        grid_multiplier = (
            4
            if K <= 256
            or config["BLOCK_N"] <= 64
            or (is_h20 and config["BLOCK_M"] <= 16)
            else 2
        )
    # For sum(group_m) == M, this bounds sum(ceil(group_m / BLOCK_M))
    # without reading offs back to the host.
    max_nonempty_groups = min(M, num_groups)
    max_m_tiles = max_nonempty_groups + (M - max_nonempty_groups) // config["BLOCK_M"]
    max_tiles = max_m_tiles * ((N + config["BLOCK_N"] - 1) // config["BLOCK_N"])
    target_ctas = max(
        1, min(_get_sm_count_for_device(a_device) * grid_multiplier, max_tiles)
    )
    # Four cohorts retained the uniform-shape gain without starving a single
    # hot group in the skewed-offset sweep.
    num_group_chunks = (
        min(4, num_groups, target_ctas)
        if is_h20 and K <= 256 and num_groups >= 32
        else 1
    )
    a_strides = A.stride()
    b_strides = B.stride()
    a_scale_strides = A_scale.stride()
    b_scale_strides = B_scale.stride()
    out_strides = out.stride()
    offs_strides = offs.stride()
    stride_am, stride_ak = a_strides
    stride_bg, stride_bk, stride_bn = b_strides
    stride_asm, stride_ask = a_scale_strides
    stride_bsg, stride_bsk, stride_bsn = b_scale_strides
    stride_cm, stride_cn = out_strides
    stride_offs = offs_strides[0]
    use_int32_addresses = _int32_addressing_is_safe(
        M,
        N,
        K,
        num_groups,
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["BLOCK_K"],
        block_m,
        block_n,
        a_scale_per_row,
        a_strides,
        b_strides,
        a_scale_strides,
        b_scale_strides,
        out_strides,
        stride_offs,
    )
    use_int32_scheduler = (
        is_h20
        and K <= 256
        and average_m <= 32
        and M <= _INT32_MAX
        and max_tiles + target_ctas - 1 <= _INT32_MAX
        and use_int32_addresses
    )
    # H20 maps the larger N axis more efficiently when small-M K-major tiles
    # accumulate B.T @ A.T and transpose once before the store.
    swap_ab = is_h20 and config["BLOCK_M"] <= 32 and stride_bk == 1
    k_aligned = K % config["BLOCK_K"] == 0
    n_aligned = N % config["BLOCK_N"] == 0
    grid = (num_group_chunks, triton.cdiv(target_ctas, num_group_chunks))
    kernel_args = (
        A,
        B,
        A_scale,
        B_scale,
        out,
        offs,
        M,
        N,
        K,
        num_groups,
        stride_offs,
        stride_am,
        stride_ak,
        stride_bg,
        stride_bk,
        stride_bn,
        stride_asm,
        stride_ask,
        stride_bsg,
        stride_bsk,
        stride_bsn,
        stride_cm,
        stride_cn,
        block_m,
        block_n,
        a_scale_per_row,
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["BLOCK_K"],
        config["GROUP_M"],
        num_group_chunks,
        use_int32_scheduler,
        use_int32_addresses,
        swap_ab,
        k_aligned,
        n_aligned,
    )
    cache_key = (
        a_device.type,
        a_device.index,
        a_dtype,
        out.dtype,
        M,
        N,
        K,
        num_groups,
        block_m,
        block_n,
        block_k,
        a_scale_per_row,
        a_strides,
        b_strides,
        a_scale_strides,
        b_scale_strides,
        out_strides,
        offs_strides,
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["BLOCK_K"],
        config["GROUP_M"],
        config["num_warps"],
        config["num_stages"],
        num_group_chunks,
        use_int32_scheduler,
        use_int32_addresses,
        swap_ab,
        k_aligned,
        n_aligned,
        grid,
        tuple(
            tensor.data_ptr() % 16 == 0
            for tensor in (A, B, A_scale, B_scale, out, offs)
        ),
    )
    with torch_device_fn.device(a_device):
        if is_h20:
            _launch_group_gemm_w8a8_fp8(
                grid,
                kernel_args,
                cache_key,
                config["num_warps"],
                config["num_stages"],
            )
        else:
            group_gemm_w8a8_fp8_kernel.fn[grid](
                *kernel_args,
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
    return out


group_mm_w8a8_fp8 = group_gemm_w8a8_fp8
