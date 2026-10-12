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
import math
import os
from typing import Any

import torch
import triton
import triton.language as tl
import yaml

import flag_gems

logger = logging.getLogger(__name__)


def _get_default_w8a8_block_fp8_config(block_n: int, block_k: int) -> dict[str, Any]:
    if flag_gems.device != "cuda":
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": min(128, block_k),
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 3,
        }

    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 2,
    }


@triton.jit
def w8a8_block_fp8_matmul_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@functools.lru_cache
def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int, device: torch.device | None = None
) -> dict[int, Any] | None:
    if not torch.cuda.is_available() or (device is not None and device.type != "cuda"):
        logger.debug(
            "CUDA is unavailable on this backend; using default W8A8 block FP8 config."
        )
        return None

    device_name = torch.cuda.get_device_name(device).replace(" ", "_")
    file_name = f"fp8_w8a8-{block_n}-{block_k}.yaml"

    config_dir = os.path.join(os.path.dirname(__file__), "..", "utils", "configs")
    cfg_file = os.path.join(config_dir, file_name)

    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            logger.info(
                "Using config from %s for W8A8 block FP8 kernel.",
                cfg_file,
            )
            dev_data = (yaml.safe_load(f) or {}).get(device_name, {})
            NK_data = dev_data.get(f"{N},{K}", {})

            result = {}
            fields = (
                "BLOCK_SIZE_M",
                "BLOCK_SIZE_N",
                "BLOCK_SIZE_K",
                "GROUP_SIZE_M",
                "num_warps",
                "num_stages",
            )
            for m, entry in NK_data.items():
                # Legacy six-element lists still use the nearest M. A mapping
                # can select a specialized kernel, but must match M exactly.
                if isinstance(entry, list):
                    tile = entry
                    kernel = "generic"
                    exact_m = False
                    swap_ab = False
                    split_k = 1
                elif isinstance(entry, dict):
                    tile = entry.get("tile")
                    kernel = entry.get("kernel", "generic")
                    exact_m = entry.get("exact_m", kernel != "generic")
                    swap_ab = entry.get("swap_ab", False)
                    split_k = entry.get("split_k", 1)
                    if set(entry) - {
                        "tile",
                        "kernel",
                        "exact_m",
                        "swap_ab",
                        "split_k",
                    }:
                        raise ValueError(f"Unknown W8A8 config fields for M={m}")
                else:
                    raise TypeError(f"Invalid W8A8 config for M={m}")
                if not isinstance(tile, list) or len(tile) != len(fields):
                    raise ValueError(f"W8A8 config for M={m} needs six tile values")
                if any(type(value) is not int or value < 1 for value in tile):
                    raise ValueError(
                        f"W8A8 config for M={m} needs positive tile integers"
                    )
                if kernel not in ("generic", "hopper"):
                    raise ValueError(f"Unknown W8A8 kernel {kernel!r} for M={m}")
                if kernel == "hopper" and not exact_m:
                    raise ValueError("Hopper W8A8 configs must match M exactly")
                if kernel == "hopper" and (
                    (block_n, block_k) != (32, 32) or tile[2] != 32
                ):
                    raise ValueError(
                        "Hopper W8A8 configs require K32 block quantization"
                    )
                if kernel != "hopper" and (swap_ab or split_k != 1):
                    raise ValueError("swap_ab and split_k require the Hopper kernel")
                if (
                    not isinstance(split_k, int)
                    or split_k < 1
                    or split_k & (split_k - 1)
                ):
                    raise ValueError("W8A8 split_k must be a positive power of two")
                result[int(m)] = dict(
                    zip(fields, tile),
                    kernel=kernel,
                    exact_m=exact_m,
                    SWAP_AB=swap_ab,
                    SPLIT_K=split_k,
                )
            if not result:
                return None
            return result

    logger.warning(
        "Using default W8A8 Block FP8 kernel config. Performance might "
        "be sub-optimal! Config file not found at %s",
        cfg_file,
    )
    return None


def _select_w8a8_block_fp8_config(configs, M, block_n, block_k):
    if configs:
        if M in configs:
            return dict(configs[M])
        approximate = {
            m: cfg for m, cfg in configs.items() if not cfg.get("exact_m", False)
        }
        if approximate:
            return dict(approximate[min(approximate, key=lambda m: abs(m - M))])
    return _get_default_w8a8_block_fp8_config(block_n, block_k)


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    if block_k < 32 or block_k & (block_k - 1):
        raise ValueError("FP8 K groups must be powers of two >= 32")
    if block_n <= 0:
        raise ValueError("block_n must be positive")
    if not As.is_floating_point() or not Bs.is_floating_point():
        raise TypeError("matmul requires numeric scales; decode UE8M0 bytes first")

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = math.prod(A.shape[:-1])

    assert B.ndim == 2 and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
    if M == 0 or N == 0:
        return C
    if K == 0:
        return C.zero_()

    configs = get_w8a8_block_fp8_configs(N, K, block_n, block_k, A.device)
    config = _select_w8a8_block_fp8_config(configs, M, block_n, block_k)
    kernel = config.pop("kernel", "generic")
    config.pop("exact_m", None)
    if kernel == "hopper":
        if (
            flag_gems.device == "cuda"
            and A.ndim == 2
            and (block_n, block_k) == (32, 32)
            and A.dtype == B.dtype == torch.float8_e4m3fn
            and output_dtype in (torch.bfloat16, torch.float32)
            and torch.cuda.get_device_capability(A.device) == (9, 0)
        ):
            from ._w8a8_block_fp8_hopper import (
                _reduce_split_k,
                _w8a8_block_fp8_matmul_hopper,
            )

            splits = config["SPLIT_K"]
            partials = (
                torch.empty((splits, M, N), device=A.device, dtype=torch.float32)
                if splits > 1
                else C
            )
            grid = (
                triton.cdiv(M, config["BLOCK_SIZE_M"])
                * triton.cdiv(N, config["BLOCK_SIZE_N"]),
                splits,
            )
            _w8a8_block_fp8_matmul_hopper[grid](
                A,
                B,
                partials,
                As,
                Bs,
                M,
                N,
                K,
                block_n,
                block_k,
                A.stride(0),
                A.stride(1),
                B.stride(1),
                B.stride(0),
                C.stride(0),
                C.stride(1),
                As.stride(0),
                As.stride(1),
                Bs.stride(1),
                Bs.stride(0),
                needs_masking=bool(K % config["BLOCK_SIZE_K"]),
                **config,
            )
            if splits > 1:
                _reduce_split_k[(triton.cdiv(M * N, 256),)](
                    partials, C, M * N, splits, 256
                )
            return C
        config = _get_default_w8a8_block_fp8_config(block_n, block_k)
    else:
        config.pop("SWAP_AB", None)
        config.pop("SPLIT_K", None)

    # One K tile loads exactly one scale per operand. Tuned configurations,
    # like defaults, must not cross a quantization-group boundary. Copy the
    # cached configuration so calls with another group size cannot mutate it.
    config = dict(config)
    tile_k = min(config["BLOCK_SIZE_K"], block_k)
    if tile_k < 32 or block_k % tile_k:
        raise ValueError("GEMM K tile must divide the quantization group")
    config["BLOCK_SIZE_K"] = tile_k

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    w8a8_block_fp8_matmul_kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C
