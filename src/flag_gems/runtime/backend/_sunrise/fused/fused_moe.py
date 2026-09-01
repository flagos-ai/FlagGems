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

import contextlib
import threading
from typing import Any

from flag_gems.fused import fused_moe as generic_fused_moe

_PATCH_LOCK = threading.RLock()
_GENERIC_GET_DEFAULT_CONFIG = generic_fused_moe.get_default_config
MOE_GEMM_TUNING_MIN_TOKENS = 4096
_HALF_GEMM_TILE_M = 128
_HALF_GEMM_TILE_K = 64
_HALF_GEMM2_TILE_N = 256
_PLAIN_HALF_CONFIG_DTYPES = ("fp16", "bf16")


def _sunrise_get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
    gemm_stage: str = "gemm1",
    enable_gemm_fast_path: bool = False,
) -> dict[str, Any]:
    # config = _GENERIC_GET_DEFAULT_CONFIG(
    #     M,
    #     E,
    #     N,
    #     K,
    #     topk,
    #     dtype,
    #     block_shape,
    #     gemm_stage,
    #     enable_gemm_fast_path,
    # )

    # # Sunrise/PTPU can exhaust registers in the generic fused MoE kernel when
    # # large-N half-precision tiles keep BLOCK_SIZE_N at 128. Narrowing the N
    # # tile to 64 avoids the inline-asm register overflow seen on PT200.
    # if dtype in _PLAIN_HALF_CONFIG_DTYPES and N >= 4096:
    #     config = config.copy()
    #     config["BLOCK_SIZE_N"] = min(config["BLOCK_SIZE_N"], 64)

    # return config

    if gemm_stage not in ("gemm1", "gemm2"):
        raise ValueError(f"Unsupported MoE GEMM stage: {gemm_stage}")

    if dtype in _PLAIN_HALF_CONFIG_DTYPES:
        # Routed rows per expert drives block_m.  Each token contributes topk
        # rows to the expert-sorted GEMM input, so M * topk / E is the relevant
        # density for high-expert-count MoE routing.
        routed_tokens_per_expert = M * max(topk, 1) // max(E, 1)
        tokens_per_expert = M // max(E, 1)

        if routed_tokens_per_expert <= 16:
            block_m = 16
        elif routed_tokens_per_expert <= 64:
            block_m = 64
        else:
            block_m = 128

        if tokens_per_expert > 128:
            group_m = 16
        elif tokens_per_expert > 32:
            group_m = 8
        else:
            group_m = 1

        block_k = 128 if M <= 64 else 64

        if N >= 4096:
            block_n = 128 if M <= 128 else 256
        else:
            block_n = 64 if M <= 64 else 128

        can_use_gemm_fast_path = (
            enable_gemm_fast_path
            and M >= MOE_GEMM_TUNING_MIN_TOKENS
            and block_m == _HALF_GEMM_TILE_M
            and block_k == _HALF_GEMM_TILE_K
        )

        use_gemm2_fast_path = (
            gemm_stage == "gemm2"
            and can_use_gemm_fast_path
            and N % _HALF_GEMM2_TILE_N == 0
        )
        use_gemm1_fast_path = (
            gemm_stage == "gemm1" and can_use_gemm_fast_path and N % block_n == 0
        )

        if gemm_stage == "gemm2" and enable_gemm_fast_path:
            block_n = (
                _HALF_GEMM2_TILE_N if use_gemm2_fast_path else (64 if M <= 64 else 128)
            )

        # Prefer 4 warps for small tiles; only use 8 for large M
        num_warps = 4 if M <= 128 else 8
        num_stages = 3

        if use_gemm1_fast_path:
            group_m = 1
            num_stages = 4
        elif use_gemm2_fast_path:
            group_m = 2
            num_stages = 4

        smem_per_stage = (block_m * block_k + block_k * block_n) * 2
        while num_stages > 2 and smem_per_stage * num_stages > 200_000:
            num_stages -= 1

        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        if use_gemm1_fast_path:
            config["PAIR_GATE_UP_DOT"] = True
    else:
        tokens_per_expert = M // max(E, 1)

        if tokens_per_expert <= 2:
            block_m = 16
        elif tokens_per_expert <= 4:
            block_m = 32
        elif tokens_per_expert <= 16:
            block_m = 64
        else:
            block_m = 128

        # Tile sizing
        if N >= 4096:
            block_n = 128 if M <= 128 else 256
        elif N >= 1024:
            block_n = 64 if M <= 64 else 128
        else:
            block_n = 64 if M <= 64 else 128

        if dtype == "fp8_w8a8":
            block_k = 128
        elif M <= 64:
            block_k = 128
        else:
            block_k = 64

        if tokens_per_expert > 128:
            group_m = 16
        elif tokens_per_expert > 32:
            group_m = 8
        else:
            group_m = 1

        # Prefer 4 warps for small tiles; only use 8 for large M
        num_warps = 4 if M <= 128 else 8
        num_stages = 3

        smem_per_stage = (block_m * block_k + block_k * block_n) * 2
        while num_stages > 2 and smem_per_stage * num_stages > 200_000:
            num_stages -= 1

        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    # sunrise
    while config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_N"] > 64 * 64:
        if config["BLOCK_SIZE_M"] > config["BLOCK_SIZE_N"]:
            config["BLOCK_SIZE_M"] //= 2
        else:
            config["BLOCK_SIZE_N"] //= 2
    if config["BLOCK_SIZE_K"] > 64:
        config["BLOCK_SIZE_K"] = 64
    if config["num_warps"] < 8:
        config["num_warps"] = 8
    if config["num_warps"] > 16:
        config["num_warps"] = 16
    return config


@contextlib.contextmanager
def _sunrise_moe_config_patch():
    with _PATCH_LOCK:
        original = generic_fused_moe.get_default_config
        generic_fused_moe.get_default_config = _sunrise_get_default_config
        try:
            yield
        finally:
            generic_fused_moe.get_default_config = original


def fused_experts_impl(*args, **kwargs):
    with _sunrise_moe_config_patch():
        return generic_fused_moe.fused_experts_impl(*args, **kwargs)


def inplace_fused_experts(*args, **kwargs):
    with _sunrise_moe_config_patch():
        return generic_fused_moe.inplace_fused_experts(*args, **kwargs)


def outplace_fused_experts(*args, **kwargs):
    with _sunrise_moe_config_patch():
        return generic_fused_moe.outplace_fused_experts(*args, **kwargs)
