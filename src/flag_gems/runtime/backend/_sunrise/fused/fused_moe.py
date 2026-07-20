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
    config = _GENERIC_GET_DEFAULT_CONFIG(
        M,
        E,
        N,
        K,
        topk,
        dtype,
        block_shape,
        gemm_stage,
        enable_gemm_fast_path,
    )

    # Sunrise/PTPU can exhaust registers in the generic fused MoE kernel when
    # large-N half-precision tiles keep BLOCK_SIZE_N at 128. Narrowing the N
    # tile to 64 avoids the inline-asm register overflow seen on PT200.
    if dtype in _PLAIN_HALF_CONFIG_DTYPES and N >= 4096:
        config = config.copy()
        config["BLOCK_SIZE_N"] = min(config["BLOCK_SIZE_N"], 64)

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
