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

"""Replace nn.Linear modules in a transformers model with TLEInt8Linear from
a pre-quantized safetensors state dict.

State dict convention (matches llm-compressor / compressed-tensors W8A8 output):
  <module_name>.weight        : int8 tensor [N, K]
  <module_name>.weight_scale  : fp32 tensor [N] or scalar

Example:
    from safetensors.torch import load_file
    from flag_gems.runtime.backend._arm.int8 import replace_linears_with_tle_int8
    state = load_file("Qwen3-1.7B-W8A8-INT8/model.safetensors")
    n = replace_linears_with_tle_int8(model, state, skip={"lm_head"})
"""

import logging
from typing import Iterable, Optional

import torch

from .tle_int8_linear import TLEInt8Linear

logger = logging.getLogger(__name__)


def replace_linears_with_tle_int8(
    model: torch.nn.Module,
    state_dict: dict,
    skip: Optional[Iterable[str]] = None,
    require_divisible_by: int = 4,
) -> int:
    """Walk model.named_modules(), replace each nn.Linear whose corresponding
    <name>.weight in state_dict is int8 with a TLEInt8Linear.

    Args:
        model: a torch.nn.Module (typically a transformers model).
        state_dict: {name.weight: int8 tensor, name.weight_scale: fp32 tensor, ...}.
        skip: iterable of module names to skip (e.g. {"lm_head"} to leave
              it as a plain Linear because it won't be followed by more
              decoder ops that can fuse it).
        require_divisible_by: minimum alignment for K and N (SDOT requires 4).

    Returns: number of Linear modules replaced.
    """
    skip_set = set(skip) if skip else set()
    n_replaced = 0
    n_skipped_dtype = 0
    n_skipped_align = 0
    n_skipped_missing = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if name in skip_set:
            continue

        w_key = f"{name}.weight"
        s_key = f"{name}.weight_scale"
        if w_key not in state_dict or s_key not in state_dict:
            n_skipped_missing += 1
            continue

        w = state_dict[w_key]
        s = state_dict[s_key]
        if w.dtype != torch.int8:
            n_skipped_dtype += 1
            continue

        N, K = w.shape
        if K % require_divisible_by != 0 or N % require_divisible_by != 0:
            logger.debug(
                "GEMS_ARM replace_linears_with_tle_int8: %s K=%d N=%d not divisible by %d",
                name,
                K,
                N,
                require_divisible_by,
            )
            n_skipped_align += 1
            continue

        # Walk dotted name to the parent module and setattr
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], TLEInt8Linear(w, s))
        n_replaced += 1

    # Engage the aten::_int_mm CPU override so TLEInt8Linear prefill's
    # torch._int_mm routes to the Triton SVE2 i8mm kernel (see quantize_live.py).
    from ..ops import apply_arm_overrides

    apply_arm_overrides(include=["_int_mm"])

    logger.info(
        "GEMS_ARM TLEInt8Linear: replaced %d Linear modules (skipped: %d dtype, %d alignment, %d missing, %d explicit)",
        n_replaced,
        n_skipped_dtype,
        n_skipped_align,
        n_skipped_missing,
        len(skip_set),
    )
    return n_replaced
