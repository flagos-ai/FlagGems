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
import os
from typing import Any, Dict, Optional

import torch
import yaml

logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "configs", "w8a8_block_fp8_matmul.yaml"
)


@functools.lru_cache
def _load_raw_configs() -> Dict[str, Any]:
    """Load the W8A8 block FP8 matmul tuned configs once per process."""
    if not os.path.exists(CONFIG_FILE):
        logger.warning(
            "W8A8 Block FP8 kernel config file not found at %s. "
            "Performance might be sub-optimal!",
            CONFIG_FILE,
        )
        return {}
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


@functools.lru_cache
def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Return tuned configs of the w8a8_block_fp8_matmul kernel for ``(N, K)``.

    The config data is stored in one extensible file (``utils/configs/``)::

        w8a8_block_fp8_matmul:            # op name
          "<block_n>-<block_k>":          # block variant
            <device_name>:                # e.g. NVIDIA_H100_80GB_HBM3
              "<N>,<K>":                  # shape
                "<M>": [BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages]

    Returns a dict mapping the closest ``M`` to a config dict, or ``None`` if
    no tuned config is available for the current device and shape.
    """
    if not torch.cuda.is_available():
        logger.debug("CUDA is unavailable; using default W8A8 block FP8 config.")
        return None

    device_name = torch.cuda.get_device_name().replace(" ", "_")
    NK_data = (
        _load_raw_configs()
        .get("w8a8_block_fp8_matmul", {})
        .get(f"{block_n}-{block_k}", {})
        .get(device_name, {})
        .get(f"{N},{K}", {})
    )

    result = {}
    for k, p in NK_data.items():
        # unpack the list into a kernel config dictionary
        result[int(k)] = {
            "BLOCK_SIZE_M": p[0],
            "BLOCK_SIZE_N": p[1],
            "BLOCK_SIZE_K": p[2],
            "GROUP_SIZE_M": p[3],
            "num_warps": p[4],
            "num_stages": p[5],
        }

    if not result:
        logger.debug(
            "No W8A8 Block FP8 tuned config for device=%s shape=(%d, %d) "
            "block=(%d, %d); using defaults.",
            device_name,
            N,
            K,
            block_n,
            block_k,
        )
        return None
    return result
