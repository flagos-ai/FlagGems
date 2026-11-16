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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

from .randn import (
    _native_cpu_randn,
    _philox_to_dsa_seed_scalars,
    _should_cpu_fallback,
    randn_kernel,
)

logger = logging.getLogger(__name__)


def randn_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_TSINGMICRO RANDN_LIKE (tle.dsa.tsingmicro.randn)")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    N = x.numel()
    if N == 0:
        return torch.empty(x.shape, device=device, dtype=dtype)

    philox_seed, philox_offset = philox_backend_seed_offset(max(N, 1))

    # float64: DSA randn is float32-only; keep CPU for correctness.
    if dtype is torch.float64:
        logger.debug("GEMS_TSINGMICRO RANDN_LIKE f64 → CPU")
        return _native_cpu_randn(x.shape, dtype, device)

    if _should_cpu_fallback(N, device):
        logger.debug("GEMS_TSINGMICRO RANDN_LIKE cpu fallback N=%s", N)
        return _native_cpu_randn(x.shape, dtype, device)

    # Prefer torch.empty over empty_like: empty_like is disproportionately
    # expensive on TXDA and dominates small-tensor latency.
    out = torch.empty(x.shape, device=device, dtype=dtype)
    seed0, seed1 = _philox_to_dsa_seed_scalars(philox_seed, philox_offset)

    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    with torch_device_fn.device(device):
        randn_kernel[grid_fn](out.view(-1), seed0, seed1, N)
    return out
