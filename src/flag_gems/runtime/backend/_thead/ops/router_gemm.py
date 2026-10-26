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

"""BF16 router GEMM with FP32 output on T-Head PPU."""

import logging

import torch

from flag_gems.ops.mm import router_gemm as _generic_router_gemm

from .mm import _dispatch_ppu_gemm

logger = logging.getLogger(__name__)


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_THEAD ROUTER_GEMM")
    if not (
        x.ndim == weight.ndim == 2
        and x.shape[1] == weight.shape[1]
        and x.dtype == weight.dtype == torch.bfloat16
    ):
        return _generic_router_gemm(x, weight)

    m, _ = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    # The MM kernels accept the dense [N, K] storage directly as an NT view.
    routed = _dispatch_ppu_gemm(x, weight.t(), out, output_dtype=torch.float32)
    if routed is not None:
        return routed
    return _generic_router_gemm(x, weight)


__all__ = ["router_gemm"]
