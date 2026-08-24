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

logger = logging.getLogger(__name__)


def _weight_norm(v: torch.Tensor, g: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """ATen entry point for the cambricon fused weight-normalization path.

    The generic ``flag_gems.ops._weight_norm`` hard-codes an import of the
    generic ``flag_gems.fused.weight_norm`` module, which in turn imports the
    generic ``weight_norm_interface`` kernels -- bypassing the cambricon
    specialization entirely. Exporting this function as ``_weight_norm`` from
    the cambricon ``ops`` package lets ``SpecOpRegistrar`` override the aten
    entry so that it dispatches to the cambricon fused/kernel implementation.
    """
    logger.debug("GEMS_CAMBRICON _WEIGHT_NORM")
    if v.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"_weight_norm does not support {v.dtype}")
    # Deferred import: the fused package is loaded after ops, so importing it at
    # call time avoids an ops <-> fused circular import at module load.
    from ..fused.weight_norm import weight_norm

    return weight_norm(v, g, dim)
