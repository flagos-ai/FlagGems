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
import triton.language as tl

from flag_gems.ops.silu import silu as _generic_silu
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_SILU_CONFIG = CodeGenConfig(
    max_tile_size=2048,
    max_grid_size=(65536, 65536, 65536),
    max_num_warps_per_cta=8,
    prefer_block_pointer=True,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
    balance_grid=True,
)

# MetaX backend modules are dynamically imported under ``_metax`` while the
# record handler listens to the ``flag_gems`` logger hierarchy.
logger = logging.getLogger("flag_gems.runtime.backend._metax.ops.silu")


@triton.jit
def _silu_forward_scalar(x):
    x_fp32 = x.to(tl.float32)
    return tl.fdiv(x_fp32, 1.0 + tl.exp(-x_fp32))


silu_forward = pointwise_dynamic(
    promotion_methods=[(0, "DEFAULT")], config=_SILU_CONFIG
)(_silu_forward_scalar)


def silu(self: torch.Tensor) -> torch.Tensor:
    if not self.is_contiguous() or self.dtype not in _SUPPORTED_DTYPES:
        return _generic_silu(self)
    logger.debug("GEMS_METAX SILU FORWARD")
    return silu_forward(self)
