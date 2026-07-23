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

from flag_gems.utils import tl_extra_shim

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

try:
    _isfinited = tl_extra_shim.isfinited
    _finitef = tl_extra_shim.finitef
except Exception:
    pass

# One shape-independent kernel (kunlunAutoGrid + prefer_1d_tile + bounded tile
# via buffer_size_limit) mirroring the healthy acos/cos/tan recipe. The previous
# override split isfinite into three kernels (bitwise uint32/uint16 variants +
# a generic fallback); the uint32 bitwise path was pathologically slow on XPU
# (the full-tile `tl.full(bits.shape, 0x7F800000)` mask materialization made
# fp32 [1024, 256] take ~2.7ms vs ~0.08ms for the plain _finitef path), while the
# uint16 f16 variant gave no measurable win over the generic kernel. Collapsing
# to a single _finitef kernel removes that slow path and two extra compilations.
_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")], config=_config
)
@triton.jit
def isfinite_func(x):
    # float64 -> float32 cast would overflow large-but-finite values to inf, so
    # keep the dedicated fp64 intrinsic; every other float dtype is exact after
    # the upcast (isfinite only inspects the exponent).
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ISFINITE")
    if A.is_floating_point():
        return isfinite_func(A)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
