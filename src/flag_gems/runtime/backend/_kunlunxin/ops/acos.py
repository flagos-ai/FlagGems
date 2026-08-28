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

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


_acos = tl_extra_shim.acos
_atan2 = tl_extra_shim.atan2

# Without an explicit CodeGenConfig, pointwise_dynamic specializes the kernel
# per input shape on XPU -> per-shape recompile -> IR explosion
# (acos_kernel_kernel recompiled ~1269x across ~1053 modules, 1.6GB IR dump).
# kunlunAutoGrid=True + prefer_1d_tile + bounded tile makes the kernel
# shape-independent so it compiles ONCE. Mirrors cos/tan/abs/sgn_.
config_ = CodeGenConfig(
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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit()
def acos_kernel(x):
    x_f32 = x.to(tl.float32)
    in_domain = tl.abs(x_f32) <= 1.0

    # The P800 acos intrinsic has a repeatable error of about 3e-3 on
    # in-domain fp32 values.  atan2(sqrt(1 - x^2), x) avoids that intrinsic;
    # clamp the radicand because fp32 roundoff can make it slightly negative
    # at the endpoints.  Keep the intrinsic for out-of-domain and NaN inputs,
    # where the identity would otherwise return a finite 0 or pi.
    radicand = tl.maximum(1.0 - x_f32 * x_f32, 0.0)
    stable = _atan2(tl.sqrt(radicand), x_f32)
    return tl.where(in_domain, stable, _acos(x_f32))


def acos(x):
    logger.debug("GEMS_KUNLUNXIN ACOS")
    y = acos_kernel(x)
    return y
