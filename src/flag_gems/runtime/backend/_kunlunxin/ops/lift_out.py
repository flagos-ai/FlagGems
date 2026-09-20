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
import triton  # noqa: F401
import triton.language as tl  # noqa: F401
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tle_copy import tle_copy

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def lift_out_func(x):
    return x


def lift_out(A, *, out=None):
    """Implements aten::lift.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!).

    Copies ``A`` into ``out`` and returns ``out``. A lift is a pure move, so it
    rides the proven copy-family recipe (same as ``alias_copy`` / ``copy_``):
    tle takes the whole transfer when it can (a TMA tile for contiguous
    same-dtype copies, an SDNN 2D row transfer for strided or broadcast layouts,
    with the dtype cast folded in), and the pointwise copy kernel keeps
    everything tle cannot express. The old ``torch.ops.aten._copy_from`` call is
    gone: it bypassed gems and dispatched straight to the vendor engine.
    """
    logger.debug("GEMS_KUNLUNXIN LIFT_OUT")
    if out is None:
        out = torch.empty_like(A, memory_format=torch.contiguous_format)
    if out.numel() == 0:
        return out
    if tle_copy(A, out):
        return out
    lift_out_func(A, out0=out)
    return out
