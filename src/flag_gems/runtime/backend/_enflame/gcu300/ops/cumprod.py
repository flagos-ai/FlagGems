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

from flag_gems.ops.cumprod import cumprod as _generic_cumprod
from flag_gems.ops.cumprod import cumprod_ as _generic_cumprod_

logger = logging.getLogger(__name__)

# The generic scan accumulates in int32, so a narrow input is widened in-kernel.
# GCU300 cannot link that widening: the int8 -> int32 conversion the scan emits
# (`tops_cvt_v1_si8_to_v2_si32`) fails with
#
#   ld.lld: error: relocation R_DTU_ADDR16_LO_ICALL cannot be used against
#   symbol tops_cvt_v1_si8_to_v2_si32__0_of_2; recompile with -fPIC
#
# so 8/16-bit inputs are widened on the host instead. int32 and wider already
# go through a conversion that links, and are left alone.
_WIDEN_TO_INT32 = (torch.int8, torch.uint8, torch.int16)


def cumprod(inp, dim, *, dtype=None):
    logger.debug("GEMS_ENFLAME CUMPROD")
    # `bool` is only widened when the requested output is not boolean; the
    # boolean-output form redispatches to aten and must see the original dtype.
    if inp.dtype in _WIDEN_TO_INT32 or (
        inp.dtype is torch.bool and dtype != torch.bool
    ):
        inp = inp.to(torch.int32)
    return _generic_cumprod(inp, dim, dtype=dtype)


def cumprod_(inp, dim, *, dtype=None):
    logger.debug("GEMS_ENFLAME CUMPROD_")
    mismatched = dtype is not None and dtype != inp.dtype
    if inp.dtype not in _WIDEN_TO_INT32 or mismatched:
        # Boolean input has to keep raising, and a dtype mismatch has to keep
        # reporting its "Bad in-place call" error, both from the generic path.
        return _generic_cumprod_(inp, dim, dtype=dtype)
    out = _generic_cumprod(inp.to(torch.int32), dim, dtype=torch.int32)
    inp.copy_(out.to(inp.dtype))
    return inp
