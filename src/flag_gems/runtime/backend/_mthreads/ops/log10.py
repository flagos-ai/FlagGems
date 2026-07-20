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
from typing import Tuple

import torch

from flag_gems.ops.log10 import log10 as default_log10
from flag_gems.ops.log10 import log10_ as default_log10_
from flag_gems.ops.log10 import log10_out as default_log10_out
from flag_gems.runtime.backend._mthreads.ops.log import _launch_log, _use_triton_kernel

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_INV_LN10 = 0.4342944819032518


def _should_use_triton(x: torch.Tensor) -> Tuple[bool, int]:
    return _use_triton_kernel(x)


def log10(x):
    logger.debug("GEMS_MTHREADS LOG10")
    use_triton, dtype_size = _should_use_triton(x)
    if not use_triton:
        return default_log10(x)

    out = torch.empty_like(x)
    return _launch_log(x, out, dtype_size, scale=_INV_LN10)


def log10_(x):
    logger.debug("GEMS_MTHREADS LOG10_")
    use_triton, dtype_size = _should_use_triton(x)
    if not use_triton:
        return default_log10_(x)

    return _launch_log(x, x, dtype_size, scale=_INV_LN10)


def log10_out(x, out):
    logger.debug("GEMS_MTHREADS LOG10_OUT")
    use_triton, dtype_size = _should_use_triton(x)
    if not use_triton:
        return default_log10_out(x, out)

    return _launch_log(x, out, dtype_size, scale=_INV_LN10)
