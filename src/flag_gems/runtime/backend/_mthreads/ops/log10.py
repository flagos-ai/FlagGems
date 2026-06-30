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
