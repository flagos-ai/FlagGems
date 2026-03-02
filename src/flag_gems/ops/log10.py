import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)


def _resolve_log10(backend):
    try:
        # Attempt to use the backend's native log10 implementation.
        log10 = backend.log10
        return log10
    except AttributeError:
        # Fallback implementation if the backend does not provide a native log10.
        # log10(x) = log2(x) * (1 / log2(10))
        @triton.jit
        def fallback_log10(x):
            INV_LOG2_10: tl.constexpr = 0.3010299956639812  #
            return tl.log2(x) * INV_LOG2_10

        return fallback_log10


_log10_impl = _resolve_log10(tl_extra_shim)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def log10_kernel(x):
    y = _log10_impl(x.to(tl.float32))
    return y


def log10(A):
    logger.debug("GEMS LOG10")
    return log10_kernel(A)


def log10_(A):
    logger.debug("GEMS LOG10_")
    if not A.is_floating_point():
        raise RuntimeError(
            f"log_ only supports floating-point dtypes, but got {A.dtype}"
        )
    log10_kernel(A, out0=A)
    return A


def log10_out(A, out):
    logger.debug("GEMS LOG10_OUT")
    if not out.is_floating_point():
        raise RuntimeError("result type Float can't be cast to the desired output type")
    log10_kernel(A, out0=out)
    return out
