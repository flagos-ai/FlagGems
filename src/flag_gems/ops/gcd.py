import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic()
@triton.jit
def gcd_func(a, b):
    # Implementación Euclidiana en int64 para seguridad
    x = a.to(tl.int64)
    y = b.to(tl.int64)

    x = tl.abs(x)
    y = tl.abs(y)

    # Euclid: while y != 0: x, y = y, x % y
    # Triton permite loops con rango constexpr; lo mantenemos acotado.
    # Para enteros 64-bit, 64 iteraciones es un bound seguro.
    for _ in range(64):
        y_is_zero = y == 0
        r = x % y
        x = tl.where(y_is_zero, x, y)
        y = tl.where(y_is_zero, 0, r)

    return x.to(a.dtype)


def gcd(A, B):
    logger.debug("GEMS GCD")
    return gcd_func(A, B)
