import logging

from flag_gems.ops.special_chebyshev_polynomial_u import (
    special_chebyshev_polynomial_u as _generic_special_chebyshev_polynomial_u,
)

logger = logging.getLogger("flag_gems." + __name__)


def special_chebyshev_polynomial_u(x, n):
    logger.debug("GEMS_METAX SPECIAL_CHEBYSHEV_POLYNOMIAL_U")
    return _generic_special_chebyshev_polynomial_u(x, n)
