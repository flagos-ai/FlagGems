import logging

from flag_gems.runtime.backend._kunlunxin.ops.addmm import addmm as _vendor_addmm

logger = logging.getLogger(__name__)


def matmuladd(input, other, bias):
    """Matrix multiplication with addition: output = matmul(input, other) + bias.

    Delegates to the vendor addmm kernel (alpha=1.0, beta=1.0).
    """
    logger.debug("GEMS_KUNLUNXIN MATMULADD")
    return _vendor_addmm(bias, input, other)
