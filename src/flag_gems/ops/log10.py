import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def log10_func(x):
    return tl.log2(x.to(tl.float32)) * 0.3010299956639812

def log10(self):
    logger.debug("GEMS LOG10")
    return log10_func(self)

def log10_(self):
    logger.debug("GEMS LOG10_")
    log10_func(self, out0=self)
    return self

def log10_out(self, out):
    logger.debug("GEMS LOG10_OUT")
    log10_func(self, out0=out)
    return out