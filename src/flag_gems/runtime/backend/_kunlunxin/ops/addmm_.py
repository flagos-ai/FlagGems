import logging

from flag_gems.utils import broadcastable_to

from .addmm import addmm_out

logger = logging.getLogger(__name__)


def addmm_(self, mat1, mat2, *, beta=1, alpha=1):
    assert self.dtype.is_floating_point, "Only floating-point dtypes are supported"
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        self.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"

    logger.debug("GEMS_KUNLUNXIN ADDMM_")
    # Write directly into self via the vendor-tuned out variant, avoiding
    # allocating a temporary and a separate copy_ (the previous generic path
    # ran addmm + copy_). The BLOCK_K_CHOICE/tile heuristics of the tuned
    # kernel apply: fp16 BK=256, bf16/fp32 BK=128, 128-tile for M,N<=512.
    return addmm_out(self, mat1, mat2, beta=beta, alpha=alpha, out=self)
