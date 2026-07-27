import logging

from flag_gems.utils import broadcastable_to

from .addmm import addmm

logger = logging.getLogger(__name__)


def addmm_(self, mat1, mat2, *, beta=1, alpha=1):
    assert self.dtype.is_floating_point, "Only floating-point dtypes are supported"
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        self.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"

    logger.debug("GEMS_KUNLUNXIN ADDMM_")
    # Write directly into self to avoid allocating and copying a temporary.
    return addmm(self, mat1, mat2, beta=beta, alpha=alpha, out=self)
