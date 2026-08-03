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
    # Route to the kunlunxin heuristic addmm (single-config, @libentry cached)
    # instead of the generic libtuner addmm. The generic path re-tunes across
    # many configs per shape -> massive IR / compile blowup (see IR dump), and
    # the imported `addmm` name in the generic ops/addmm_.py binds to the generic
    # (un-specialized) kernel, bypassing this backend's fast kernel entirely.
    result = addmm(self, mat1, mat2, beta=beta, alpha=alpha)
    return self.copy_(result)
