import logging

from .nonzero import nonzero

logger = logging.getLogger(__name__)


def nonzero_numpy(inp):
    """
    Returns a tuple of 1D tensors, one for each dimension of the input,
    containing the indices of the non-zero elements in that dimension.

    This is equivalent to torch.nonzero(...).T or numpy.nonzero() behavior.
    """
    logger.debug("GEMS_ENFLAME GCU300 NONZERO_NUMPY")

    out = nonzero(inp, as_tuple=False)

    return list(out.unbind(dim=1))
