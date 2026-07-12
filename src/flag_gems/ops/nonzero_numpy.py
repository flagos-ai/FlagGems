import logging

from flag_gems.ops.nonzero import nonzero

logger = logging.getLogger(__name__)


def nonzero_numpy(inp):
    """
    Returns a tuple of 1D tensors, one for each dimension of the input,
    containing the indices of the non-zero elements in that dimension.

    This is equivalent to torch.nonzero(...).T or numpy.nonzero() behavior.
    """
    logger.debug("GEMS NONZERO_NUMPY")

    # Use the existing nonzero implementation which returns shape [N, ndim]
    out = nonzero(inp, as_tuple=False)

    # Unbind along dim=1 to get ndim tensors of shape [N]
    # Convert to list since aten::nonzero_numpy returns Tensor[]
    return list(out.unbind(dim=1))
