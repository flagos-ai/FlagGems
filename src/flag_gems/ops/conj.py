import logging
import torch

logger = logging.getLogger(__name__)

def conj(input: torch.Tensor) -> torch.Tensor:
    """
    Returns a view of the input tensor with the conjugate flag set.

    For real (non-complex) tensors, this is a no-op and returns the input itself.
    For complex tensors, this returns a view that shares the same underlying storage
    but has is_conj() == True. When the view is later materialized (e.g. by
    resolve_conj), the imaginary part is negated lazily.

    This matches the semantics of torch.conj exactly: zero memory copy,
    output and input share the same data pointer.
    """
    logger.debug("GEMS CONJ")
    if not input.is_complex():
        return input
    # _conj() is a low-level TensorImpl method that toggles the conjugate flag
    return input._conj()
