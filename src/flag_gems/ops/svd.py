import logging

import torch

logger = logging.getLogger(__name__)


def _svd_with_uv(input_tensor: torch.Tensor, some: bool):
    full_matrices = not bool(some)
    u, s, vh = torch.linalg.svd(input_tensor, full_matrices=full_matrices)
    v = vh.transpose(-2, -1).conj()
    return u, s, v


def _svd_without_uv(input_tensor: torch.Tensor):
    s = torch.linalg.svdvals(input_tensor)
    batch_shape = input_tensor.shape[:-2]
    m = input_tensor.size(-2)
    n = input_tensor.size(-1)
    u = torch.zeros(
        (*batch_shape, m, m),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    v = torch.zeros(
        (*batch_shape, n, n),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    return u, s, v


def svd(self: torch.Tensor, some: bool = True, compute_uv: bool = True):
    logger.debug("GEMS SVD")
    if compute_uv:
        return _svd_with_uv(self, some)
    return _svd_without_uv(self)
