import logging

import torch

logger = logging.getLogger(__name__)


# median.dim: Computes median along a dimension using on-device
# sort + index selection. The sort is backed by Triton kernels.
def median_dim(self, dim=-1, keepdim=False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.ndim
    sorted_tensor, sorted_indices = torch.sort(self, dim=dim, stable=True)
    n = self.shape[dim]
    k = (n - 1) // 2
    values = sorted_tensor.select(dim, k)
    indices = sorted_indices.select(dim, k)
    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)
    return values, indices


def median(self):
    logger.debug("GEMS MEDIAN")
    return median_dim(self.flatten(), dim=0, keepdim=False)[0]
