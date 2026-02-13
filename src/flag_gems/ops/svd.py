import logging

import torch

logger = logging.getLogger(__name__)


def _maybe_upcast(input_tensor: torch.Tensor):
    if input_tensor.dtype in (torch.float16, torch.bfloat16):
        return input_tensor.float(), True
    return input_tensor, False


def _cast_back(tensor: torch.Tensor, dtype: torch.dtype):
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


def _svd_with_uv(input_tensor: torch.Tensor, some: bool):
    full_matrices = not bool(some)
    compute_tensor, need_cast_back = _maybe_upcast(input_tensor)
    u, s, vh = torch.linalg.svd(compute_tensor, full_matrices=full_matrices)
    v = vh.transpose(-2, -1).conj()
    if need_cast_back:
        u = _cast_back(u, input_tensor.dtype)
        s = _cast_back(s, input_tensor.dtype)
        v = _cast_back(v, input_tensor.dtype)
    return u, s, v


def _svd_without_uv(input_tensor: torch.Tensor):
    compute_tensor, need_cast_back = _maybe_upcast(input_tensor)
    s = torch.linalg.svdvals(compute_tensor)
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
    if need_cast_back:
        s = _cast_back(s, input_tensor.dtype)
    return u, s, v


def svd(self: torch.Tensor, some: bool = True, compute_uv: bool = True):
    logger.debug("GEMS SVD")
    if compute_uv:
        return _svd_with_uv(self, some)
    return _svd_without_uv(self)
