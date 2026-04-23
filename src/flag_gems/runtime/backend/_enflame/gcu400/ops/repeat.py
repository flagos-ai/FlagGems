import logging

import torch

logger = logging.getLogger(__name__)


def _repeat_optimized(inp, sizes):
    ndim = max(inp.ndim, len(sizes))
    sizes_list = list(sizes)

    if inp.ndim < ndim:
        inp = inp.reshape([1] * (ndim - inp.ndim) + list(inp.shape))
    if len(sizes_list) < ndim:
        sizes_list = [1] * (ndim - len(sizes_list)) + sizes_list

    if all(s == 1 for s in sizes_list):
        return inp.clone()

    out_shape = [sizes_list[i] * inp.shape[i] for i in range(ndim)]
    for s in out_shape:
        if s == 0:
            return torch.empty(out_shape, device=inp.device, dtype=inp.dtype)

    if not inp.is_contiguous():
        inp = inp.contiguous()

    repeat_dims = sum(1 for s in sizes_list if s > 1)
    if repeat_dims == 1:
        dim_idx = next(i for i, s in enumerate(sizes_list) if s > 1)
        r = sizes_list[dim_idx]
        return torch.cat([inp] * r, dim=dim_idx)

    reshape_shape = []
    expand_shape = []
    for i in range(ndim):
        reshape_shape.extend([1, inp.shape[i]])
        expand_shape.extend([sizes_list[i], inp.shape[i]])

    return inp.reshape(reshape_shape).expand(expand_shape).contiguous().view(out_shape)


def repeat(inp: torch.Tensor, sizes) -> torch.Tensor:
    logger.debug("GEMS REPEAT GCU400")
    return _repeat_optimized(inp, sizes)
