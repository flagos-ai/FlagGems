import logging

import torch

from flag_gems.runtime import device

device_ = device

logger = logging.getLogger(
    f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}'
)


def unsqueeze_copy(inp, dim):
    logger.debug("GEMS_ASCEND UNSQUEEZE_COPY")

    result = torch.unsqueeze(inp, dim)
    out = torch.empty_like(result)
    out.copy_(result)
    return out


def unsqueeze_copy_out(inp, dim, *, out):
    logger.debug("GEMS_ASCEND UNSQUEEZE_COPY_OUT")

    result = torch.unsqueeze(inp, dim)
    out.resize_(result.shape)
    out.copy_(result)

    return out
