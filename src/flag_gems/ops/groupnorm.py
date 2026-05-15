import logging

import torch

from flag_gems.utils import libentry  # noqa: F401

logger = logging.getLogger(__name__)


def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS GROUP_NORM")
    if x.numel() == 0:
        return x.clone()
    return torch.nn.functional.group_norm(x, num_groups, weight, bias, eps)


def group_norm_backward(dy, x, mean, rstd, weight, num_groups):
    logger.debug("GEMS GROUP_NORM BACKWARD")
    x_req = x.detach().requires_grad_(True)
    w_req = weight.detach().requires_grad_(True) if weight is not None else None
    with torch.enable_grad():
        out = torch.nn.functional.group_norm(x_req, num_groups, w_req)
    out.backward(dy)
    return x_req.grad, w_req.grad if w_req is not None else None, None
