import logging

import torch

from flag_gems.runtime import device

logger = logging.getLogger("flag_gems." + __name__)
device_ = device


def _make_dep_token(
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    logger.debug("GEMS_METAX _MAKE_DEP_TOKEN")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    # Create a scalar tensor (0-dimensional) as dependency token
    out = torch.empty([], device=device, dtype=dtype)
    return out
