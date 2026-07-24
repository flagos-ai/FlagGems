import logging

import torch

logger = logging.getLogger(__name__)


def imag(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS IMAG")

    if not input.is_complex():
        return torch.tensor(0, dtype=input.dtype, device=input.device).expand(
            input.shape
        )

    if input.is_contiguous():
        return torch.view_as_real(input)[..., 1]
    return input.imag
