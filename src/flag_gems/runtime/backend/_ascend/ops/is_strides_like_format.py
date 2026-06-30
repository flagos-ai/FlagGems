import logging
import torch
from typing import Union

from flag_gems.runtime import device

device_ = device
logger = logging.getLogger(
    f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}'
)

SUPPORTED_FORMATS = {"contiguous", "channels_last", "any"}

def is_strides_like_format(
    x: torch.Tensor,
    format: str
) -> Union[bool, torch.Tensor]:
    """
    Checks whether the stride pattern of the input tensor matches the specified memory format.

    Args:
        x: Input tensor.
        format: Target memory format. Supported values:
                "contiguous" - standard contiguous memory layout.
                "channels_last" - channels-last (NHWC) format (only valid for 4D tensors).
                "any" - always returns True.

    Returns:
        A scalar boolean tensor indicating whether the stride matches the target format.
    """
    logger.debug(f"GEMS_ASCEND IS_STRIDES_LIKE_FORMAT: format={format}")

    if format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported stride format '{format}'. "
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    if format == "any":
        return torch.tensor(True, dtype=torch.bool, device=x.device)

    if format == "contiguous":
        result = x.is_contiguous()
    else:  # format == "channels_last"
        if x.dim() != 4:
            result = False
        else:
            result = x.is_contiguous(memory_format=torch.channels_last)

    return torch.tensor(result, dtype=torch.bool, device=x.device)
