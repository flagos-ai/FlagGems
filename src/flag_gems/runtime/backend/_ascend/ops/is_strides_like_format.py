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
    检查张量 x 的 stride 是否与指定内存格式对齐

    Args:
        x (torch.Tensor): 输入张量
        format (str): 目标内存格式，支持：
                     - "contiguous": 连续内存
                     - "channels_last": 通道后置（仅 4D）
                     - "any": 任意格式（直接通过）

    Returns:
        torch.Tensor: 标量布尔张量
    """
    logger.debug(f"GEMS_ASCEND IS_STRIDES_LIKE_FORMAT: format={format}")

    if format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported stride format '{format}'. "
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    # case 1：any，dirctely pass
    if format == "any":
        return torch.tensor(True, dtype=torch.bool, device=x.device)

    # case 2: contiguous check
    if format == "contiguous":
        result = x.is_contiguous()

    # case 3：Channels Last check（only  4D vialied）
    elif format == "channels_last":
        if x.dim() != 4:
            result = False
        else:
            result = x.is_contiguous(memory_format=torch.channels_last)

    # return PyTorch biaoliang bool tensor
    return torch.tensor(result, dtype=torch.bool, device=x.device)