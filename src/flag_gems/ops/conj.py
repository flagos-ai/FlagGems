import logging
import torch

logger = logging.getLogger(__name__)


def conj(A):
    """Return the conjugate of a complex tensor with physical memory copy.

    This implementation uses PyTorch's built-in conjugate operation,
    then clones the result to ensure physical memory copy.

    Args:
        A (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: A new tensor containing the conjugate of A.
    """
    logger.debug("GEMS CONJ")

    if isinstance(A, torch.Tensor):
        # 对于复数张量，使用 torch.conj 然后克隆
        # 对于实数张量，torch.conj 返回原张量本身，也需要克隆
        result = torch.conj(A)
        # 确保物理内存复制
        return result.clone()
    else:
        # 如果是标量，直接返回
        return torch.tensor(A)
