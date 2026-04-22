import logging

import torch


def view_as_complex(input: torch.Tensor) -> torch.Tensor:
    """
    Convert a real tensor with last dimension 2 to a complex tensor.

    Args:
        input: Input tensor with shape (..., 2) and dtype float32 or float64

    Returns:
        Complex tensor with shape (...) and dtype complex64 or complex128
    """
    logging.debug("GEMS_ILUVATAR VIEW_AS_COMPLEX")

    # Use PyTorch's native implementation which is highly optimized
    return torch.view_as_complex(input.contiguous())
