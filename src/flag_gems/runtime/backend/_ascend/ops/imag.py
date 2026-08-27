import torch


def imag(input: torch.Tensor) -> torch.Tensor:
    """
    Ascend backend implementation for imag.
    Returns the imaginary part of a complex tensor as a view sharing storage.
    For real tensors, returns a zero tensor.
    """
    if not input.is_complex():
        return torch.zeros_like(input)
    real_view = torch.view_as_real(input)
    return real_view[..., 1]
