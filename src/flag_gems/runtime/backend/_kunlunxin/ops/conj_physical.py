import torch

from ._conj import _conj


def conj_physical(input: torch.Tensor) -> torch.Tensor:
    """Materialize the complex conjugate with the vendor stride-1 kernel."""
    if not input.is_complex():
        return input
    return _conj(input)
