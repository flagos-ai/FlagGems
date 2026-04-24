import logging

import torch
import triton
from torch import Tensor

from flag_gems.utils import pointwise_dynamic

from .dot import dot

# Pointwise function for complex multiplication (vdot case: first argument needs conjugation)


@pointwise_dynamic(
    is_tensor=[True, True, True, True], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def complex_mult_conj_real_part(a_real, a_imag, b_real, b_imag):
    return a_real * b_real + a_imag * b_imag


@pointwise_dynamic(
    is_tensor=[True, True, True, True], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def complex_mult_conj_imag_part(a_real, a_imag, b_real, b_imag):
    return a_real * b_imag - a_imag * b_real


def vdot(input: Tensor, other: Tensor):
    """
    Compute the dot product of two vectors, conjugating the first vector for complex inputs.

    vdot(a, b) = sum(conj(a) * b)

    For complex: Split into real and imaginary parts, compute multiplication separately, then sum.
    For real: Directly call dot.
    """
    logging.debug("GEMS VDOT")

    assert (
        input.dtype == other.dtype
    ), f"Input tensors must have the same dtype. Got {input.dtype} and {other.dtype}."
    assert (
        input.ndim == 1 and other.ndim == 1
    ), f"Input tensors must be 1D. Got {input.ndim}D and {other.ndim}D."
    assert (
        input.size() == other.size()
    ), f"Input tensors must have the same size. Got {input.size()} and {other.size()}."

    if input.is_complex():
        # Handle complex case
        # vdot definition: vdot(a, b) = sum(conj(a[i]) * b[i])

        # Handle conjugation flag
        inp = input
        if inp.is_conj():
            # Remove conjugation flag (vdot conjugates again, so they cancel out)
            inp = inp.conj()  # Remove conjugation flag

        oth = other
        if oth.is_conj():
            oth = oth.conj()  # Remove conjugation flag

        # Split complex numbers into real and imaginary parts on CPU (avoid view_as_real issues on TPU)
        device = inp.device
        inp_cpu = inp.cpu()
        oth_cpu = oth.cpu()

        # Split real and imaginary parts
        inp_real_imag = torch.view_as_real(inp_cpu)  # shape: [N, 2]
        oth_real_imag = torch.view_as_real(oth_cpu)  # shape: [N, 2]

        inp_real = inp_real_imag[..., 0].to(device)  # shape: [N]
        inp_imag = inp_real_imag[..., 1].to(device)  # shape: [N]
        oth_real = oth_real_imag[..., 0].to(device)  # shape: [N]
        oth_imag = oth_real_imag[..., 1].to(device)  # shape: [N]

        # Use pointwise_dynamic to compute element-wise multiplication
        # Real and imaginary parts of conj(a) * b
        prod_real = complex_mult_conj_real_part(inp_real, inp_imag, oth_real, oth_imag)
        prod_imag = complex_mult_conj_imag_part(inp_real, inp_imag, oth_real, oth_imag)

        # Sum real and imaginary parts separately (using dot operator)
        ones = torch.ones_like(prod_real)
        sum_real = dot(prod_real, ones)
        sum_imag = dot(prod_imag, ones)

        # Combine into complex result
        result_real_imag = torch.stack([sum_real, sum_imag], dim=-1).cpu()
        result = (
            torch.view_as_complex(result_real_imag.unsqueeze(0)).squeeze().to(device)
        )

        return result
    else:
        # Real case, vdot is equivalent to dot
        return dot(input, other)
