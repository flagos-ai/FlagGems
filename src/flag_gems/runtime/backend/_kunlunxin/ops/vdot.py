import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def compute_vdot(
    inp_real, inp_imag, other_real, other_imag, inp_is_conj, other_is_conj
):
    # # Given inp storage: [inp_real, inp_imag], other: [other_real, other_imag]

    # # Case 1: inp_is_conj = False, other_is_conj = False
    # out_real = inp_real * other_real + inp_imag * other_imag
    # out_imag = inp_real * other_imag - inp_imag * other_real

    # # Case 2: inp_is_conj = True, other_is_conj = False
    # out_real = inp_real * other_real - inp_imag * other_imag
    # out_imag = inp_real * other_imag + inp_imag * other_real

    # # Case 3: inp_is_conj = False, other_is_conj = True
    # out_real = inp_real * other_real - inp_imag * other_imag
    # out_imag = -inp_real * other_imag - inp_imag * other_real

    # # Case 4: inp_is_conj = True, other_is_conj = True
    # out_real = inp_real * other_real + inp_imag * other_imag
    # out_imag = inp_real * other_imag - inp_imag * other_real
    if not inp_is_conj and not other_is_conj:  # Case 1
        out_real = tl.sum(inp_real * other_real + inp_imag * other_imag)
        out_imag = tl.sum(inp_real * other_imag - inp_imag * other_real)
    elif inp_is_conj and not other_is_conj:  # Case 2
        out_real = tl.sum(inp_real * other_real - inp_imag * other_imag)
        out_imag = tl.sum(inp_real * other_imag + inp_imag * other_real)
    elif not inp_is_conj and other_is_conj:  # Case 3
        out_real = tl.sum(inp_real * other_real - inp_imag * other_imag)
        out_imag = tl.sum(-inp_real * other_imag - inp_imag * other_real)
    else:  # Case 4
        out_real = tl.sum(inp_real * other_real + inp_imag * other_imag)
        out_imag = tl.sum(-inp_real * other_imag + inp_imag * other_real)

    return out_real, out_imag


# support old version triton which do not support tl.split
@libentry()
@triton.heuristics(runtime.get_heuristic_config("vdot"))
@triton.jit()
def vdot_kernel_complex(
    inp_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    inp_is_conj: tl.constexpr,
    other_is_conj: tl.constexpr,
    inp_stride: tl.constexpr,
    other_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    base_offset = 2 * pid * BLOCK_SIZE + 2 * tl.arange(0, BLOCK_SIZE) + tl.arange(0, 1)

    inp_real_offset = inp_stride * base_offset
    inp_imag_offset = inp_real_offset + 1

    other_real_offset = other_stride * base_offset
    other_imag_offset = other_real_offset + 1

    mask = base_offset < n_elements

    inp_real = tl.load(inp_ptr + inp_real_offset, mask=mask)
    inp_imag = tl.load(inp_ptr + inp_imag_offset, mask=mask)

    other_real = tl.load(other_ptr + other_real_offset, mask=mask)
    other_imag = tl.load(other_ptr + other_imag_offset, mask=mask)

    # Compute based on conjugate flags
    out_real, out_imag = compute_vdot(
        inp_real, inp_imag, other_real, other_imag, inp_is_conj, other_is_conj
    )

    tl.atomic_add(out_ptr, out_real)
    tl.atomic_add(out_ptr + 1, out_imag)


# only support real number
@libentry()
@triton.heuristics(runtime.get_heuristic_config("vdot"))
@triton.jit()
def dot_kernel(
    inp_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    inp_stride: tl.constexpr,
    other_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp = tl.load(inp_ptr + inp_stride * offset, mask=mask).to(tl.float32)
    other = tl.load(other_ptr + other_stride * offset, mask=mask).to(tl.float32)

    out = tl.sum(inp * other)
    tl.atomic_add(out_ptr, out)


def vdot(input: Tensor, other: Tensor):
    logger.debug("GEMS VDOT")

    assert (
        input.dtype == other.dtype
    ), f"Input tensors must have the same dtype. Got {input.dtype} and {other.dtype}."
    assert (
        input.ndim == 1 and other.ndim == 1
    ), f"Input tensors must be 1D. Got {input.ndim}D and {other.ndim}D."
    assert (
        input.size() == other.size()
    ), f"Input tensors must have the same size. Got {input.size()} and {other.size()}."

    inp = input
    inp_stride = inp.stride()[0]
    other_stride = other.stride()[0]

    if inp.is_complex():
        inp_is_conj = False
        other_is_conj = False

        if inp.is_conj():
            inp_is_conj = True
            inp = inp.conj()

        if other.is_conj():
            other_is_conj = True
            other = other.conj()

        inp_real = torch.view_as_real(inp)
        other_real = torch.view_as_real(other)

        n_elements = inp_real.numel()
        n_complex = inp.numel()

        output_real = torch.zeros(2, dtype=inp_real.dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(n_complex, meta["BLOCK_SIZE"]),)

        vdot_kernel_complex[grid](
            inp_real,
            other_real,
            output_real,
            n_elements=n_elements,
            inp_is_conj=inp_is_conj,
            other_is_conj=other_is_conj,
            inp_stride=inp_stride,
            other_stride=other_stride,
        )

        return torch.view_as_complex(output_real)
    else:
        output = torch.zeros([], dtype=torch.float32, device=inp.device)
        n_elements = inp.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        dot_kernel[grid](
            inp,
            other,
            output,
            n_elements=n_elements,
            inp_stride=inp_stride,
            other_stride=other_stride,
        )
        return output.to(inp.dtype)
