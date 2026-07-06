import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def conj_physical_kernel_vec4(
    input_ptr,
    output_ptr,
    num_complex,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 4
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 4
    mask = offsets < num_complex

    in_offsets = offsets * 2
    r0 = tl.load(input_ptr + in_offsets, mask=mask)
    i0 = tl.load(input_ptr + in_offsets + 1, mask=mask)
    r1 = tl.load(input_ptr + in_offsets + 2, mask=mask)
    i1 = tl.load(input_ptr + in_offsets + 3, mask=mask)
    r2 = tl.load(input_ptr + in_offsets + 4, mask=mask)
    i2 = tl.load(input_ptr + in_offsets + 5, mask=mask)
    r3 = tl.load(input_ptr + in_offsets + 6, mask=mask)
    i3 = tl.load(input_ptr + in_offsets + 7, mask=mask)

    out_offsets = in_offsets
    tl.store(output_ptr + out_offsets, r0, mask=mask)
    tl.store(output_ptr + out_offsets + 1, -i0, mask=mask)
    tl.store(output_ptr + out_offsets + 2, r1, mask=mask)
    tl.store(output_ptr + out_offsets + 3, -i1, mask=mask)
    tl.store(output_ptr + out_offsets + 4, r2, mask=mask)
    tl.store(output_ptr + out_offsets + 5, -i2, mask=mask)
    tl.store(output_ptr + out_offsets + 6, r3, mask=mask)
    tl.store(output_ptr + out_offsets + 7, -i3, mask=mask)


@triton.jit
def conj_physical_kernel_vec2(
    input_ptr,
    output_ptr,
    num_complex,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 2
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 2
    mask = offsets < num_complex

    in_offsets = offsets * 2
    r0 = tl.load(input_ptr + in_offsets, mask=mask)
    i0 = tl.load(input_ptr + in_offsets + 1, mask=mask)
    r1 = tl.load(input_ptr + in_offsets + 2, mask=mask)
    i1 = tl.load(input_ptr + in_offsets + 3, mask=mask)

    out_offsets = in_offsets
    tl.store(output_ptr + out_offsets, r0, mask=mask)
    tl.store(output_ptr + out_offsets + 1, -i0, mask=mask)
    tl.store(output_ptr + out_offsets + 2, r1, mask=mask)
    tl.store(output_ptr + out_offsets + 3, -i1, mask=mask)


@triton.jit
def conj_physical_kernel_vec1(
    input_ptr,
    output_ptr,
    num_complex,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_complex

    in_offsets = offsets * 2
    r = tl.load(input_ptr + in_offsets, mask=mask)
    i = tl.load(input_ptr + in_offsets + 1, mask=mask)

    out_offsets = in_offsets
    tl.store(output_ptr + out_offsets, r, mask=mask)
    tl.store(output_ptr + out_offsets + 1, -i, mask=mask)


def conj_physical(A):
    logger.debug("GEMS CONJ_PHYSICAL")
    if not isinstance(A, torch.Tensor):
        return torch.tensor(A)
    if not A.is_complex():
        return A.clone()
    if A.device.type == "cpu":
        return torch.conj(A).clone()

    A = A.contiguous()
    A_flat = torch.view_as_real(A).contiguous()
    output_flat = torch.empty_like(A_flat)

    num_complex = A.numel()
    BLOCK_SIZE = 256

    if num_complex < 1000:
        grid = (triton.cdiv(num_complex, BLOCK_SIZE * 4),)
        conj_physical_kernel_vec4[grid](A_flat, output_flat, num_complex, BLOCK_SIZE)
    elif num_complex < 10000:
        grid = (triton.cdiv(num_complex, BLOCK_SIZE * 2),)
        conj_physical_kernel_vec2[grid](A_flat, output_flat, num_complex, BLOCK_SIZE)
    else:
        grid = (triton.cdiv(num_complex, BLOCK_SIZE),)
        conj_physical_kernel_vec1[grid](A_flat, output_flat, num_complex, BLOCK_SIZE)

    return torch.view_as_complex(output_flat)


def conj(A):
    logger.debug("GEMS CONJ")
    if not isinstance(A, torch.Tensor):
        return torch.tensor(A)
    if not A.is_complex():
        return A.clone()
    return torch.conj(A)
