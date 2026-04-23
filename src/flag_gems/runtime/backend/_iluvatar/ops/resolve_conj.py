import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems." + __name__)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def _resolve_conj_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused resolve_conj: single-pass int32 XOR to negate imag parts.

    Complex64 is stored as interleaved [real0, imag0, real1, imag1, ...] in float32.
    Viewing as int32, odd-indexed elements are imaginary parts.
    XOR with 0x80000000 on odd indices flips the sign bit (negates) only imag parts.
    This is a single-kernel approach vs PyTorch's multi-kernel resolve_conj.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(input_ptr + offsets, mask=mask)
    sign_flip = tl.cast((offsets & 1), tl.int32) << 31
    vals = vals ^ sign_flip
    tl.store(output_ptr + offsets, vals, mask=mask)


def resolve_conj(A: torch.Tensor):
    logger.debug("ILUVATAR GEMS RESOLVE_CONJ")

    if not A.is_conj():
        return A

    # Remove the conj bit to get the physical tensor
    phys = A.conj()
    if not phys.is_contiguous():
        phys = phys.contiguous()

    numel = phys.numel()
    n_ints = numel * 2

    input_i32 = phys.view(torch.int32)
    output_i32 = torch.empty_like(input_i32)

    grid = lambda meta: ((n_ints + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _resolve_conj_kernel[grid](
        input_i32,
        output_i32,
        n_ints,
    )

    return output_i32.view(torch.complex64).view_as(phys)
