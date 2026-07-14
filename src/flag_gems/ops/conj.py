import logging
import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("conj_physical"),
    key=["n_elements"],
)
@triton.jit
def conj_complex64_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # complex64 每个元素由两个连续 float32 组成: [real, imag]
    float_offsets = offsets * 2

    real = tl.load(x_ptr + float_offsets, mask=mask)
    imag = tl.load(x_ptr + float_offsets + 1, mask=mask)

    tl.store(out_ptr + float_offsets, real, mask=mask)
    tl.store(out_ptr + float_offsets + 1, -imag, mask=mask)


def conj(input: torch.Tensor) -> torch.Tensor:
    """
    FlagGems Triton implementation of torch.conj.
    Performs physical conjugate computation via custom GPU kernel.
    """
    logger.debug("GEMS CONJ")
    if not input.is_complex():
        return input

    input_contig = input.contiguous()
    n_elements = input_contig.numel()
    output = torch.empty_like(input_contig)

    x_real = torch.view_as_real(input_contig)
    out_real = torch.view_as_real(output)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    conj_complex64_kernel[grid](
        x_ptr=x_real,
        out_ptr=out_real,
        n_elements=n_elements,
    )

    return output
