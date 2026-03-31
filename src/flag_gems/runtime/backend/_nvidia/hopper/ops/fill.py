import logging

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

fill_autotune = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["n_elements"],
)


@fill_autotune
@triton.jit
def fill_scalar_kernel(
    inp_ptr,
    out_ptr,
    value_scalar,
    n_elements,
    stride_inp,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    out_offsets = offsets * stride_out

    fill_val = tl.full([BLOCK_SIZE], value_scalar, dtype=tl.float32)

    tl.store(out_ptr + out_offsets, fill_val, mask=mask)


def fill_scalar_(self, value):
    logger.debug("GEMS FILL_SCALAR_")
    n_elements = self.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    with torch_device_fn.device(self.device):
        fill_scalar_kernel[grid](
            self,
            self,
            value,
            n_elements,
            self.stride(0) if self.ndim > 0 else 1,
            self.stride(0) if self.ndim > 0 else 1,
        )
    return self
