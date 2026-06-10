import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# Kunlunxin XPU has 12 compute clusters; distribute work evenly across them.
CLUSTER_NUM = 12


@libentry()
@triton.jit
def zero_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Write-only kernel: no dummy load, stores 0 directly with dtype handled by Triton."""
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, 0.0, mask=mask)


def _launch_zero_kernel(tensor: torch.Tensor) -> torch.Tensor:
    n_elements = tensor.numel()
    if n_elements == 0:
        return tensor
    # BLOCK_SIZE: distribute n_elements evenly across CLUSTER_NUM clusters,
    # rounded up to the next power of 2 for aligned vectorised stores.
    block_size = triton.next_power_of_2(triton.cdiv(n_elements, CLUSTER_NUM))
    grid = (CLUSTER_NUM, 1, 1)
    with torch_device_fn.device(tensor.device):
        zero_kernel[grid](
            tensor,
            n_elements,
            BLOCK_SIZE=block_size,
            buffer_size_limit=2048,
            isCloseDtypeConvert=True,
        )
    return tensor


def zero(self: torch.Tensor) -> torch.Tensor:
    """aten::zero(Tensor self) -> Tensor  — in-place zero-fill, returns self."""
    logger.debug("GEMS_KUNLUNXIN ZERO")
    return _launch_zero_kernel(self)


def zero_out(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """aten::zero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)  — writes zeros to out."""
    logger.debug("GEMS_KUNLUNXIN ZERO_OUT")
    return _launch_zero_kernel(out)
