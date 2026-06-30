import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def _assert_async_kernel(x_ptr, scratch_ptr, MSG: tl.constexpr):
    val = tl.load(x_ptr)
    cond = val != 0
    tl.device_assert(cond, MSG)
    tl.store(scratch_ptr, cond)


def _assert_async(tensor: torch.Tensor, msg: str = "Assertion failed"):
    logger.debug("GEMS_CAMBRICON ASSERT_ASYNC")
    if tensor.numel() != 1:
        raise RuntimeError(
            f"Boolean value of Tensor with shape {list(tensor.shape)} is ambiguous"
        )
    scratch = torch.empty((), dtype=torch.bool, device=tensor.device)
    with torch_device_fn.device(tensor.device):
        _assert_async_kernel[(1,)](tensor, scratch, MSG=msg)
