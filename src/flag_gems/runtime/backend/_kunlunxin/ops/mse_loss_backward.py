import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _mse_loss_backward_kernel(
    grad_output,
    self,
    target,
    output,
    n_elements,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    self_value = tl.load(self + offsets, mask=mask, other=0.0).to(tl.float32)
    target_value = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_value = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = 2.0 / n_elements if REDUCTION == 1 else 2.0
    result = grad_value * scale * (self_value - target_value)
    tl.store(output + offsets, result, mask=mask)


def mse_loss_backward(grad_output, self, target, reduction=1):
    logger.debug("GEMS_KUNLUNXIN MSE_LOSS_BACKWARD")
    self_contiguous = self.contiguous()
    target_contiguous = target.contiguous()
    grad_contiguous = grad_output.contiguous()
    output = torch.empty_like(self_contiguous)
    n_elements = self_contiguous.numel()
    if n_elements == 0:
        return output

    block_size = min(8192, triton.next_power_of_2(n_elements))
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(self.device):
        _mse_loss_backward_kernel[grid](
            grad_contiguous,
            self_contiguous,
            target_contiguous,
            output,
            n_elements,
            REDUCTION=reduction,
            BLOCK_SIZE=block_size,
        )
    return output
