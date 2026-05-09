import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime

log = logging.getLogger(__name__)


@triton.autotune(
    configs=runtime.get_tuned_config("smooth_l1_loss_backward"),
    key=["n_elements"],
)
@triton.jit
def _smooth_l1_loss_backward_kernel(
    grad_output_ptr,
    input_ptr,
    target_ptr,
    grad_input_ptr,
    n_elements,
    inv_n,
    beta,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    go = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inp = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    diff = inp - target
    abs_diff = tl.abs(diff)

    grad = tl.where(abs_diff < beta, diff / beta, tl.where(diff > 0.0, 1.0, -1.0))
    grad = go * grad

    if REDUCTION == 1:
        grad = grad * inv_n

    tl.store(grad_input_ptr + offsets, grad, mask=mask)


def smooth_l1_loss_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    target: torch.Tensor,
    reduction: int,
    beta: float,
):
    log.debug("GEMS SMOOTH_L1_LOSS BACKWARD")

    device = self.device
    if not (isinstance(device, torch.device) and device.type == "cuda"):
        return torch.ops.aten.smooth_l1_loss_backward(
            grad_output, self, target, reduction, beta
        )

    if not self.is_contiguous():
        self = self.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    n_elements = self.numel()
    grad_input = torch.empty_like(self)

    if n_elements == 0:
        return grad_input

    inv_n = 1.0 / n_elements
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _smooth_l1_loss_backward_kernel[grid](
        grad_output,
        self,
        target,
        grad_input,
        n_elements,
        inv_n,
        beta,
        REDUCTION=reduction,
    )

    return grad_input


def smooth_l1_loss_backward_out(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    target: torch.Tensor,
    reduction: int,
    beta: float,
    *,
    grad_input: torch.Tensor,
):
    log.debug("GEMS SMOOTH_L1_LOSS BACKWARD GRAD_INPUT")
    result = smooth_l1_loss_backward(grad_output, self, target, reduction, beta)
    grad_input.copy_(result)
    return grad_input
