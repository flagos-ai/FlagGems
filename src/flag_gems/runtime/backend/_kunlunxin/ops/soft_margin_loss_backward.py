import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# soft_margin_loss_backward is a pure memory-bound elementwise op:
#   grad = grad_output * (-target) / (1 + exp(target * self)) * norm
# where norm = 1/N for 'mean' reduction, else 1.  The generic (H20) kernel is a
# raw @triton.jit with a fixed BLOCK_SIZE=1024 AND passes n_elements_reduced as a
# tl.constexpr, so every distinct N recompiles a fresh module -> IR explosion
# (116MB / 8256 modules on XPU) plus the generic slow path.  Fix: XPU
# pointwise_dynamic (tuned codegen) with norm fed as a RUNTIME scalar, so no
# per-shape recompile.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, True, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def _soft_margin_loss_backward_func(grad_output, self_val, target, norm):
    grad_output_f = grad_output.to(tl.float32)
    self_f = self_val.to(tl.float32)
    target_f = target.to(tl.float32)
    exp_term = tl.exp(target_f * self_f)
    grad = grad_output_f * (-target_f) / (1.0 + exp_term)
    return grad * norm


def _normalize_reduction(reduction):
    # Accept both string and enum/int forms: 0=none, 1=mean, 2=sum
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def soft_margin_loss_backward(
    grad_output: torch.Tensor,
    self_tensor: torch.Tensor,
    target: torch.Tensor,
    reduction="mean",
):
    logger.debug("GEMS_KUNLUNXIN SOFT_MARGIN_LOSS_BACKWARD")
    if not self_tensor.is_contiguous():
        self_tensor = self_tensor.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    n_elements = self_tensor.numel()
    if n_elements == 0:
        return torch.empty_like(self_tensor)

    red = _normalize_reduction(reduction)
    norm = 1.0 / n_elements if red == 1 else 1.0

    out = torch.empty_like(self_tensor)
    _soft_margin_loss_backward_func(grad_output, self_tensor, target, norm, out0=out)
    return out
