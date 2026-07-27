import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_kernel(x, y):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    return tl.where(
        x_f32 >= 0,
        x_f32 - x_f32 * y_f32 + tl.log(1.0 + tl.exp(-x_f32)),
        tl.log(1.0 + tl.exp(x_f32)) - x_f32 * y_f32,
    )


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_weight_kernel(x, y, weight):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    w_f32 = weight.to(tl.float32)
    loss = tl.where(
        x_f32 >= 0,
        x_f32 - x_f32 * y_f32 + tl.log(1.0 + tl.exp(-x_f32)),
        tl.log(1.0 + tl.exp(x_f32)) - x_f32 * y_f32,
    )
    return loss * w_f32


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_pos_weight_kernel(x, y, pos_weight):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    pw_f32 = pos_weight.to(tl.float32)
    log_1p_exp_neg_x = tl.log(1.0 + tl.exp(-x_f32))
    log_1p_exp_x = tl.log(1.0 + tl.exp(x_f32))
    x_pos = y_f32 * pw_f32 * log_1p_exp_neg_x + (1.0 - y_f32) * (
        x_f32 + log_1p_exp_neg_x
    )
    x_neg = y_f32 * (-pw_f32 * x_f32 + pw_f32 * log_1p_exp_x) + (
        1.0 - y_f32
    ) * log_1p_exp_x
    return tl.where(x_f32 >= 0, x_pos, x_neg)


@pointwise_dynamic(
    is_tensor=[True, True, True, True], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def _bce_weight_pos_weight_kernel(x, y, weight, pos_weight):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    w_f32 = weight.to(tl.float32)
    pw_f32 = pos_weight.to(tl.float32)
    log_1p_exp_neg_x = tl.log(1.0 + tl.exp(-x_f32))
    log_1p_exp_x = tl.log(1.0 + tl.exp(x_f32))
    x_pos = y_f32 * pw_f32 * log_1p_exp_neg_x + (1.0 - y_f32) * (
        x_f32 + log_1p_exp_neg_x
    )
    x_neg = y_f32 * (-pw_f32 * x_f32 + pw_f32 * log_1p_exp_x) + (
        1.0 - y_f32
    ) * log_1p_exp_x
    return tl.where(x_f32 >= 0, x_pos, x_neg) * w_f32


def binary_cross_entropy_with_logits(
    self, target, weight=None, pos_weight=None, reduction=1
):
    logger.debug("GEMS_KUNLUNXIN BINARY_CROSS_ENTROPY_WITH_LOGITS")
    if weight is not None and pos_weight is not None:
        out = _bce_weight_pos_weight_kernel(self, target, weight, pos_weight)
    elif weight is not None:
        out = _bce_weight_kernel(self, target, weight)
    elif pos_weight is not None:
        out = _bce_pos_weight_kernel(self, target, pos_weight)
    else:
        out = _bce_kernel(self, target)

    if reduction == 2:
        flat = out.to(torch.float32).reshape(-1)
        chunk_size = 65536
        full_chunks = flat.numel() // chunk_size
        if full_chunks:
            total = flat[: full_chunks * chunk_size].reshape(full_chunks, chunk_size).sum(
                dim=1
            ).sum()
        else:
            total = torch.zeros((), dtype=torch.float32, device=flat.device)
        if full_chunks * chunk_size < flat.numel():
            total = total + flat[full_chunks * chunk_size :].sum()
        return total.to(self.dtype)
    if reduction == 1:
        return out.mean()
    return out
