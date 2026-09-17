import logging

import triton
import triton.language as tl

import flag_gems

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(
    is_tensor=[True, True, False],
    num_outputs=2,
    promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
    dtypes=[None, None, float],
)
@triton.jit
def _prelu_kernel_backward_scalar_func(grad_output, x, weight):
    pos = tl.minimum(1.0, tl.maximum(0.0, x.to(tl.float32) * 1.0e30)).to(x.dtype)
    x_neg = tl.minimum(x, 0.0)
    grad_input = grad_output * (pos + (1.0 - pos) * weight)
    grad_weight = grad_output * x_neg * (1.0 - pos)
    return grad_input, grad_weight


@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
)
@triton.jit
def _prelu_kernel_backward_channel_func(grad_output, x, weight):
    pos = tl.minimum(1.0, tl.maximum(0.0, x.to(tl.float32) * 1.0e30)).to(x.dtype)
    x_neg = tl.minimum(x, 0.0)
    grad_input = grad_output * (pos + (1.0 - pos) * weight)
    grad_weight = grad_output * x_neg * (1.0 - pos)
    return grad_input, grad_weight


def _prelu_kernel_backward(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN _PRELU_KERNEL_BACKWARD")
    if len(args) >= 3:
        grad_output, x, weight = args[0], args[1], args[2]
    else:
        grad_output = kwargs.get("grad_output")
        x = kwargs.get("self")
        weight = kwargs.get("weight")

    if grad_output is None or x is None or weight is None:
        raise ValueError(
            "_prelu_kernel_backward expects (grad_output, self, weight) as arguments."
        )

    if (
        grad_output.device.type != flag_gems.device
        or x.device.type != flag_gems.device
        or weight.device.type != flag_gems.device
    ):
        raise RuntimeError(
            f"_prelu_kernel_backward: all tensors must be "
            f"{flag_gems.device} tensors for Triton kernel."
        )

    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    if grad_output.dtype != x.dtype:
        grad_output = grad_output.to(dtype=x.dtype)

    grad_output = grad_output.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()

    ndim = x.dim()
    if weight.numel() == 1:
        return _prelu_kernel_backward_scalar_func(grad_output, x, float(weight))
    if ndim == 0:
        raise AssertionError("Non-scalar weight provided for a 0-dim input.")
    C = x.shape[-1]
    if weight.numel() != C:
        raise AssertionError(
            f"Weight numel ({weight.numel()}) must equal last dimension size ({C})."
        )
    if ndim == 1:
        w_shape = [C]
    else:
        w_shape = [1] * (ndim - 1) + [C]
    w = weight.reshape(w_shape)
    return _prelu_kernel_backward_channel_func(grad_output, x, w)
