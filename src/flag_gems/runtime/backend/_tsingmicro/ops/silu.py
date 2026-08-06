import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.codegen_config_utils import CodeGenConfig

div_rn = tl_extra_shim.div_rn
exp2 = tl_extra_shim.exp2

logger = logging.getLogger(__name__)


def _collapse_contiguous_dims(t):
    """Merge adjacent dims that are physically contiguous.

    For dim d and d+1, if stride[d] == stride[d+1] * shape[d+1], the two
    dims can be fused into one (shape[d]*shape[d+1], stride[d+1]) without
    changing the memory layout.  Reducing ndim cuts pointwise_dynamic's
    nD index calculation -- which on Tx81 turns into one
    sitofp->reciprocal->mul->trunc->...->mask_move chain per dim.

    Example for silu's input (4096, 4, 3072) stride (24576, 6144, 1):
        24576 == 6144 * 4  -> dims 0,1 collapse to (16384, 3072) stride (6144, 1)
        6144  != 1    * 3072 -> dim 2 stays separate
    Result: rank_3 -> rank_2, saves one reciprocal chain.
    """
    if t.dim() <= 1:
        return t
    shape = list(t.shape)
    stride = list(t.stride())
    new_shape = [shape[-1]]
    new_stride = [stride[-1]]
    for i in range(len(shape) - 2, -1, -1):
        if stride[i] == new_stride[0] * new_shape[0]:
            new_shape[0] = shape[i] * new_shape[0]
        else:
            new_shape.insert(0, shape[i])
            new_stride.insert(0, stride[i])
    if len(new_shape) == len(shape):
        return t
    return torch.as_strided(t, new_shape, new_stride)


my_config = CodeGenConfig(
    max_tile_size=65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

# @pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=my_config)
# @triton.jit
# def silu_forward(x):
#     x_fp32 = x.to(tl.float32)
#     y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
#     return y


# Variant for triggering Tx81 backend SigmoidFusionPattern.
# Matches sub(0, x) -> exp -> add(1, ...) -> div(1, ...) chain so that
# LinalgToMK rewrites the whole chain into a single mk.sigmoid op, then
# `x * sigma` becomes one mulvv.  Eliminates the reciprocal-corrective
# chain (negf/exp/sub/abs/cmpf/bit2fp/mask_move/...) that otherwise
# produces ~13 live SPM buffers.
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def silu_forward_sigmoid(x):
    x_fp32 = x.to(tl.float32)
    # 0.0 - x_fp32  (kept as sub, not folded to neg) → exp → 1 + exp → 1 / (...)
    sigma = 1.0 / (1.0 + tl.exp(0.0 - x_fp32))
    return x_fp32 * sigma


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    return x_fp32 / (1 + exp2(-x.to(tl.float32) * log2e))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def silu_backward_kernel(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


def silu(self):
    logger.debug("GEMS_TSINGMICRO SILU FORWARD")
    collapsed = _collapse_contiguous_dims(self)
    output = silu_forward_sigmoid(collapsed)
    return output.view(self.shape)


def silu_backward(grad_output, self):
    logger.debug("GEMS_TSINGMICRO SILU BACKWARD")
    grad_input = silu_backward_kernel(self, grad_output)
    return grad_input


def silu_(A):
    logger.debug("GEMS_TSINGMICRO SILU_ FORWARD")
    out = silu_forward(A, out0=A)
    return out
