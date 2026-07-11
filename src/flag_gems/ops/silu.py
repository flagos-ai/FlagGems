import dataclasses
import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import get_codegen_config
from flag_gems.utils.triton_lang_extension import div_rn

logger = logging.getLogger(__name__)

# Local, per-op tuning. The default NVIDIA max_tile_size (512) is too small for
# this memory-bound elementwise op on newer GPUs (e.g. H20), leaving it slower
# than torch on large tensors. Bump the tile to >=1024 for silu only, starting
# from the current device's default config so vendors whose default is already
# >=1024 (Iluvatar/MetaX/Mthreads/Hygon/...) are unchanged, and the global config
# and all other ops are untouched.
_base_config = get_codegen_config()
_silu_config = dataclasses.replace(
    _base_config, max_tile_size=max(1024, _base_config.max_tile_size)
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_silu_config)
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return y


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_silu_config)
@triton.jit
def silu_backward_kernel(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


def silu(self):
    logger.debug("GEMS SILU")
    output = silu_forward(self)
    return output


def silu_backward(grad_output, self):
    logger.debug("GEMS SILU_BACKWARD")
    grad_input = silu_backward_kernel(self, grad_output)
    return grad_input


def silu_(A):
    logger.debug("GEMS SILU_")
    out = silu_forward(A, out0=A)
    return out


def silu_out(self, *, out):
    logger.debug("GEMS SILU_OUT")
    return silu_forward(self, out0=out)
