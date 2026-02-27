import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.randn import randn_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)
UNROLL = 4


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_log_normal(val, std, mean):
    # log_normal: exp(val * std + mean)
    # where val ~ N(0, 1), so val * std + mean ~ N(mean, std^2)
    # and exp(val * std + mean) ~ LogNormal(mean, std)
    return tl.exp(val * std + mean)


def log_normal_(self, mean=1.0, std=2.0, *, generator=None):
    logger.debug("GEMS LOG_NORMAL_")
    shape = self.shape
    device = self.device
    N = volume(shape)

    # Generate standard normal distribution in float32
    temp = torch.empty(shape, device=device, dtype=torch.float32)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        randn_kernel[grid_fn](temp, N, philox_seed, philox_offset)

    # Apply log_normal transformation: exp(val * std + mean)
    transform_log_normal(temp, std, mean, out0=self)
    return self
