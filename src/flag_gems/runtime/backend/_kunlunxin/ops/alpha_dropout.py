import logging

import torch
import triton

from flag_gems.ops.alpha_dropout import (
    _alpha_dropout_affine,
    alpha_dropout_forward_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger("flag_gems.ops.alpha_dropout")

_UNROLL = 4
_TILE = 1024 * _UNROLL


def _alpha_dropout_grid(meta):
    return (triton.cdiv(meta["N"], meta["BLOCK"] * _UNROLL),)


def alpha_dropout(input, p=0.5, train=True):
    logger.debug("GEMS_KUNLUNXIN ALPHA_DROPOUT FORWARD")
    if not train or p == 0:
        return input.clone()
    if p == 1:
        return torch.zeros_like(input)

    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    device = input.device
    input = input.contiguous()
    N = input.numel()

    a, b = _alpha_dropout_affine(p)

    if N % _TILE == 0:
        launch_input = input
        out = torch.empty_like(input)
        N_launch = N
    else:
        N_launch = triton.cdiv(N, _TILE) * _TILE
        launch_input = torch.zeros(N_launch, dtype=input.dtype, device=device)
        launch_input[:N] = input.view(-1)
        out = torch.empty_like(launch_input)

    increment = triton.cdiv(N_launch, _UNROLL)

    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        alpha_dropout_forward_kernel[_alpha_dropout_grid](
            launch_input, out, N_launch, p, a, b, philox_seed, philox_offset
        )
    if N_launch != N:
        out = out[:N].view(input.shape)
    return out
