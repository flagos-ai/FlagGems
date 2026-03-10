import logging

import torch

logger = logging.getLogger(__name__)


def pixel_unshuffle(inp, downscale_factor):
    logger.debug("GEMS PIXEL UNSHUFFLE")
    assert inp.dim() == 4, "pixel_unshuffle expects 4D input"
    N, C_in, H_in, W_in = inp.shape
    r = downscale_factor
    assert H_in % r == 0, "H must be divisible by downscale_factor"
    assert W_in % r == 0, "W must be divisible by downscale_factor"
    C_out = C_in * r * r
    H_out = H_in // r
    W_out = W_in // r
    x = inp.reshape(N, C_in, H_out, r, W_out, r)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(N, C_out, H_out, W_out).contiguous()
