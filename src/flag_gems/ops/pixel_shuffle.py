import logging

logger = logging.getLogger(__name__)


def pixel_shuffle(inp, upscale_factor):
    logger.debug("GEMS PIXEL SHUFFLE")
    assert inp.dim() == 4, "pixel_shuffle expects 4D input"
    N, C_in, H_in, W_in = inp.shape
    r = upscale_factor
    assert C_in % (r * r) == 0, "C_in must be divisible by upscale_factor^2"
    C_out = C_in // (r * r)
    H_out = H_in * r
    W_out = W_in * r
    x = inp.reshape(N, C_out, r, r, H_in, W_in)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(N, C_out, H_out, W_out).contiguous()
