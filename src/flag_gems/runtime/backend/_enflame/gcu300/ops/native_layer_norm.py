import logging

from .layernorm import layer_norm

logger = logging.getLogger(__name__)


def native_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Run native layer normalization through the existing implementation."""
    logger.debug("GEMS NATIVE_LAYER_NORM")
    output, mean, rstd = layer_norm(input, normalized_shape, weight, bias, eps)
    stats_shape = input.shape[: -len(normalized_shape)] + (1,) * len(normalized_shape)
    return output, mean.view(stats_shape), rstd.view(stats_shape)
