import logging

from flag_gems.ops.fft_irfftn import fft_irfftn as _generic_fft_irfftn

logger = logging.getLogger("flag_gems." + __name__)


def fft_irfftn(input, s=None, dim=None, norm=None, out=None):
    """Metax specialized implementation for fft_irfftn.

    This is an inverse real FFT operation. The input is interpreted as a
    one-sided Hermitian signal in the Fourier domain (as produced by rfftn).
    The output will be real-valued.

    Args:
        input: the input tensor (complex)
        s: signal size in the transformed dimensions
        dim: dimensions to be transformed
        norm: normalization mode ("forward", "backward", "ortho")
        out: the output tensor

    Returns:
        Real-valued tensor after inverse FFT
    """
    logger.debug("GEMS_METAX FFT_IRFFTN")
    return _generic_fft_irfftn(input, s=s, dim=dim, norm=norm)
