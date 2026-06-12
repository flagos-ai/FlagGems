import logging

import torch

from flag_gems.runtime import torch_device_fn

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

    # Delegate to torch's irfftn - this uses optimized cuFFT/cuFFT-like implementations
    # The Metax GPU can leverage similar FFT libraries through the torch backend
    with torch_device_fn.device(input.device):
        result = torch.fft.irfftn(input, s=s, dim=dim, norm=norm, out=out)

    return result
