import logging

import torch

logger = logging.getLogger(__name__)


def _conj(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CONJ")
    if not input.is_complex():
        raise RuntimeError("_conj only supports complex tensors")

    # The interleaved real/imag negate kernel miscompiles on XPU at some tensor
    # sizes, so materialize via the native elementwise path (correct for both
    # complex dtypes at every size). input.imag already reflects a pending conj
    # bit, so this is correct for an unresolved conj view too.
    return torch.complex(input.real, -input.imag).contiguous()
