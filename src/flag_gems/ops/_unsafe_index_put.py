import logging

import torch

from flag_gems import config

logger = logging.getLogger(__name__)


def _unsafe_index_put(inp, indices, values, accumulate=False):
    """_unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor

    Functional advanced indexing scatter. Returns a new tensor.
    All parameter forms (bool masks, None indices, basic+advanced mixed) are
    handled in the C++ wrapper. Accumulate for dtypes without native Triton
    atomic_add (fp16/bf16, int8/int16/uint8/bool) uses the C++ widened-scratch
    scheme (fp32 / int32 scratch, opmath_t-equivalent).
    """
    logger.debug("GEMS _UNSAFE_INDEX_PUT")

    if not indices:
        raise ValueError("At least one index tensor is required")

    if config.has_c_extension:
        try:
            from flag_gems import c_operators

            return c_operators.unsafe_index_put(inp, indices, values, accumulate)
        except ImportError:
            pass

    # Fallback when the C extension is unavailable: PyTorch native via CPU to
    # bypass flag_gems' own _index_put_impl_ patch (same atomic_add limits).
    indices = list(indices)
    out = inp.clone()
    inp_device = inp.device
    if inp_device.type == "cuda":
        cpu_indices = [idx.cpu() for idx in indices]
        cpu_out = out.cpu()
        cpu_values = values.cpu()
        torch._index_put_impl_(
            cpu_out, cpu_indices, cpu_values, accumulate, unsafe=True
        )
        out = cpu_out.to(inp_device)
    else:
        torch._index_put_impl_(out, indices, values, accumulate, unsafe=True)
    return out
