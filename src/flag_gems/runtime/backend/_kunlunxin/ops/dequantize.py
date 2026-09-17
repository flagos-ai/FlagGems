import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_QINT_ALIAS = {
    torch.qint8: (torch.int8, False),
    torch.quint8: (torch.int8, True),
    torch.qint32: (torch.int32, False),
}

_BLOCK = 8192


@triton.jit
def _dequantize_kernel(
    x_ptr,
    out_ptr,
    scale,
    zero_point,
    n_elements,
    UNSIGNED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    values = raw.to(tl.int32).to(tl.float32)
    if UNSIGNED:
        values = tl.where(values < 0.0, values + 256.0, values)
    tl.store(out_ptr + offsets, (values - zero_point) * scale)


def _int_repr_view(inp: torch.Tensor, int_dtype: torch.dtype) -> torch.Tensor:
    """Reinterpret a quantized tensor's storage as its integer representation.

    ``Tensor.int_repr()`` has no working QuantizedCUDA kernel on this platform
    (it raises ``CUDA error: invalid device function``), so the integer payload
    is obtained by re-typing the very same device storage. This is pure metadata
    work: no copy, no host round trip, no ATen compute kernel.
    """
    view = torch.empty(0, dtype=int_dtype, device=inp.device)
    view.set_(
        inp.untyped_storage(),
        inp.storage_offset(),
        tuple(inp.shape),
        tuple(inp.stride()),
    )
    return view


def dequantize(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DEQUANTIZE")
    if not input.is_quantized:
        raise RuntimeError("dequantize expects a quantized tensor")
    if input.qscheme() not in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        raise NotImplementedError(
            "Kunlunxin dequantize supports per-tensor quantization only."
        )
    if input.dtype not in _QINT_ALIAS:
        raise NotImplementedError(
            f"Kunlunxin dequantize does not support {input.dtype}."
        )

    int_dtype, unsigned = _QINT_ALIAS[input.dtype]
    int_repr = _int_repr_view(input, int_dtype)
    if not int_repr.is_contiguous():
        int_repr = int_repr.contiguous()

    n_elements = int_repr.numel()
    if n_elements == 0:
        return torch.empty(input.shape, dtype=torch.float32, device=input.device)

    BLOCK = _BLOCK
    n_tiles = triton.cdiv(n_elements, BLOCK)
    padded = torch.empty(n_tiles * BLOCK, dtype=torch.float32, device=input.device)

    with torch_device_fn.device(input.device):
        _dequantize_kernel[(n_tiles,)](
            int_repr,
            padded,
            float(input.q_scale()),
            int(input.q_zero_point()),
            n_elements,
            UNSIGNED=unsigned,
            BLOCK=BLOCK,
        )
    return padded[:n_elements].view(input.shape)
