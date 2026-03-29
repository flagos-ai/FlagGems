import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _view_as_complex_kernel(inp_ptr, out_ptr, n_elem, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elem

    in_real_offsets = offsets * 2
    in_imag_offsets = in_real_offsets + 1
    out_real_offsets = offsets * 2
    out_imag_offsets = out_real_offsets + 1

    real = tl.load(inp_ptr + in_real_offsets, mask=mask)
    imag = tl.load(inp_ptr + in_imag_offsets, mask=mask)
    tl.store(out_ptr + out_real_offsets, real, mask=mask)
    tl.store(out_ptr + out_imag_offsets, imag, mask=mask)


def view_as_complex(A: torch.Tensor):
    logger.debug("GEMS VIEW_AS_COMPLEX")
    if A.dtype not in (torch.float16, torch.float32, torch.float64):
        raise RuntimeError(
            "view_as_complex is only supported for float16, float32 and float64 tensors"
        )
    if A.ndim < 1 or A.shape[-1] != 2:
        raise RuntimeError("Tensor must have a last dimension of size 2")

    if A.stride(-1) != 1:
        raise RuntimeError("Tensor must have a last dimension with stride 1")

    if A.dtype == torch.float16:
        out_dtype = torch.complex32
    elif A.dtype == torch.float32:
        out_dtype = torch.complex64
    else:
        out_dtype = torch.complex128
    out = torch.empty(A.shape[:-1], device=A.device, dtype=out_dtype)
    n_elem = out.numel()
    if n_elem == 0:
        return out

    if out.ndim == 0:
        flat = A.contiguous().view(-1)
        return torch.complex(flat[0], flat[1])

    inp = A.contiguous().view(-1)
    out_view = out.view(A.dtype).view(-1)
    grid = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device):
        _view_as_complex_kernel[grid](inp, out_view, n_elem, BLOCK_SIZE=1024)
    return out
