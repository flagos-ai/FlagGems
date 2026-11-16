import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _conj_flat_kernel(fin, fout, n2, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n2
    x = tl.load(fin + i, mask=m)
    out = tl.where((i % 2) == 1, -x, x)
    tl.store(fout + i, out, mask=m)


@triton.jit
def _conj_flat_copy_kernel(fin, fout, n2, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n2
    x = tl.load(fin + i, mask=m)
    tl.store(fout + i, x, mask=m)


def _flatten_storage(input: torch.Tensor) -> torch.Tensor:
    """Return a contiguous 1D view of the interleaved real/imag storage.

    ``view_as_real`` strips the complex dtype, so materializing a
    non-contiguous input goes through the vendor ``copy_`` for a *real* tensor
    (the complex ``copy_`` is rejected by the vendor on this stack).
    """
    rv = torch.view_as_real(input)
    if not rv.is_contiguous():
        rv = rv.contiguous()
    return rv.reshape(-1)


def _conj_block_size(n2: int) -> int:
    return 32768 if n2 >= 1 << 20 else 8192


def _conj_from_storage(input: torch.Tensor) -> torch.Tensor:
    """Materialize ``conj`` of a tensor whose storage is the physical value."""
    out = torch.empty_like(input, memory_format=torch.contiguous_format)
    fin = _flatten_storage(input)
    fout = torch.view_as_real(out).reshape(-1)
    n2 = fin.numel()

    BLOCK = _conj_block_size(n2)
    grid = (triton.cdiv(n2, BLOCK),)
    with torch_device_fn.device(input.device):
        _conj_flat_kernel[grid](fin, fout, n2, BLOCK=BLOCK, num_warps=8)

    return out


def _conj_materialize(input: torch.Tensor) -> torch.Tensor:
    """Physically materialize ``conj(input)`` into a fresh tensor.

    This is *not* ``aten::_conj`` semantics (that op is a zero-copy view); it
    is the private helper used by ``conj_physical``, which must write the
    conjugated values to memory.
    """
    if input.is_conj():
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
        fin = _flatten_storage(input)
        fout = torch.view_as_real(out).reshape(-1)
        n2 = fin.numel()
        BLOCK = _conj_block_size(n2)
        grid = (triton.cdiv(n2, BLOCK),)
        with torch_device_fn.device(input.device):
            _conj_flat_copy_kernel[grid](fin, fout, n2, BLOCK=BLOCK, num_warps=8)
        return out

    return _conj_from_storage(input)


def _conj(input: torch.Tensor) -> torch.Tensor:
    """``aten::_conj``: zero-copy lazy conjugate view.

    Native ``aten::_conj`` returns a view of ``input`` (same storage, same
    shape/stride/storage_offset) with the per-tensor lazy conjugate bit
    toggled; it reads and writes no element.  Materializing a copy here (the
    old behaviour) both loses the alias contract and launches a device kernel
    for an op that should be metadata-only, which also broke ``use_gems``
    consumers such as ``adjoint``/``mH`` that rely on ``conj`` being a view.

    Build the view from raw metadata (``torch.as_strided``) and flip the flag
    with ``torch._C._set_conj`` -- exactly like ``adjoint`` does; calling
    ``Tensor.conj()``/``aten::_conj`` from here would recurse.
    """
    logger.debug("GEMS_KUNLUNXIN CONJ")
    if not input.is_complex():
        raise RuntimeError("_conj only supports complex tensors")

    out = torch.as_strided(input, input.shape, input.stride(), input.storage_offset())
    torch._C._set_conj(out, not input.is_conj())
    return out
