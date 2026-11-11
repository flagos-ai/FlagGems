# Kunlunxin (XPU) override of ravel.
#
# The generic KernelGen implementation (src/flag_gems/ops/ravel.py) selects
# between three kernels on the COPY path:
#
#   * _ravel_tile_kernel    -- 2-D transposing tile, used when the outermost
#     canonical dim has unit stride and the rank is >= 2;
#   * _ravel_strided1d_kernel -- constant-stride rank-1 rows;
#   * _ravel_copy_kernel    -- rank-agnostic flat gather, everything else.
#
# On the XPU backend (torch 2.9 + torch_xmlir / XTriton 3.6):
#
#   * _ravel_tile_kernel does not legalize -- ``PassManager::run failed:
#     'arith.remsi' op requires the same type`` (int64 tile indices mixed
#     with int32 stride scalars; reported through OutOfResources anyway);
#   * _ravel_strided1d_kernel produces numerically wrong output for
#     bfloat16 with T0 = 2 (the ``strided_last`` test layout fails
#     ``torch.testing.assert_close``).
#
# The rank-agnostic flat gather (_ravel_copy_kernel / _ravel_copy_cplx_kernel)
# compiles and is exact for every layout on XPU, so the vendor override routes
# the whole COPY path through it.  The VIEW path (contiguous input) and the
# autograd wrapper are unchanged from the generic implementation.

import logging

import torch
import triton

from flag_gems.ops.ravel import (
    _MAX_DIMS,
    _canonical_dims,
    _pack,
    _ravel_copy_cplx_kernel,
    _ravel_copy_kernel,
)

logger = logging.getLogger(__name__)

__all__ = ["ravel"]

# Same geometry as the generic copy path (fully coalesced 16 B stores).
_BLOCK = 1024
_NUM_WARPS = 4


def _ravel_copy_impl(inp):
    """Materialise a contiguous 1-D flat copy for any strided layout.

    The generic implementation special-cases unit-stride rank-2 (tile) and
    constant-stride rank-1 (strided 1-D) layouts for bandwidth; those kernels
    are not usable on XPU (see the module docstring), so every copy goes
    through the rank-agnostic flat gather.
    """
    numel = inp.numel()
    out = torch.empty(numel, dtype=inp.dtype, device=inp.device)
    if numel == 0:
        return out

    dims = _canonical_dims(inp)
    rank = len(dims)
    if rank == 0:  # a single element is always contiguous: unreachable
        return out
    if rank > _MAX_DIMS:
        raise RuntimeError(
            "ravel: unsupported strided layout rank %d (max %d)" % (rank, _MAX_DIMS)
        )

    neg = bool(inp.is_neg())
    packed = _pack(dims)

    if inp.dtype.is_complex:
        _ravel_copy_cplx_kernel[(triton.cdiv(numel, _BLOCK),)](
            inp.data_ptr(),
            out.data_ptr(),
            numel,
            *packed,
            R=rank,
            WIDE=(inp.dtype == torch.complex128),
            NEG=neg,
            CONJ=bool(inp.is_conj()),
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
        return out

    _ravel_copy_kernel[(triton.cdiv(numel, _BLOCK),)](
        inp,
        out,
        numel,
        *packed,
        R=rank,
        NEG=neg,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out


class _RavelCopy(torch.autograd.Function):
    """Copy path with the native backward: the gradient is only rearranged
    back to the input shape and touches no backing storage."""

    @staticmethod
    def forward(ctx, inp):
        ctx.input_shape = inp.shape
        return _ravel_copy_impl(inp)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.reshape(ctx.input_shape)


def ravel(self):
    """``aten::ravel(Tensor(a) self) -> Tensor(a)`` (kunlunxin override)."""
    logger.debug("GEMS_KUNLUNXIN RAVEL")
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "ravel(): argument 'self' must be a Tensor, not %s" % type(self).__name__
        )
    if self.layout != torch.strided:
        raise RuntimeError("unsupported memory format option Contiguous")

    if self.is_contiguous():
        return self.view(-1)

    if self.requires_grad and torch.is_grad_enabled():
        return _RavelCopy.apply(self)
    return _ravel_copy_impl(self)