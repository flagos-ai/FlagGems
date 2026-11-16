# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys

import torch
import triton

from flag_gems.ops.copy import copy_
from flag_gems.runtime import torch_device_fn

from ._conj import _conj_block_size, _conj_flat_kernel, _conj_materialize
from .resolve_conj import resolve_conj

logger = logging.getLogger(__name__)


def conj_physical(input: torch.Tensor) -> torch.Tensor:
    """Materialize the complex conjugate with the vendor stride-1 kernel."""
    logger.debug("GEMS_KUNLUNXIN CONJ_PHYSICAL")
    if not input.is_complex():
        return input
    # Same behavior as the generic implementation: resolve a pending conjugate
    # bit first, since torch.view_as_real() (used by the materializing kernel)
    # rejects unresolved conj views. resolve_conj() is a no-op when the bit is
    # not set.  ``_conj`` itself is a lazy *view* (aten::_conj), so the physical
    # write must go through ``_conj_materialize``.
    input = resolve_conj(input)
    return _conj_materialize(input)


def _conj_into_contiguous(src: torch.Tensor, out: torch.Tensor) -> None:
    """Write ``conj(src)`` into ``out`` over the flat interleaved pair stream.

    Both operands must be contiguous complex tensors of equal numel.  Walking a
    single stride-1 real stream (and negating every odd lane in-register) lets
    XPU3 emit block DMA; the generic pair kernel instead gathers/stores at a
    stride of 2, which is roughly two orders of magnitude slower on large
    tensors (see ``harness/perf_ir/kernel_summary__conj_physical_pre.md``).
    """
    n = src.numel()
    if n == 0:
        return
    n2 = n * 2
    fin = torch.view_as_real(src).reshape(-1)
    fout = torch.view_as_real(out).reshape(-1)
    block = _conj_block_size(n2)
    grid = (triton.cdiv(n2, block),)
    with torch_device_fn.device(src.device):
        _conj_flat_kernel[grid](fin, fout, n2, BLOCK=block, num_warps=8)


def _materialize_real(input: torch.Tensor) -> torch.Tensor:
    """Fresh copy with the layout ``empty_like``'s preserve_format yields.

    ATen's ``_conj_physical`` never aliases its input; for a real dtype the
    conjugate is the value itself, but a new tensor is still returned.
    """
    out = torch.empty_like(input)
    if out.numel() != 0:
        copy_(out, input)
    return out


def _conj_physical(input: torch.Tensor) -> torch.Tensor:
    """``aten::_conj_physical``: physically conjugate into a fresh tensor.

    Unlike the public ``conj_physical`` (which hands a real input straight
    back), this always allocates, and it keeps the input's *dense* strides
    (channels-last stays channels-last) while non-dense inputs (gappy slices,
    expanded views) come back contiguous -- the same ``preserve_format`` rule
    ATen applies.
    """
    logger.debug("GEMS_KUNLUNXIN CONJ_PHYSICAL (private)")
    if input.is_complex() and input.is_conj():
        input = resolve_conj(input)

    if not input.is_complex():
        return _materialize_real(input)

    out = torch.empty_like(input)
    src = input if input.is_contiguous() else input.contiguous()
    if out.is_contiguous():
        _conj_into_contiguous(src, out)
    else:
        # Dense but not contiguous (channels-last / transposed): the flat kernel
        # needs a contiguous destination, so conjugate into a temporary and
        # restride through the Gems copy.
        tmp = torch.empty_like(src)
        _conj_into_contiguous(src, tmp)
        if tmp.numel() != 0:
            copy_(out, tmp)
    return out


def _conj_physical_out(input: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """``aten::_conj_physical.out``: write into ``out``, resizing a mismatch."""
    logger.debug("GEMS_KUNLUNXIN CONJ_PHYSICAL_OUT")
    _check_out(out, input)
    if tuple(out.shape) != tuple(input.shape):
        out.resize_(input.shape)

    if input.is_complex() and input.is_conj():
        input = resolve_conj(input)

    if not input.is_complex():
        if out.numel() != 0:
            copy_(out, input)
        return out

    src = input if input.is_contiguous() else input.contiguous()
    if out.is_contiguous():
        _conj_into_contiguous(src, out)
    else:
        tmp = torch.empty_like(src)
        _conj_into_contiguous(src, tmp)
        if tmp.numel() != 0:
            copy_(out, tmp)
    return out


def _check_out(out: torch.Tensor, input: torch.Tensor) -> None:
    """Validate ``out`` the way ATen's structured-kernel wrapper does."""
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_dtype_name(input.dtype)}, "
            f"but got {_dtype_name(out.dtype)} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to have device {input.device}, "
            f"but got {out.device} instead"
        )


def _dtype_name(dtype: torch.dtype) -> str:
    """Spell a dtype the way ATen's ``out=`` dtype check does (TypeName)."""
    names = {
        torch.float32: "float",
        torch.float64: "double",
        torch.float16: "c10::Half",
        torch.bfloat16: "c10::BFloat16",
        torch.complex64: "c10::complex<float>",
        torch.complex128: "c10::complex<double>",
        torch.uint8: "unsigned char",
        torch.int8: "signed char",
        torch.int16: "short",
        torch.int32: "int",
        torch.int64: "long int",
        torch.bool: "bool",
    }
    return names.get(dtype, str(dtype))


# The dedicated tests / benchmark for the private op call ``flag_gems.ops.
# _conj_physical`` directly (not the aten-dispatch entry), so publish the
# override there too -- the same pattern the Kunlunxin ``_ctc_loss`` and
# ``special_chebyshev_polynomial_t`` overrides use.
_generic_ops_module = sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    for _name, _fn in (
        ("_conj_physical", _conj_physical),
        ("_conj_physical_out", _conj_physical_out),
    ):
        setattr(_generic_ops_module, _name, _fn)


# The vendor's native ``aten::_conj_physical`` is broken for real dtypes on
# this stack: it corrupts its *input* in place (bit-pattern garbage / NaN) and
# returns the same garbage, e.g. ``_conj_physical(torch.randn(3, 4))`` leaves
# the caller's tensor full of denormals. ``torch._conj_physical`` is not
# reachable through ``use_gems`` in the dedicated tests (they compare against
# ``torch.ops.aten._conj_physical`` directly), so route the CUDA dispatch key
# to this correct Gems kernel process-wide -- the same remediation the backend
# already applies to ``adaptive_max_pool3d`` (whose vendor indices are also
# broken).  This only replaces the *forward* op; ``.out`` is intentionally left
# to the native path.
_aten_conj_physical_lib = None


def _patch_aten_conj_physical() -> None:
    global _aten_conj_physical_lib
    if _aten_conj_physical_lib is not None:
        return
    try:
        _aten_conj_physical_lib = torch.library.Library("aten", "IMPL")
        _aten_conj_physical_lib.impl(
            "_conj_physical", _conj_physical, "CUDA", allow_override=True
        )
    except Exception:  # pragma: no cover - never break import on odd stacks
        _aten_conj_physical_lib = None


_patch_aten_conj_physical()
