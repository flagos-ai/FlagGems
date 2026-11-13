# Copyright 2026, The FlagOS Contributors.
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
"""gcu400 ``_conj_physical``: bit-safe word-stream conjugation.

Why the generic implementation cannot run here (both measured, P04/P21):

* complex128 views lower to fp64: the generic pair kernel loads fp64 words
  (``view_as_real``/``view(dtype)``), and gcu400 has no fp64 arithmetic or
  f64 vector memory ops, so the compile dies with "Pipeline run failed:
  PassManager execution failed" (P04).  On this backend complex128 is
  complex64 in disguise (P21): every creation path stores fp32 (real, imag)
  pairs at an effective 8 bytes per element while the dtype label,
  ``element_size()`` and ``data_ptr()`` keep the 16B claim, and torch_gcu
  reads position elements by the effective 8B itemsize.

* the conjugate bit cannot be resolved: ``Tensor.resolve_conj()`` and
  ``copy_`` from a conj source both leave ``is_conj() == True`` on the
  result (torch_gcu metadata defect, P21), so any ``view``-family op on a
  derived tensor raises "view_as_real doesn't work on unresolved conjugated
  tensors".  The generic host resolves the bit via copy and then views the
  result, which dies before any kernel runs.

The fix mirrors ``isreal.py``: the complex storage is rebound into a flat
fp32-word carrier (``torch.empty(0).set_(storage, offset * 2, ...)`` — pure
metadata, strips the lazy conj/neg bits, and lands exactly where torch_gcu's
aten reads land for both complex64 and complex128).  The kernel then walks
the interleaved (real, imag, ...) word stream and flips signs, so the conj
bit never needs resolving: the flags below encode the bit effects directly.

Host policy (user ruling, 2026-11-12): the wrapper only allocates
(``empty_like``/``empty``/factory), launches this module's kernels, and
composes the FlagGems ``copy_`` op (backend-local ``ops/copy.py``) for
strided writes; no aten compute op is called from this host.

The comparison side of this op's accuracy gate is handled by ``add.py``'s
``torch.isclose`` shim (this backend installs it for every process):
``torch.isclose`` on fp8 gcu operands aborts the process (no topsaten fp8
comparison kernel), and torch.testing compares fp8 bitwise through isclose,
which kills the accuracy file outright.  The fp8 route goes to CPU *with an
fp32 upcast* -- CPU isclose itself raises NotImplementedError for fp8
("mul_cpu_reduced_float") -- which is exact for tolerance-free comparisons
(isclose(0,0) == eq and fp8->fp32 is injective).

Finally, this module publishes itself over the generic package attribute
(``flag_gems.ops._conj_physical``), the same pattern as
``_linalg_det.py``/``quantized_max_pool2d``: the accuracy tests and the perf
benchmark resolve the op through that attribute, which otherwise keeps
pointing at the generic implementation that cannot compile here.
"""

import logging
import sys

import torch
import triton
import triton.language as tl

import flag_gems.ops
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .copy import copy_ as _flaggems_copy_

logger = logging.getLogger(__name__)

__all__ = ["_conj_physical", "_conj_physical_out"]

# Flat word-stream tile/grid, the backend-standard shape-independent launch
# (copy.py / isreal.py family): at most 48 programs loop over the stream, so
# the grid never depends on the problem size (P19 grid.x cap is safe).
_BLOCK_WORDS = 16384
_GRID_CAP = 48
# 2 * element offsets must stay inside int32; keep one full tile of headroom.
_MAX_CHUNK_WORDS = 2**31 - 1 - _BLOCK_WORDS

_COMPLEX_WORD_DTYPES = (torch.complex64, torch.complex128)


@libentry()
@triton.jit(do_not_specialize=["n_words"])
def _conj_physical_words_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    BLOCK: tl.constexpr,
    NEGATE_REAL: tl.constexpr,
    NEGATE_IMAG: tl.constexpr,
):
    """Conjugate an interleaved fp32 (real, imag, ...) word stream in place.

    The stream is the effective storage of a contiguous complex64 or
    complex128 tensor (both store fp32 pairs at 8 bytes per element on this
    backend).  Odd words are imaginary parts; IEEE negation of an fp32 value
    is an exact sign-bit flip, so ``-value`` is bit-exact against aten for
    every value including +-0.0 and NaN payloads.  The flag combinations are
    constexpr so only the needed variant is compiled.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_tiles = tl.cdiv(n_words, BLOCK)
    for tile_id in tl.range(pid, num_tiles, num_pids):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_words
        value = tl.load(src_ptr + offsets, mask=mask, other=0)
        if NEGATE_REAL and NEGATE_IMAG:
            value = -value
        elif NEGATE_REAL:
            value = tl.where((offsets & 1) == 0, -value, value)
        elif NEGATE_IMAG:
            value = tl.where((offsets & 1) == 1, -value, value)
        tl.store(dst_ptr + offsets, value, mask=mask)


def _launch_words(src_words, dst_words, n_words, negate_real, negate_imag):
    """Launch the word kernel, retrying cold compiles (P02 family)."""
    for attempt in range(3):
        try:
            _conj_physical_words_kernel[
                (min(_GRID_CAP, triton.cdiv(n_words, _BLOCK_WORDS)),)
            ](
                src_words,
                dst_words,
                n_words,
                BLOCK=_BLOCK_WORDS,
                NEGATE_REAL=negate_real,
                NEGATE_IMAG=negate_imag,
                num_warps=8,
                num_stages=1,
            )
            return
        except Exception:
            if attempt == 2:
                raise
            logger.warning(
                "_conj_physical kernel launch failed (attempt %d)", attempt + 1
            )


def _words_view(tensor: torch.Tensor) -> torch.Tensor:
    """Rebind a contiguous complex tensor's storage as a flat fp32 stream.

    The offset is scaled by the effective two words per element, which is
    where torch_gcu's aten reads actually land (complex128's claimed 16B
    itemsize would double-count; measured in compiler_problems/P21).  The
    rebind is pure metadata and strips the lazy conj/neg bits that block
    ``view(dtype)`` entirely (and that ``resolve_conj`` cannot clear).
    """
    carrier = torch.empty(0, dtype=torch.float32, device=tensor.device)
    carrier.set_(
        tensor.untyped_storage(),
        tensor.storage_offset() * 2,
        (tensor.numel() * 2,),
        (1,),
    )
    return carrier


def _conj_flags(input: torch.Tensor):
    """Per-lane negation flags for the *stored* words of ``input``.

    ``_conj_physical`` returns conj(logical(input)).  With the conj bit A and
    the neg bit N the logical value is conj_A(neg_N(stored)), so the output
    is (r, -i) / (r, i) / (-r, i) / (-r, -i) for (A,N) = (F,F)/(T,F)/(F,T)/
    (T,T) respectively: negate real iff N, negate imag iff A == N.  (Complex
    neg views cannot be created on this backend -- ``.neg()`` aborts -- but
    the algebra is free, so keep it exact.)
    """
    negate_real = bool(input.is_neg())
    negate_imag = bool(input.is_conj()) == negate_real
    return negate_real, negate_imag


def _contiguous_source(input: torch.Tensor) -> torch.Tensor:
    """Materialize a non-contiguous complex input into logical order.

    The FlagGems copy op carries the conj/neg bits through (raw values, bit
    propagated -- measured), so the flags still come from the original input
    and the word view strips the bits before the kernel reads raw storage.
    """
    contiguous = torch.empty_like(input, memory_format=torch.contiguous_format)
    if contiguous.numel() != 0:
        _flaggems_copy_(contiguous, input)
    return contiguous


def _conj_into(src: torch.Tensor, dst: torch.Tensor, negate_real, negate_imag):
    """Conjugate contiguous complex ``src`` into contiguous complex ``dst``."""
    n_words = src.numel() * 2
    if n_words == 0:
        return
    src_words = _words_view(src)
    dst_words = _words_view(dst)
    with torch_device_fn.device(src.device):
        start = 0
        while start < n_words:
            segment = min(_MAX_CHUNK_WORDS, n_words - start)
            _launch_words(
                src_words[start : start + segment],
                dst_words[start : start + segment],
                segment,
                negate_real,
                negate_imag,
            )
            start += segment


def _handle_locally(input: torch.Tensor) -> bool:
    """This backend owns complex64/128 inputs and every real input.

    Real inputs must also stay local: the generic copy that materializes
    them mis-addresses strided destinations on this backend (measured in
    ops/copy.py), while the backend-local copy op guards that case.  Only
    complex32 and non-gcu inputs keep the generic contract.
    """
    if not (isinstance(input, torch.Tensor) and input.device.type == "gcu"):
        return False
    return input.dtype in _COMPLEX_WORD_DTYPES or not input.is_complex()


def _conj_physical(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME _CONJ_PHYSICAL")
    if not _handle_locally(input):
        # complex32, non-gcu inputs and anything exotic keep the generic
        # dtype validation and error contract.
        return _GENERIC_CONJ_PHYSICAL(input)

    output = torch.empty_like(input)
    if not input.is_complex():
        # ATen's _conj_physical always hands back a fresh tensor; for a real
        # input conjugation is the identity, so this is a plain materialize
        # through the FlagGems copy op.
        if output.numel() != 0:
            _flaggems_copy_(output, input)
        return output

    src = input if input.is_contiguous() else _contiguous_source(input)
    negate_real, negate_imag = _conj_flags(input)
    if output.is_contiguous():
        _conj_into(src, output, negate_real, negate_imag)
    else:
        # Dense-but-strided output (channels_last & friends keep their
        # strides under preserve_format): conjugate into a flat temporary
        # and scatter through the copy op, which honours the target strides.
        tmp = torch.empty_like(src)
        _conj_into(src, tmp, negate_real, negate_imag)
        if tmp.numel() != 0:
            _flaggems_copy_(output, tmp)
    return output


def _conj_physical_out(input: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME _CONJ_PHYSICAL_OUT")
    if not _handle_locally(input):
        return _GENERIC_CONJ_PHYSICAL_OUT(input, out=out)

    _check_out(out, input)

    # ATen resizes a mismatched `out`; this is metadata-only.
    if tuple(out.shape) != tuple(input.shape):
        out.resize_(input.shape)

    if not input.is_complex():
        if out.numel() != 0:
            _flaggems_copy_(out, input)
        return out

    src = input if input.is_contiguous() else _contiguous_source(input)
    negate_real, negate_imag = _conj_flags(input)
    if out.is_contiguous():
        _conj_into(src, out, negate_real, negate_imag)
    else:
        # `out` may be an arbitrary strided view (ATen accepts those and
        # writes through them); the flat kernel cannot address it directly.
        tmp = torch.empty_like(src)
        _conj_into(src, tmp, negate_real, negate_imag)
        if tmp.numel() != 0:
            _flaggems_copy_(out, tmp)
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
    """Spell a dtype the way ATen's `out=` dtype check does (TypeName, not ScalarType)."""
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


# ---------------------------------------------------------------------------
# Generic-path module alias (vendor injection).
#
# SpecOpRegistrar wires ``flag_gems._conj_physical`` and the aten-registered
# entry to this module, but the accuracy tests and the perf benchmark resolve
# the op through the generic package attribute
# (``flag_gems.ops._conj_physical(...)``), which keeps pointing at the generic
# implementation -- a complex128 or conj-bit input then reaches the fp64-less
# Triton kernels / the unresolved-conj view and fails, even though the
# dispatcher path already runs this backend copy.  Publish this module under
# the generic path (the same pattern as ``_linalg_det`` /
# ``quantized_max_pool2d``) and rebind the package attribute so the module
# under test IS the implementation that runs.
#
# The fp8 side of the comparison gate (torch.isclose aborts on fp8 gcu
# operands) is handled by the fp8 route in add.py's isclose shim, which this
# backend installs for every process.
# ---------------------------------------------------------------------------
_GENERIC_CONJ_PHYSICAL = flag_gems.ops._conj_physical
_GENERIC_CONJ_PHYSICAL_OUT = flag_gems.ops._conj_physical_out

sys.modules["flag_gems.ops._conj_physical"] = sys.modules[__name__]
flag_gems.ops._conj_physical = _conj_physical
flag_gems.ops._conj_physical_out = _conj_physical_out
