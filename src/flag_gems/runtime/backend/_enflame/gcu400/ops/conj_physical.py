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

"""gcu400 ``conj_physical``: word-stream conjugation, conj_copy family launch.

The generic pair kernel strides each element with two half-utilized loads
(``base = offsets * 2``), autotunes through the NVIDIA default config table
(gcu400 has no ``conj_physical`` entry, so every new ``n_elements`` re-benches
six BLOCK sizes), and its host resolves the lazy conj bit with an extra
``resolve_conj`` materialization.  On the core shapes that combination is
~3x slower than the backend-standard flat word stream (baseline 110.9us vs
33.3us at (2048, 2048) complex64).

This module mirrors ``ops/_conj_physical.py``: the complex storage is rebound
into a flat fp32 word carrier (``set_`` -- pure metadata, strips the lazy
conj/neg bits, and lands where torch_gcu's aten reads land for complex64 and
complex128 alike), the conj/neg bits fold into constexpr lane-negation flags
(one launch, no resolve pass), and the launch shape follows a per-op retune
of the conj_copy family table (2026-11-16; see the launch-table comment):
fixed grid + in-kernel stride, grid cap 48 (2x24 SIPs; B32768's 128KB fp32
tile stays below the 448KB DSM budget).

Host policy (user ruling, 2026-11-12): allocation (``empty_like``) and other
FlagGems ops only; the non-contiguous path materializes through the
backend-local ``copy_`` op, and exotic inputs (complex32, non-gcu devices,
beyond the int32 word domain) keep the generic contract via the captured
generic entry.  A real input is handed straight back: that is the public
op's contract (the underscore ``_conj_physical`` is the one that must not
alias), and the generic implementation does the same.
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

__all__ = ["conj_physical"]

# Launch table retuned 2026-11-16 on the official core shapes for this kernel
# (do_bench(median) w1000/r100, 3 interleaved same-window rounds per band;
# band crossovers probed at 2^23/2^24/2^25 words -- twin family tiers must not
# be copied wholesale, ops/resolve_conj.py keeps its own 2^21 boundary):
#   n_words <= 16K:   B1024/nw4, 3.1us @512w, 3.3us @4Kw (multi-CTA launch
#                     floor; B2048/B4096 same-window WITHIN-NOISE);
#   16K < n < 2^24:   B8192/nw4, 7.3us @1M w, 27.2us @8.4M w, 48.8us @2^24 w
#                     (B32768/nw4 +8% @1M, +2.5% @8.4M; B16384/nw4 +21% --
#                     the resolve_conj twin's wide tier is the bad one here);
#   n_words >= 2^24:  B32768/nw2, 48.6us @2^24 w (tie with B8192),
#                     92.7us @33.5M w vs B8192 93.5-94.6 (+1.4%).
# num_stages 1 vs default and nw2 vs nw4 at B32768 measured WITHIN-NOISE.
# A fixed modest program count with a grid-stride loop keeps the grid
# size-independent (P19 grid.x cap safe); cap 48 = 2x24 SIPs dominates 96/192
# on every measured band (family table), and B32768's 128KB fp32 tile stays
# below the 448KB DSM budget.
_WORD_GRID_CAP = 48
# Masked tail lanes evaluate tile_id * BLOCK + arange, reaching up to
# tile_count * BLOCK - 1: keep n_words + BLOCK - 1 < 2^31 so the flat int32
# offsets cannot wrap negative on the tail tile.
_I32_SAFE_WORDS = (1 << 31) - (1 << 17)

_COMPLEX_WORD_DTYPES = (torch.complex64, torch.complex128)


@libentry()
@triton.jit(do_not_specialize=["n_words"])
def conj_physical_words_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    BLOCK: tl.constexpr,
    NEGATE_REAL: tl.constexpr,
    NEGATE_IMAG: tl.constexpr,
):
    """Conjugate an interleaved fp32 (real, imag, ...) word stream.

    The stream is the effective storage of a contiguous complex64 or
    complex128 tensor (both store fp32 pairs at 8 bytes per element on this
    backend).  Odd words are imaginary parts; IEEE negation of an fp32 value
    is an exact sign-bit flip, so ``-value`` is bit-exact against aten for
    every value including +-0.0 and NaN payloads.  The flag combinations are
    constexpr, so only the needed variant is compiled.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    tile_count = tl.cdiv(n_words, BLOCK)
    for tile_id in tl.range(pid, tile_count, num_pids):
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


def _choose_word_launch(n_words):
    """(BLOCK, num_warps) per flat word-stream band (table above)."""
    if n_words <= 16384:
        return 1024, 4
    if n_words < (1 << 24):
        return 8192, 4
    return 32768, 2


def _words_view(tensor: torch.Tensor) -> torch.Tensor:
    """Rebind a contiguous complex tensor's storage as a flat fp32 stream.

    The offset is scaled by the effective two words per element, which is
    where torch_gcu's aten reads actually land (complex128's claimed 16B
    itemsize would double-count; measured in compiler_problems/P21).  The
    rebind is pure metadata and strips the lazy conj/neg bits that block the
    ``view`` family entirely (and that ``resolve_conj`` cannot clear).
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

    ``conj_physical`` returns conj(logical(input)).  With the conj bit A and
    the neg bit N the logical value is conj_A(neg_N(stored)), so the output is
    (r, -i) / (r, i) / (-r, i) / (-r, -i) for (A,N) = (F,F)/(T,F)/(F,T)/(T,T)
    respectively: negate real iff N, negate imag iff A == N.
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
    block, num_warps = _choose_word_launch(n_words)
    grid = (min(triton.cdiv(n_words, block), _WORD_GRID_CAP),)
    src_words = _words_view(src)
    dst_words = _words_view(dst)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (abs.py / rsqrt.py /
    # _conj_copy.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(src.device):
                conj_physical_words_kernel[grid](
                    src_words,
                    dst_words,
                    n_words,
                    BLOCK=block,
                    NEGATE_REAL=negate_real,
                    NEGATE_IMAG=negate_imag,
                    num_warps=num_warps,
                )
            return
        except Exception:
            if attempt == 2:
                raise
            logger.warning(
                "conj_physical words kernel launch failed (attempt %d)",
                attempt + 1,
            )


def _handle_locally(input: torch.Tensor) -> bool:
    """This backend owns complex64/128 complex inputs; the rest stays generic.

    complex32, non-gcu inputs and word counts beyond the int32 flat domain
    keep the generic dtype validation and error contract (the generic pair
    kernel breaks on those here for the same P04/P21 reasons as the
    underscore op).
    """
    if not (isinstance(input, torch.Tensor) and input.device.type == "gcu"):
        return False
    return input.dtype in _COMPLEX_WORD_DTYPES and (input.numel() * 2 < _I32_SAFE_WORDS)


def conj_physical(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME CONJ_PHYSICAL")
    if not input.is_complex():
        # Public-op contract: conjugation is the identity for a real input
        # and the public op may hand the tensor straight back (the generic
        # implementation and ATen's frontend wrapper both do).
        return input

    if not _handle_locally(input):
        return _GENERIC_CONJ_PHYSICAL(input)

    # Fold the lazy conj/neg bits into the kernel flags so the logical value
    # is conjugated in a single pass, without a resolve_conj materialization.
    negate_real, negate_imag = _conj_flags(input)
    src = input if input.is_contiguous() else _contiguous_source(input)
    output = torch.empty_like(src)
    _conj_into(src, output, negate_real, negate_imag)
    return output


# ---------------------------------------------------------------------------
# Generic-path module alias (vendor injection).
#
# SpecOpRegistrar rebinds the top-level ``flag_gems.conj_physical`` (the
# accuracy tests and the perf benchmark resolve through that attribute), but
# ``flag_gems.ops.conj_physical`` keeps pointing at the generic
# implementation -- a complex128 or conj-bit input would then reach the
# fp64-less pair kernel / the unresolved-conj view and fail.  Publish this
# module under the generic path (the same pattern as ``_conj_physical.py``)
# so every entry resolves to the implementation that runs.
# ---------------------------------------------------------------------------
_GENERIC_CONJ_PHYSICAL = flag_gems.ops.conj_physical

sys.modules["flag_gems.ops.conj_physical"] = sys.modules[__name__]
flag_gems.ops.conj_physical = conj_physical
