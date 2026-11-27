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

"""gcu400 isreal: complex inputs through a local int32 bit-domain kernel.

The host-function policy (user ruling, 2026-11-12) forbids routing the
computation to aten, so the previous override (``torch.imag(input) == 0``,
which delegated to torch_gcu's native imag/eq kernels) is replaced by a
Triton kernel owned by this backend.  The generic pair-load kernel cannot
be reused verbatim here:

* its complex128 path lowers to fp64 loads, which fail to compile on this
  backend ("Pipeline run failed: PassManager execution failed" -- P04 fp64
  family; gcu400 has no fp64 arithmetic and no f64 vector memory ops);
* the generic host calls ``view_as_real``, which rejects a lazy conjugate,
  and ``resolve_conj`` is broken on torch_gcu (the result still reports
  ``is_conj() == True``), so the conj path dies before any kernel runs.

On this backend complex128 is complex64 in disguise (measured, see
flaggems_isreal.md in compiler_problems/P21): every creation path
(``torch.complex`` on device parts, casts, ``copy_``, ``zeros``/``ones``,
H2D transfer) stores fp32 (real, imag) pairs at an effective 8 bytes per
element, while the dtype label, ``element_size()`` and -- crucially --
``data_ptr()`` keep the 16B claim.  torch_gcu's aten reads use the
effective 8B itemsize, so a dtype-converting ``view()`` (whose data_ptr
is scaled by the claimed itemsize) points at the wrong words as soon as
``storage_offset != 0``.  The kernel therefore never uses ``view``;
instead the raw storage is rebound into an int64 carrier whose offset is
the effective one qword per element (``torch.empty(0).set_(...)`` -- pure
metadata, no kernel).  This also drops the lazy conj/neg bits, which
``view(dtype)`` rejects outright; those bits only flip the sign of the
imaginary part, which cannot move a zero, so the predicate does not
care.

The kernel then tests the imaginary bits in the integer domain: each
int64 qword packs one element's fp32 (real, imag) pair with the
imaginary half in the high 32 bits, so ``(q >> 32) & 0x7fffffff == 0``
is bit-exact against what aten's value-level ComplexFloat replacement
(and hence the CPU reference, which upcasts the very same fp32 values)
computes, for both complex64 and complex128.

The previous revision read the imaginary word with stride-2 int32 loads
(``words_ptr + 2 * off + 1``).  That lowering truncates the byte offset
at 2^31: every element from 2^28 onward silently reads garbage, so the
512M-element core benchmark shapes were computed wrong (half the output
False, True count halved; CPU-side full compare, P17 >=2^31 offset
truncation family -- see
compiler_problems/P17_int64_narrow_and_2p31_offset/sources/flaggems_enflame_isreal.md).
Fully contiguous 8-byte qword loads do not truncate, and measured faster
anyway (2.15ms vs 3.66ms at 512M, 73us vs 118us at 16M): one 8B request
per element halves the DMA request count of two 4B-strided words.

Real dtypes are unconditionally True, so they only need a bool fill; the
generic fill kernel launches one program per 4096 elements with no cap,
which exceeds the hardware grid.x limit of 65535 on the 512M-element
core benchmark shape (P19).  Both local kernels therefore use the
backend-standard grid-stride pattern with the grid capped at
``NUM_SIPS * 2`` programs, which is shape-independent.

Tuning (GCU400/Libra, kernel-mode do_bench w1000/r100, grid cap 48):
* complex: BLOCK 8192 / num_warps 4 at >= 16384 elements (per-element
  lane count stays at 32 -- >= 64 elements per lane falls into a ~12x
  slower strided-load path), BLOCK 2048 below that (launch-floor sized);
  full blocks are stored unmasked, the single tail block is masked;
* fill: BLOCK 16384 / num_warps 2 at >= 65536 elements, BLOCK 8192
  below that, with the per-block store base advanced in int64.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.isreal import _REAL_DTYPES as _GENERIC_REAL_DTYPES
from flag_gems.ops.isreal import isreal as _generic_isreal
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

__all__ = ["isreal"]

NUM_SIPS = 24

_COMPLEX_DTYPES = (torch.complex64, torch.complex128)


def _launch(kernel, grid_size, *args, **kwargs):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; converterResult=-2 family, see P02); retrying the
    # launch recompiles and succeeds (add.py / sinh.py precedent).
    kwargs.setdefault("num_warps", 4)
    kwargs.setdefault("num_stages", 1)
    for attempt in range(3):
        try:
            kernel[(grid_size,)](*args, **kwargs)
            return
        except Exception:
            if attempt == 2:
                raise
            logger.warning("isreal kernel launch failed (attempt %d)", attempt + 1)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def isreal_fill_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    """Fill True for every element; real dtypes are always "real".

    Full blocks store unmasked (wider store vectors); the single partial
    tail block, if any, is stored masked by program 0.  The per-block
    base advances in int64 so byte offsets beyond 2^31 stay exact.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    full_blocks = n_elements // BLOCK
    tail = n_elements - full_blocks * BLOCK
    for block_id in tl.range(pid, full_blocks, num_pids):
        base = out_ptr + (block_id * BLOCK).to(tl.int64)
        tl.store(base + arange, 1)
    if pid == 0 and tail > 0:
        base = out_ptr + (full_blocks * BLOCK).to(tl.int64)
        tl.store(base + arange, 1, mask=arange < tail)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def isreal_complex_bits_kernel(qwords_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """Store ``imag == 0`` for each complex element from its int64 qword.

    ``qwords_ptr`` addresses the storage as int64 words, one per complex
    element (the effective fp32 (real, imag) pair; complex128 is stored
    the same way on this backend).  The imaginary fp32 word sits in the
    high 32 bits, and the sign bit is masked off because IEEE +-0.0 both
    satisfy ``== 0``, while every other fp32 value (denormals, inf, NaN)
    keeps nonzero payload bits, so the integer-domain test is exact.
    Loads are fully contiguous 8B words: the previous stride-2 int32
    lowering truncated byte offsets at 2^31 (P17), silently corrupting
    every element from 2^28 onward.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    full_blocks = n_elements // BLOCK
    tail = n_elements - full_blocks * BLOCK
    for block_id in tl.range(pid, full_blocks, num_pids):
        off = block_id * BLOCK + arange
        q = tl.load(qwords_ptr + off)
        is_real = ((q >> 32) & 0x7FFFFFFF) == 0
        tl.store(out_ptr + off, is_real)
    if pid == 0 and tail > 0:
        off = full_blocks * BLOCK + arange
        mask = arange < tail
        q = tl.load(qwords_ptr + off, mask=mask, other=1)
        is_real = ((q >> 32) & 0x7FFFFFFF) == 0
        tl.store(out_ptr + off, is_real, mask=mask)


def _qwords_view(input):
    """Rebind a contiguous complex tensor's storage as flat int64 words.

    The offset is the effective one 8-byte qword per element, which is
    where torch_gcu's aten reads actually land; ``data_ptr()``-based
    dtype views would use the claimed (complex128: doubled) itemsize and
    read the wrong words for any non-zero storage offset.  The rebind
    also strips the lazy conj/neg bits, which block ``view(dtype)``
    entirely (and cannot be resolved: torch_gcu's ``resolve_conj``
    leaves the bit set).
    """
    carrier = torch.empty(0, dtype=torch.int64, device=input.device)
    carrier.set_(
        input.untyped_storage(),
        input.storage_offset(),
        (input.numel(),),
        (1,),
    )
    return carrier


def _isreal_real(input):
    n_elements = input.numel()
    out = torch.empty(input.size(), dtype=torch.bool, device=input.device)
    if n_elements == 0:
        return out
    BLOCK = 16384 if n_elements >= 65536 else 8192
    grid_size = min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(input.device):
        _launch(
            isreal_fill_kernel, grid_size, out, n_elements, BLOCK=BLOCK, num_warps=2
        )
    return out


def _isreal_complex(input):
    n_elements = input.numel()
    out = torch.empty(input.size(), dtype=torch.bool, device=input.device)
    if n_elements == 0:
        return out
    if not input.is_contiguous():
        input = input.contiguous()
    qwords = _qwords_view(input)
    BLOCK = 2048 if n_elements < 16384 else 8192
    grid_size = min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(input.device):
        _launch(
            isreal_complex_bits_kernel, grid_size, qwords, out, n_elements, BLOCK=BLOCK
        )
    return out


def isreal(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME ISREAL")
    if isinstance(input, torch.Tensor) and input.device.type == "gcu":
        if input.dtype in _COMPLEX_DTYPES:
            return _isreal_complex(input)
        if input.dtype in _GENERIC_REAL_DTYPES:
            return _isreal_real(input)
        # Unsupported dtypes (e.g. complex32) and non-gcu inputs keep the
        # generic dtype validation and error contract.
    return _generic_isreal(input)
