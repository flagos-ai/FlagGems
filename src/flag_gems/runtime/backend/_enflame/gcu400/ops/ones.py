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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)

# Pure-write stream tuning (GCU400, sip_count=24): grids above the 24
# schedulable SIPs never help stores, and non-multiples of 24 (e.g. 32) pay
# ~20% on mid shapes from half-speed co-residency.
_FLAT_MAX_GRID = 24
_FLAT_MIN_BLOCK = 2048
_FLAT_SMALL_MAX = 8192


@libentry()
@triton.jit
def ones_flat_kernel(
    output_ptr,
    n_units,
    high_word,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # Flat grid-stride fill capped at the 24 schedulable SIPs. Small sizes
    # launch a single CTA with an exact tile so they sit on the launch floor;
    # stream sizes tie the DRAM write wall (~3.25 TB/s, BLOCK-insensitive).
    # For int64/uint64 outputs n_units counts int32 words
    # (_int64_words_per_element words per element).
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    n_tiles = tl.cdiv(n_units, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        # torch_gcu has shipped both compact (1 int32 word/element, P17 era)
        # and full-width (2 int32 words/element) int64/uint64 storage across
        # runtime releases. Element stores of 8B interleave on the compact
        # model, while single-word fills splat into both halves
        # (0x100000001) on the full-width model, so fill through an int32
        # view with a runtime-detected word count: full-width needs
        # low=1/high=0 per element (high_word=0, parity pattern), compact
        # needs a single 1 word per element (high_word=1 collapses the
        # pattern to a constant fill).
        if (
            output_ptr.dtype.element_ty == tl.int64
            or output_ptr.dtype.element_ty == tl.uint64
        ):
            word_ptr = output_ptr.to(tl.pointer_type(tl.int32))
            val = tl.where((offsets % 2) == 1, high_word, 1)
            if EXACT:
                tl.store(word_ptr + offsets, val)
            else:
                mask = offsets < n_units
                tl.store(word_ptr + offsets, val, mask=mask)
        else:
            if EXACT:
                tl.store(output_ptr + offsets, 1.0)
            else:
                mask = offsets < n_units
                tl.store(output_ptr + offsets, 1.0, mask=mask)


def _ones_flat_block(n_units):
    if n_units <= _FLAT_SMALL_MAX:
        # Small sizes sit on the launch floor: one CTA with an exact tile.
        return max(2, triton.next_power_of_2(n_units))
    # 32Ki-unit tiles are the mid-band sweet spot (ramp-friendly) and tie the
    # DRAM write wall at stream sizes, so the smaller tile also keeps DSM
    # headroom per program.
    return min(32768, max(_FLAT_MIN_BLOCK, triton.next_power_of_2(-(-n_units // 48))))


_I64_WORDS_PER_ELEMENT = None


def _int64_words_per_element(device):
    # Probe the live int64 storage model once per process by filling 4
    # int32 words with 1 through an int32 view of a small int64 tensor and
    # reading element 0 back: full-width storage widens the word into both
    # halves (0x100000001), compact storage returns the plain word (1). The
    # 4-word fill stays inside the allocation under both models.
    global _I64_WORDS_PER_ELEMENT
    if _I64_WORDS_PER_ELEMENT is None:
        probe = torch.empty(4, dtype=torch.int64, device=device)
        _fill_ones_flat(probe, 4, 1)
        first = int(probe.reshape(-1)[0].item())
        # full-width: (1 << 32) | 1; compact: 1
        _I64_WORDS_PER_ELEMENT = 2 if first != 1 else 1
    return _I64_WORDS_PER_ELEMENT


def _ones_fill_params(n_elements, dtype, device):
    # Returns (n_units, high_word) for ones_flat_kernel: n_units counts int32
    # words for int64/uint64 outputs (runtime storage model dependent) and
    # elements for every other dtype.
    if dtype == torch.int64 or dtype == torch.uint64:
        words_per_element = _int64_words_per_element(device)
        if words_per_element == 1:
            # Compact storage: every word is a whole element (constant fill).
            return n_elements, 1
        # Full-width storage: low word 1 / high word 0 per element.
        return 2 * n_elements, 0
    return n_elements, 0


def _fill_ones_flat(out, n_units, high_word):
    block = _ones_flat_block(n_units)
    grid = (min(_FLAT_MAX_GRID, -(-n_units // block)),)
    exact = n_units % block == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level flake windows on shared nodes); retrying
    # the launch recompiles and succeeds.  3 attempts keep deterministic
    # breakage loud while riding out the flake (new_full.py / atan2_.py /
    # sinh.py precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(out.device):
                ones_flat_kernel[grid](
                    out,
                    n_units,
                    high_word,
                    BLOCK=block,
                    EXACT=exact,
                    num_warps=1,
                )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "ones flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ENFLAME ONES")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    n_elements = volume(size)
    if n_elements == 0:
        return out
    n_units, high_word = _ones_fill_params(n_elements, dtype, device)
    return _fill_ones_flat(out, n_units, high_word)
