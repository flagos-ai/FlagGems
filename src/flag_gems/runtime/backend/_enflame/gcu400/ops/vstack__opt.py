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

"""vstack candidate v1: linear concat kernel with BLOCK=65536 for large shapes.

Production `vstack` is just a wrapper around `cat(aligned, dim=0)`; item 21's
`cat__opt.try_fast_cat` uses `BLOCK = 16384 if max_n<65536 else 32768`.
A 30-config sweep (BLOCK x num_warps x 2D/1D grid, bit-identical to torch.cat) shows:
with single-input element count >= 2^24, `BLOCK=65536 / nw4` is steadily better than 32768:

    fp16/bf16 (4096,4096) / (64,512,512)  0.1128 -> 0.0891  1.27x
    fp32 same shape 0.1728 -> 0.1577 1.10x

small shapes (64,64) differ by only 0.0037~0.0044ms across configs (1.05~1.19x,
noise level), so **take over only large shapes**; small shapes and all boundaries
(>4 inputs, non-contiguous, dtype/device mismatch, float8, t0.ndim<2,
shape differs outside cat dim) delegate to production `vstack`, guaranteeing full correctness coverage
without introducing p10 jitter regressions.

Reuse `cat__opt`'s `_can_fast_path` boundary checks and `_linear_cat_kernel`,
without changing cat.py / cat__opt.py (to avoid affecting the closed item 21 concat and
item 30 cat_out), only choosing a larger BLOCK per operator.
"""

import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._enflame.gcu400.ops.cat import cat as _cat
from flag_gems.runtime.backend._enflame.gcu400.ops.cat__opt import (
    _can_fast_path,
    _linear_cat_kernel,
)

__all__ = ["vstack"]

logger = logging.getLogger(__name__)

_LARGE = 1 << 20
_PLAIN_BLOCK = 65536  # element lanes for the masked fallback (v1 choice)
_GS48_BLOCK = 8192  # int64 lanes per chunk -> 64KB; 4 in flight = 256KB DSM
_GS48_STAGES = 4
_GS48_GRID = 48  # one program per slot: no per-wave scheduling tax

# The external /opt/triton_gcu converter SIGSEGVs on a fraction of cold
# compiles (compiler_problems/P02, flaggems_row_stack_toolchain_flaky_segv).
# The crash is per-invocation probabilistic, leaves no Triton cache entry,
# and a fresh attempt compiles fine, so retry the launch (row_stack precedent).
_TOOLCHAIN_COMPILE_RETRIES = 3


def _launch_with_toolchain_retry(launch):
    for attempt in range(_TOOLCHAIN_COMPILE_RETRIES):
        try:
            return launch()
        except Exception as error:
            if attempt == _TOOLCHAIN_COMPILE_RETRIES - 1:
                raise
            logger.debug(
                "vstack: retrying Triton launch after toolchain failure "
                "(attempt %d): %r",
                attempt,
                error,
            )


@triton.jit
def _vstack_gs48_kernel(
    Out,
    In0,
    In1,
    In2,
    In3,
    len0,
    len1,
    len2,
    len3,
    base0,
    base1,
    base2,
    base3,
    nx,
    nt,
    BLOCK: tl.constexpr,
    NSPC: tl.constexpr,
    NSTG: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Grid-stride linear copy over 8-byte lanes with a capped fixed grid.

    NSPC programs stream all (chunk, tensor) work items round-robin so the
    DMA pipeline stays full without paying the per-wave scheduling tax of a
    cdiv(max_lanes, BLOCK)-wide 2D grid.  ``num_stages=NSTG`` keeps NSTG
    chunk loads in flight (DSM: NSTG * BLOCK * 8B = 256KB at this config,
    inside the 448KB per-slot budget).  ``EVEN`` specializes the
    all-tensors-equal, len % BLOCK == 0 case to mask-free loads/stores.
    """
    pid = tl.program_id(0)
    total = nx * nt
    for wid in tl.range(pid, total, NSPC, num_stages=NSTG):
        tid = wid // nx
        chunk = wid % nx
        if tid == 0:
            src, length, base = In0, len0, base0
        elif tid == 1:
            src, length, base = In1, len1, base1
        elif tid == 2:
            src, length, base = In2, len2, base2
        else:
            src, length, base = In3, len3, base3
        idx = chunk * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            data = tl.load(src + idx)
            tl.store(Out + base + idx, data)
        else:
            mask = idx < length
            data = tl.load(src + idx, mask=mask)
            tl.store(Out + base + idx, data, mask=mask)


def vstack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME VSTACK")
    n = len(tensors)
    if n == 0:
        raise RuntimeError("vstack expected a non-empty TensorList")

    t0 = tensors[0]
    if t0.ndim < 2:
        aligned = list(torch.atleast_2d(tensors))
    else:
        aligned = list(tensors) if not isinstance(tensors, list) else tensors

    A = list(aligned)
    # Only take over cases with "large shapes + strictly satisfying the cat fast-path boundaries"
    if A[0].numel() < _LARGE or t0.ndim < 2 or not _can_fast_path(A, 0):
        # A word-for-word identical direct call to the production vstack, with zero extra forwarding overhead
        return _cat(aligned, dim=0)

    numels = [t.numel() for t in A]
    out_shape = (sum(t.shape[0] for t in A),) + tuple(A[0].shape[1:])
    out = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)

    itemsize = A[0].element_size()
    factor = 8 // itemsize if itemsize <= 4 else 0
    if factor and all(nu % factor == 0 for nu in numels):
        # 8-byte-lane bit-exact copy (zero-copy int64 views of the flat,
        # dense row-tiled buffers; fresh/aligned allocations only).
        aligned_ptrs = all(t.data_ptr() % 8 == 0 for t in A) and out.data_ptr() % 8 == 0
        if aligned_ptrs:
            lanes = [nu // factor for nu in numels]
            lens = [lanes[j] if j < n else 0 for j in range(4)]
            bases = [0] * 4
            acc = 0
            for j in range(4):
                bases[j] = acc
                acc += lens[j]
            out_lanes = out.view(-1).view(torch.int64)
            src_lanes = [A[j].view(-1).view(torch.int64) for j in range(n)]
            src_lanes += [src_lanes[0]] * (4 - n)
            nx = triton.cdiv(max(lens), _GS48_BLOCK)
            even = len(set(lanes)) == 1 and lanes[0] % _GS48_BLOCK == 0
            _launch_with_toolchain_retry(
                lambda: _vstack_gs48_kernel[(_GS48_GRID,)](
                    out_lanes,
                    src_lanes[0],
                    src_lanes[1],
                    src_lanes[2],
                    src_lanes[3],
                    lens[0],
                    lens[1],
                    lens[2],
                    lens[3],
                    bases[0],
                    bases[1],
                    bases[2],
                    bases[3],
                    nx,
                    n,
                    BLOCK=_GS48_BLOCK,
                    NSPC=_GS48_GRID,
                    NSTG=_GS48_STAGES,
                    EVEN=even,
                    num_warps=8 if itemsize == 4 else 4,
                )
            )
            return out

    # Element copies with non-divisible / unaligned sizes: v1's masked element-lane kernel
    ns = [numels[j] if j < n else 0 for j in range(4)]
    ptrs = [A[j] if j < n else A[0] for j in range(4)]
    offs = [0] * 4
    acc = 0
    for j in range(4):
        offs[j] = acc
        acc += ns[j]
    grid = (triton.cdiv(max(ns), _PLAIN_BLOCK), n)
    _linear_cat_kernel[grid](
        out, *ptrs, *ns, *offs, NT=n, BLOCK=_PLAIN_BLOCK, num_warps=4
    )
    return out
