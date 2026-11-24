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
#
"""GCU400 subtract/subtract_: flat grid-stride tier (2026-11-18 retune).

The KernelGen thin shell used to re-export the generic ``flag_gems.ops.sub``
implementation, whose ENFLAME codegen config locks the pointwise tile grid at
min(12, num_tiles) with tile 4096 — a quarter-machine path that pinned the
official bench (kernel mode, do_bench median w1000/r100) at 0.47-0.48x of
torch native (arithmean; 1Gi fp16 13.3ms vs native 2.35ms). Repointing to the
local ``.sub`` (48-CTA pointwise) recovers ~1.00-1.02x; this module adds a
dedicated flat 1D tier measured on the 2026-11-17 toolchain (Triton GCU 4.1.6):

* mid 2^24 out-of-place: f16 41.0us / bf16 40.5us / f32 73.1us vs local-sub
  ref 48.6/49.8/80.2us (native 43.3/42.9/77.4) -> ~1.04-1.09x native.
* huge 2^30 (5-round interleaved paired arb): chunk-gs best or tied in 5/6
  cells (f16 out 2284.7 vs flat 2315.6us; f32 out 4539.6 vs 4579.0). The one
  structural split is f32 in-place at n>=2^27, where the contiguous-chunk arm
  collapses (4579.5 vs flat 4485.8us, +2.1%) — same pattern as the less_
  f32-chunk huge-band veto, so in-place f32 huge keeps the masked flat arm.
  num_stages=1 vetoed everywhere (+3.5-4.9% at 2^30).
* tiny n<2^16 stays on the local sub wrapper: probes tied at ~3.2us (the
  wrapper is already at the device-time floor; less_/square family finding).
* DSM budget (2R1W dual-load: 2*BLOCK*itemsize, probe-verified via LLIR
  gcu.dsm_memory_size): f16/bf16 CHUNK 65536 = 256KB, f32 32768 = 256KB —
  both inside the 448KB 48-way line (copysign_ family rule). BLOCK 131072
  (512KB -> 24-way lock) probed worse and is not used.
* the chunk ladder shrinks CHUNK to keep >=48 tiles below the sweep points
  so the grid stays saturated for 2^16<=n<2^24; odd n falls back to the
  masked flat arm (within 1.5% of chunk at probed sizes).
* fp16/bf16 load-cast to fp32 and store-cast back (opmath "hp" body) —
  bit-exact (torch.equal) vs the generic path across the test alphas.
* int32 offsets stay in-domain via the square-style host guard
  n < 2^31 - 2^17; complex alpha / complex / non-contiguous / mixed-dtype /
  broadcast inputs fall back to the local sub/sub_ (which routes complex to
  the add pair-domain path).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .sub import sub, sub_

logger = logging.getLogger(__name__)

_MIN_N = 1 << 16
_GRID_CAP = 48
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)
_F32_HUGE_N = 1 << 27
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _subtract_flat_kernel(X, Y, OUT, alpha, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(X + off, mask=mask).to(tl.float32)
        y = tl.load(Y + off, mask=mask).to(tl.float32)
        tl.store(OUT + off, (x - y * alpha).to(OUT.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _subtract_chunk_kernel(
    X, Y, OUT, alpha, N_total, CHUNK: tl.constexpr, CAP: tl.constexpr
):
    pid = tl.program_id(0)
    arange = tl.arange(0, CHUNK)
    for s in tl.range(pid * CHUNK, N_total, CAP * CHUNK):
        off = s + arange
        x = tl.load(X + off).to(tl.float32)
        y = tl.load(Y + off).to(tl.float32)
        tl.store(OUT + off, (x - y * alpha).to(OUT.dtype.element_ty))


def _fast_params(A, B, alpha, inplace):
    if not (
        isinstance(A, torch.Tensor)
        and isinstance(B, torch.Tensor)
        and isinstance(alpha, (int, float))
        and A.dtype in _FAST_DTYPES
        and A.dtype == B.dtype
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        return None
    n = A.numel()
    if n < _MIN_N or n >= _INT32_SAFE_LIMIT:
        return None
    chunk = 32768 if A.element_size() == 4 else 65536
    n_per_grid = n // _GRID_CAP
    if n_per_grid < chunk:
        chunk = 1 << (n_per_grid.bit_length() - 1) if n_per_grid else 1
    if A.element_size() == 4 and n >= (1 << 18):
        # f32 ladder floor: 8192-wide chunks lose ~8% to the wrapper at 2^19
        # (8.50 vs 7.94us) while 16384 wins both 2^18 (5.69) and 2^19 (7.39);
        # 16-bit keeps the raw ladder (measured winning at 2^17/2^19).
        chunk = max(chunk, 16384)
    use_chunk = (n % chunk == 0) and not (
        inplace and A.dtype == torch.float32 and n >= _F32_HUGE_N
    )
    return n, chunk, use_chunk


def _launch(A, B, OUT, alpha, n, chunk, use_chunk):
    grid = (min(triton.cdiv(n, chunk), _GRID_CAP),)
    # The external llvm converter intermittently fails on cold compiles
    # (shared-node userland flake windows); retrying recompiles and succeeds
    # (square/atan2_/sinh/rad2deg precedents for this toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(A.device):
                if use_chunk:
                    _subtract_chunk_kernel[grid](
                        A,
                        B,
                        OUT,
                        alpha,
                        n,
                        CHUNK=chunk,
                        CAP=_GRID_CAP,
                        num_warps=2,
                        num_stages=2,
                    )
                else:
                    _subtract_flat_kernel[grid](
                        A, B, OUT, alpha, n, BLOCK=chunk, num_warps=2
                    )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "subtract flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def subtract(A, B, *, alpha=1):
    """Subtracts `other`, scaled by `alpha`, from `input`."""
    logger.debug("GEMS SUBTRACT")
    params = _fast_params(A, B, alpha, inplace=False)
    if params is not None:
        n, chunk, use_chunk = params
        out = torch.empty_like(A)
        _launch(A, B, out, float(alpha), n, chunk, use_chunk)
        return out
    return sub(A, B, alpha=alpha)


def subtract_(A, B, *, alpha=1):
    """In-place subtraction: A.subtract_(B, alpha=alpha) is equivalent to A -= B * alpha"""
    logger.debug("GEMS SUBTRACT_")
    params = _fast_params(A, B, alpha, inplace=True)
    if params is not None:
        n, chunk, use_chunk = params
        _launch(A, B, A, float(alpha), n, chunk, use_chunk)
        return A
    return sub_(A, B, alpha=alpha)
