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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Raw (non-FlagGems) native nonzero kernel on the vendor CUDA-style key. The
# FlagGems override at the "Autograd" dispatching key intercepts torch.nonzero
# inside use_gems scopes, and the flag nonzero override is ~20x slower than
# the vendor kernel (16.5ms vs 0.8ms for a 4096^2 mask), so masked_select
# deliberately fetches the raw kernel once at import time and calls it with a
# CUDA-only key set (same pattern as ops/to.py `_XPU_NATIVE_CPU_COPY`). The
# output is a plain tensor of ascending flat indices; the compaction itself
# (the gather) is still done by the Triton kernel below.
try:
    _NATIVE_NONZERO = torch.library.get_kernel(torch.ops.aten.nonzero.default, "CUDA")
    _VENDOR_KS = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
except Exception:  # pragma: no cover
    logger.warning("masked_select: unable to acquire raw native nonzero kernel")
    _NATIVE_NONZERO = None
    _VENDOR_KS = None


def _native_nonzero(x):
    if _NATIVE_NONZERO is not None:
        return _NATIVE_NONZERO.call_boxed(_VENDOR_KS, x)
    return x.nonzero()


# The old kunlunxin kernel materialized a full global `cumsum` (int64) and did a
# data-dependent DISCRETE masked STORE (`out_ptr + out_offset`). On XPU a masked
# scatter store with per-element offsets serializes -> a fixed ~0.029 GBPS wall
# (4096^2 = 1273ms, [1024,65536] = 5090ms, speedup ~0.001) REGARDLESS of offset
# dtype or the *_SIM env flags (verified: int32/int64 and sim/no-sim identical).
#
# Fix: turn the compaction into a GATHER instead of a scatter. `nonzero` gives the
# ascending flat positions of the selected elements (torch native, fast), then a
# Triton kernel does out[j] = inp[idx[j]] -> a CONTIGUOUS store + a discrete (but
# monotonic) load. On XPU a discrete READ is ~2 orders of magnitude cheaper than a
# discrete masked STORE, so this lifts the huge shapes from ~1273/5090ms to
# ~6.4/25ms (~200x) at speedup ~0.09-0.11 (large) up to ~0.45 (tiny).


@libentry()
@triton.jit
def masked_select_gather_kernel(
    inp_ptr,
    idx_ptr,
    out_ptr,
    n_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_out
    idx = tl.load(idx_ptr + offsets, mask=mask, other=0)
    vals = tl.load(inp_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, vals, mask=mask)


def masked_select(inp, mask):
    logger.debug("GEMS_KUNLUNXIN MASKED_SELECT")

    inp_shape = tuple(inp.shape)
    mask_shape = tuple(mask.shape)

    assert broadcastable(
        inp_shape, mask_shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"
    inp, mask = torch.broadcast_tensors(inp, mask)

    inp = inp.contiguous()
    mask = mask.contiguous()

    flat_inp = inp.ravel()
    # ascending flat positions of the selected elements (row-major == the order
    # masked_select must preserve). Keep int64 as returned by nonzero: converting
    # to int32 costs more (extra full pass) than the contiguous idx-read it saves.
    idx = _native_nonzero(mask.ravel()).ravel()
    n_out = idx.numel()

    out = torch.empty(n_out, dtype=inp.dtype, device=inp.device)
    if n_out == 0:
        return out

    BLOCK_SIZE = 8192
    grid = lambda meta: (triton.cdiv(n_out, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(inp.device):
        masked_select_gather_kernel[grid](
            flat_inp, idx, out, n_out, BLOCK_SIZE=BLOCK_SIZE, num_warps=16
        )

    return out
