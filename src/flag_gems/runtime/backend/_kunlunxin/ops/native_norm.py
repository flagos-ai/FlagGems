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

"""Kunlunxin XPU3 override for ``aten::native_norm(Tensor self, Scalar p=2)``.

The generic implementation delegates every non-L2 order to
``flag_gems.ops.vector_norm``, whose general-p stage-1 kernel calls
``tl_extra_shim.pow``. On this backend that lowers to an external symbol that
the XPU linker rejects (``ld.lld: error: undefined symbol: Unsupported``), so
``native_norm`` fails to compile for every ``p`` other than 2.  The generic L2
path also only reaches ~0.04-0.15x of torch on the large shapes because its
two-kernel scheme uses a tiny masked tile.

This override performs the full-tensor reduction with the proven two-stage
pattern from ``_kunlunxin/ops/l1_loss.py``:

* unmasked full-lane stage-1 tiles (contiguous block DMA) with a single
  in-bounds window program for the residue, so no lane ever reads out of range;
* an fp32 (fp64 for fp64 input) stage-2 reduction launched with
  ``buffer_size_limit=2048`` -- the largest exact ``tl.sum`` tile here; a
  32768-lane single-tile ``tl.sum`` without it silently returned ~0.

General orders use ``exp(p * log(|x|))`` instead of ``pow`` so they compile on
XPU; ``p == 0`` counts non-zeros and ``p == +-inf`` take the max/min of ``|x|``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Reduction modes (constexpr switch in the kernels).
_M_COUNT = 0  # p == 0: number of non-zero elements
_M_SUM_ABS = 1  # p == 1: sum |x|
_M_SUM_SQ = 2  # p == 2: sum x^2
_M_SUM_POW = 3  # general finite p: sum |x|^p via exp/log
_M_MAX = 4  # p == +inf: max |x|
_M_MIN = 5  # p == -inf: min |x|

# Largest unmasked stage-1 tile verified exact on XPU3 with
# buffer_size_limit=2048 for a load + reduce pattern (see l1_loss / mse_loss).
_MAX_TILE = 131072
# Smallest stage-1 tile used by the parallel path.
_MIN_TILE = 8192
# Target number of stage-1 programs; also caps the stage-2 tile at 32768.
_TARGET_PROGRAMS = 128
_MAX_MID = 32768
# Below this numel a single program covers the whole tensor (launch-bound).
_SMALL_MAX = 16384
# Exact tl.sum tile ceiling on this backend.
_BUFFER_SIZE_LIMIT = 2048


def _acc_torch_dtype(dtype):
    return torch.float64 if dtype == torch.float64 else torch.float32


def _mode_for(p):
    pf = float(p)
    if pf == 0.0:
        return _M_COUNT, pf
    if pf == 1.0:
        return _M_SUM_ABS, pf
    if pf == 2.0:
        return _M_SUM_SQ, pf
    if pf == float("inf"):
        return _M_MAX, pf
    if pf == float("-inf"):
        return _M_MIN, pf
    return _M_SUM_POW, pf


@libentry()
@triton.jit
def _nn_stage1_kernel(x, mid, ord, BLOCK: tl.constexpr, MODE: tl.constexpr):
    # Full lane blocks only (grid = M // BLOCK): every lane is in-bounds, so the
    # load is plain/unconditional. Masked stage-1 tiles are a slow non-coalesced
    # path here, and out-of-range lanes must never enter the reduction.
    pid = ext.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if x.type.element_ty == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    v = tl.load(x + off).to(acc_dtype)
    a = tl.abs(v)
    if MODE == 0:
        r = tl.sum((v != 0.0).to(acc_dtype))
    elif MODE == 1:
        r = tl.sum(a)
    elif MODE == 2:
        r = tl.sum(v * v)
    elif MODE == 3:
        pos = a != 0.0
        lg = tl.log(tl.where(pos, a, 1.0))
        r = tl.sum(tl.where(pos, tl.exp(ord * lg), 0.0))
    elif MODE == 4:
        r = tl.max(a)
    else:
        r = tl.min(a)
    tl.store(mid + pid, r)


@libentry()
@triton.jit
def _nn_tail_kernel(
    x,
    mid,
    base,
    n_keep,
    g,
    ord,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    # Reduces the in-bounds window [M - BLOCK, M). Its trailing `n_keep` lanes
    # are the elements past the full blocks; the leading (BLOCK - n_keep) lanes
    # were already counted by _nn_stage1_kernel and are neutralised BY VALUE
    # (tl.where selects values, never addresses), so this never reads out of
    # range and is exact for any tensor.
    lane = tl.arange(0, BLOCK)
    if x.type.element_ty == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    keep = lane >= BLOCK - n_keep
    v = tl.load(x + base + lane).to(acc_dtype)
    a = tl.abs(v)
    if MODE == 0:
        r = tl.sum(tl.where(keep, (v != 0.0).to(acc_dtype), 0.0))
    elif MODE == 1:
        r = tl.sum(tl.where(keep, a, 0.0))
    elif MODE == 2:
        r = tl.sum(tl.where(keep, v * v, 0.0))
    elif MODE == 3:
        pos = a != 0.0
        lg = tl.log(tl.where(pos, a, 1.0))
        contrib = tl.where(pos, tl.exp(ord * lg), 0.0)
        r = tl.sum(tl.where(keep, contrib, 0.0))
    elif MODE == 4:
        r = tl.max(tl.where(keep, a, 0.0))
    else:
        r = tl.min(tl.where(keep, a, float("inf")))
    tl.store(mid + g, r)


@libentry()
@triton.jit
def _nn_small_kernel(
    x,
    out,
    M,
    ord,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
    MODE: tl.constexpr,
):
    # Single-program whole-tensor reduction for tiny inputs. When M is a power
    # of two the load is unmasked; otherwise a value-predicated masked load is
    # used (single `other`, then tl.where), which is safe on this backend and
    # where the launch cost dominates anyway.
    off = tl.arange(0, BLOCK)
    if x.type.element_ty == tl.float64:
        acc_dtype = tl.float64
        other = 0.0
    else:
        acc_dtype = tl.float32
        other = 0.0
    if MASKED:
        m = off < M
        v = tl.load(x + off, mask=m, other=other).to(acc_dtype)
    else:
        v = tl.load(x + off).to(acc_dtype)
    a = tl.abs(v)
    if MODE == 0:
        r = tl.sum((v != 0.0).to(acc_dtype))
    elif MODE == 1:
        r = tl.sum(a)
    elif MODE == 2:
        r = tl.sum(v * v)
    elif MODE == 3:
        pos = a != 0.0
        lg = tl.log(tl.where(pos, a, 1.0))
        r = tl.sum(tl.where(pos, tl.exp(ord * lg), 0.0))
    elif MODE == 4:
        r = tl.max(a)
    else:
        # Masked lanes load as |0| = 0, which would poison a min; neutralise
        # them by value with the min identity.
        if MASKED:
            r = tl.min(tl.where(off < M, a, float("inf")))
        else:
            r = tl.min(a)
    if MODE == 2:
        r = tl.sqrt(r)
    elif MODE == 3:
        r = tl.exp(tl.log(r) / ord)
    tl.store(out, r.to(out.type.element_ty))


@libentry()
@triton.jit
def _nn_final_kernel(
    mid,
    out,
    mid_size,
    ord,
    BLOCK_MID: tl.constexpr,
    MODE: tl.constexpr,
):
    # `mid` is allocated BLOCK_MID slots but only the first `mid_size` hold
    # partials; padding lanes are dropped BY VALUE (never an address mask).
    off = tl.arange(0, BLOCK_MID)
    if mid.type.element_ty == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    mv = tl.load(mid + off).to(acc_dtype)
    if MODE == 4:
        mv = tl.where(off < mid_size, mv, 0.0)
        r = tl.max(mv)
    elif MODE == 5:
        mv = tl.where(off < mid_size, mv, float("inf"))
        r = tl.min(mv)
    else:
        mv = tl.where(off < mid_size, mv, 0.0)
        r = tl.sum(mv)
        if MODE == 2:
            r = tl.sqrt(r)
        elif MODE == 3:
            r = tl.exp(tl.log(r) / ord)
    tl.store(out, r.to(out.type.element_ty))


def _pick_block(m):
    # Largest tile in [MIN_TILE, MAX_TILE] that still yields at least
    # _TARGET_PROGRAMS full programs; if even MIN_TILE cannot, use MIN_TILE and
    # let the residue window absorb the remainder.
    block = _MAX_TILE
    while block > _MIN_TILE and (m // block) < _TARGET_PROGRAMS:
        block //= 2
    if (m // block) == 0:
        block = _MIN_TILE
    # Never let the stage-2 tile exceed the exact tl.sum ceiling.
    if triton.cdiv(m, block) > _MAX_MID:
        block = triton.next_power_of_2(triton.cdiv(m, _MAX_MID))
    return block


def native_norm(self, p=2):
    """Compute the p-norm over all elements of ``self`` (0-D output)."""
    logger.debug("GEMS_KUNLUNXIN NATIVE_NORM")

    if self.dtype not in [torch.float16, torch.float32, torch.bfloat16, torch.float64]:
        raise NotImplementedError(f"native_norm not implemented for {self.dtype}")

    x = self.contiguous()
    M = x.numel()
    mode, ord = _mode_for(p)

    out = torch.empty((), dtype=x.dtype, device=x.device)

    if M == 0:
        # Full-tensor norm of an empty tensor: 0 for the usual orders.
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        return zero

    with torch_device_fn.device(x.device):
        if M <= _SMALL_MAX:
            block = triton.next_power_of_2(M)
            masked = block != M
            _nn_small_kernel[(1,)](
                x,
                out,
                M,
                ord,
                BLOCK=block,
                MASKED=masked,
                MODE=mode,
                buffer_size_limit=_BUFFER_SIZE_LIMIT,
            )
            return out

        block = _pick_block(M)
        n_full = M // block
        rem = M - n_full * block
        mid_size = n_full + (1 if rem else 0)
        block_mid = triton.next_power_of_2(mid_size)

        mid = torch.empty((block_mid,), dtype=_acc_torch_dtype(x.dtype), device=x.device)

        if n_full:
            _nn_stage1_kernel[(n_full,)](
                x,
                mid,
                ord,
                BLOCK=block,
                MODE=mode,
                buffer_size_limit=_BUFFER_SIZE_LIMIT,
            )
        if rem:
            _nn_tail_kernel[(1,)](
                x,
                mid,
                M - block,
                rem,
                n_full,
                ord,
                BLOCK=block,
                MODE=mode,
                buffer_size_limit=_BUFFER_SIZE_LIMIT,
            )
        if mid_size == 1:
            # Single partial already holds the fully reduced value; apply the
            # final root/transform on the host-free path via the final kernel.
            _nn_final_kernel[(1,)](
                mid,
                out,
                mid_size,
                ord,
                BLOCK_MID=1,
                MODE=mode,
                buffer_size_limit=_BUFFER_SIZE_LIMIT,
            )
            return out
        _nn_final_kernel[(1,)](
            mid,
            out,
            mid_size,
            ord,
            BLOCK_MID=block_mid,
            MODE=mode,
            buffer_size_limit=_BUFFER_SIZE_LIMIT,
        )
    return out
