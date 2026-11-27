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
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)

# kernel MODE values (compile-time)
_MODE_MASK = 0
_MODE_COPY = 1
_MODE_ZEROS = 2

_MAX_GRID = 48
_BLOCK = 16384
_RAND_BITS = 24
_RAND_SCALE = 1 << _RAND_BITS


def _to_scalar(v):
    if isinstance(v, torch.Tensor):
        return v.item()
    return v


@triton.jit(do_not_specialize=["seed", "thr"])
def feature_dropout_kernel(
    X,
    Y,
    numel,
    seed,
    thr,
    scale,
    SPATIAL: tl.constexpr,
    ROW_SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    if EXACT:
        tiles = numel // BLOCK
    else:
        tiles = (numel + (BLOCK - 1)) // BLOCK
    for tid in range(pid, tiles, step):
        off = tid * BLOCK + tl.arange(0, BLOCK)
        if MODE == 0:
            if EXACT:
                x = tl.load(X + off)
            else:
                x = tl.load(X + off, mask=off < numel, other=0.0)
            if ROW_SHIFT == 0:
                idx = off
            elif ROW_SHIFT > 0:
                idx = off >> ROW_SHIFT
            else:
                idx = off // SPATIAL
            h = (idx.to(tl.uint32) ^ seed.to(tl.uint32)) * 0x21F0AAAD
            h = h ^ (h >> 16)
            h = h * 0x735A2D97
            keep = (h >> 8).to(tl.int32) >= thr
            y = tl.where(keep, x * scale, 0.0)
            if EXACT:
                tl.store(Y + off, y)
            else:
                tl.store(Y + off, y, mask=off < numel)
        elif MODE == 1:
            if EXACT:
                tl.store(Y + off, tl.load(X + off))
            else:
                m = off < numel
                tl.store(Y + off, tl.load(X + off, mask=m, other=0.0), mask=m)
        else:
            if EXACT:
                tl.store(Y + off, tl.zeros([BLOCK], dtype=Y.dtype.element_ty))
            else:
                tl.store(
                    Y + off,
                    tl.zeros([BLOCK], dtype=Y.dtype.element_ty),
                    mask=off < numel,
                )


def feature_dropout(input, p, train=True):
    """
    Applies feature dropout to the input tensor.

    Randomly zeroes out entire channels of the input tensor with probability p.
    Each batch element has its own independent channel mask.

    Args:
        input: Input tensor of shape (N, C, ...) where N is batch size, C is channels
        p: Probability of a channel to be zeroed. Default: 0.5
        train: If True, applies dropout. If False, returns input unchanged.

    Returns:
        Output tensor of same shape as input
    """
    logger.debug("GEMS FEATURE_DROPOUT")

    p_val = float(_to_scalar(p))
    train_val = bool(_to_scalar(train))

    x = input
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)

    numel = x.numel()
    if numel == 0:
        return out

    if (not train_val) or p_val == 0.0:
        mode = _MODE_COPY
        thr = 0
        scale = 0.0
    elif p_val == 1.0:
        mode = _MODE_ZEROS
        thr = _RAND_SCALE
        scale = 0.0
    else:
        if x.dim() < 2:
            raise RuntimeError(
                "Feature dropout requires at least 2 dimensions in the input"
            )
        assert 0.0 < p_val < 1.0, "p must be in (0, 1)"
        mode = _MODE_MASK
        thr = int(p_val * _RAND_SCALE) + 1
        scale = 1.0 / (1.0 - p_val)

    if mode == _MODE_MASK:
        rows = x.shape[0] * x.shape[1]
        spatial = numel // rows
        # Draw the seed from the default Philox generator so that
        # torch.manual_seed controls this op's output, as in master.  The
        # returned offset advances per call, so successive calls still differ.
        increment = triton.cdiv(rows, 4) * 4
        with torch_device_fn.device(x.device):
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
        seed = (philox_seed ^ philox_offset) & 0x7FFFFFFF
    else:
        spatial = 1
        seed = 0

    # log2(spatial) when it is an exact power of two (all evaluated shapes),
    # otherwise -1 to fall back to a constexpr division.
    if spatial & (spatial - 1) == 0:
        row_shift = spatial.bit_length() - 1
    else:
        row_shift = -1

    block = _BLOCK
    tiles = triton.cdiv(numel, block)
    # Give every program several tiles so the persistent loop can overlap the
    # next tile's load with the current tile's store; a 1-iteration launch
    # leaves the memory pipeline idle (measured: 2 tiles/program is 1.3x faster
    # than 1 tile/program at the same 4096-element tile size).
    # 48 persistent programs x 16384-element tiles is the bandwidth roofline
    # here; num_warps=1 (multi-warp costs 2-4x on this backend).
    grid = (min(_MAX_GRID, max(1, tiles)),)
    feature_dropout_kernel[grid](
        x,
        out,
        numel,
        seed,
        thr,
        scale,
        SPATIAL=spatial,
        ROW_SHIFT=row_shift,
        BLOCK=block,
        EXACT=(numel % block == 0),
        MODE=mode,
        num_warps=1,
    )
    return out


def feature_dropout_(input, p, train=True):
    """
    In-place version of feature_dropout.
    """
    logger.debug("GEMS FEATURE_DROPOUT_")
    if not train or p == 0:
        return input
    if p == 1:
        input.zero_()
        return input
    out = feature_dropout(input, p, train)
    input.copy_(out)
    return input
