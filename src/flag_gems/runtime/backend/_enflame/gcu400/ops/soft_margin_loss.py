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

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
BLOCK = 32768


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _soft_margin_loss_elementwise_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N_total,
    EVEN_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # reduction='none': persistent 2-read/1-write streaming kernel. EVEN_N
    # elides the per-element mask, which breaks 16-bit vectorized loads on
    # this backend (smooth_l1_loss precedent: fp16 16.7M 1.5x).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        if EVEN_N:
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask, other=0.0)
            y = tl.load(y_ptr + off, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        yf = y.to(tl.float32)
        z = -xf * yf
        vals = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(z)))
        if EVEN_N:
            tl.store(out_ptr + off, vals)
        else:
            tl.store(out_ptr + off, vals, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _soft_margin_loss_partial_sum_kernel(
    x_ptr,
    y_ptr,
    mid_ptr,
    N_total,
    REDUCTION_MEAN: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Persistent stage 1 for reduction='mean'/'sum': each program grid-strides
    # over BLOCK_SIZE tiles, accumulating the per-element losses into a [BLOCK]
    # fp32 vector and reducing with ONE tl.sum tree at the end. The per-block
    # tree plus one atomic_add per block of the previous implementation was the
    # measured bottleneck on this backend (smooth_l1_loss precedent); a plain
    # mid[pid] store removes every atomic. The mean scaling is folded into the
    # partial (each program divides its own partial by N_total, the finalize
    # kernel just sums them).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        if EVEN_N:
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask, other=0.0)
            y = tl.load(y_ptr + off, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        yf = y.to(tl.float32)
        z = -xf * yf
        vals = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(z)))
        if EVEN_N:
            acc += vals
        else:
            acc += tl.where(mask, vals, 0.0)
    total = tl.sum(acc, axis=0)
    if REDUCTION_MEAN:
        total = total / N_total
    tl.store(mid_ptr + pid, total)


@libentry()
@triton.jit(do_not_specialize=["mid_size"])
def _soft_margin_loss_sum_kernel(mid_ptr, out_ptr, mid_size, BLOCK_MID: tl.constexpr):
    # Stage 2: single program reduces the mid partials to the scalar result,
    # casting to the output dtype in-kernel.
    acc = tl.zeros([], dtype=tl.float32)
    num_blocks = (mid_size + BLOCK_MID - 1) // BLOCK_MID
    for i in tl.range(0, num_blocks):
        offset = i * BLOCK_MID + tl.arange(0, BLOCK_MID)
        mask = offset < mid_size
        vals = tl.load(mid_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
    tl.store(out_ptr, acc)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _check_tensors(input: torch.Tensor, target: torch.Tensor):
    if input.device.type != flag_gems.device or target.device.type != flag_gems.device:
        raise AssertionError(
            f"soft_margin_loss: input and target must be {flag_gems.device} tensors for Triton kernel."
        )
    if input.device != target.device:
        raise AssertionError(
            "soft_margin_loss: input and target must be on the same device."
        )
    if input.numel() != target.numel():
        raise AssertionError(
            "soft_margin_loss: input and target must have the same number of elements."
        )
    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    return input, target


def _none_config(n_elements, itemsize):
    """(BLOCK_SIZE, num_warps, grid) for the reduction='none' streaming kernel.

    DSM per program: 3 * BLOCK * itemsize (2 read streams + 1 write stream),
    same ~384KB cap as the smooth_l1_loss none path, so fp32 tops out at
    16384 and 16-bit at 65536. Ladder values from the smooth_l1_loss sweep
    (do_bench median, GCU400).
    """
    if itemsize >= 4:
        return 16384, 2, NUM_SIPS * 2
    if n_elements <= (1 << 20):
        return 8192, 2, NUM_SIPS * 2
    return 65536, 2, NUM_SIPS * 4


def _reduce_config(n_elements, itemsize):
    """(BLOCK_SIZE, num_warps, grid) for the reduction='mean'/'sum' kernel.

    The body carries a [BLOCK] fp32 vector accumulator across the persistent
    loop, so its DSM footprint is 2 * itemsize * BLOCK (loads) + 4 * BLOCK
    (accumulator); every bucket below stays well under the 917504B dsm limit
    (max here: fp32 16384 -> 196KB, 16-bit 32768 -> 196KB). Ladder follows the
    measured smooth_l1_loss sweep on this backend; exp/log add transcendental
    ALU per element, so the ladder is re-validated by the soft_margin_loss
    core benchmark (see git history of this file).
    """
    if itemsize >= 4:
        if n_elements <= 4096:
            return triton.next_power_of_2(max(n_elements, 1)), 4, NUM_SIPS * 2
        if n_elements <= (1 << 21):
            return 4096, 4, NUM_SIPS * 2
        return 16384, 2, NUM_SIPS * 2
    if n_elements <= 4096:
        return triton.next_power_of_2(max(n_elements, 1)), 4, NUM_SIPS * 2
    if n_elements <= (1 << 23):
        return 8192, 2, NUM_SIPS * 2
    return 32768, 2, NUM_SIPS * 2


def _soft_margin_loss_reduce(input, target, red, out=None):
    n_elements = input.numel()
    if n_elements == 0:
        if out is None:
            if red == 2:
                return torch.zeros((), device=input.device, dtype=input.dtype)
            return torch.full((), float("nan"), device=input.device, dtype=input.dtype)
        if out.device.type != flag_gems.device:
            raise AssertionError(
                f"soft_margin_loss_out: out must be a {flag_gems.device} tensor."
            )
        if out.device != input.device:
            raise AssertionError(
                "soft_margin_loss_out: out must be on the same device as input."
            )
        if out.numel() != 1:
            raise AssertionError(
                "soft_margin_loss_out: for reduction='sum' or 'mean', out must be a scalar tensor."
            )
        if red == 2:
            out.fill_(0)
        else:
            out.fill_(float("nan"))
        return out

    block_size, num_warps, grid_cap = _reduce_config(n_elements, input.element_size())
    grid = min(triton.cdiv(n_elements, block_size), grid_cap)

    result = out
    if result is None:
        result = torch.empty((), device=input.device, dtype=input.dtype)
    else:
        if result.device.type != flag_gems.device:
            raise AssertionError(
                f"soft_margin_loss_out: out must be a {flag_gems.device} tensor."
            )
        if result.numel() != 1:
            raise AssertionError(
                "soft_margin_loss_out: for reduction='sum' or 'mean', out must be a scalar tensor."
            )
        if result.device != input.device:
            raise AssertionError(
                "soft_margin_loss_out: out must be on the same device as input."
            )

    # With a single program the partial sum is already the final scalar, so
    # the kernel can store straight into the 0-dim result and the mid buffer
    # plus the second launch disappear.
    if grid == 1:
        mid = result
    else:
        mid = torch.empty((grid,), device=input.device, dtype=torch.float32)

    # EVEN_N (mask-free) is a stable win on 16-bit and small/mid fp32, but was
    # measured 1.35x SLOWER on large fp32 for smooth_l1_loss on this backend,
    # so keep it off there pending a soft_margin_loss A/B.
    even_n = n_elements % block_size == 0
    if input.element_size() >= 4 and n_elements > (1 << 21):
        even_n = False

    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; acosh.py / sinh.py / smooth_l1_loss.py precedent for
    # this toolchain family); retrying the launch recompiles and succeeds.
    for attempt in range(3):
        try:
            with torch_device_fn.device(input.device):
                _soft_margin_loss_partial_sum_kernel[(grid,)](
                    input,
                    target,
                    mid,
                    n_elements,
                    REDUCTION_MEAN=red == 1,
                    EVEN_N=even_n,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
            break
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "soft_margin_loss partial sum launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )

    if mid is not result:
        block_mid = min(triton.next_power_of_2(grid), BLOCK)
        with torch_device_fn.device(input.device):
            _soft_margin_loss_sum_kernel[(1,)](
                mid, result, grid, BLOCK_MID=block_mid, num_warps=4
            )
    return result


def _soft_margin_loss_none(input, target, out=None):
    n_elements = input.numel()
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.device.type != flag_gems.device:
            raise AssertionError(
                f"soft_margin_loss_out: out must be a {flag_gems.device} tensor."
            )
        if out.device != input.device:
            raise AssertionError(
                "soft_margin_loss_out: out must be on the same device as input."
            )
        if out.numel() != n_elements:
            raise AssertionError(
                "soft_margin_loss_out: for reduction='none', out must match input shape."
            )
    if n_elements > 0:
        block_size, num_warps, grid = _none_config(n_elements, input.element_size())
        grid = min(triton.cdiv(n_elements, block_size), grid)
        with torch_device_fn.device(input.device):
            _soft_margin_loss_elementwise_kernel[(grid,)](
                input,
                target,
                out,
                n_elements,
                EVEN_N=n_elements % block_size == 0,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
    return out


def soft_margin_loss(input: torch.Tensor, target: torch.Tensor, reduction="mean"):
    logger.debug("GEMS_ENFLAME SOFT_MARGIN_LOSS")
    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    if red == 0:
        return _soft_margin_loss_none(input, target)
    return _soft_margin_loss_reduce(input, target, red)


def soft_margin_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction="mean",
    out: torch.Tensor = None,
):
    logger.debug("GEMS_ENFLAME SOFT_MARGIN_LOSS_OUT")
    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    if red == 0:
        return _soft_margin_loss_none(input, target, out=out)
    return _soft_margin_loss_reduce(input, target, red, out=out)
