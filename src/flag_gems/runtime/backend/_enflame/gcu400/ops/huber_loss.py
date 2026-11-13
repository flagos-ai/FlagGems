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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Single-program fused reduction is only profitable while the whole tensor
# fits one block: past this size the lone program serializes and wastes the
# device (measured on GCU400: n=65536 single-program = 55us vs flat 5.4us).
_FUSED_LIMIT = 8192

# Flat two-stage BLOCK ladder (measured on GCU400/L600, kernel-mode do_bench
# median): the sweet spot keeps ~256KB of input bytes in flight per program
# (x and y streams), i.e. BLOCK * 2 * itemsize ~= 256KB at the cap, and
# roughly n/16 (16-bit dtypes) or n/32 (32-bit dtypes) elements per program
# below it. num_warps barely matters (<=3%); 4 matches the sibling loss ops.
_BLOCK_MIN = 4096
_IN_FLIGHT_BYTES = 262144
_GRID_X_MAX = 65535

# Past this size the per-block tl.sum tree becomes the ALU wall (fp16 at 67M
# ran at only ~34% of its memory floor). Switch to a persistent grid-stride
# kernel with a [BLOCK] vector accumulator and ONE tree per program
# (smooth_l1_loss precedent): fp16 67M 316->158us, fp32 67M 278->217us.
_VECACC_MIN = 1 << 20
_VECACC_WIDE_BLOCK = 16384
_VECACC_WIDE_GRID = 48
_VECACC_NARROW_MID_BLOCK = 16384
_VECACC_NARROW_MID_GRID = 24
_VECACC_NARROW_BIG_BLOCK = 32768
_VECACC_NARROW_BIG_GRID = 48
_VECACC_NARROW_BIG_MIN = 1 << 24


@libentry()
@triton.jit
def _huber_loss_element_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, delta, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    diff = x - y
    absdiff = tl.abs(diff)

    # loss = 0.5 * diff^2                    if |diff| <= delta
    #        delta * (|diff| - 0.5 * delta)  otherwise
    loss_quad = 0.5 * diff * diff
    loss_linear = delta * (absdiff - 0.5 * delta)
    loss = tl.where(absdiff <= delta, loss_quad, loss_linear)

    tl.store(out_ptr + offsets, loss, mask=mask)


@libentry()
@triton.jit
def _huber_loss_reduce_kernel(
    x_ptr,
    y_ptr,
    mid_ptr,
    n_elements,
    delta,
    inv_div,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Stage 1: one program per BLOCK_SIZE-tile computes the per-element huber
    # loss, reduces it to a partial sum (scaled by 1/N for mean) and stores it
    # to `mid`. No atomics -- avoids serialization. EVEN elides the mask,
    # which otherwise costs up to 2.4x on large shapes on this backend.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if EVEN:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    diff = x - y
    absdiff = tl.abs(diff)
    loss_quad = 0.5 * diff * diff
    loss_linear = delta * (absdiff - 0.5 * delta)
    loss = tl.where(absdiff <= delta, loss_quad, loss_linear)
    if not EVEN:
        loss = tl.where(offsets < n_elements, loss, 0.0)

    tl.store(mid_ptr + pid, tl.sum(loss, axis=0) * inv_div)


@libentry()
@triton.jit
def _reduce_sum_kernel(mid_ptr, out_ptr, mid_size, BLOCK_MID: tl.constexpr):
    # Stage 2: single program reduces the `mid` partial sums to the scalar
    # result (already carries the mean scaling), casting to the output dtype
    # in-kernel. Loops so arbitrarily large mid arrays stay in one launch.
    acc = tl.zeros((), dtype=tl.float32)
    num_blocks = (mid_size + BLOCK_MID - 1) // BLOCK_MID
    for i in tl.range(0, num_blocks):
        offsets = i * BLOCK_MID + tl.arange(0, BLOCK_MID)
        mask = offsets < mid_size
        vals = tl.load(mid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
    tl.store(out_ptr, acc)


@libentry()
@triton.jit
def _huber_loss_vecacc_kernel(
    x_ptr,
    y_ptr,
    mid_ptr,
    n_elements,
    delta,
    inv_div,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Persistent stage 1 for large tensors: each program grid-strides over
    # BLOCK_SIZE-tiles, accumulating the per-element losses into a [BLOCK]
    # fp32 vector and reducing with ONE tl.sum tree at the end -- the tree
    # per block is the ALU wall this avoids. One partial sum per program
    # goes to `mid` (mid has num_programs entries).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK_SIZE + arange
        if EVEN:
            x = tl.load(x_ptr + offsets)
            y = tl.load(y_ptr + offsets)
        else:
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = y.to(tl.float32)
        diff = x - y
        absdiff = tl.abs(diff)
        loss_quad = 0.5 * diff * diff
        loss_linear = delta * (absdiff - 0.5 * delta)
        loss = tl.where(absdiff <= delta, loss_quad, loss_linear)
        if not EVEN:
            loss = tl.where(offsets < n_elements, loss, 0.0)
        acc += loss

    tl.store(mid_ptr + pid, tl.sum(acc, axis=0) * inv_div)


@libentry()
@triton.jit
def _huber_loss_fused_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, delta, inv_div, BLOCK_SIZE: tl.constexpr
):
    # Single-launch fused reduction for tiny tensors (numel <= _FUSED_LIMIT,
    # so BLOCK_SIZE >= numel and the loop runs exactly once): reduces to a
    # scalar, applies the mean/sum scaling, and casts to the output dtype
    # in-kernel -- ONE launch with NO intermediate `mid` allocation.
    acc = tl.zeros((), dtype=tl.float32)
    for base in range(0, n_elements, BLOCK_SIZE):
        offsets = base + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - y
        absdiff = tl.abs(diff)
        loss_quad = 0.5 * diff * diff
        loss_linear = delta * (absdiff - 0.5 * delta)
        loss = tl.where(absdiff <= delta, loss_quad, loss_linear)
        loss = tl.where(mask, loss, 0.0)
        acc += tl.sum(loss, axis=0)

    tl.store(out_ptr, acc * inv_div)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        r = reduction.lower()
        mapping = {"none": 0, "mean": 1, "sum": 2}
        if r not in mapping:
            raise ValueError(f"Unsupported reduction: {reduction}")
        return mapping[r]
    if isinstance(reduction, int) and reduction in (0, 1, 2):
        return reduction
    raise ValueError(f"Unsupported reduction: {reduction}")


def _huber_loss_compute(input, target, reduction, delta, result_dtype):
    device = input.device
    x_b, y_b = torch.broadcast_tensors(input.to(result_dtype), target.to(result_dtype))
    x_b = x_b.contiguous()
    y_b = y_b.contiguous()
    numel = x_b.numel()

    with torch_device_fn.device(device):
        if reduction == 0:  # none
            out = torch.empty_like(x_b, dtype=result_dtype, device=device)
            grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
            _huber_loss_element_kernel[grid](
                x_b, y_b, out, numel, float(delta), BLOCK_SIZE=1024
            )
            return out

        if numel == 0:
            # Matches aten: mean of empty is NaN, sum of empty is 0.
            fill = float("nan") if reduction == 1 else 0.0
            return torch.full((), fill, dtype=result_dtype, device=device)

        inv_div = 1.0 / numel if reduction == 1 else 1.0  # mean vs sum
        out = torch.empty((), dtype=result_dtype, device=device)

        if numel <= _FUSED_LIMIT:
            block_size = min(_FUSED_LIMIT, triton.next_power_of_2(numel))
            _huber_loss_fused_kernel[(1, 1, 1)](
                x_b,
                y_b,
                out,
                numel,
                float(delta),
                inv_div,
                BLOCK_SIZE=block_size,
                num_warps=4,
            )
            return out

        if numel >= _VECACC_MIN:
            # Persistent vector-accumulator path for large tensors (one
            # reduction tree per program instead of one per block).
            if x_b.element_size() >= 4:
                block_size = _VECACC_WIDE_BLOCK
                grid_n = _VECACC_WIDE_GRID
                launch_kwargs = {"num_warps": 2, "num_stages": 2}
            elif numel >= _VECACC_NARROW_BIG_MIN:
                block_size = _VECACC_NARROW_BIG_BLOCK
                grid_n = _VECACC_NARROW_BIG_GRID
                launch_kwargs = {"num_warps": 2}
            else:
                block_size = _VECACC_NARROW_MID_BLOCK
                grid_n = _VECACC_NARROW_MID_GRID
                launch_kwargs = {"num_warps": 4}

            mid = torch.empty((grid_n,), dtype=torch.float32, device=device)
            _huber_loss_vecacc_kernel[(grid_n, 1, 1)](
                x_b,
                y_b,
                mid,
                numel,
                float(delta),
                inv_div,
                BLOCK_SIZE=block_size,
                EVEN=(numel % block_size == 0),
                **launch_kwargs,
            )
            block_mid = min(triton.next_power_of_2(grid_n), 4096)
            _reduce_sum_kernel[(1, 1, 1)](
                mid, out, grid_n, BLOCK_MID=block_mid, num_warps=4
            )
            return out

        # Flat two-stage tree reduction. BLOCK ladder keeps ~256KB of input
        # bytes in flight per program; the grid guard keeps grid.x within the
        # backend launch limit for absurdly large tensors.
        itemsize = x_b.element_size()
        block_max = max(_BLOCK_MIN, _IN_FLIGHT_BYTES // (2 * itemsize))
        shift = 5 if itemsize >= 4 else 4
        block_size = max(
            _BLOCK_MIN, min(block_max, triton.next_power_of_2(numel) >> shift)
        )
        mid_size = triton.cdiv(numel, block_size)
        while mid_size > _GRID_X_MAX and block_size < _BLOCK_MIN * 16:
            block_size = min(_BLOCK_MIN * 16, block_size * 2)
            mid_size = triton.cdiv(numel, block_size)

        mid = torch.empty((mid_size,), dtype=torch.float32, device=device)

        _huber_loss_reduce_kernel[(mid_size, 1, 1)](
            x_b,
            y_b,
            mid,
            numel,
            float(delta),
            inv_div,
            BLOCK_SIZE=block_size,
            EVEN=(numel % block_size == 0),
            num_warps=4,
        )
        block_mid = min(triton.next_power_of_2(mid_size), 4096)
        _reduce_sum_kernel[(1, 1, 1)](
            mid, out, mid_size, BLOCK_MID=block_mid, num_warps=4
        )
        return out


def huber_loss(self: torch.Tensor, target: torch.Tensor, reduction=1, delta=1.0):
    logger.debug("GEMS HUBER_LOSS")
    reduction = _normalize_reduction(reduction)
    result_dtype = torch.result_type(self, target)
    return _huber_loss_compute(self, target, reduction, delta, result_dtype)


def huber_loss_out(
    self: torch.Tensor,
    target: torch.Tensor,
    reduction=1,
    delta=1.0,
    *,
    out: torch.Tensor,
):
    logger.debug("GEMS HUBER_LOSS_OUT")
    reduction = _normalize_reduction(reduction)
    val = _huber_loss_compute(self, target, reduction, delta, out.dtype)
    if reduction == 0:
        if out.shape != val.shape:
            out.resize_(val.shape)
    out.copy_(val)
    return out
