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
#
# GCU400 specialization of ``aten::_cast_Float`` (to float32). The cast is a
# 1:1 elementwise copy, so the whole job is bandwidth: a flat grid-stride kernel
# with a mask-free hot loop for dense inputs, a row-tile kernel for non-dense
# (transposed / sliced) inputs, and a general div/mod kernel for the shapes the
# row-tile form cannot express.
#
# float64 and int64 sources are handed to the native composite kernel: torch_gcu
# stores float64 as fp32 with no fp64 vector lane (a Triton fp64 load fails the
# compiler outright), and the int64->fp32 vector convert returns garbage.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

# Four waves of programs per GCU400 side: probed on 1M/16M element casts, where
# two waves left the copy engine idle between grid strides.
_NUM_SIPS = 24
_GRID_CAP = _NUM_SIPS * 4

_INT32_SAFE_LIMIT = 1 << 31
_STRIDED_BYTE_LIMIT = 1 << 31


@triton.jit
def _cast_float_value(x):
    # bool/int8/uint8 all have an exact fp32 image, so every supported source
    # widens in one step.
    return x.to(tl.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _cast_float_flat_even_kernel(X, Y, N_total, BLOCK: tl.constexpr):
    # Host guarantees N_total % BLOCK == 0, so the hot loop carries no mask.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = N_total // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        tl.store(Y + off, _cast_float_value(tl.load(X + off)))


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _cast_float_flat_tail_kernel(X, Y, N_total, BLOCK: tl.constexpr):
    # Unmasked full tiles in the grid-stride loop plus one masked tail block.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_full = N_total // BLOCK
    for block_id in tl.range(pid, num_full, num_pids):
        off = block_id * BLOCK + arange
        tl.store(Y + off, _cast_float_value(tl.load(X + off)))
    if pid == 0 and N_total % BLOCK != 0:
        off = num_full * BLOCK + arange
        mask = off < N_total
        tl.store(Y + off, _cast_float_value(tl.load(X + off, mask=mask)), mask=mask)


@libentry()
@triton.jit
def _cast_float_rowtile_kernel(
    X,
    Y,
    R,
    C,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Non-dense fast path: one [TILE_R, BLOCK_C] tile per program, with the
    # per-row div/mod hoisted out of the inner vector loop.
    rows = tl.program_id(0) * TILE_R + tl.arange(0, TILE_R)
    col = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    src = tl.zeros((TILE_R,), dtype=tl.int32)
    remaining = rows
    for dim in tl.static_range(len(SHAPE) - 2, -1, -1):
        src += (remaining % SHAPE[dim]) * STRIDES[dim]
        remaining //= SHAPE[dim]
    src = src[:, None] + col[None, :] * STRIDES[len(SHAPE) - 1]
    out_offs = rows[:, None] * C + col[None, :]
    if EVEN:
        values = tl.load(X + src)
        tl.store(Y + out_offs, _cast_float_value(values))
    else:
        mask = (rows[:, None] < R) & (col[None, :] < C)
        values = tl.load(X + src, mask=mask)
        tl.store(Y + out_offs, _cast_float_value(values), mask=mask)


@libentry()
@triton.jit
def _cast_float_strided_kernel(
    X, Y, N, SHAPE: tl.constexpr, STRIDES: tl.constexpr, BLOCK: tl.constexpr
):
    # General ND fallback: one div/mod chain per element, grid-stride so small
    # tensors still spread across programs.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK + arange
        remaining = offsets
        src = tl.zeros((BLOCK,), dtype=tl.int32)
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            src += (remaining % SHAPE[dim]) * STRIDES[dim]
            remaining //= SHAPE[dim]
        mask = offsets < N
        values = tl.load(X + src, mask=mask)
        tl.store(Y + offsets, _cast_float_value(values), mask=mask)


def _choose_flat_launch(n_total, itemsize):
    """Pick (BLOCK, num_warps) for the flat path.

    Probed on GCU400: a 32K tile with four warps is the sweet spot once the
    tensor is larger than the 64K launch-overhead regime, for every source width
    (16M fp16 / int32 sources both reach the native copy bandwidth). Each BLOCK
    maps to exactly one num_warps because the launch cache key does not include
    num_warps, so a BLOCK shared by two arms would silently reuse the first
    arm's compiled kernel.
    """
    if n_total <= 8192:
        block = max(triton.next_power_of_2(n_total), 64)
        return block, 8 if block == 8192 else 4
    if n_total <= 65536:
        return 8192, 8
    return 32768, 4


def _launch_with_retry(kernel, grid, *args, **kwargs):
    # The external llvm converter intermittently fails on a cold compile; a
    # retry recompiles and succeeds (cast_Byte / cast_Char precedent).
    for attempt in range(3):
        try:
            kernel[grid](*args, **kwargs)
            return
        except Exception as error:
            if attempt == 2:
                raise
            logger.warning(
                "cast_Float kernel %s launch failed (attempt %d): %s",
                kernel,
                attempt + 1,
                error,
            )


def _launch_flat(x, out, n):
    block, num_warps = _choose_flat_launch(n, x.element_size())
    grid = (min(triton.cdiv(n, block), _GRID_CAP),)
    kernel = (
        _cast_float_flat_even_kernel if n % block == 0 else _cast_float_flat_tail_kernel
    )
    _launch_with_retry(kernel, grid, x, out, n, BLOCK=block, num_warps=num_warps)


def _launch_strided(x, out, n):
    order = sorted(range(x.ndim), key=lambda dim: out.stride(dim), reverse=True)
    shape = tuple(x.shape[dim] for dim in order)
    strides = tuple(x.stride(dim) for dim in order)
    max_src = sum((shape[dim] - 1) * strides[dim] for dim in range(x.ndim))
    if max_src * x.element_size() >= _STRIDED_BYTE_LIMIT or n >= _INT32_SAFE_LIMIT:
        return False
    inner = shape[-1]
    rows = n // inner
    if inner >= 16 and 32 <= rows <= (1 << 21):
        itemsize = x.element_size()
        cap = 2048 if itemsize <= 2 else (1024 if itemsize == 4 else 256)
        block_c = min(triton.next_power_of_2(inner), cap)
        _launch_with_retry(
            _cast_float_rowtile_kernel,
            (triton.cdiv(rows, 64), triton.cdiv(inner, block_c)),
            x,
            out,
            rows,
            inner,
            shape,
            strides,
            TILE_R=64,
            BLOCK_C=block_c,
            EVEN=(rows % 64 == 0 and inner % block_c == 0),
            num_warps=4,
        )
    else:
        if n <= 4096:
            block, num_warps = max(triton.next_power_of_2(n), 64), 8
        else:
            block, num_warps = 4096, 8
        _launch_with_retry(
            _cast_float_strided_kernel,
            (min(triton.cdiv(n, block), _GRID_CAP),),
            x,
            out,
            n,
            SHAPE=shape,
            STRIDES=strides,
            BLOCK=block,
            num_warps=num_warps,
        )
    return True


def cast_Float(self, non_blocking=False):
    logger.debug("GEMS_ENFLAME CAST_FLOAT")
    if self.dtype == torch.float32:
        return self
    if self.is_complex() and self.device.type == "gcu":
        # No native complex -> float32 conversion in hardware: on device aten's
        # _to_copy returns TOPSATEN_STATUS_NOT_SUPPORT and aborts the whole process
        # (gcu_utils.cpp Check failed), so it cannot be used directly. Go through
        # CPU to take the real part, then copy it back to the device (same approach
        # as the cast_Char complex bridge).
        cpu_out = torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET, self.cpu(), dtype=torch.float32, non_blocking=non_blocking
        )
        return cpu_out.to(self.device)
    if (
        self.device.type == "cpu"
        or self.layout != torch.strided
        or self.is_complex()
        or self.is_quantized
        or self.is_neg()
        # No working GCU400 Triton lowering for these two source dtypes.
        or self.dtype in (torch.float64, torch.int64)
        or self.dtype
        not in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        )
    ):
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            self,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
    out = torch.empty_like(self, dtype=torch.float32)
    n = self.numel()
    if n == 0:
        return out
    with torch_device_fn.device(self.device):
        if self.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(self):
            # Dense inputs (contiguous, transposed, channels-last, ...) keep
            # their strides in `out`, so both sides share one memory
            # permutation and the cast is a flat memory-order copy.
            if n >= _INT32_SAFE_LIMIT:
                return torch.ops.aten._to_copy.default.redispatch(
                    _FALLBACK_KEYSET,
                    self,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )
            _launch_flat(self, out, n)
        elif not _launch_strided(self, out, n):
            return torch.ops.aten._to_copy.default.redispatch(
                _FALLBACK_KEYSET,
                self,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
    return out
