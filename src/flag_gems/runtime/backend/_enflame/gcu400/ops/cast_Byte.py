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

from flag_gems import runtime  # noqa: F401
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_NATIVE_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd)

_NUM_SIPS = 24
_GRID_CAP = _NUM_SIPS * 2

# int32 offset safe domain: masked tail blocks evaluate block offsets past the
# tensor end (up to num_blocks * BLOCK - 1), so keep n under 2^31 - 2^17 to
# keep flat int32 offsets from wrapping negative (abs.py / neg.py precedent).
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

# Strided address chains stay in 32-bit arithmetic on this backend: element
# offsets being i32-safe is NOT enough, the derived byte offset must also stay
# below 2^31 (strided-path silent truncation trap, compiler_traps 2026-11-16).
_STRIDED_BYTE_LIMIT = 1 << 31


def _cast_byte_complex_bridge(self, non_blocking=False):
    # The hardware has no native complex -> uint8 conversion and ATen semantics
    # only keep the real part (the conj bit does not affect it). On device, aten's
    # _to_copy aborts the whole process with TOPSATEN_STATUS_NOT_SUPPORT
    # (gcu_utils.cpp Check failed), so it cannot be used as a direct fallback, and
    # torch_gcu's resolve_conj does not clear the conj bit, so view_as_real raises
    # "unresolved conjugated tensors" and the real-part view cannot be taken on
    # device. Instead go through CPU: ATen's CPU _to_copy handles complex->uint8
    # correctly, then the result is copied back to the device (same approach as the
    # cast_Char complex bridge).
    cpu_out = torch.ops.aten._to_copy.default.redispatch(
        _FALLBACK_KEYSET, self.cpu(), dtype=torch.uint8, non_blocking=non_blocking
    )
    return cpu_out.to(self.device)


@triton.jit
def _cast_byte_value(x):
    # GCU400 cannot select an i64->i8 vector truncate (LLVM fails with
    # "Cannot select: v512i8 = truncate v512i64"), so narrow through int32:
    # uint8 keeps only the low 8 bits, which int32 preserves for in-range
    # integer and floating values.
    if x.dtype.is_floating():
        if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
            x = x.to(tl.float32)
        # ATen narrows float -> uint8 through int64, so values outside the int32
        # domain do not wrap like the low byte of a saturating int32 conversion:
        # measured against aten._cast_Byte on GCU400, finite |x| >= 2**31 maps
        # to 0x00 (int64 low byte of 2**31 / 2**32 + 256 / 1e20), NaN and -inf
        # map to 0x00, and +inf maps to 0xFF (int64 saturation to INT64_MAX).
        # Reproduce that explicitly; the int32 path cannot carry those bits.
        pos_inf = x > 3.4028234663852886e38
        out_of_range = (x != x) | (x < -3.4028234663852886e38)
        out_of_range = out_of_range | ((x >= 2147483648.0) & ~pos_inf)
        int_val = x.to(tl.int32)
        # Select in the int32 domain: 0x00 for NaN / -inf / finite overflow,
        # 0xFF for +inf. Masking after the uint8 cast needs per-dtype literals;
        # int32 keeps the lowered code dtype-agnostic.
        int_val = tl.where(out_of_range, int_val * 0, int_val)
        int_val = tl.where(pos_inf, int_val | 255, int_val)
        return (int_val & 0xFF).to(tl.uint8)
    # An explicit mask keeps LLVM from folding i64->i32->i8 back into the
    # illegal i64->i8 vector truncate; uint8 is exactly the low 8 bits.
    return (x.to(tl.int32) & 0xFF).to(tl.uint8)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _cast_byte_flat_even_kernel(X, Y, N_total, BLOCK: tl.constexpr):
    # Single-wave grid-stride over full tiles; host guarantees N_total % BLOCK == 0,
    # so the hot loop carries no mask and no `other=` select (masked + nw4 at
    # BLOCK=16384 measured +28% vs this form).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = N_total // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        x = tl.load(X + off)
        tl.store(Y + off, _cast_byte_value(x))


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _cast_byte_flat_tail_kernel(X, Y, N_total, BLOCK: tl.constexpr):
    # Non-divisible sizes: unmasked full tiles in the grid-stride loop plus a
    # single masked tail block on program 0 (full+tail split, hot loop clean).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_full = N_total // BLOCK
    for block_id in tl.range(pid, num_full, num_pids):
        off = block_id * BLOCK + arange
        x = tl.load(X + off)
        tl.store(Y + off, _cast_byte_value(x))
    if pid == 0 and N_total % BLOCK != 0:
        off = num_full * BLOCK + arange
        mask = off < N_total
        x = tl.load(X + off, mask=mask)
        tl.store(Y + off, _cast_byte_value(x), mask=mask)


@libentry()
@triton.jit
def _cast_byte_rowtile_kernel(
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
    # Non-dense strided fast path (cast_Char's rowtile form): each program casts
    # a [TILE_R, BLOCK_C] tile of the output (empty_like lays non-dense inputs
    # out dense in the stride-sorted basis), hoisting the per-row div/mod out of
    # the inner vector loop so the per-element cost is one mul-add. Measured on
    # the bench slice case (1024,512): fp16 6.7us / fp32 8.0us vs 13.7-17.0us
    # for flat div/mod blocks and 31.4us for the previous one-block-per-program
    # strided kernel.
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
        x = tl.load(X + src)
        tl.store(Y + out_offs, _cast_byte_value(x))
    else:
        mask = (rows[:, None] < R) & (col[None, :] < C)
        x = tl.load(X + src, mask=mask)
        tl.store(Y + out_offs, _cast_byte_value(x), mask=mask)


@libentry()
@triton.jit
def _cast_byte_strided_kernel(
    X, Y, N, SHAPE: tl.constexpr, STRIDES: tl.constexpr, BLOCK: tl.constexpr
):
    # General ND fallback for strided inputs that do not fit the rowtile form
    # (tiny inner dim / too many rows). Grid-stride so small tensors still get
    # multi-program parallelism without unbounded grids.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK + arange
        remaining = offsets
        src = tl.full((BLOCK,), 0, tl.int32)
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            src += (remaining % SHAPE[dim]) * STRIDES[dim]
            remaining //= SHAPE[dim]
        mask = offsets < N
        x = tl.load(X + src, mask=mask)
        tl.store(Y + offsets, _cast_byte_value(x), mask=mask)


# GCU400 retune (2026-11-16, same-process interleaved do_bench(median) A/B,
# 3 rounds per config, official bench core w1000/r100):
# - flat: grid-stride capped at 48 programs beats one-block-per-program by
#   20-56% on the 16M bucket; the EVEN (mask-free) form beats the masked
#   grid-stride by another 1-4% at divisible sizes (fp32 16M +3.8%), and the
#   masked+nw4 combination at BLOCK=16384 is a +28% pathology, so divisible
#   sizes dispatch the pure even kernel and the rest a full+tail split.
# - ladder boundaries were probed at 2^20/2^21/2^22/2^23/2^24 per itemsize
#   bucket; 8-byte dtypes cap BLOCK at 32768 to keep the per-program DSM
#   footprint within the 448KB 48-way residency budget.
def _choose_flat_launch(n_total, itemsize):
    # NOTE: the libentry launch cache key covers (dtype, BLOCK) but NOT
    # num_warps, so every BLOCK value must map to a single num_warps per
    # dtype bucket (floor.py precedent) or the second arm silently reuses
    # the first arm's compiled kernel.
    if n_total <= 8192:
        block = max(triton.next_power_of_2(n_total), 64)
        # BLOCK=8192 is shared with the 64K bucket below -> same nw there.
        return block, 8 if block == 8192 else 4
    if n_total <= 65536:
        return 8192, 8
    if itemsize >= 8:
        if n_total <= (1 << 21):
            return 16384, 4
        return 32768, 2
    if itemsize == 4:
        if n_total <= (1 << 21):
            # nw4 == nw8 measured tie at 2^20/65536; unified to 8 to keep
            # the (fp32, 8192) cache entry single.
            return 8192, 8
        if n_total <= (1 << 23):
            return 32768, 2
        return 65536, 8
    if n_total <= (1 << 20):
        return 16384, 4
    if n_total <= (1 << 21):
        return 32768, 2
    return 65536, 2


def _launch_with_retry(kernel, grid, *args, **kwargs):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic host-level flake on shared nodes); retrying the launch
    # recompiles and succeeds (abs.py / neg.py / cast_Char precedents).
    for attempt in range(3):
        try:
            kernel[grid](*args, **kwargs)
            return
        except Exception as error:
            if attempt == 2:
                raise
            logger.warning(
                "cast_Byte kernel %s launch failed (attempt %d): %s",
                kernel,
                attempt + 1,
                error,
            )


def _launch_flat(x, out, n):
    block, num_warps = _choose_flat_launch(n, x.element_size())
    grid = (min(triton.cdiv(n, block), _GRID_CAP),)
    if n % block == 0:
        _launch_with_retry(
            _cast_byte_flat_even_kernel,
            grid,
            x,
            out,
            n,
            BLOCK=block,
            num_warps=num_warps,
        )
    else:
        _launch_with_retry(
            _cast_byte_flat_tail_kernel,
            grid,
            x,
            out,
            n,
            BLOCK=block,
            num_warps=num_warps,
        )


def _launch_strided(x, out, n):
    order = sorted(range(x.ndim), key=lambda dim: out.stride(dim), reverse=True)
    shape = tuple(x.shape[dim] for dim in order)
    strides = tuple(x.stride(dim) for dim in order)
    max_src = sum((shape[dim] - 1) * strides[dim] for dim in range(x.ndim))
    if max_src * x.element_size() >= _STRIDED_BYTE_LIMIT or n >= _INT32_SAFE_LIMIT:
        # 32-bit strided address chains silently truncate above 2^31 bytes;
        # hand such huge views to the native composite instead of corrupting.
        return False
    inner = shape[-1]
    rows = n // inner
    if inner >= 16 and 32 <= rows <= (1 << 21):
        itemsize = x.element_size()
        if itemsize <= 2:
            cap = 2048
        elif itemsize == 4:
            cap = 1024
        else:
            cap = 256
        block_c = min(triton.next_power_of_2(inner), cap)
        _launch_with_retry(
            _cast_byte_rowtile_kernel,
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
        # num_warps unified to 8 for every BLOCK of this kernel: the libentry
        # cache key ignores num_warps, so BLOCK=4096 must not appear with two
        # different warp counts (tiny np2 arm vs the 4096 streaming arm).
        if n <= 4096:
            block, num_warps = max(triton.next_power_of_2(n), 64), 8
        else:
            block, num_warps = 4096, 8
        _launch_with_retry(
            _cast_byte_strided_kernel,
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


def cast_Byte(self, non_blocking=False):
    logger.debug("GEMS_ENFLAME CAST_BYTE")
    if self.is_complex():
        return _cast_byte_complex_bridge(self, non_blocking=non_blocking)
    if self.dtype == torch.uint8:
        return self
    if (
        self.device.type == "cpu"
        or self.layout != torch.strided
        or self.is_complex()
        or self.is_quantized
        or self.is_neg()
        # GCU400 has no FP64 vector path: torch_gcu stores float64 tensors as
        # fp32, so a Triton fp64 load/convert lowers to an illegal
        # elementwise_fusion_region. Let aten handle the downgraded dtype.
        or (self.device.type == "gcu" and self.dtype == torch.float64)
        # int64 is also emulated on this stack: the storage is 32-bit while the
        # dtype reports int64, so a Triton int64 load reads garbage (verified:
        # every int64 accuracy case returned wrong bytes). Let aten convert.
        or (self.device.type == "gcu" and self.dtype == torch.int64)
        or self.dtype
        not in (
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
    ):
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET, self, dtype=torch.uint8, non_blocking=non_blocking
        )
    out = torch.empty_like(self, dtype=torch.uint8)
    n = self.numel()
    if n == 0:
        return out
    with torch_device_fn.device(self.device):
        if self.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(self):
            # Dense inputs (contiguous, transposed, channels-last, ...) keep
            # their strides in `out`, so input and output share one memory
            # permutation and the cast is a flat memory-order copy.
            if n >= _INT32_SAFE_LIMIT:
                return torch.ops.aten._to_copy.default.redispatch(
                    _FALLBACK_KEYSET,
                    self,
                    dtype=torch.uint8,
                    non_blocking=non_blocking,
                )
            _launch_flat(self, out, n)
        elif not _launch_strided(self, out, n):
            return torch.ops.aten._to_copy.default.redispatch(
                _FALLBACK_KEYSET,
                self,
                dtype=torch.uint8,
                non_blocking=non_blocking,
            )
    return out


# ---------------------------------------------------------------------------
# torch_gcu dispatch adaptation: aten::_cast_Byte aborts the process on complex inputs.
#
# Same cause as cast_Char: torch_gcu has no PrivateUse1 kernel registered for aten::_cast_Byte,
# so GCU complex tensors dispatch to CompositeImplicitAutograd -> _to_copy -> topsatenCopy,
# which returns TOPSATEN_STATUS_NOT_SUPPORT and csrc Check terminates the process (no Python exception).
# The accuracy test reference leg calls torch.ops.aten._cast_Byte(x) directly, so fixing only the flag_gems side
# still crashes; following the existing cast_Char pattern, idempotently register the complex narrow domain on the device
# dispatch
# key and redispatch all other dtypes back to the native composite, keeping oracle semantics unchanged.
# ---------------------------------------------------------------------------
_CAST_BYTE_KEYS_FLAG = "_flag_gems_gcu400_cast_byte_keys_registered"
_CAST_BYTE_LIBRARY = None


def _cast_byte_device_impl(self, non_blocking=False):
    if self.device.type == "gcu" and self.is_complex():
        return _cast_byte_complex_bridge(self, non_blocking)
    return torch.ops.aten._cast_Byte.default.redispatch(
        _NATIVE_KEYSET, self, non_blocking=non_blocking
    )
