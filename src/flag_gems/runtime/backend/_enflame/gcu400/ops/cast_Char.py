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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_NATIVE_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd)


def _cast_char_complex_bridge(self, non_blocking=False):
    # torch_gcu's native _to_copy (topsatenCopy) has no ComplexFloat /
    # ComplexDouble -> Char conversion: it fails with
    # TOPSATEN_STATUS_NOT_SUPPORT and the csrc check then aborts the whole
    # process instead of raising a Python error (P21 archive). ATen casts
    # complex to int8 through the real part (float -> int32 -> low 8 bits,
    # conjugate bit ignored), so compute that on CPU and move the result back.
    cpu_out = torch.ops.aten._to_copy.default.redispatch(
        _FALLBACK_KEYSET, self.cpu(), dtype=torch.int8, non_blocking=non_blocking
    )
    return cpu_out.to(self.device)


@triton.jit
def _cast_char_value(x):
    # Match CUDA's float-to-int32 conversion before narrowing to signed int8.
    # GCU400 cannot select an i64->i8 vector truncate (LLVM fails with
    # "Cannot select: v512i8 = truncate v512i64"), and a plain x.to(tl.int8)
    # is folded back into that illegal op. Narrowing through int32 with an
    # explicit uint8 mask keeps the low 8 bits, which is exactly int8's
    # bit pattern for both int64 inputs and in-range floating values.
    if x.dtype.is_floating():
        if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
            x = x.to(tl.float32)
        x = x.to(tl.int32)
    else:
        x = x.to(tl.int32)
    return (x & 0xFF).to(tl.uint8).to(tl.int8)


@libentry()
@triton.jit
def _cast_char_strided(
    X, Y, N, SHAPE: tl.constexpr, STRIDES: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remaining = offsets
    src = tl.full((BLOCK,), 0, tl.int64)
    for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
        src += (remaining % SHAPE[dim]) * STRIDES[dim]
        remaining //= SHAPE[dim]
    x = tl.load(X + src, offsets < N, other=0)
    tl.store(Y + offsets, _cast_char_value(x), offsets < N)


@libentry()
@triton.jit
def _cast_char_kernel(X, Y, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, _cast_char_value(x), offsets < N)


@libentry()
@triton.jit
def _cast_char_gridstride(X, Y, N, NPROG: tl.constexpr, BLOCK: tl.constexpr):
    # Single-wave grid-stride form for large dense casts: NPROG programs (at
    # most one per SPC on GCU400's 48-SPC fixed grid) interleave full tiles in
    # a rolled loop, amortizing the large per-program launch cost while
    # keeping the per-iteration DSM footprint small enough to pipeline. The
    # last partial tile, if any, is handled by program 0 with a masked block.
    pid = tl.program_id(0)
    n_full = N // BLOCK
    for tile in tl.range(pid, n_full, NPROG):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offsets)
        tl.store(Y + offsets, _cast_char_value(x))
    if pid == 0 and N % BLOCK != 0:
        offsets = n_full * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=0)
        tl.store(Y + offsets, _cast_char_value(x), mask=mask)


@libentry()
@triton.jit
def _cast_char_rowtile(
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
    # Non-dense strided fast path: each program casts a [TILE_R, BLOCK_C] tile
    # of the output (which empty_like lays out dense in the stride-sorted
    # basis), addressing the source with per-row scalar div/mod hoisted out of
    # the inner vector loop, so the per-element cost is one mul-add instead of
    # the full div/mod linearization of _cast_char_strided.
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
        tl.store(Y + out_offs, _cast_char_value(x))
    else:
        mask = (rows[:, None] < R) & (col[None, :] < C)
        x = tl.load(X + src, mask=mask, other=0)
        tl.store(Y + out_offs, _cast_char_value(x), mask=mask)


def _cast_char_flat_launch(x, out, n):
    # GCU400 charges a large fixed cost per program (~52ns measured), so the
    # dense cast wants few, fat programs: a single-wave grid-stride loop at
    # roughly 64 tiles spread over <=48 programs for large sizes, and one
    # flat block per program for small ones. Tile caps keep the per-program
    # DSM footprint under the 917504B hard limit (larger fp16 load tiles
    # raise a device Sip error).
    itemsize = x.element_size()
    if n > 131072:
        cap = 65536 if itemsize <= 2 else 32768
        block = min(max(triton.next_power_of_2(n // 64), 16384), cap)
        nprog = min(48, n // block)
        _cast_char_gridstride[(nprog,)](
            x, out, n, NPROG=nprog, BLOCK=block, num_warps=2
        )
    else:
        block = 8192 if n > 8192 else triton.next_power_of_2(max(int(n), 16))
        _cast_char_kernel[(triton.cdiv(n, block),)](x, out, n, BLOCK=block, num_warps=4)


def cast_Char(self, non_blocking=False):
    logger.debug("GEMS CAST_CHAR")
    if self.dtype == torch.int8:
        return self
    if self.is_complex() and self.device.type == "gcu":
        return _cast_char_complex_bridge(self, non_blocking)
    if (
        self.device.type == "cpu"
        or self.layout != torch.strided
        or self.is_complex()
        or self.is_quantized
        or self.is_neg()
        # torch_gcu stores float64 tensors as fp32 on GCU, so a Triton fp64
        # load/convert lowers to an illegal elementwise_fusion_region. Let
        # aten handle the downgraded dtype.
        or (self.device.type == "gcu" and self.dtype == torch.float64)
        or self.dtype
        not in (
            torch.bool,
            torch.uint8,
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
            _FALLBACK_KEYSET, self, dtype=torch.int8, non_blocking=non_blocking
        )
    out = torch.empty_like(self, dtype=torch.int8)
    n = self.numel()
    if n == 0:
        return out
    with torch_device_fn.device(self.device):
        if self.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(self):
            # Dense inputs (contiguous, transposed, channels-last, ...) keep
            # their strides in `out`, so input and output share one memory
            # permutation and the cast is a flat memory-order copy.
            _cast_char_flat_launch(self, out, n)
            return out
        order = sorted(range(self.ndim), key=lambda dim: out.stride(dim), reverse=True)
        shape = tuple(self.shape[dim] for dim in order)
        strides = tuple(self.stride(dim) for dim in order)
        inner = shape[-1]
        rows = n // inner
        max_src = sum((shape[dim] - 1) * strides[dim] for dim in range(self.ndim))
        if (
            n < 2**31
            and max_src < 2**31
            and inner >= 16
            and rows >= 32
            and rows <= 1 << 21
        ):
            itemsize = self.element_size()
            if itemsize <= 2:
                cap = 2048
            elif itemsize == 4:
                cap = 1024
            else:
                cap = 256
            block_c = min(triton.next_power_of_2(inner), cap)
            _cast_char_rowtile[(triton.cdiv(rows, 64), triton.cdiv(inner, block_c))](
                self,
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
            grid = (triton.cdiv(n, 16384),)
            _cast_char_strided[grid](
                self, out, n, shape, strides, BLOCK=16384, num_warps=4
            )
    return out


# ---------------------------------------------------------------------------
# torch_gcu dispatch adaptation（aten::_cast_Char 的 complex→int8 进程 abort）
#
# torch_gcu 未为 `aten::_cast_Char` 注册 PrivateUse1 kernel，GCU 张量派发到
# CompositeImplicitAutograd 原生 composite → `_to_copy` → topsatenCopy，对
# ComplexFloat/ComplexDouble → Char 变换返回 TOPSATEN_STATUS_NOT_SUPPORT 并
# 由 csrc Check failed 直接终止进程（无 Python 异常可捕获，P21 档案）。
# enable()/use_gems() 之外直接调 torch.ops.aten.* 的精度测试参考腿因此整
# 文件崩挂。照 `flash_attention_backward.py` / `resize_as_sparse_.py` 的既
# 定制式把 complex 窄域实现幂等注册到 device key：complex → CPU 桥；其余
# 一律 redispatch 原生 composite（real dtype 参考腿 oracle 与 bench base
# 腿语义保持不变）。后续 enable() 对同一 key 的重复注册被 registrar 捕获
# 为 warning，real dtype 的 aten 调用退回 native composite 行为（与未
# enable 时一致）。
# ---------------------------------------------------------------------------
_CAST_CHAR_KEYS_FLAG = "_flag_gems_gcu400_cast_char_keys_registered"
_CAST_CHAR_LIBRARY = None


def _cast_char_device_impl(self, non_blocking=False):
    if self.device.type == "gcu" and self.is_complex():
        return _cast_char_complex_bridge(self, non_blocking)
    return torch.ops.aten._cast_Char.default.redispatch(
        _NATIVE_KEYSET, self, non_blocking=non_blocking
    )


def _ensure_cast_char_key_registrations():
    global _CAST_CHAR_LIBRARY
    if getattr(torch, _CAST_CHAR_KEYS_FLAG, False):
        return
    _CAST_CHAR_LIBRARY = torch.library.Library("aten", "IMPL")
    try:
        _CAST_CHAR_LIBRARY.impl(
            "_cast_Char", _cast_char_device_impl, runtime.device.dispatch_key
        )
    except RuntimeError as error:
        logger.warning(
            "_cast_Char: aten registration _cast_Char@%s skipped: %s",
            runtime.device.dispatch_key,
            error,
        )
    setattr(torch, _CAST_CHAR_KEYS_FLAG, True)


_ensure_cast_char_key_registrations()
