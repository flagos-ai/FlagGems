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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24
_INT32_MAX = torch.iinfo(torch.int32).max
# Grid cap and tile are tiered by lane width (alternating A/B on 2026-11-17, same board and same do_bench settings,
# median, with 3 rounds each of cur/new per tier, taking the median):
# - 4-byte lanes (fp32 elements / i32 words packing bools) switch to a single wave of
#   24 CTAs beyond >=2^18 elements: fp32 2^22 0.955x -> 1.049x (-9.0%), 2^24 0.974x -> 1.004x;
#   packed bool 2^22 1.139x -> 1.379x (-17.4%), 2^24 1.403x -> 1.511x.
# - 2-byte lanes (fp16/bf16) also converge to a single wave of 24 CTAs beyond >=2^22, and raise the tile from
#   65536 to 131072 (from 2^24): fp16 2^22 1.153x -> 1.291x (-10.6%),
#   2^24 1.073x -> 1.106x; bf16 2^22 1.014x -> 1.147x, 2^24 0.913x -> 0.970x.
_WIDE_LANE_GRID_CAP = NUM_SIPS
_NARROW_LANE_GRID_CAP = NUM_SIPS * 2
_WIDE_LANE_MIN_N = 1 << 18
_WIDE_LANE_MIN_WORDS = 8192
_NARROW_LANE_MIN_N = 1 << 22
_NARROW_LANE_BIG_N = 1 << 24
# torch_gcu process flag that switches int64 storage to real 8-byte lanes;
# when unset, int64 tensors are fake: a compact N x int32 word payload.
_INT64_ENV = "TORCH_GCU_ENABLE_INT64_AND_UINT64"
_FAST_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


# GCU400 override of the generic pointwise_dynamic implementation.
#
# The generic wrapper lowers bool stores through the i1 lane (~35% off the
# byte-stream memory floor, bitwise_and precedent) and pays the block-pointer
# tile / GSL structure tax on top; the mid and large core shapes measured
# 0.81-0.91x vs torch native.  The flat grid-stride kernel below stores the
# 0/1 result through an int8 view of the bool output instead, saturating the
# memory floor.  Semantics of `(x == 0)` are bit-identical to `not bool(x)`
# for every IEEE/integer value (NaN is truthy, -0.0 == 0).  Exotic inputs
# (fp64/complex, strided, out-of-int32-range sizes) fall back to the backend
# pointwise path.
#
# int64: without TORCH_GCU_ENABLE_INT64_AND_UINT64 torch_gcu stores int64
# tensors as a fake -- a compact N x int32 word payload while the metadata
# still reports 8 bytes/element (P17 narrow-storage family).  A value-level
# ATen op (native logical_not, randint, D2H) sees element i as the i-th
# int32 word, while an i64-typed Triton pointer interleaves two elements
# per lane and drifts past the dense payload -- which is why the pointwise
# `not x.to(tl.int1)` arm and every i64/strided-pair flat variant
# misclassify values.  The word-lane kernels below bitcast the pointer once
# and load the i-th int32 word with stride-1 contiguous addressing (the
# safe bitcast form, same as the bool packed lane), reading exactly the
# value native sees, so `(w == 0)` matches bit-for-bit.  With the env flag
# set (real 8-byte int64) the fast path is skipped and the pointwise
# fallback runs (correct under the env per the bitwise_left_shift finding).
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def logical_not_flat_kernel(
    x_ptr, out_ptr, N_total, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
        if x_ptr.dtype.element_ty == tl.int16:
            v = (~(x | -x) >> 15) & 1
        elif x_ptr.dtype.element_ty == tl.int8:
            v = (~(x | -x) >> 7) & 1
        else:
            v = (x == 0).to(out_ptr.dtype.element_ty)
        v = v.to(out_ptr.dtype.element_ty)
        if EVEN:
            tl.store(out_ptr + off, v)
        else:
            tl.store(out_ptr + off, v, mask=mask)


# Bool fast lane: 1-byte loads/stores top out at ~2.4TB/s on GCU400 (the
# byte-lane flat kernel measured 439us vs the 345us native floor on the
# 512M-element core shape) while 4-byte lanes stream at ~3.1TB/s.  Packing
# four bools per i32 word through an in-kernel pointer bitcast recovers the
# floor: t = (w & 0x7f7f7f7f) + 0x7f7f7f7f | w sets bit 7 of every byte
# iff that byte is nonzero (the masked add cannot carry across bytes, so
# each byte is classified independently), and (~t >> 7) & 0x01010101 is
# exactly the logical_not of the four packed booleans.  NOTE: the tempting
# 5-op `(~(w | -w) >> 7) & 0x01010101` per-element trick is INVALID here --
# the word-wide two's-complement borrow chain crosses byte boundaries and
# misclassifies zero bytes above a nonzero byte (-0x01000100 = 0xFEFFFF00),
# value-probe caught it before delivery.  Contiguous word loads only --
# strided bitcast loads miscompile (see the job record).
@libentry()
@triton.jit(do_not_specialize=["N_words"])
def logical_not_bool_packed_kernel(
    x_ptr, out_ptr, N_words, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    x32 = x_ptr.to(tl.pointer_type(tl.int32))
    o32 = out_ptr.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_words + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            w = tl.load(x32 + off)
        else:
            mask = off < N_words
            w = tl.load(x32 + off, mask=mask, other=1)
        t = (w & 0x7F7F7F7F) + 0x7F7F7F7F
        t = t | w
        v = (~t >> 7) & 0x01010101
        if EVEN:
            tl.store(o32 + off, v)
        else:
            tl.store(o32 + off, v, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_words"])
def logical_not_bool_packed_inplace_kernel(
    x_ptr, N_words, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    x32 = x_ptr.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_words + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            w = tl.load(x32 + off)
        else:
            mask = off < N_words
            w = tl.load(x32 + off, mask=mask, other=1)
        t = (w & 0x7F7F7F7F) + 0x7F7F7F7F
        t = t | w
        v = (~t >> 7) & 0x01010101
        if EVEN:
            tl.store(x32 + off, v)
        else:
            tl.store(x32 + off, v, mask=mask)


# int64 (fake storage) out-of-place arm: element i's value is the i-th
# int32 word; stride-1 contiguous word load through the bitcast pointer.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def logical_not_i64_word_kernel(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    x32 = x_ptr.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        w = tl.load(x32 + off, mask=mask, other=1)
        v = (w == 0).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + off, v, mask=mask)


# int64 (fake storage) in-place arm: writes the 0/1 result back into the
# same word slot; the value-level D2H readback then sees 0/1 per element.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def logical_not_i64_word_inplace_kernel(x_ptr, N_total, BLOCK: tl.constexpr):
    x32 = x_ptr.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        w = tl.load(x32 + off, mask=mask, other=1)
        v = (w == 0).to(tl.int32)
        tl.store(x32 + off, v, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def logical_not_func(x):
    return not x.to(tl.int1)


def _choose_block(n_total, itemsize, packed=False):
    # do_bench(median) ladder shared with the sibling flat kernels
    # (rad2deg / bitwise_and, same 1-read/1-write trivial-ALU body):
    # tiny shapes want the exact pow2 tile, mid shapes peak at 32768, and
    # only >2^28-element 1/2-byte streams gain from 131072 (a 4-byte
    # 131072 tile overflows tile resources and segfaults in ld.lld).
    # Packed bool-word streams (itemsize lane = 4 bytes per 4 bools)
    # measure best at 32768 across the whole mid+large band (2026-11-16:
    # 4.2M words -5.6% vs 65536 -- the 64-block/48-CTA stride loop leaves a
    # 2:1 tail imbalance at 65536 -- and 134M words ties 65536 within noise).
    if packed:
        if n_total <= 8192:
            return max(triton.next_power_of_2(n_total), 2)
        if n_total <= 65536:
            return 8192
        return 32768
    if n_total <= 8192:
        return max(triton.next_power_of_2(n_total), 2)
    if n_total <= 65536:
        return 8192
    if n_total <= 524288:
        return 32768
    if itemsize >= 4:
        # For 4-byte lanes on large shapes, 32768 is consistently better than 65536 (see the grid-cap comment above)
        return 32768
    if n_total >= _NARROW_LANE_BIG_N:
        return 131072
    return 65536


def _lane_view(t):
    # Bool tensors ride the int8 lane: 0/1 bytes compare identically and the
    # i1 lowering is ~35% off the byte-stream memory floor.
    return t.view(torch.int8) if t.dtype == torch.bool else t


def _launch_flat(kern, grid, args, block, even=None):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic host-level flake on shared nodes); retrying the
    # launch recompiles and succeeds (rad2deg / atan2_ / sinh precedents).
    # even=None -> kernel has no EVEN arm (the int64 word kernels keep the
    # masked-only form; they are unreachable under the service int64 env and
    # stay bit-identical to their 2026-11-11 validated build).
    kwargs = (
        {"num_warps": 2, "BLOCK": block}
        if even is None
        else {
            "num_warps": 2,
            "BLOCK": block,
            "EVEN": even,
        }
    )
    for attempt in range(3):
        try:
            with torch_device_fn.device(args[0].device):
                kern[grid](*args, **kwargs)
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "logical_not flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _fast_tensor(t):
    # NOTE: check the device *type*, not Tensor.is_cuda -- torch_gcu only
    # flips is_cuda on gcu tensors after the first Triton launch completes
    # its lazy patching, so an is_cuda gate would wrongly route the very
    # first op call in a process to the pointwise fallback (broken for the
    # fake-int64 word lane).
    return (
        isinstance(t, torch.Tensor)
        and t.device.type in ("cuda", "gcu")
        and t.is_contiguous()
        and t.dtype in _FAST_DTYPES
        and t.numel() <= _INT32_MAX
    )


def _flat_config(n_total, itemsize, packed=False):
    block = _choose_block(n_total, itemsize, packed=packed)
    if triton.cdiv(n_total, block) * block > _INT32_MAX:
        return None
    return (
        block,
        (min(triton.cdiv(n_total, block), NUM_SIPS * 2),),
        n_total % block == 0,
    )


def _fake_i64_words(t):
    # Without the env flag, torch_gcu stores int64 tensors as a compact
    # N x int32 word payload (value-level element == the i-th word); with
    # the flag set the storage is real 8-byte int64 and the word lane would
    # be wrong, so those stay on the pointwise path (correct under env).
    return (
        t.dtype == torch.int64
        and not os.environ.get(_INT64_ENV, "")
        and t.data_ptr() % 4 == 0
    )


def logical_not(A):
    logger.debug("GEMS_ENFLAME LOGICAL_NOT")
    if _fast_tensor(A):
        out = torch.empty_like(A, dtype=torch.bool)
        n_total = out.numel()
        if n_total == 0:
            return out
        if A.dtype == torch.int64:
            if _fake_i64_words(A):
                config = _flat_config(n_total, 4)
                if config is not None:
                    block, grid, _ = config
                    _launch_flat(
                        logical_not_i64_word_kernel,
                        grid,
                        (A, out.view(torch.int8), n_total),
                        block,
                    )
                    return out
            return logical_not_func(A)
        if (
            A.dtype == torch.bool
            and n_total % 4 == 0
            and A.data_ptr() % 4 == 0
            and out.data_ptr() % 4 == 0
        ):
            n_words = n_total // 4
            config = _flat_config(n_words, 4, packed=True)
            if config is not None:
                block, grid, even = config
                _launch_flat(
                    logical_not_bool_packed_kernel,
                    grid,
                    (A, out, n_words),
                    block,
                    even,
                )
                return out
        config = _flat_config(n_total, A.element_size())
        if config is not None:
            block, grid, even = config
            _launch_flat(
                logical_not_flat_kernel,
                grid,
                (_lane_view(A), out.view(torch.int8), n_total),
                block,
                even,
            )
            return out
    return logical_not_func(A)


# ---------------------------------------------------------------------------
# torch.logical_not reference-arm proxy (torch_gcu P21)
#
# topsatenLogicalNot refuses Long inputs (TOPSATEN_STATUS_NOT_SUPPORT,
# op_aten_unary.h:347) once TORCH_GCU_ENABLE_INT64_AND_UINT64 turns on the
# real 8-byte layout, and torch_gcu csrc turns that into a hard process abort
# (gcu_utils.cpp:74 "Check failed: 0") with no Python exception -- the
# accuracy harness's reference arm dies on the very int64 cases it is
# supposed to check, for both the functional and the in-place entry.  Route
# gcu int64 inputs through CPU aten for an independent ground truth (the
# D2H/H2D round trip is value-exact under both the fake 4-byte-slot and the
# real 8-byte storage) and move the result back to the original device.
# Every other call shape -- all bench/native arms included -- passes straight
# through to the original callable unchanged.  Drop once topsatenLogicalNot
# handles Long.
# ---------------------------------------------------------------------------


def _gcu_long_tensor(t):
    return (
        isinstance(t, torch.Tensor)
        and t.device.type in ("gcu", "cuda")
        and t.dtype == torch.int64
    )


def _install_aten_logical_not_ref_arm_proxy():
    current_fn = torch.logical_not
    if not getattr(current_fn, "_flaggems_gcu_logical_not_proxy", False):

        def _logical_not_proxy(self, *args, **kwargs):
            if _gcu_long_tensor(self):
                host = self.cpu()
                out = kwargs.get("out")
                if out is None:
                    return current_fn(host).to(self.device)
                if (
                    isinstance(out, torch.Tensor)
                    and out.device.type in ("gcu", "cuda")
                    and out.dtype == torch.bool
                ):
                    out.copy_(current_fn(host))
                    return out
            return current_fn(self, *args, **kwargs)

        _logical_not_proxy._flaggems_gcu_logical_not_proxy = True
        torch.logical_not = _logical_not_proxy

    current_method = torch.Tensor.logical_not
    if not getattr(current_method, "_flaggems_gcu_logical_not_proxy", False):

        def _logical_not_method_proxy(self, *args, **kwargs):
            if _gcu_long_tensor(self):
                host = self.cpu()
                out = kwargs.get("out")
                if out is None:
                    return current_method(host).to(self.device)
                if (
                    isinstance(out, torch.Tensor)
                    and out.device.type in ("gcu", "cuda")
                    and out.dtype == torch.bool
                ):
                    out.copy_(current_method(host))
                    return out
            return current_method(self, *args, **kwargs)

        _logical_not_method_proxy._flaggems_gcu_logical_not_proxy = True
        torch.Tensor.logical_not = _logical_not_method_proxy

    current_inplace = torch.Tensor.logical_not_
    if not getattr(current_inplace, "_flaggems_gcu_logical_not_proxy", False):

        def _logical_not_inplace_proxy(self, *args, **kwargs):
            if _gcu_long_tensor(self):
                host = self.cpu()
                current_inplace(host, *args, **kwargs)
                return self.copy_(host)
            return current_inplace(self, *args, **kwargs)

        _logical_not_inplace_proxy._flaggems_gcu_logical_not_proxy = True
        torch.Tensor.logical_not_ = _logical_not_inplace_proxy


_install_aten_logical_not_ref_arm_proxy()
