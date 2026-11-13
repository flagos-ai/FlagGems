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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24
_INT32_MAX = torch.iinfo(torch.int32).max
_FAST_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32)


# GCU400 override of the generic pointwise_dynamic implementation (bitwise_and
# precedent).  The generic path is a block-pointer kernel (grid capped at 48,
# i1 loads for bool, and -- for the scalar variants -- the Python scalar wrapped
# into a 0-dim device tensor whose stride-0 broadcast is materialized per
# tile).  Measured on the core bench shapes (L600) that costs bool 0.60-0.85x
# and mid-size shapes 0.82-0.93x vs torch native, and for the (bool tensor,
# int scalar) mix it even produces wrong values: torch promotes that mix to an
# int64 result, the generated kernel stores 8 bytes per element into the
# fake-int64 (4-byte-slot) output buffer, so readback is interleaved garbage.
#
# The flat grid-stride kernels below saturate the memory floor instead:
#   * tensor pairs: 2-read/1-write flat stream, bool handled through an int8
#     lane view (0/1 bytes XOR identically, and the i1 lowering loses ~35%
#     bandwidth);
#   * scalar variants: the scalar rides as a kernel argument (no 0-dim tensor,
#     no broadcast materialization);
#   * (bool tensor, int scalar): torch promotes to int64; the int64 output is
#     a fake container whose value channel is one 32-bit word per element, so
#     the kernel bitcasts the output pointer to int32 and stores one word per
#     element (kthvalue precedent).  Unlike AND the XOR result keeps the full
#     scalar value (x ^ s in {s, s^1}), so this arm requires the scalar to
#     fit int32; wider scalars fall back.
# Everything exotic (broadcasting, strided inputs, mixed tensor dtypes,
# out-of-int32-range sizes, int64 tensors) falls back to the generic kernels.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def bitwise_xor_flat_kernel(x_ptr, y_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        y = tl.load(y_ptr + off, mask=mask)
        tl.store(out_ptr + off, x ^ y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total", "y"])
def bitwise_xor_scalar_flat_kernel(x_ptr, out_ptr, y, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(out_ptr + off, x ^ y.to(x_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total", "y"])
def bitwise_xor_bool_int_flat_kernel(x_ptr, out_ptr, y, N_total, BLOCK: tl.constexpr):
    out32 = out_ptr.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        v = x.to(tl.int32) ^ y.to(tl.int32)
        tl.store(out32 + off, v, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_xor_func(x, y):
    return x ^ y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_xor_func_scalar(x, y):
    return x ^ y


def _choose_block(n_total, itemsize):
    # do_bench(median, warmup=1000ms/rep=100ms) sweep on GCU400 (L600):
    # num_warps=2 beats 4 on every bucket; tiny shapes want the exact pow2
    # tile, mid shapes peak at 65536 lanes, and only >2^28-element 1/2-byte
    # streams gain from 131072 (num_stages and a 2-block unroll both lose).
    if n_total <= 8192:
        return max(triton.next_power_of_2(n_total), 2)
    if n_total <= 65536:
        return 8192
    if n_total <= 524288:
        return 32768
    if n_total > (1 << 28) and itemsize <= 2:
        return 131072
    return 65536


def _lane_view(t):
    # Bool tensors ride the int8 lane: 0/1 bytes XOR identically and the
    # generic i1 lowering is ~35% off the byte-stream memory floor.
    return t.view(torch.int8) if t.dtype == torch.bool else t


def _launch_flat(kern, grid, args, block):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic host-level flake on shared nodes); retrying the
    # launch recompiles and succeeds (rad2deg / atan2_ / sinh precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(args[0].device):
                kern[grid](*args, num_warps=2, BLOCK=block)
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "bitwise_xor flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _fast_tensor(t):
    return (
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.is_contiguous()
        and t.dtype in _FAST_DTYPES
        and t.numel() <= _INT32_MAX
    )


def _flat_config(n_total, itemsize):
    block = _choose_block(n_total, itemsize)
    if triton.cdiv(n_total, block) * block > _INT32_MAX:
        return None
    return block, (min(triton.cdiv(n_total, block), NUM_SIPS * 2),)


def _scalar_lane_value(dtype, scalar):
    # Bitwise XOR commutes with truncation: (x ^ s) mod 2^L == x ^ (s mod 2^L),
    # so wrap the scalar into the lane dtype's range on the host (two's
    # complement for signed lanes) exactly like the AND precedent.
    if dtype == torch.bool:
        return 1 if scalar else 0
    bits = torch.iinfo(dtype).bits
    value = int(scalar) & ((1 << bits) - 1)
    if value >= (1 << (bits - 1)) and dtype.is_signed:
        value -= 1 << bits
    return value


def _is_python_scalar(v):
    return isinstance(v, bool) or isinstance(v, int)


def _xor_scalar_impl(tensor, scalar, out=None):
    # Shared by bitwise_xor_scalar / bitwise_xor_scalar_tensor (XOR commutes)
    # and by the in-place variant (out=tensor).  Returns None when the input
    # pair is not eligible for the flat path.
    if not _fast_tensor(tensor) or not _is_python_scalar(scalar):
        return None
    mixed = not isinstance(scalar, bool) and tensor.dtype == torch.bool
    if mixed and not (-(1 << 31) <= int(scalar) < (1 << 31)):
        # x ^ s keeps the full scalar (result in {s, s^1}); the mixed arm
        # stores one int32 word, so wider scalars stay on the generic path.
        return None
    out_dtype = torch.int64 if mixed else tensor.dtype
    if out is None:
        out = torch.empty_like(tensor, dtype=out_dtype)
    elif out.dtype != out_dtype:
        return None
    n_total = tensor.numel()
    if n_total == 0:
        return out
    config = _flat_config(n_total, tensor.element_size())
    if config is None:
        return None
    block, grid = config
    x_lanes = _lane_view(tensor)
    if mixed:
        # (bool, int) promotes to int64; store one int32 word per element into
        # the fake-int64 output (its value channel is a 4-byte slot).
        _launch_flat(
            bitwise_xor_bool_int_flat_kernel,
            grid,
            (x_lanes, out, int(scalar), n_total),
            block,
        )
    else:
        _launch_flat(
            bitwise_xor_scalar_flat_kernel,
            grid,
            (
                x_lanes,
                _lane_view(out),
                _scalar_lane_value(tensor.dtype, scalar),
                n_total,
            ),
            block,
        )
    return out


def bitwise_xor_tensor(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_XOR")
    if (
        _fast_tensor(A)
        and _fast_tensor(B)
        and A.dtype == B.dtype
        and A.shape == B.shape
        and A.device == B.device
    ):
        out = torch.empty_like(A)
        n_total = out.numel()
        if n_total == 0:
            return out
        config = _flat_config(n_total, A.element_size())
        if config is not None:
            block, grid = config
            _launch_flat(
                bitwise_xor_flat_kernel,
                grid,
                (_lane_view(A), _lane_view(B), _lane_view(out), n_total),
                block,
            )
            return out
    return bitwise_xor_func(A, B)


def bitwise_xor_tensor_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_XOR_")
    if (
        _fast_tensor(A)
        and _fast_tensor(B)
        and A.dtype == B.dtype
        and A.shape == B.shape
        and A.device == B.device
        and (A is B or A.untyped_storage().data_ptr() != B.untyped_storage().data_ptr())
    ):
        n_total = A.numel()
        if n_total == 0:
            return A
        config = _flat_config(n_total, A.element_size())
        if config is not None:
            block, grid = config
            _launch_flat(
                bitwise_xor_flat_kernel,
                grid,
                (_lane_view(A), _lane_view(B), _lane_view(A), n_total),
                block,
            )
            return A
    return bitwise_xor_func(A, B, out0=A)


def bitwise_xor_scalar(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_XOR_SCALAR")
    out = _xor_scalar_impl(A, B)
    if out is not None:
        return out
    return bitwise_xor_func_scalar(A, B)


def bitwise_xor_scalar_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_XOR_SCALAR_")
    if isinstance(A, torch.Tensor) and _is_python_scalar(B):
        if A.dtype == torch.bool and not isinstance(B, bool):
            # torch raises for the (bool, int) in-place promotion; the generic
            # pointwise path silently truncates instead, so keep aten parity.
            raise RuntimeError(
                "result type Long can't be cast to the desired output type Bool"
            )
        out = _xor_scalar_impl(A, B, out=A)
        if out is not None:
            return A
    return bitwise_xor_func_scalar(A, B, out0=A)


def bitwise_xor_scalar_tensor(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_XOR_SCALAR_TENSOR")
    # XOR commutes, so (scalar, tensor) reuses the (tensor, scalar) path.
    if _is_python_scalar(A):
        out = _xor_scalar_impl(B, A)
        if out is not None:
            return out
    return bitwise_xor_func_scalar(B, A)
