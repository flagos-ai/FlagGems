# Copyright (c) FlagGems Team. All Rights Reserved.
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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def _c_rem(a, b):
    # C-style signed remainder: sign follows the dividend a, magnitude is
    # computed via unsigned abs so INT_MIN never overflows.
    ua = tl.where(a < 0, 0 - a.to(tl.uint64), a.to(tl.uint64))
    ub = tl.where(b < 0, 0 - b.to(tl.uint64), b.to(tl.uint64))
    mag = ua % ub
    rem = mag.to(a.dtype)
    return tl.where((a < 0) & (mag != 0), -rem, rem)


@libentry()
@triton.jit
def gcd_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr,
               MINV: tl.constexpr, NITER: tl.constexpr):
    pid = ext.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int64)
    y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.int64)

    # torch.gcd semantics: gcd(x,0)=|x|, gcd(0,y)=|y|, gcd(0,0)=0.
    # For INT_MIN, abs overflows in the native dtype; keeping the value
    # (like std::abs) makes the signed-modulo chain below reproduce
    # torch's sign propagation, e.g. gcd(INT_MIN, 6) == -2.
    ax = tl.where(x == MINV, x, tl.abs(x))
    ay = tl.where(y == MINV, y, tl.abs(y))

    # Fixed-iteration signed-modulo Euclid. The XPU backend cannot compile
    # the generic kernel's data-dependent while tl.sum(active) > 0
    # loops (scalar reductions are unsupported), so iterate a worst-case
    # bound and gate each step with active (Lamé: <= 2*bitwidth steps).
    sa = ax
    sb = ay
    active = mask & (sa != 0)
    for _ in range(0, NITER):
        bb = tl.where(sa != 0, sa, 1)
        r = _c_rem(sb, bb)
        next_sa = tl.where(active, r, sa)
        sb = tl.where(active, sa, sb)
        sa = next_sa
        active = active & (sa != 0)

    tl.store(out_ptr + offsets, sb.to(out_ptr.type.element_ty), mask=mask)


def _kernel_meta(dtype):
    # BLOCK is capped at 128: the XPU compiler cannot lower this int64-heavy
    # fixed-iteration Euclid loop for larger blocks. BLOCK=256 fails in the
    # uni_sram pass (OutOfResources: PassManager::run failed), BLOCK=512 fails
    # on the int64 comparison inside the loop
    # (arith.cmpi: result type must be i1 with same shape as operands).
    if dtype == torch.int16:
        return 128, 4, -(1 << 15), 64
    if dtype == torch.int32:
        return 128, 4, -(1 << 31), 64
    return 128, 4, -(1 << 63), 128


_TL_DTYPE = {torch.int16: tl.int16, torch.int32: tl.int32, torch.int64: tl.int64}


@triton.jit
def _bcast_copy_kernel(src_ptr, dst_ptr, n_elements, RANK: tl.constexpr,
                       S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                       S3: tl.constexpr, S4: tl.constexpr, S5: tl.constexpr,
                       T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr,
                       T3: tl.constexpr, T4: tl.constexpr, T5: tl.constexpr,
                       OUT_DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    # Gather-materialize a possibly-strided / broadcast (stride-0) source into a
    # contiguous buffer. Torch-level copy_ kernels are broken on this XPU for
    # strided sources (CUDA error: invalid device function), so we compute the
    # source offset ourselves: decompose the flat output index into
    # multi-dimensional coordinates and dot with the source strides. The rank is
    # unrolled into constexpr branches because constexpr loop arithmetic is
    # unreliable in this Triton build.
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    rem = idx
    src_off = tl.zeros([BLOCK], dtype=tl.int32)
    if RANK == 0:
        src_off = tl.zeros([BLOCK], dtype=tl.int32)
    elif RANK == 1:
        src_off += rem * T0
    elif RANK == 2:
        c = rem % S1
        rem = rem // S1
        src_off += c * T1
        src_off += rem * T0
    elif RANK == 3:
        c = rem % S2
        rem = rem // S2
        src_off += c * T2
        c = rem % S1
        rem = rem // S1
        src_off += c * T1
        src_off += rem * T0
    elif RANK == 4:
        c = rem % S3
        rem = rem // S3
        src_off += c * T3
        c = rem % S2
        rem = rem // S2
        src_off += c * T2
        c = rem % S1
        rem = rem // S1
        src_off += c * T1
        src_off += rem * T0
    elif RANK == 5:
        c = rem % S4
        rem = rem // S4
        src_off += c * T4
        c = rem % S3
        rem = rem // S3
        src_off += c * T3
        c = rem % S2
        rem = rem // S2
        src_off += c * T2
        c = rem % S1
        rem = rem // S1
        src_off += c * T1
        src_off += rem * T0
    elif RANK == 6:
        c = rem % S5
        rem = rem // S5
        src_off += c * T5
        c = rem % S4
        rem = rem // S4
        src_off += c * T4
        c = rem % S3
        rem = rem // S3
        src_off += c * T3
        c = rem % S2
        rem = rem // S2
        src_off += c * T2
        c = rem % S1
        rem = rem // S1
        src_off += c * T1
        src_off += rem * T0
    val = tl.load(src_ptr + src_off, mask=mask, other=0).to(OUT_DTYPE)
    tl.store(dst_ptr + idx, val, mask=mask)


def _to_full_contiguous(t, shape, dtype):
    # Return a contiguous tensor of the given shape / dtype, broadcasting
    # (via the gem expand override, a pure as_strided view) and materializing
    # with our own triton gather kernel. Empty inputs short-circuit.
    if t.numel() == 0:
        return torch.empty(shape, dtype=dtype, device=t.device)
    if t.shape == shape and t.is_contiguous() and t.dtype == dtype:
        return t
    if t.shape != shape:
        t = t.expand(shape)
    out = torch.empty(shape, dtype=dtype, device=t.device)
    numel = out.numel()
    rank = len(shape)
    if rank > 6:
        raise NotImplementedError(f"gcd broadcast rank {rank} > 6 not supported")
    shape_t = tuple(shape) + (1,) * 6
    stride_t = tuple(t.stride()) + (1,) * 6
    with torch_device_fn.device(out.device):
        grid = (triton.cdiv(numel, 128),)
        _bcast_copy_kernel[grid](
            t, out, numel, RANK=rank,
            S0=shape_t[0], S1=shape_t[1], S2=shape_t[2],
            S3=shape_t[3], S4=shape_t[4], S5=shape_t[5],
            T0=stride_t[0], T1=stride_t[1], T2=stride_t[2],
            T3=stride_t[3], T4=stride_t[4], T5=stride_t[5],
            OUT_DTYPE=_TL_DTYPE[dtype], BLOCK=128, num_warps=4)
    return out


def _materialize_inputs(self, other):
    promoted_dtype = torch.promote_types(self.dtype, other.dtype)
    shape = torch.broadcast_shapes(self.shape, other.shape)
    lhs = _to_full_contiguous(self, shape, promoted_dtype)
    rhs = _to_full_contiguous(other, shape, promoted_dtype)
    return lhs, rhs, promoted_dtype


def _launch_gcd(lhs, rhs, out):
    numel = out.numel()
    if numel == 0:
        return out
    block, num_warps, minv, niter = _kernel_meta(out.dtype)
    grid = (triton.cdiv(numel, block),)
    with torch_device_fn.device(out.device):
        gcd_kernel[grid](lhs, rhs, out, numel, BLOCK=block, MINV=minv,
                         NITER=niter, num_warps=num_warps)
    return out


def gcd(self, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GCD")
    lhs, rhs, promoted_dtype = _materialize_inputs(self, other)
    result = torch.empty_like(lhs, dtype=promoted_dtype)
    _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), result.reshape(-1))
    result = result.view(lhs.shape)
    if out is None:
        return result

    out.copy_(result)
    return out


def gcd_out(self, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GCD_OUT")
    if out is None:
        return gcd(self, other)
    return gcd(self, other, out=out)


def gcd_(A, B):
    logger.debug("GEMS_KUNLUNXIN GCD_")
    lhs, rhs, promoted_dtype = _materialize_inputs(A, B)
    flat_out = torch.empty(lhs.numel(), dtype=promoted_dtype, device=A.device)
    _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), flat_out)
    A.copy_(flat_out.view(A.shape))
    return A
