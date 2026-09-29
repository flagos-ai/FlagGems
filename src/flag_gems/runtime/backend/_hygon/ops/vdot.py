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
from torch import Tensor

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Below these many (complex) elements, one program with an internal loop beats
# any multi-program scheme: exactly one kernel launch, no partial buffer and
# no zero-init fill. vdot is launch/latency-bound in this regime.
_SINGLE_PROG_THRESHOLD = 262144
_SINGLE_PROG_THRESHOLD_COMPLEX = 65536

_MULTI_GRID_CAP = 2048


@libentry()
@triton.jit()
def dot_kernel_single(
    inp_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    inp_stride: tl.constexpr,
    other_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program reduces the whole tensor and stores the scalar directly.
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(0, n_elements, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(inp_ptr + offs * inp_stride, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(other_ptr + offs * other_stride, mask=mask, other=0.0).to(
            tl.float32
        )
        acc += a * b
    tl.store(out_ptr, tl.sum(acc).to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit()
def dot_kernel_multi(
    inp_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    inp_stride: tl.constexpr,
    other_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid-stride loop; each program accumulates a partial and atomically adds
    # it to out_ptr (the wrapper must zero-initialise it).
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(pid * BLOCK_SIZE, n_elements, num_progs * BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(inp_ptr + offs * inp_stride, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(other_ptr + offs * other_stride, mask=mask, other=0.0).to(
            tl.float32
        )
        acc += a * b
    tl.atomic_add(out_ptr, tl.sum(acc))


@triton.jit
def _c64_dot_body(a64, b64, acc_re, acc_im):
    # Deinterleave [re, im] pairs from 8-byte words: coalesced wide loads,
    # no strided access. Bit-exact reinterpretation of the f32 lanes.
    ar = (a64 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    ai = (a64 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    br = (b64 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    bi = (b64 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    # conj(a) * b = (ar - i*ai) * (br + i*bi)
    acc_re += ar * br + ai * bi
    acc_im += ar * bi - ai * br
    return acc_re, acc_im


@libentry()
@triton.jit()
def vdot_c64_kernel_single(
    a64_ptr,
    b64_ptr,
    out_ptr,
    n_complex,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    acc_re = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_im = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(0, n_complex, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_complex
        a64 = tl.load(a64_ptr + offs * stride, mask=mask, other=0)
        b64 = tl.load(b64_ptr + offs * stride, mask=mask, other=0)
        acc_re, acc_im = _c64_dot_body(a64, b64, acc_re, acc_im)
    tl.store(out_ptr, tl.sum(acc_re).to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + 1, tl.sum(acc_im).to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit()
def vdot_c64_kernel_multi(
    a64_ptr,
    b64_ptr,
    out_ptr,
    n_complex,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    acc_re = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_im = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(pid * BLOCK_SIZE, n_complex, num_progs * BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_complex
        a64 = tl.load(a64_ptr + offs * stride, mask=mask, other=0)
        b64 = tl.load(b64_ptr + offs * stride, mask=mask, other=0)
        acc_re, acc_im = _c64_dot_body(a64, b64, acc_re, acc_im)
    tl.atomic_add(out_ptr, tl.sum(acc_re))
    tl.atomic_add(out_ptr + 1, tl.sum(acc_im))


@libentry()
@triton.jit()
def vdot_c128_kernel_single(
    inp_ptr,
    other_ptr,
    out_ptr,
    n_complex,
    inp_stride: tl.constexpr,
    other_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    acc_re = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    acc_im = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    for start in range(0, n_complex, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_complex
        ar = tl.load(inp_ptr + 2 * offs * inp_stride, mask=mask, other=0.0)
        ai = tl.load(inp_ptr + 2 * offs * inp_stride + 1, mask=mask, other=0.0)
        br = tl.load(other_ptr + 2 * offs * other_stride, mask=mask, other=0.0)
        bi = tl.load(other_ptr + 2 * offs * other_stride + 1, mask=mask, other=0.0)
        acc_re += ar * br + ai * bi
        acc_im += ar * bi - ai * br
    tl.store(out_ptr, tl.sum(acc_re))
    tl.store(out_ptr + 1, tl.sum(acc_im))


@libentry()
@triton.jit()
def vdot_c128_kernel_multi(
    inp_ptr,
    other_ptr,
    out_ptr,
    n_complex,
    inp_stride: tl.constexpr,
    other_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    acc_re = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    acc_im = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    for start in range(pid * BLOCK_SIZE, n_complex, num_progs * BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_complex
        ar = tl.load(inp_ptr + 2 * offs * inp_stride, mask=mask, other=0.0)
        ai = tl.load(inp_ptr + 2 * offs * inp_stride + 1, mask=mask, other=0.0)
        br = tl.load(other_ptr + 2 * offs * other_stride, mask=mask, other=0.0)
        bi = tl.load(other_ptr + 2 * offs * other_stride + 1, mask=mask, other=0.0)
        acc_re += ar * br + ai * bi
        acc_im += ar * bi - ai * br
    tl.atomic_add(out_ptr, tl.sum(acc_re))
    tl.atomic_add(out_ptr + 1, tl.sum(acc_im))


def vdot(input: Tensor, other: Tensor):
    logger.debug("GEMS VDOT")

    assert (
        input.dtype == other.dtype
    ), f"Input tensors must have the same dtype. Got {input.dtype} and {other.dtype}."
    assert (
        input.ndim == 1 and other.ndim == 1
    ), f"Input tensors must be 1D. Got {input.ndim}D and {other.ndim}D."
    assert (
        input.size() == other.size()
    ), f"Input tensors must have the same size. Got {input.size()} and {other.size()}."

    inp = input
    if inp.is_complex():
        # resolve_conj physically materializes the logical values (with conj
        # applied), so afterwards both tensors are plain conjugate-free views.
        if inp.is_conj():
            inp = torch.resolve_conj(inp)
        if other.is_conj():
            other = torch.resolve_conj(other)

        n_complex = inp.numel()
        inp_stride = inp.stride(0)
        other_stride = other.stride(0)
        single = n_complex <= _SINGLE_PROG_THRESHOLD_COMPLEX

        if inp.dtype == torch.complex64:
            # Reinterpret the interleaved [re, im] pairs as int64 lanes so the
            # kernel issues wide coalesced loads (strides are in elements and
            # complex64 pairs are 8 bytes, so the element stride carries over).
            if inp_stride == other_stride:
                a64 = inp.view(torch.int64)
                b64 = other.view(torch.int64)
                if single:
                    buf = torch.empty(2, dtype=torch.float32, device=inp.device)
                    vdot_c64_kernel_single[(1,)](
                        a64,
                        b64,
                        buf,
                        n_complex,
                        stride=inp_stride,
                        BLOCK_SIZE=4096,
                        num_warps=8,
                    )
                else:
                    buf = torch.zeros(2, dtype=torch.float32, device=inp.device)
                    grid = (min(triton.cdiv(n_complex, 4096), _MULTI_GRID_CAP),)
                    vdot_c64_kernel_multi[grid](
                        a64,
                        b64,
                        buf,
                        n_complex,
                        stride=inp_stride,
                        BLOCK_SIZE=4096,
                        num_warps=8,
                    )
                return torch.view_as_complex(buf)

        # complex128, or complex64 with mismatched strides: element-wise access
        # via the real view (pointer dtype comes from the view: f32 for c64,
        # f64 for c128, so one kernel source covers both).
        inp_real = torch.view_as_real(inp)
        other_real = torch.view_as_real(other)
        real_dtype = torch.float32 if inp.dtype == torch.complex64 else torch.float64
        if single:
            buf = torch.empty(2, dtype=real_dtype, device=inp.device)
            vdot_c128_kernel_single[(1,)](
                inp_real,
                other_real,
                buf,
                n_complex,
                inp_stride=inp_stride,
                other_stride=other_stride,
                BLOCK_SIZE=2048,
                num_warps=8,
            )
        else:
            buf = torch.zeros(2, dtype=real_dtype, device=inp.device)
            grid = (min(triton.cdiv(n_complex, 2048), _MULTI_GRID_CAP),)
            vdot_c128_kernel_multi[grid](
                inp_real,
                other_real,
                buf,
                n_complex,
                inp_stride=inp_stride,
                other_stride=other_stride,
                BLOCK_SIZE=2048,
                num_warps=8,
            )
        return torch.view_as_complex(buf)

    n_elements = inp.numel()
    inp_stride = inp.stride(0)
    other_stride = other.stride(0)
    if n_elements <= _SINGLE_PROG_THRESHOLD:
        output = torch.empty([], dtype=input.dtype, device=inp.device)
        dot_kernel_single[(1,)](
            inp,
            other,
            output,
            n_elements,
            inp_stride=inp_stride,
            other_stride=other_stride,
            BLOCK_SIZE=4096,
            num_warps=8,
        )
        return output

    acc = torch.zeros([], dtype=torch.float32, device=inp.device)
    grid = (min(triton.cdiv(n_elements, 4096), _MULTI_GRID_CAP),)
    dot_kernel_multi[grid](
        inp,
        other,
        acc,
        n_elements,
        inp_stride=inp_stride,
        other_stride=other_stride,
        BLOCK_SIZE=4096,
        num_warps=8,
    )
    if input.dtype == torch.float32:
        return acc
    return acc.to(input.dtype)
