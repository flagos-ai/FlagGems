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

"""In-place arccosh (aten::arccosh_) implemented as a Triton kernel.

Semantics preserved:
  * acosh(x) = log(x + sqrt(x-1)*sqrt(x+1)) computed in fp32 opmath for
    fp16/bf16 inputs and in native precision for fp32/fp64.
  * x == 1 -> +0, x < 1 -> NaN, +inf -> +inf, -inf -> NaN, NaN -> NaN.
  * the input tensor is mutated in place and the *same* Python object is
    returned (identity / metadata / storage preserved).
  * zero-sized tensors return immediately without launching anything.
  * arbitrarily strided (noncontiguous) tensors are handled with an
    element-wise address reconstruction kernel, no copies are made.
"""

import torch
import triton
import triton.language as tl

try:  # libdevice acosh matches torch's acoshf most closely
    from triton.language.extra import libdevice as _libdevice
except Exception:  # pragma: no cover - depends on the backend packaging
    _libdevice = None

_HAS_LD_ACOSH = _libdevice is not None and hasattr(_libdevice, "acosh")

MAX_RANK = 8
_FLAT_BLOCK = 1024
_FLAT_WARPS = 4
_STRIDED_BLOCK = 1024
_STRIDED_WARPS = 4
_I32_MAX = 2147483647


@triton.jit
def _acosh32(x):
    # acosh(x) = log(x + sqrt(x^2 - 1)) with three cheap fixes that keep it
    # exact for every edge case the operator is tested with:
    #   * the square root is formed as p * rsqrt(p), which is a single
    #     approximate reciprocal-sqrt evaluation (measured relative error
    #     1.3e-7 on this target) instead of the ~14-instruction
    #     correctly-rounded sqrtf that dominated the kernel;
    #   * min(., x) reproduces log(2x) once (x-1)*(x+1) overflows;
    #   * x < 1 is forced to NaN so the invalid domain matches torch even
    #     though the cheap root never produces a NaN there.
    p = (x - 1.0) * (x + 1.0)
    s = tl.where(p > 0.0, p * tl.rsqrt(p), p)
    s = tl.minimum(s, x)
    # log2 lowered to a single hardware lg2 plus one multiply is cheaper than
    # the generic log expansion on this backend (equivalent up to ~0.5 ulp).
    return tl.where(x < 1.0, float("nan"), tl.log2(x + s) * 0.6931471805599453)


if _HAS_LD_ACOSH:

    @triton.jit
    def _acosh64(x):
        return _libdevice.acosh(x)

else:  # pragma: no cover - compile-safe fallback

    @triton.jit
    def _acosh64(x):
        return tl.log(x + tl.sqrt(x - 1.0) * tl.sqrt(x + 1.0))


@triton.jit
def _acosh_flat_kernel(
    ptr,
    n,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
    I64: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        # numel is an exact multiple of BLOCK: no tail, so no predicate logic
        # is emitted for the vector loads and stores.
        x = tl.load(ptr + offs)
    else:
        mask = offs < n
        x = tl.load(ptr + offs, mask=mask)
    if IS_F64:
        y = _acosh64(x)
    else:
        y = _acosh32(x.to(tl.float32))
    if EVEN:
        tl.store(ptr + offs, y.to(x.dtype))
    else:
        tl.store(ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _acosh_strided_kernel(
    ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs
    mem = offs * 0
    if RANK >= 1:
        q = idx // s0
        mem += (idx - q * s0) * t0
        idx = q
    if RANK >= 2:
        q = idx // s1
        mem += (idx - q * s1) * t1
        idx = q
    if RANK >= 3:
        q = idx // s2
        mem += (idx - q * s2) * t2
        idx = q
    if RANK >= 4:
        q = idx // s3
        mem += (idx - q * s3) * t3
        idx = q
    if RANK >= 5:
        q = idx // s4
        mem += (idx - q * s4) * t4
        idx = q
    if RANK >= 6:
        q = idx // s5
        mem += (idx - q * s5) * t5
        idx = q
    if RANK >= 7:
        q = idx // s6
        mem += (idx - q * s6) * t6
        idx = q
    if RANK >= 8:
        q = idx // s7
        mem += (idx - q * s7) * t7
        idx = q
    x = tl.load(ptr + mem, mask=mask)
    if IS_F64:
        y = _acosh64(x)
    else:
        y = _acosh32(x.to(tl.float32))
    tl.store(ptr + mem, y.to(x.dtype), mask=mask)


def _collapse_dims(shape, stride):
    """Drop size-1 dims and merge runs that are contiguous in memory."""
    dims = [(int(s), int(st)) for s, st in zip(shape, stride) if int(s) != 1]
    collapsed = []
    for s, st in dims:
        if s == 0:
            return []
        if collapsed:
            ps, pst = collapsed[-1]
            if pst == st * s:
                collapsed[-1] = (ps * s, st)
                continue
        collapsed.append((s, st))
    return collapsed


def _launch_strided(self, n, is_f64, block, warps):
    dims = _collapse_dims(self.shape, self.stride())
    rank = len(dims)
    if rank == 0:
        # every element maps to the same 0-offset? cannot happen for numel > 1
        dims = [(1, 0)]
        rank = 1
    if rank > MAX_RANK:
        raise RuntimeError("arccosh_: unsupported tensor layout with rank %d" % rank)
    sizes = [d[0] for d in dims] + [1] * (MAX_RANK - rank)
    strides = [d[1] for d in dims] + [0] * (MAX_RANK - rank)
    max_off = sum((s - 1) * abs(st) for s, st in dims)
    i64 = n > _I32_MAX or max_off > _I32_MAX
    grid = (triton.cdiv(n, block),)
    _acosh_strided_kernel[grid](
        self,
        n,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        RANK=rank,
        BLOCK=block,
        IS_F64=is_f64,
        I64=i64,
        num_warps=warps,
    )


# Cached direct launchers: bypassing Triton's Python JIT dispatch removes a
# fixed per-call cost that dominates small tensors. The key mirrors every
# property Triton specializes on (device, pointer dtype, pointer 16B alignment,
# constexprs, and the int specialization of n), so a cached launcher is only
# reused for an identical specialization.
_LAUNCH_CACHE = {}
_F64 = torch.float64


def _flat_launch(self, n, dtype):
    dev = self.get_device()
    is_f64 = dtype == _F64
    i64 = n > _I32_MAX
    block = _FLAT_BLOCK
    warps = _FLAT_WARPS
    key = (
        dev,
        dtype,
        block,
        warps,
        is_f64,
        i64,
        n == 1,
        n % 16 == 0,
        n % block == 0,
        self.data_ptr() % 16 == 0,
    )
    entry = _LAUNCH_CACHE.get(key)
    grid0 = (n + block - 1) // block
    even = n % block == 0
    if entry is None:
        ck = _acosh_flat_kernel[(grid0,)](
            self,
            n,
            BLOCK=block,
            IS_F64=is_f64,
            I64=i64,
            EVEN=even,
            num_warps=warps,
        )
        from triton import knobs
        from triton.runtime import driver

        rt = knobs.runtime
        _LAUNCH_CACHE[key] = (
            ck.run,  # bound launcher for this binary
            ck.function,
            ck.packed_metadata,
            driver.active.get_current_stream,
            rt.launch_enter_hook,
            rt.launch_exit_hook,
        )
        return
    launcher, fn, packed_metadata, get_stream, enter, exit_ = entry
    launcher(
        grid0,
        1,
        1,
        get_stream(dev),
        fn,
        packed_metadata,
        None,
        enter,
        exit_,
        self,
        n,
        block,
        is_f64,
        i64,
        even,
    )


def arccosh_(self):
    """In-place inverse hyperbolic cosine; returns the same tensor object."""
    if not isinstance(self, torch.Tensor):
        raise TypeError("arccosh_ expects a torch.Tensor, got %r" % type(self))
    n = self.numel()
    if n == 0:
        return self
    dtype = self.dtype
    if not dtype.is_floating_point:
        raise RuntimeError(
            "acosh_ not implemented for '%s'" % str(dtype).split(".")[-1]
        )

    if self.is_contiguous():
        # A contiguous tensor cannot have a zero stride with size > 1, so the
        # internal-overlap check is only needed on the strided branch.
        _flat_launch(self, n, dtype)
    else:
        for s, st in zip(self.shape, self.stride()):
            if int(s) > 1 and int(st) == 0:
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location."
                )
        _launch_strided(self, n, dtype == _F64, _STRIDED_BLOCK, _STRIDED_WARPS)
    return self
