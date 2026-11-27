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
"""_foreach_sqrt_ : in-place elementwise sqrt over a TensorList.

Architecture (round 4): hybrid launches + a lean host launch path.

  * tensors are bucketed by dtype
  * a bucket of 1, or a bucket whose read+write traffic is large, runs one flat
    launch per tensor: the pointer is a real kernel argument, so Triton proves
    16-byte alignment and the accesses stay vectorized at ~1.2 TB/s
  * a small bucket with >= 2 members runs ONE fused 16-slot launch
    (slot = program_id(1), grid = (cdiv(max_numel, BLOCK), n_slots)); the
    numel/pointer select is a tl.where chain over int64 arguments, so there is
    no host->device metadata copy
  * gappy views use a generic strided kernel
  * per-call host cost is cut by caching the CompiledKernel that the normal JIT
    launch returns and re-invoking it through the same call that
    JITFunction.run makes at its tail (launch_metadata + CompiledKernel.run).
    The cache key reproduces Triton's own specialization classes, because a
    kernel compiled under a divisibility assumption is not valid for other
    values (measured: reusing an n%16==0 kernel for n=4097 wrote past the end).
  * f16/bf16 use f32 opmath with a single rounding back to the storage dtype
    (verified bit-identical to ATen's _foreach_sqrt_ on this target);
    f32/f64 use tl.sqrt directly (also bit-identical)
"""

import logging

import torch
import triton
import triton.language as tl
from triton import knobs as _knobs
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_unary import _foreach_sqrt_ as default__foreach_sqrt_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_BLOCK = 1024
_MAX_SLOTS = 16
# Fuse only when the bucket's read+write traffic is small enough that the scalar
# (non-vectorized) fused access pattern cannot cost more than the per-launch
# dispatch it removes.
_FUSE_MAX_TRAFFIC = 6e7
_I32_MAX = 2147483648

_HOOK_ENTER = _knobs.runtime.launch_enter_hook
_HOOK_EXIT = _knobs.runtime.launch_exit_hook


def _hooks_idle(hook):
    """True only when we can prove a launch hook cannot observe a launch."""
    if hook is None:
        return True
    hooks = getattr(hook, "hooks", None)
    if hooks is None:
        return False
    try:
        return len(hooks) == 0
    except TypeError:
        return False


# Only build the per-launch metadata dict when a hook could actually read it.
_NEED_LAUNCH_METADATA = not (_hooks_idle(_HOOK_ENTER) and _hooks_idle(_HOOK_EXIT))
_CK_CACHE = {}

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

_UPCONV = {
    torch.float16: True,
    torch.bfloat16: True,
    torch.float32: False,
    torch.float64: False,
}

_ITEMSIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}

_INT_NAMES = {
    torch.int64: "Long",
    torch.int32: "Int",
    torch.int16: "Short",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.bool: "Bool",
}

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


@triton.jit
def _sqrt_flat_kernel(x_ptr, n, UPCONV: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    # Pure in-place stream: neither the fetched line nor the written line is
    # ever re-read, so both are marked evict-first.
    x = tl.load(x_ptr + offs, mask=mask, eviction_policy="evict_first")
    if UPCONV:
        y = tl.sqrt(x.to(tl.float32)).to(x.dtype)
    else:
        y = tl.sqrt(x)
    tl.store(x_ptr + offs, y, mask=mask, eviction_policy="evict_first")


@triton.jit(
    do_not_specialize=[
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
        "p11",
        "p12",
        "p13",
        "p14",
        "p15",
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "n8",
        "n9",
        "n10",
        "n11",
        "n12",
        "n13",
        "n14",
        "n15",
    ]
)
def _sqrt_multi_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    DT: tl.constexpr,
    UPCONV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(1)
    base = tl.program_id(0) * BLOCK

    ptr_i = tl.where(
        slot == 0,
        p0,
        tl.where(
            slot == 1,
            p1,
            tl.where(
                slot == 2,
                p2,
                tl.where(
                    slot == 3,
                    p3,
                    tl.where(
                        slot == 4,
                        p4,
                        tl.where(
                            slot == 5,
                            p5,
                            tl.where(
                                slot == 6,
                                p6,
                                tl.where(
                                    slot == 7,
                                    p7,
                                    tl.where(
                                        slot == 8,
                                        p8,
                                        tl.where(
                                            slot == 9,
                                            p9,
                                            tl.where(
                                                slot == 10,
                                                p10,
                                                tl.where(
                                                    slot == 11,
                                                    p11,
                                                    tl.where(
                                                        slot == 12,
                                                        p12,
                                                        tl.where(
                                                            slot == 13,
                                                            p13,
                                                            tl.where(
                                                                slot == 14, p14, p15
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    n = tl.where(
        slot == 0,
        n0,
        tl.where(
            slot == 1,
            n1,
            tl.where(
                slot == 2,
                n2,
                tl.where(
                    slot == 3,
                    n3,
                    tl.where(
                        slot == 4,
                        n4,
                        tl.where(
                            slot == 5,
                            n5,
                            tl.where(
                                slot == 6,
                                n6,
                                tl.where(
                                    slot == 7,
                                    n7,
                                    tl.where(
                                        slot == 8,
                                        n8,
                                        tl.where(
                                            slot == 9,
                                            n9,
                                            tl.where(
                                                slot == 10,
                                                n10,
                                                tl.where(
                                                    slot == 11,
                                                    n11,
                                                    tl.where(
                                                        slot == 12,
                                                        n12,
                                                        tl.where(
                                                            slot == 13,
                                                            n13,
                                                            tl.where(
                                                                slot == 14, n14, n15
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    x_ptr = ptr_i.to(tl.pointer_type(DT))
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if UPCONV:
        y = tl.sqrt(x.to(tl.float32)).to(x.dtype)
    else:
        y = tl.sqrt(x)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit(
    do_not_specialize=[
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
        "p11",
        "p12",
        "p13",
        "p14",
        "p15",
        "n",
    ]
)
def _sqrt_multi_same_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n,
    DT: tl.constexpr,
    UPCONV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Same as _sqrt_multi_kernel but one numel shared by every slot."""
    slot = tl.program_id(1)
    base = tl.program_id(0) * BLOCK

    ptr_i = tl.where(
        slot == 0,
        p0,
        tl.where(
            slot == 1,
            p1,
            tl.where(
                slot == 2,
                p2,
                tl.where(
                    slot == 3,
                    p3,
                    tl.where(
                        slot == 4,
                        p4,
                        tl.where(
                            slot == 5,
                            p5,
                            tl.where(
                                slot == 6,
                                p6,
                                tl.where(
                                    slot == 7,
                                    p7,
                                    tl.where(
                                        slot == 8,
                                        p8,
                                        tl.where(
                                            slot == 9,
                                            p9,
                                            tl.where(
                                                slot == 10,
                                                p10,
                                                tl.where(
                                                    slot == 11,
                                                    p11,
                                                    tl.where(
                                                        slot == 12,
                                                        p12,
                                                        tl.where(
                                                            slot == 13,
                                                            p13,
                                                            tl.where(
                                                                slot == 14, p14, p15
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    x_ptr = ptr_i.to(tl.pointer_type(DT))
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if UPCONV:
        y = tl.sqrt(x.to(tl.float32)).to(x.dtype)
    else:
        y = tl.sqrt(x)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit
def _sqrt_strided_kernel(
    x_ptr,
    n,
    d0,
    t0,
    d1,
    t1,
    d2,
    t2,
    d3,
    t3,
    d4,
    t4,
    d5,
    t5,
    d6,
    t6,
    d7,
    t7,
    UPCONV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    q = idx
    off = (q % d0) * t0
    q = q // d0
    off += (q % d1) * t1
    q = q // d1
    off += (q % d2) * t2
    q = q // d2
    off += (q % d3) * t3
    q = q // d3
    off += (q % d4) * t4
    q = q // d4
    off += (q % d5) * t5
    q = q // d5
    off += (q % d6) * t6
    q = q // d6
    off += (q % d7) * t7
    x = tl.load(x_ptr + off, mask=mask)
    if UPCONV:
        y = tl.sqrt(x.to(tl.float32)).to(x.dtype)
    else:
        y = tl.sqrt(x)
    tl.store(x_ptr + off, y, mask=mask)


def _is_flat_addressable(t):
    """True when [data_ptr, data_ptr + numel) covers exactly this tensor."""
    if t.is_contiguous():
        return True
    expected = 1
    for stride, size in sorted(
        (s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _has_internal_overlap(t):
    fn = getattr(torch, "_debug_has_internal_overlap", None)
    if fn is not None:
        return fn(t) == 1
    for stride, size in zip(t.stride(), t.shape):
        if stride == 0 and size > 1:
            return True
    return False


def _stride_dims(t):
    """Up to 8 (size, stride) pairs, outermost first, padded with (1, 0)."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if len(dims) > 8:
        raise RuntimeError("_foreach_sqrt_: rank > 8 strided tensor unsupported")
    pad = [(1, 0)] * (8 - len(dims))
    return pad + dims


def _ptr_type_key(ptrs):
    """Specialization class of the int64 pointer arguments' mangled type.

    Triton mangles an int argument as i32 when its value fits in int32 and as
    i64 otherwise, and the mangled type is part of the compiled signature, so a
    cached kernel may only be reused when every pointer keeps the same class.
    """
    if min(ptrs) >= _I32_MAX:
        return 1
    mask = 0
    for ptr in ptrs:
        mask = (mask << 1) | (1 if ptr >= _I32_MAX else 0)
    return mask | (1 << 17)


def _launch(jit_fn, key, grid, args):
    """Launch through a cached precompiled kernel.

    The cached CompiledKernel is only reused when the specialization key matches
    exactly; otherwise the ordinary JIT path compiles/looks it up and the kernel
    it returns is cached. `args` are the kernel arguments in declaration order
    (Triton's own bound-argument values).
    """
    device = _driver.active.get_current_device()
    stream = _driver.active.get_current_stream(device)
    device_cache = _CK_CACHE.get(device)
    if device_cache is None:
        device_cache = _CK_CACHE[device] = {}
    ck = device_cache.get(key)
    if ck is None:
        ck = jit_fn[grid](*args, num_warps=4)
        device_cache[key] = ck
        return
    metadata = (
        ck.launch_metadata(grid, stream, *args) if _NEED_LAUNCH_METADATA else None
    )
    ck.run(
        grid[0],
        grid[1] if len(grid) > 1 else 1,
        1,
        stream,
        ck.function,
        ck.packed_metadata,
        metadata,
        _HOOK_ENTER,
        _HOOK_EXIT,
        *args,
    )


def _foreach_sqrt_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_SQRT_")
    if not isinstance(self, (list, tuple)) or any(
        not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in self
    ):
        return default__foreach_sqrt_(self)
    if isinstance(self, torch.Tensor) or not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    groups = {}
    strided = []
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dtype = t.dtype
        upconv = _UPCONV.get(dtype)
        if upconv is None:
            name = _INT_NAMES.get(dtype)
            if name is None:
                name = str(dtype).replace("torch.", "").capitalize()
            raise RuntimeError(
                "result type Float can't be cast to the desired output type " + name
            )
        numel = t.numel()
        if numel == 0:
            continue
        if numel == 1 or _is_flat_addressable(t):
            # one metadata pass: (tensor, numel, data_ptr) is everything the
            # launches below need
            item = (t, numel, t.data_ptr())
            bucket = groups.get(dtype)
            if bucket is None:
                groups[dtype] = [item]
            else:
                bucket.append(item)
        else:
            if _has_internal_overlap(t):
                raise RuntimeError(_OVERLAP_MSG)
            strided.append((t, numel, _stride_dims(t)))

    for dtype, bucket in groups.items():
        upconv = _UPCONV[dtype]
        count = len(bucket)
        traffic = 0
        for item in bucket:
            traffic += item[1]
        traffic *= _ITEMSIZE[dtype] * 2
        if count < 2 or traffic > _FUSE_MAX_TRAFFIC:
            # per-tensor flat launches: real pointer arguments keep the
            # 16-byte vectorized access pattern
            for t, n, ptr in bucket:
                key = (0, dtype, (n & 15) == 0, n == 1, (ptr & 15) == 0, n >> 31)
                _launch(
                    _sqrt_flat_kernel,
                    key,
                    ((n + _BLOCK - 1) // _BLOCK,),
                    (t, n, upconv, _BLOCK),
                )
            continue

        tl_dtype = _TL_DTYPE[dtype]
        start = 0
        while start < count:
            stop = start + _MAX_SLOTS
            if stop > count:
                stop = count
            chunk = bucket[start:stop]
            start = stop
            slots = len(chunk)
            ptrs = [item[2] for item in chunk]
            numels = [item[1] for item in chunk]
            max_n = max(numels)
            grid = ((max_n + _BLOCK - 1) // _BLOCK, slots)
            ptr_key = _ptr_type_key(ptrs)
            pad = _MAX_SLOTS - slots
            if pad:
                ptrs += [ptrs[-1]] * pad
            if min(numels) == max_n:
                key = (2, dtype, ptr_key, max_n >> 31)
                args = (
                    ptrs[0],
                    ptrs[1],
                    ptrs[2],
                    ptrs[3],
                    ptrs[4],
                    ptrs[5],
                    ptrs[6],
                    ptrs[7],
                    ptrs[8],
                    ptrs[9],
                    ptrs[10],
                    ptrs[11],
                    ptrs[12],
                    ptrs[13],
                    ptrs[14],
                    ptrs[15],
                    max_n,
                    tl_dtype,
                    upconv,
                    _BLOCK,
                )
                _launch(_sqrt_multi_same_kernel, key, grid, args)
            else:
                chunk_numels = numels + [0] * pad
                key = (1, dtype, ptr_key, max_n >> 31)
                args = (
                    ptrs[0],
                    ptrs[1],
                    ptrs[2],
                    ptrs[3],
                    ptrs[4],
                    ptrs[5],
                    ptrs[6],
                    ptrs[7],
                    ptrs[8],
                    ptrs[9],
                    ptrs[10],
                    ptrs[11],
                    ptrs[12],
                    ptrs[13],
                    ptrs[14],
                    ptrs[15],
                    chunk_numels[0],
                    chunk_numels[1],
                    chunk_numels[2],
                    chunk_numels[3],
                    chunk_numels[4],
                    chunk_numels[5],
                    chunk_numels[6],
                    chunk_numels[7],
                    chunk_numels[8],
                    chunk_numels[9],
                    chunk_numels[10],
                    chunk_numels[11],
                    chunk_numels[12],
                    chunk_numels[13],
                    chunk_numels[14],
                    chunk_numels[15],
                    tl_dtype,
                    upconv,
                    _BLOCK,
                )
                _launch(_sqrt_multi_kernel, key, grid, args)

    for t, numel, dims in strided:
        _sqrt_strided_kernel[(triton.cdiv(numel, _BLOCK),)](
            t,
            numel,
            dims[0][0],
            dims[0][1],
            dims[1][0],
            dims[1][1],
            dims[2][0],
            dims[2][1],
            dims[3][0],
            dims[3][1],
            dims[4][0],
            dims[4][1],
            dims[5][0],
            dims[5][1],
            dims[6][0],
            dims[6][1],
            dims[7][0],
            dims[7][1],
            _UPCONV[t.dtype],
            _BLOCK,
            num_warps=4,
        )
    return None
