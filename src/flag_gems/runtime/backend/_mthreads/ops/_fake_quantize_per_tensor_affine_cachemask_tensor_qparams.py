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

"""Triton implementation of
aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.

Reference semantics verified on the target (aten, MUSA MTT S5000):

    s = float32(scale[0]); z = float32(zero_point[0]); en = (enabled[0] != 0)
    xf = float32(x)                      # fp16/bf16 elements go through fp32
    q  = rint_half_even(xf / s) + z      # NOTE: rint BEFORE adding zero_point
    in_range = (q >= quant_min) & (q <= quant_max)   # pre-clamp
    qc = min(max(q, quant_min), quant_max)
    out  = en ? cast_to_input_dtype((qc - z) * s) : x   (bitwise identity of x)
    mask = en ? in_range : true

Because scale/zero_point/quant_min/quant_max are all exactly representable
fp32 integers or quotient-rounded values, the kernel below uses the exactly
equivalent form

    r  = rint_half_even(xf / s);  lo = quant_min - z;  hi = quant_max - z
    in_range = (r >= lo) & (r <= hi)
    out = clamp(r, lo, hi) * s

(all intermediate integer-valued arithmetic is exact in fp32), which drops one
add and one subtract per element, and selects the enabled/disabled behaviour
with a uniform branch instead of two per-element selects.  Full blocks are also
emitted without tail predicates; only the single partial tail program pays for
masking.

Layout rules are aten's `empty_like` (MemoryFormat::Preserve) rules: output and
mask keep the input strides when the input is non-overlapping-and-dense, and are
contiguous otherwise.  For a dense tensor the (index -> storage offset) map is a
bijection onto [0, numel) relative to the view's data_ptr, so the dense path is a
plain memory-order stream with no index arithmetic; only non-dense inputs need a
per-element stride decomposition.
"""

import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

# The generic implementation lives in a module whose name shadows the function,
# so import it via the module object rather than `from flag_gems.ops import ...`
# (the package re-export binds the function, not the module).
_default_mod = importlib.import_module(
    "flag_gems.ops._fake_quantize_per_tensor_affine_cachemask_tensor_qparams"
)
default_fake_quantize_per_tensor_affine_cachemask_tensor_qparams = (
    _default_mod._fake_quantize_per_tensor_affine_cachemask_tensor_qparams
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_ld = tl_extra_shim

_MAX_DIMS = 8


@triton.jit
def _fq_body(
    x_ptr,
    scale_ptr,
    zp_ptr,
    en_ptr,
    out_ptr,
    mask_ptr,
    in_off,
    out_off,
    m,
    qmin_f,
    qmax_f,
    BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    s = tl.load(scale_ptr).to(tl.float32)
    z = tl.load(zp_ptr).to(tl.float32)
    en = tl.load(en_ptr) != 0
    if HAS_MASK:
        x_raw = tl.load(x_ptr + in_off, mask=m, other=0.0)
    else:
        x_raw = tl.load(x_ptr + in_off)
    if en:
        r = _ld.rint(x_raw.to(tl.float32) / s)
        lo = qmin_f - z
        hi = qmax_f - z
        y = tl.minimum(tl.maximum(r, lo), hi) * s
        v = (r >= lo) & (r <= hi)
        if HAS_MASK:
            tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty), mask=m)
            tl.store(mask_ptr + out_off, v, mask=m)
        else:
            tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty))
            tl.store(mask_ptr + out_off, v)
    else:
        v = tl.full([BLOCK], 1, tl.int1)
        if HAS_MASK:
            tl.store(out_ptr + out_off, x_raw, mask=m)
            tl.store(mask_ptr + out_off, v, mask=m)
        else:
            tl.store(out_ptr + out_off, x_raw)
            tl.store(mask_ptr + out_off, v)


@triton.jit(do_not_specialize=["n_elements"])
def _fq_flat_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    en_ptr,
    out_ptr,
    mask_ptr,
    n_elements,
    qmin_f,
    qmax_f,
    BLOCK: tl.constexpr,
):
    base = tl.program_id(0) * BLOCK
    offs = base + tl.arange(0, BLOCK)
    if base + BLOCK <= n_elements:
        _fq_body(
            x_ptr,
            scale_ptr,
            zp_ptr,
            en_ptr,
            out_ptr,
            mask_ptr,
            offs,
            offs,
            offs,
            qmin_f,
            qmax_f,
            BLOCK,
            False,
        )
    else:
        _fq_body(
            x_ptr,
            scale_ptr,
            zp_ptr,
            en_ptr,
            out_ptr,
            mask_ptr,
            offs,
            offs,
            offs < n_elements,
            qmin_f,
            qmax_f,
            BLOCK,
            True,
        )


@triton.jit(do_not_specialize=["n_elements", "size_class", "dtype_tag"])
def _fq_flat_auto_body(
    x_ptr,
    scale_ptr,
    zp_ptr,
    en_ptr,
    out_ptr,
    mask_ptr,
    n_elements,
    qmin_f,
    qmax_f,
    size_class,
    dtype_tag,
    BLOCK: tl.constexpr,
):
    base = tl.program_id(0) * BLOCK
    offs = base + tl.arange(0, BLOCK)
    if base + BLOCK <= n_elements:
        _fq_body(
            x_ptr,
            scale_ptr,
            zp_ptr,
            en_ptr,
            out_ptr,
            mask_ptr,
            offs,
            offs,
            offs,
            qmin_f,
            qmax_f,
            BLOCK,
            False,
        )
    else:
        _fq_body(
            x_ptr,
            scale_ptr,
            zp_ptr,
            en_ptr,
            out_ptr,
            mask_ptr,
            offs,
            offs,
            offs < n_elements,
            qmin_f,
            qmax_f,
            BLOCK,
            True,
        )


# Bounded, explicit configuration search for the bandwidth-bound large shapes.
# BLOCK=1024 / num_warps=4 is the round-1 baseline and stays a candidate.
_LARGE_CONFIGS = [
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 8192}, num_warps=4),
    triton.Config({"BLOCK": 512}, num_warps=8),
    triton.Config({"BLOCK": 1024}, num_warps=8),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=8),
    triton.Config({"BLOCK": 16384}, num_warps=8),
    triton.Config({"BLOCK": 32768}, num_warps=8),
    triton.Config({"BLOCK": 2048}, num_warps=16),
    triton.Config({"BLOCK": 4096}, num_warps=16),
    triton.Config({"BLOCK": 8192}, num_warps=16),
    triton.Config({"BLOCK": 16384}, num_warps=16),
    triton.Config({"BLOCK": 32768}, num_warps=16),
]

_fq_flat_auto_kernel = triton.autotune(
    configs=_LARGE_CONFIGS, key=["size_class", "dtype_tag"]
)(_fq_flat_auto_body)


@triton.jit(
    do_not_specialize=[
        "n_elements",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "t7",
    ]
)
def _fq_strided_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    en_ptr,
    out_ptr,
    mask_ptr,
    n_elements,
    qmin_f,
    qmax_f,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
    SHARED_OUT: tl.constexpr,
):
    lin = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = lin < n_elements

    rem = lin
    i7 = rem % d7
    rem = rem // d7
    off = i7 * t7
    i6 = rem % d6
    rem = rem // d6
    off += i6 * t6
    i5 = rem % d5
    rem = rem // d5
    off += i5 * t5
    i4 = rem % d4
    rem = rem // d4
    off += i4 * t4
    i3 = rem % d3
    rem = rem // d3
    off += i3 * t3
    i2 = rem % d2
    rem = rem // d2
    off += i2 * t2
    i1 = rem % d1
    rem = rem // d1
    off += i1 * t1
    i0 = rem % d0
    off += i0 * t0

    if SHARED_OUT:
        out_off = off
    else:
        out_off = lin

    _fq_body(
        x_ptr,
        scale_ptr,
        zp_ptr,
        en_ptr,
        out_ptr,
        mask_ptr,
        off,
        out_off,
        m,
        qmin_f,
        qmax_f,
        BLOCK,
        True,
    )


def _launch_strided(x, out, mask, scale, zero_point, enabled, qmin_f, qmax_f, n):
    shape = list(x.shape)
    stride = list(x.stride())
    nd = len(shape)
    # Drop trivial dims and merge adjacent dims that are contiguous in memory.
    dims = []
    for d in range(nd):
        if shape[d] == 1:
            continue
        if dims and stride[dims[-1]] == stride[d] * shape[d]:
            pd = dims[-1]
            shape[pd] = shape[pd] * shape[d]
            stride[pd] = stride[d]
        else:
            dims.append(d)
    sizes = [shape[d] for d in dims]
    strides = [stride[d] for d in dims]
    if len(sizes) > _MAX_DIMS:
        for i in range(x.shape[0]):
            sub_n = x[i].numel()
            if sub_n:
                _launch_strided(
                    x[i],
                    out[i],
                    mask[i],
                    scale,
                    zero_point,
                    enabled,
                    qmin_f,
                    qmax_f,
                    sub_n,
                )
        return
    pad = _MAX_DIMS - len(sizes)
    sizes = sizes + [1] * pad
    strides = strides + [0] * pad
    shared = out.stride() == x.stride()

    if n <= 1024:
        block = max(triton.next_power_of_2(n), 16)
        grid = (1,)
    else:
        block = 1024
        grid = (triton.cdiv(n, block),)

    _fq_strided_kernel[grid](
        x,
        scale,
        zero_point,
        enabled,
        out,
        mask,
        n,
        qmin_f,
        qmax_f,
        *sizes,
        *strides,
        BLOCK=block,
        SHARED_OUT=shared,
        num_warps=4,
    )


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
):
    logger.debug(
        "GEMS_MTHREADS _FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS"
    )

    if not _use_triton_kernel(self):
        return default_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
            self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
        )

    with torch_device_fn.device(self.device):
        return _launch_tensor_qparams(
            self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
        )


def _launch_tensor_qparams(
    self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
):
    out = torch.empty_like(self)
    mask = torch.empty_like(self, dtype=torch.bool)
    n = self.numel()
    if n == 0:
        return out, mask

    qmin_f = float(quant_min)
    qmax_f = float(quant_max)

    # `empty_like` preserves the input strides exactly when the input is
    # non-overlapping-and-dense (contiguous, transposed, permuted, row-sliced)
    # and falls back to contiguous strides otherwise (e.g. x[::2]).  For a
    # dense tensor the (index -> storage offset) map is a bijection onto
    # [0, numel) relative to the view's data_ptr, so equality of the strides is
    # exactly the condition for the plain memory-order stream.
    if out.stride() == self.stride():
        if n <= 1024:
            block = max(triton.next_power_of_2(n), 16)
            _fq_flat_kernel[(1,)](
                self,
                scale,
                zero_point,
                fake_quant_enabled,
                out,
                mask,
                n,
                qmin_f,
                qmax_f,
                BLOCK=block,
                num_warps=4,
            )
            return out, mask
        if n < 65536:
            block = 1024
            _fq_flat_kernel[(triton.cdiv(n, block),)](
                self,
                scale,
                zero_point,
                fake_quant_enabled,
                out,
                mask,
                n,
                qmin_f,
                qmax_f,
                BLOCK=block,
                num_warps=4,
            )
            return out, mask

        dt = self.dtype
        if dt is torch.float32:
            tag = 0
        elif dt is torch.float16:
            tag = 1
        elif dt is torch.bfloat16:
            tag = 2
        else:
            tag = 3
        if n < (1 << 22):
            size_class = 1
        elif n < (1 << 26):
            size_class = 2
        else:
            size_class = 3
        _fq_flat_auto_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
            self,
            scale,
            zero_point,
            fake_quant_enabled,
            out,
            mask,
            n,
            qmin_f,
            qmax_f,
            size_class,
            tag,
        )
    else:
        _launch_strided(
            self, out, mask, scale, zero_point, fake_quant_enabled, qmin_f, qmax_f, n
        )

    return out, mask
