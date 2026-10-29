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

"""Kunlunxin (XPU) implementation of
``fake_quantize_per_tensor_affine_cachemask`` / ``..._out``.

At benchmark scale this operator is **host bound, not memory bound**, and the
route is chosen by counting host calls rather than bytes.  ``benchmark/base.py``
OPERATOR mode times ``n_rep`` calls between two synchronizes, so every
microsecond of Python counts, while ``triton.testing.do_bench`` records an event
pair per iteration and reports the median, which hides host cost entirely --
the same call measures 10.4us one way and 71.6us the other.  Priced with the
harness's own wall clock on P800/KL3:

    triton launch   17.4us      torch.eq         11.6us
    empty_like       4.3us      empty             3.7us
    device guard     2.1us      view              1.1us
    contiguous       0.1us      is_contiguous     0.1us

torch answers the whole operator in 15.5us, and its ``out`` variant in 25.3us.
One Triton launch already costs more than torch's entire call, so below a few
hundred thousand elements nothing matters except how many host calls the route
makes.  Hence it switches on ``n``:

* **``n <= 131072``: one launch**, the ``i1`` mask stored straight from the
  kernel.  That per-lane byte store is pathological on this hardware (~3.2ns
  per element against ~0.1ns for a contiguous wide store) but here the device
  is idle waiting for Python, and skipping the scratch, the ``view`` and
  ``torch.eq`` saves 16.4us of host -- more than the store costs.
* **``n > 131072``: the scratch route.**  One flat kernel writes ``output``
  plus a float scratch that is ``0.0`` exactly where the mask is true, and
  ``torch.eq`` turns that into ``bool``.

The crossover is measured, not estimated: at n=98304 the one-launch route is
43.0us against the scratch route's 48.2us, and at n=262144 it is 84.1us
against 48.2us.  On the ``_out`` entry the switch is worth 0.485 -> 0.754 over
the eleven benchmark shapes, and it is worth 1.11 against torch on the four
smallest.

**The chain never materialises ``zero_point``.**  The reference is
``clamp(rne(x * inv_scale) + zp, qmin, qmax)``, but ``rne(...)`` and ``zp`` are
both integers far inside ``2**22``, so shifting the *bounds* instead of the
value is bit-identical::

    r   = (x * inv_scale + 1.5*2**23) - 1.5*2**23      # round-half-to-even
    c   = min(max(r, qmin - zp), qmax - zp)            # bounds shifted on host
    out = c * scale

For the scratch route the mask is ``d = min(|r - c|, 1.0)``, which is ``0.0``
iff the value is in range: ``r`` and ``c`` are integer-valued, so they differ
by at least 1 whenever the clamp bit and by 0 otherwise.  That makes the mask a
by-product of the clamp rather than eight extra vector ops -- nine ops per
element against seventeen -- and it is worth 18% of the whole route at 67M
elements (585us against 718us for the saturating-product form, f16).  The mask
comes out inverted, hence ``torch.eq`` rather than ``torch.ne``; the two cost
the same.

Unlike the per-channel operator, the reciprocal here *is* the reference: ATen
computes ``inv_scale = 1.0f / scale`` on the host and multiplies, so the
host-side ``struct.pack('f')`` round trip reproduces it bit-for-bit and there
is no accuracy argument for dividing.

Other measured choices:

* **``BLOCK = 16384``** for fp16 and bf16 above the crossover, **8192 for
  fp32**.  fp32 is 10% faster at 8192 on every large shape (741.6us against
  821.0us at 67M) where fp16 is 60% worse there (752.3us against 467.8us), and
  on bf16 a BLOCK of 32768 or more does not merely slow down, it raises a
  hardware kernel exception (``kl3ChannelCheckErrors ... status=700``) that
  poisons the device for the rest of the process.
* **``MASKED`` and ``BOOL`` are ``tl.constexpr``** so the bounds check and the
  whole mask form specialise away.
* **For fp32 input the scratch is fp16**, halving its share of the traffic
  (17n bytes per element down to 13n), worth 0.469 against 0.444 over the
  large shapes.  A scratch narrower than the output fails to compile at
  ``BLOCK=1024`` (``OutOfResources: uni_sram`` in ``TritonXPUVectorize``), but
  the scratch route only ever runs with ``BLOCK=16384``, so that limit cannot
  bite here.
* **``torch.eq``/``torch.ne`` into a ``bool`` out is the cheapest converter.**
  Gems overrides both, so it is a ``pointwise_dynamic`` launch costing 11.6us
  of host and ~2us on the device.  ``aten._copy_from`` is *not* overridden but
  does not perform this conversion at all (every element mismatches), and
  ``aten._to_copy`` costs 219us on the device where ``eq`` costs 10us.
* **``libentry`` is not a cheaper launch here, it is a dearer one** -- 33.5us
  against 17.2us for the same kernel, and it loses on every shape.  What *is*
  cheaper is caching the ``CompiledKernel`` by hand and calling its launcher
  directly, which skips the per-call ``binder`` + cache-key work inside
  ``JITFunction.run`` and takes the whole call from 30.5us to 7.3us (f16) /
  19.1us to 8.9us (f32) at n=16, bit identically.  See ``_run_kernel`` for why
  the hand cache key is complete.
* **``resize_`` is a dispatch each**, so the ``_out`` entry only resizes when
  the shape actually differs, and the device guard is only entered when the
  input is not already on the current device.  Together those are 5us of the
  22.9us the small shapes cost.

What the route cannot beat: above the crossover it moves 4nw + n bytes where
torch moves 2nw + n, and a zero-arithmetic kernel with the same traffic scores
only 0.60 (f16) / 0.53 (f32) -- that is the ceiling there.  Writing the 1-byte
``bool`` straight from the kernel is what would fix it, and it needs two or
more elements per lane: packing two mask bytes into an ``int16`` via
``1.0 + V * 2**-23`` and a bitcast *is* exact and avoids the pathological
``float -> int`` convert, but it forces stride-2 lane *loads*, which cost 130us
against an 8.0us contiguous floor at n=32768 and 479us against 9.5us at
n=262144.

bf16 is the weak dtype for a reason that is not the arithmetic: a kernel that
only copies costs 1049us through ``bfloat16`` pointers and 349us through an
``int16`` view of the same bytes -- three times, and the 349us is exactly the
f16 copy cost.  The XPU backend models bf16 as 32 bits internally (an
``i16 -> bf16`` bitcast fails to compile with "Cannot bitcast data-type of size
16 to data-type of size 32"), so every bf16 memory op pays a convert.  The
obvious fix -- keep the memory in ``int16`` and do the convert in registers --
is unavailable from Triton: the 16-bit bitcast needed to recover the float is
per-lane, 17x slower, and it is per-lane for f16 too (17697us against 493us at
67M), so this is the same trap as every other bit-manipulation route here.
Doing the convert as integer shifts on ``int32`` instead is bit-exact and just
as slow (18423us).
"""

import logging
import math
import struct
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Below this many elements one launch with the i1 store beats a launch plus a
# converter, because only host calls count there.  Measured: 43.0us vs 48.2us
# at n=98304, 84.1us vs 48.2us at n=262144.
_ONE_LAUNCH_MAX = 131072


@triton.jit(
    do_not_specialize=["n_elements", "scale", "inv_scale", "lower", "upper"]
)
def _fake_quantize_per_tensor_affine_cachemask_kernel(
    input_ptr,
    output_ptr,
    mask_ptr,
    n_elements,
    scale,
    inv_scale,
    lower,
    upper,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
    BOOL: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        valid = offs < n_elements
        x = tl.load(input_ptr + offs, mask=valid, other=0.0).to(tl.float32)
    else:
        x = tl.load(input_ptr + offs).to(tl.float32)
    # Round half to even without a convert: exact for |v| <= 2**22, which the
    # quantised range always is.
    rounded = (x * inv_scale + 12582912.0) - 12582912.0
    # The bounds carry zero_point, so the value never has to.
    clamped = tl.minimum(tl.maximum(rounded, lower), upper)
    if BOOL:
        m = rounded == clamped
    else:
        # Both are integer valued, so they differ by at least 1 exactly when
        # the clamp bit and by 0 otherwise.
        m = tl.minimum(tl.abs(rounded - clamped), 1.0)
    if MASKED:
        tl.store(output_ptr + offs, clamped * scale, mask=valid)
        tl.store(mask_ptr + offs, m, mask=valid)
    else:
        tl.store(output_ptr + offs, clamped * scale)
        tl.store(mask_ptr + offs, m)


def _pick_block(n_elements, dtype):
    if n_elements <= 1024:
        return 1024
    if n_elements < 98304:
        return 4096
    # fp32 runs 10% faster at 8192 on every large shape -- 741.6us against
    # 821.0us at 67M, 186.7 against 208.8 at 16.7M -- while fp16 is 60% worse
    # there (752.3us against 467.8us), and bf16 both prefers 16384 and raises a
    # hardware exception at 32768 or more.
    return 8192 if dtype == torch.float32 else 16384


def _validate_and_convert_qparams(scale, zero_point, quant_min, quant_max):
    if quant_min > quant_max:
        raise RuntimeError("`quant_min` should be less than or equal to `quant_max`.")
    if zero_point < quant_min or zero_point > quant_max:
        raise RuntimeError("`zero_point` must be between `quant_min` and `quant_max`.")
    # ATen computes inv_scale as a float32 reciprocal on the host, so rounding
    # both through float32 here reproduces the reference bit for bit.
    scale = struct.unpack("f", struct.pack("f", float(scale)))[0]
    if scale == 0.0:
        inv_scale = math.copysign(math.inf, scale)
    else:
        inv_scale = struct.unpack("f", struct.pack("f", 1.0 / scale))[0]
    return scale, inv_scale


# ``JITFunction.run`` spends most of the 17.4us launch *before* the kernel:
# ``binder(*args, **kwargs)`` rebinds and re-specialises every argument on every
# call, then the result is hashed into a cache key.  Once the kernel is
# compiled, the only thing that has to happen is
#
#     kernel.run(g0, g1, g2, stream, kernel.function, kernel.packed_metadata,
#                launch_metadata, enter_hook, exit_hook, *args)
#
# so caching the CompiledKernel by hand and calling that directly takes the
# whole launch from 17.4us to ~8us -- measured 30.5us -> 7.3us (f16) and
# 19.1us -> 8.9us (f32) for the complete call at n=16.  ``libentry`` does the
# opposite: it caches *around* ``run`` and its key inspects every argument, so
# it measures 33.5us against 17.2us for the plain path.
#
# The hand key has to cover everything the binder would have put in
# ``specialization``, or a later call could reuse the wrong binary.  Measured,
# the specialisation is a pure function of
#
#     pointer    ('*fp16', 'D')      'D' iff data_ptr() % 16 == 0
#     int        ('i32', 'D')        'D' iff v % 16 == 0, ('constexpr', 1) if v == 1
#     float      ('fp32', None)      never specialised under do_not_specialize
#     constexpr  ('constexpr', v)
#
# and ``n_elements`` is in ``do_not_specialize`` precisely so the int collapses
# to ``('i32', None)`` in every case -- no divisibility bit, and no surprise
# constexpr at n == 1.  That leaves the dtypes, one alignment bit per pointer
# and the three constexprs, which cost 0.54us + 0.18us to assemble and look up.
# The stream is *not* cached: ``get_current_stream`` is 0.27us and a stale
# stream would reorder work.
try:
    from triton import knobs as _triton_knobs
    from triton.runtime import driver as _triton_driver

    _LAUNCH_EXIT_HOOK = _triton_knobs.runtime.launch_exit_hook
except Exception:  # pragma: no cover - fall back to the plain JIT path
    _triton_driver = None
    _LAUNCH_EXIT_HOOK = None

_COMPILED_CACHE = {}


def _run_kernel(grid_0, argv, block, masked, bool_mask, device_index):
    kwargs = {
        "BLOCK": block,
        "MASKED": masked,
        "BOOL": bool_mask,
        "num_warps": 8,
    }
    if _triton_driver is None:
        _fake_quantize_per_tensor_affine_cachemask_kernel[(grid_0,)](
            *argv, **kwargs
        )
        return
    input, output, mask = argv[0], argv[1], argv[2]
    key = (
        device_index,
        input.dtype,
        output.dtype,
        mask.dtype,
        block,
        masked,
        bool_mask,
        input.data_ptr() % 16 == 0,
        output.data_ptr() % 16 == 0,
        mask.data_ptr() % 16 == 0,
    )
    entry = _COMPILED_CACHE.get(key)
    if entry is None:
        # First call for this specialisation: go through the JIT, which
        # compiles if needed and hands back the CompiledKernel it launched.
        compiled = _fake_quantize_per_tensor_affine_cachemask_kernel[(grid_0,)](
            *argv, **kwargs
        )
        triton_device = _triton_driver.active.get_current_device()
        if hasattr(compiled, "run") and hasattr(compiled, "function"):
            _COMPILED_CACHE[key] = (compiled, triton_device)
        return
    compiled, triton_device = entry
    stream = _triton_driver.active.get_current_stream(triton_device)
    compiled.run(
        grid_0,
        1,
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        _LAUNCH_EXIT_HOOK,
        *argv,
        block,
        masked,
        bool_mask,
    )


def _launch(input, output, cachemask, scale, zero_point, quant_min, quant_max):
    scale, inv_scale = _validate_and_convert_qparams(
        scale, zero_point, quant_min, quant_max
    )
    contiguous_input = input.contiguous()
    n_elements = contiguous_input.numel()
    if n_elements == 0:
        return
    contiguous_output = output
    if not output.is_contiguous():
        contiguous_output = torch.empty_like(contiguous_input)
    contiguous_mask = cachemask
    if not cachemask.is_contiguous():
        contiguous_mask = torch.empty_like(contiguous_input, dtype=torch.bool)
    block = _pick_block(n_elements, contiguous_input.dtype)
    grid_0 = triton.cdiv(n_elements, block)
    one_launch = n_elements <= _ONE_LAUNCH_MAX
    if one_launch:
        # The per-lane byte store is slow on this hardware, but below the
        # crossover the device is idle waiting for Python anyway.
        scratch = contiguous_mask
    else:
        scratch_dtype = (
            torch.float16
            if contiguous_input.dtype == torch.float32
            else contiguous_input.dtype
        )
        scratch = torch.empty(
            contiguous_input.shape, device=input.device, dtype=scratch_dtype
        )
    argv = (
        contiguous_input,
        contiguous_output,
        scratch,
        n_elements,
        scale,
        inv_scale,
        float(quant_min - zero_point),
        float(quant_max - zero_point),
    )
    current = torch_device_fn.current_device()
    device = input.device
    if device.index is None or device.index == current:
        guard = nullcontext()
    else:
        guard = torch_device_fn.device(device)
        current = device.index
    with guard:
        _run_kernel(
            grid_0, argv, block, n_elements % block != 0, one_launch, current
        )
        if not one_launch:
            # scratch is exactly 0.0 where the mask is true, so eq is exact.
            torch.eq(scratch, 0, out=contiguous_mask)
    if contiguous_output is not output:
        output.copy_(contiguous_output)
    if contiguous_mask is not cachemask:
        cachemask.copy_(contiguous_mask)


def fake_quantize_per_tensor_affine_cachemask(
    input, scale, zero_point, quant_min, quant_max
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK")
    contiguous_input = input.contiguous()
    output = torch.empty_like(contiguous_input)
    cachemask = torch.empty_like(contiguous_input, dtype=torch.bool)
    _launch(
        contiguous_input, output, cachemask, scale, zero_point, quant_min, quant_max
    )
    return output, cachemask


def fake_quantize_per_tensor_affine_cachemask_out(
    input, scale, zero_point, quant_min, quant_max, *, out0, out1
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_OUT")
    # resize_ is a dispatch each, and the benchmark hands us tensors that are
    # already the right shape, so only resize when it actually differs.
    if out0.shape != input.shape:
        out0.resize_(input.shape)
    if out1.shape != input.shape:
        out1.resize_(input.shape)
    _launch(input, out0, out1, scale, zero_point, quant_min, quant_max)
    return out0, out1
