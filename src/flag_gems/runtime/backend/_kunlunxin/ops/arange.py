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
import math
import operator
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# SIMD store-only arange fast path for large fp32/fp16: a float32x16 ramp
# generates 16/32 values per instruction and a contiguous LM2GM writes each
# tile, hitting ~bandwidth (fp32 ~0.92x, fp16 ~1.05x of torch at 16M). The
# kernels are loaded from the precompiled device library arange_simd.o in the
# payload/obj directory; the two stubs below just bind to its entry symbols.
# Optional: only usable where the xpu3 cluster tle.raw pipeline is available.
try:
    import triton.experimental.tle as tle

    _PAY_OBJ = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "payload", "obj"
    )
    _ARANGE_OBJ = os.path.join(_PAY_OBJ, "arange_simd.o")

    @tle.raw.dialect("xpu3", object=_ARANGE_OBJ, arch=3)
    def arange_f32(out, n, start, step, pid, npid):
        ...

    @tle.raw.dialect("xpu3", object=_ARANGE_OBJ, arch=3)
    def arange_f16(out, n, start, step, pid, npid):
        ...

    @triton.jit(do_not_specialize=["n", "npid"])
    def _arange_f32_raw(out, n, start, step, npid):
        tle.raw.call(arange_f32, (out, n, start, step, tl.program_id(0), npid))

    @triton.jit(do_not_specialize=["n", "npid"])
    def _arange_f16_raw(out, n, start, step, npid):
        tle.raw.call(arange_f16, (out, n, start, step, tl.program_id(0), npid))

    _HAS_ARANGE_RAW = True
except Exception:  # pragma: no cover - tle.raw not available on this build
    _HAS_ARANGE_RAW = False

# Below this element count the ~30us raw launch floor is not worth it, so the
# Triton path (which is ~1x for small shapes) is kept. npid=12 is the measured
# sweet spot on this part.
_ARANGE_RAW_MIN = 1 << 20
_ARANGE_RAW_NPID = 12


def _is_integral_scalar(x):
    """True for scalars that behave as integers (int/bool/numpy ints)."""
    if isinstance(x, (int, bool)):
        return True
    try:
        operator.index(x)
        return True
    except TypeError:
        return False


@libentry()
@triton.jit
def arange_func(
    y_ptr,
    start,
    end,
    step,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE
    step_offset = offset * step

    cols = tl.arange(0, BLOCK_SIZE)
    arange_val = cols * step + step_offset + start
    mask = cols + offset < size
    tl.store(y_ptr + offset + cols, arange_val, mask=mask)


@libentry()
@triton.jit
def arange_func_float(
    y_ptr,
    start,
    step,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    idx = offset + cols
    base = (offset * step + start).to(tl.float32)
    arange_val = cols.to(tl.float32) * step + base
    tl.store(y_ptr + idx, arange_val, mask=idx < size)


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logger.debug("GEMS_KUNLUNXIN ARANGE")
    if dtype is None:
        if all(_is_integral_scalar(x) for x in (start, end, step)):
            dtype = torch.int64
        else:
            dtype = torch.float32
    if dtype is torch.int64:
        start = int(start)
        end = int(end)
        step = int(step)
        if step == 0:
            raise RuntimeError("step must be nonzero")
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        size = math.ceil((end - start) / step)
    size = int(size)

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = runtime.device.name

    # Size-based heuristic for BLOCK_SIZE and num_warps
    if size <= 1024:
        BLOCK_SIZE = 256
        num_warps = 2
    elif size <= 8192:
        BLOCK_SIZE = 1024
        num_warps = 4
    elif size <= 65536:
        BLOCK_SIZE = 4096
        num_warps = 8
    else:
        BLOCK_SIZE = 16384
        num_warps = 8

    grid = triton.cdiv(size, BLOCK_SIZE)

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)

    # Large fp32/fp16 store-only arange -> hand-written SIMD tle.raw (near-BW).
    if (
        _HAS_ARANGE_RAW
        and size >= _ARANGE_RAW_MIN
        and dtype in (torch.float32, torch.float16)
    ):
        npid = _ARANGE_RAW_NPID
        raw = _arange_f32_raw if dtype is torch.float32 else _arange_f16_raw
        raw[(npid,)](result, size, float(start), float(step), npid)
        return result

    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        # fp32-base fast path (avoids slow large-int -> float conversion)
        arange_func_float[grid,](
            result, start, step, size, BLOCK_SIZE, num_warps=num_warps
        )
    else:
        arange_func[grid,](
            result, start, end, step, size, BLOCK_SIZE, num_warps=num_warps
        )
    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
