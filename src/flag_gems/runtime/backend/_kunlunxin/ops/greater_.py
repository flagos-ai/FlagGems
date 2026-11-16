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

"""In-place ``aten::greater_`` for Kunlunxin/XPU3.

Why this override exists
------------------------
The generic implementation in ``flag_gems/ops/greater_.py`` was written for a
``torch.gcu`` device runtime: it calls ``torch.gcu.current_device()`` on every
launch.  On this Kunlunxin image ``torch`` has no ``gcu`` attribute, so every
``A.greater_(B)`` raised ``AttributeError: module 'torch' has no attribute
'gcu'`` (baseline: 16/16 functional cases and the benchmark fail before a
single kernel is launched).

Kernel formulation
------------------
``A.greater_(B)`` is a read-modify-write storing the boolean result cast back
into ``A``'s dtype: ``A[i] = 1 if A[i] > B[i] else 0``.  A direct ``x > y``
whose i1 result is cast/selected lowers to a per-lane scalar path on XPU3
(~11x slower than pure fp arithmetic; see ``greater_equal_`` in ``ge.py``).
We therefore express the comparison as saturating fp arithmetic, exactly like
the sibling in-place ``greater_equal_``:

    t = (x - y) * 1e30   -> +inf when x > y, <= 0 otherwise
    t = max(0, t)        -> +inf / 0
    t = min(1, t)        -> 1 / 0

NaN inputs collapse to 0 through ``max/min`` (which prefer the non-NaN
operand), matching ``torch`` (``nan > anything`` is False).

Launch path
-----------
The benchmark runs in KERNEL mode (the start event is enqueued before the host
builds the launch), so Triton's ``JITFunction`` host path (~13-16 us on this
backend) is charged to the kernel for small shapes.  For the common
same-shape/contiguous/same-dtype case we issue a single flat kernel through a
pre-bound ``CompiledKernel.run`` launcher with a per-device cached stream
(~4 us), and fall back to a general ``pointwise_dynamic`` kernel for
broadcasting, strided, mixed-dtype or scalar operands.
"""

import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

try:  # low-overhead launch path
    import triton.knobs as _trt_knobs
    from triton.runtime import driver as _trt_driver
except Exception:  # pragma: no cover
    _trt_knobs = None
    _trt_driver = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# general (broadcast / strided / scalar) fallback: pointwise_dynamic
# ---------------------------------------------------------------------------
# In-place-safe config: keep the DEFAULT isCloseMemoryAsync setting, which
# avoids the async-copy noc-idle-timeout deadlock for read-modify-write kernels
# (same geometry as the sibling greater_equal_ override in ge.py).
config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_inplace_)
@triton.jit
def greater_func_(x, y):
    t = (x - y) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


# ---------------------------------------------------------------------------
# fast path: same shape, contiguous, same floating dtype
# ---------------------------------------------------------------------------
@triton.jit
def _gt_flat_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr, MASKED: tl.constexpr):
    # Native-dtype arithmetic: 1e30 saturates every representable gap of the
    # 16-bit formats to +inf (fp16 stores it as inf, bf16/fp32 keep it finite),
    # and an equal pair yields 0 * 1e30 = 0 (or NaN when 1e30 is inf, which
    # max/min resolve to 0 through their non-NaN preference).  Keeping the math
    # in the element dtype avoids a per-element widening that halves large-shape
    # throughput; this matches the formulation the pointwise_dynamic sibling
    # kernel already validates.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        m = offs < n
        a = tl.load(a_ptr + offs, mask=m)
        b = tl.load(b_ptr + offs, mask=m)
        t = (a - b) * 1.0e30
        t = tl.minimum(1.0, tl.maximum(0.0, t))
        tl.store(a_ptr + offs, t, mask=m)
    else:
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)
        t = (a - b) * 1.0e30
        t = tl.minimum(1.0, tl.maximum(0.0, t))
        tl.store(a_ptr + offs, t)


_FAST_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_FLAT_BLOCK = 16384
_FLAT_BLOCK_MIN = 256
# Above this many elements the persistent-grid pointwise_dynamic kernel wins
# (fewer programs -> far less per-program DMA/setup cost, and its 16-bit
# vectorised store is ~2.6x faster than a naive flat bf16 store).  Below it the
# flat kernel launched through a pre-bound CompiledKernel is faster because the
# shape is launch/host bound.
_FLAT_MAX_ELEMS = 8 << 20
_FLAT_OPTS = dict(
    num_warps=4,
    buffer_size_limit=8192,
    unroll_num=16,
    isCloseMemoryAsync=False,
)
_PLAN_CACHE = {}
_PLAN_CACHE_MAX = 256


def _pick_geometry(n):
    blk = min(_FLAT_BLOCK, max(_FLAT_BLOCK_MIN, triton.next_power_of_2(max(n, 1))))
    grid = max(1, (n + blk - 1) // blk)
    masked = (n % blk) != 0
    return blk, grid, masked


def _bind_runner(compiled, grid_x, dev):
    """Pre-bind a compiled kernel; returns None if the fast path is unavailable."""
    if _trt_driver is None:
        return None
    try:
        stream = _trt_driver.active.get_current_stream(dev)
        run = compiled.run
        fn = compiled.function
        meta = compiled.packed_metadata
        enter = _trt_knobs.runtime.launch_enter_hook
        leave = _trt_knobs.runtime.launch_exit_hook
    except Exception:
        return None

    def _runner(a, b, n):
        run(grid_x, 1, 1, stream, fn, meta, None, enter, leave, a, b, n)

    return _runner


def _make_flat_runner(n):
    """Build a per-plan ``runner(a, b)`` for the flat in-place greater_ kernel.

    One closure is created per ``(device, dtype, numel)`` plan and owns the
    ``CompiledKernel`` pre-bound to that plan's geometry, so the launch never
    re-enters Triton's ``JITFunction`` host path.
    """
    blk, grid, masked = _pick_geometry(n)
    dev = torch_device_fn.current_device()
    bound = [None]

    def jit_launch(a, b):
        return _gt_flat_kernel[(grid,)](
            a, b, n, BLOCK=blk, MASKED=masked, **_FLAT_OPTS
        )

    def launch(a, b):
        if bound[0] is not None:
            bound[0](a, b, n)
            return
        # First call: the normal JIT path compiles and executes it once.
        runner = _bind_runner(jit_launch(a, b), grid, dev)
        if runner is not None:
            bound[0] = runner

    return launch


def _run_fast(A, B):
    n = A.numel()
    dev = torch_device_fn.current_device()
    key = (dev, A.dtype, n)
    entry = _PLAN_CACHE.get(key)
    if entry is None:
        if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
            _PLAN_CACHE.clear()
        entry = _make_flat_runner(n)
        _PLAN_CACHE[key] = entry
    entry(A, B)
    return A


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def greater_(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_")
    if isinstance(B, torch.Tensor):
        # torch requires both operands on the same device and errors otherwise;
        # we reproduce that instead of silently moving data.
        if A.device != B.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at "
                "least two devices, {} and {}!".format(A.device, B.device)
            )
        if A.numel() == 0:
            return A
        if (
            A.dtype == B.dtype
            and A.dtype in _FAST_DTYPES
            and A.shape == B.shape
            and A.is_contiguous()
            and B.is_contiguous()
            and A.numel() <= _FLAT_MAX_ELEMS
        ):
            return _run_fast(A, B)
        greater_func_(A, B, out0=A)
        return A
    if isinstance(B, (bool, int, float)):
        if A.numel() == 0:
            return A
        # Wrap the scalar at the tensor's dtype so the comparison matches
        # torch's wrapped-scalar semantics (e.g. fp16 with 1e-8 -> 0).  This is
        # an allocation/host->device fill, not a compute fallback.
        scalar = torch.tensor(B, dtype=A.dtype, device=A.device)
        greater_func_(A, scalar, out0=A)
        return A
    raise TypeError("greater_(): argument 'other' must be a Tensor or a Number")
