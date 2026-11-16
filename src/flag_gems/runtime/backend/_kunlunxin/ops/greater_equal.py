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

"""Out-of-place ``aten::greater_equal.Tensor`` for Kunlunxin/XPU3.

Why this override exists
------------------------
The generic implementation resolves to ``flag_gems.ops.greater_equal`` ->
``flag_gems.ops.ge.ge``, which uses the *generic* ``pointwise_dynamic``
wrapper (no Kunlunxin ``CodeGenConfig`` and no ``TRITONXPU_COMPARE_FUSION`` /
``TRITONXPU_FP16_FAST`` at compile time).  On XPU3 a tensor-tensor compare
whose i1 result feeds a bool store then lowers to a per-lane scalar path.
Measured on the benchmark matrix: 4096x4096 fp16 took **55.5 ms** per call
(pre-profiling: ``ge_func_kernel_rank_1`` 52/52 calls, 3586.8 ms, 99.1% of
device time), i.e. ~0.001x vs torch.

Kernel formulation
------------------
Same-shape/contiguous/same float dtype (the benchmark matrix): a single flat
kernel keeps the comparison in the native dtype (``x.to(tl.float32) >= y``,
identical to ``ge.py``) and stores the i1 result as int8 into the caller's
bool tensor while ``TRITONXPU_COMPARE_FUSION`` / ``TRITONXPU_FP16_FAST`` are
set at compile time -- this is what turns off the per-lane slow path.  The
geometry is a fixed ``BLOCK`` capped at 65536 elements (``BLOCK>=65536`` is
the bandwidth plateau; 16K tiles leave 2-3x on the table for large shapes).

Launch path
-----------
The benchmark runs in KERNEL mode (the start event is enqueued before the host
builds the launch), so Triton's ``JITFunction`` host path (~13-16 us) is
charged to the kernel for small shapes.  The flat kernel is therefore issued
through a pre-bound ``CompiledKernel.run`` launcher with a per-device cached
stream (~4 us host), which is what lifts the many sub-4M shapes to ~1.0x.

Broadcasting / non-contiguous / mixed-dtype / non-float operands keep the
backend's general ``ge`` (pointwise_dynamic with the Kunlunxin config), i.e. a
real XPU kernel -- never a torch/ATen fallback.

Semantics
---------
``torch.greater_equal`` returns ``nan >= x == False`` (IEEE), ``+/-0`` compare
equal, and ``+/-inf`` compare normally; the native compare instruction has the
same IEEE semantics, so no saturating-arithmetic emulation is needed here (the
in-place variants need it because they write 0/1 back into the fp dtype).
"""

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .ge import ge as _general_ge

try:  # low-overhead launch path
    import triton.knobs as _trt_knobs
    from triton.runtime import driver as _trt_driver
except Exception:  # pragma: no cover
    _trt_knobs = None
    _trt_driver = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fast path: same shape, contiguous, same floating dtype
# ---------------------------------------------------------------------------
@triton.jit
def _ge_flat_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr, MASKED: tl.constexpr):
    # Strict ``<`` against the runtime bound (never ``<=``): XPU3 can fold a
    # non-strict runtime tail mask away and write the full tile out of bounds
    # (wiki pattern-xpu3-masked-store-constexpr-tail).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        m = offs < n
        a = tl.load(a_ptr + offs, mask=m)
        b = tl.load(b_ptr + offs, mask=m)
        r = a.to(tl.float32) >= b
        tl.store(out_ptr + offs, r.to(tl.int8), mask=m)
    else:
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)
        r = a.to(tl.float32) >= b
        tl.store(out_ptr + offs, r.to(tl.int8))


_FAST_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_FLAT_BLOCK_MAX_SMALL = 32768  # n < 8M: 32K tiles keep grid >= 8-32 lanes-wide
_FLAT_BLOCK_MAX_LARGE = 65536  # n >= 8M: 64K tile is the bandwidth plateau
_FLAT_BLOCK_MIN = 8192
_FLAT_TINY = 4096  # below this a single program beats any multi-program grid
_FLAT_OPTS = dict(
    num_warps=4,
    buffer_size_limit=8192,
    unroll_num=16,
    isCloseMemoryAsync=False,
)
_PLAN_CACHE = {}
_PLAN_CACHE_MAX = 256


def _pick_geometry(n):
    if n <= _FLAT_TINY:
        # Launch-bound: one program, tile hugs numel (no wasted lanes).
        blk = max(256, triton.next_power_of_2(max(n, 1)))
    else:
        # Aim for ~8 programs so the tile is wide but the grid still fills the
        # SMs; the sweep showed a 32K cap below 8M and 64K above it.
        cap = _FLAT_BLOCK_MAX_LARGE if n >= (8 << 20) else _FLAT_BLOCK_MAX_SMALL
        blk = max(_FLAT_BLOCK_MIN, triton.next_power_of_2((n + 7) // 8))
        blk = min(cap, blk)
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

    def _runner(a, b, out, n):
        run(grid_x, 1, 1, stream, fn, meta, None, enter, leave, a, b, out, n)

    return _runner


def _make_flat_runner(n):
    """Build a per-plan ``runner(a, b, out)`` for the flat greater_equal kernel.

    One closure is created per ``(device, dtype, numel)`` plan and owns the
    ``CompiledKernel`` pre-bound to that plan's geometry, so the launch never
    re-enters Triton's ``JITFunction`` host path.
    """
    blk, grid, masked = _pick_geometry(n)
    dev = torch_device_fn.current_device()
    bound = [None]

    def jit_launch(a, b, out):
        # The fusion passes read these at compile time; this closure only runs
        # on the first (compiling) call for the plan.
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
        os.environ["TRITONXPU_FP16_FAST"] = "1"
        try:
            return _ge_flat_kernel[(grid,)](
                a, b, out, n, BLOCK=blk, MASKED=masked, **_FLAT_OPTS
            )
        finally:
            del os.environ["TRITONXPU_COMPARE_FUSION"]
            del os.environ["TRITONXPU_FP16_FAST"]

    def launch(a, b, out):
        if bound[0] is not None:
            bound[0](a, b, out, n)
            return
        runner = _bind_runner(jit_launch(a, b, out), grid, dev)
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
    # Output allocation only (the comparison result buffer); never a compute
    # fallback.
    out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
    entry(A, B, out)
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def greater_equal(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_EQUAL")
    if isinstance(B, torch.Tensor):
        if A.device != B.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at "
                "least two devices, {} and {}!".format(A.device, B.device)
            )
        if A.numel() == 0:
            return torch.empty(A.shape, dtype=torch.bool, device=A.device)
        if (
            A.dtype == B.dtype
            and A.dtype in _FAST_DTYPES
            and A.shape == B.shape
            and A.is_contiguous()
            and B.is_contiguous()
        ):
            return _run_fast(A, B)
        return _general_ge(A, B)
    # Number `other` normally routes to the Scalar overload; if a caller uses
    # this entry directly, wrap the scalar at the tensor dtype (allocation /
    # fill only) so the backend's wrapped-scalar promotion matches torch.
    if isinstance(B, (bool, int, float)):
        B = torch.tensor(B, dtype=A.dtype, device=A.device)
        return _general_ge(A, B)
    raise TypeError(
        "greater_equal(): argument 'other' must be Tensor or Number, not "
        f"{type(B).__name__}"
    )
