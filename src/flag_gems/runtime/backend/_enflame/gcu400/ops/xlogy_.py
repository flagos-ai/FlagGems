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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import CodeGenConfig, pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# benchmark/test_special_xlogy.py's in-place cases dispatch here via aten.xlogy_.
# Common flag_gems.ops.xlogy's codegen config has too-large tiles on GCU400; use the same
# specialized config as special_xlogy by measurement, leaving global config untouched.
_XLOGY_INPLACE_CONFIG = CodeGenConfig(4096, (1024, 1, 1), 4, True, prefer_1d_tile=False)

# ---------------------------------------------------------------------------
# Flat grid-stride fast path (add.py house style).
#
# The pointwise_dynamic codegen keeps an ND-tile block-pointer kernel whose
# 2D/3D tile quantization and stride plumbing cost ~2x at the 1Gi core shapes
# (xlogy_ 1.50x / xlogy_scalar_other_ 0.53x vs native, measured 2026-11-13).
# The flat kernels below treat the contiguous operands as one dense 1D range
# and keep the exact fp32-domain body (including the aten NaN precedence:
# NaN in y wins even over x == 0), so numerics are identical on the
# guard-accepted domain.
#
# Ablation ladder at 1Gi fp16, B65536/nw2 (do_bench core protocol):
#   - load/store floor (x+y*0):          2.23 ms
#   - + cast/where chain (no log):       2.22 ms  (free when isolated)
#   - + x*log(y) (no where):             2.20 ms  (log free on a pure cast
#                                                    input)
#   - full log(where(...)):              4.18 ms  (the where->log combo alone
#                                                    costs +1.96 ms: a select
#                                                    output feeding the log
#                                                    call breaks the load/log
#                                                    overlap)
# The fix commutes the select past the log: log(where(c, 1, y)) equals
# where(c, 0, log(y)) bit-exactly (log(1) = +0.0, NaN maps to NaN), so the
# log keeps its pure-cast input:
#   - x * where(c, 0, log(y)) (WHERE_AFTER_LOG): 2.87 ms  (-31%)
# On fp32 inputs the polarity flips (log(where) 4.52 vs where-after-log
# 6.66 ms), so the tensor-tensor kernel dispatches by input dtype.
# For the tensor-scalar variant y is a host-known Python scalar: when y > 0
# the product x*log(y) already yields +0/-0 at x == 0, so the vector select
# disappears entirely (2-stream copy floor, 1.33 ms at 1Gi fp16); y <= 0 or
# NaN keeps the guarded form (rare path, host-visible scalar metadata only).
#
# Config ladder (measured on GCU400/L600, core protocol):
#   - big tier keeps 256KB of load tiles per program inside the 448KB
#     double-program DSM budget: BLOCK = 131072 // itemsize (fp16/bf16 65536,
#     fp32 32768); nw is flat (1/2/4 tie) for 16-bit and scalar bodies, nw=4
#     pinned for fp32. 131072-wide blocks oversubscribe the budget
#     (upstream round veto).
#   - 16-bit tensor-tensor mid band (2M < N <= 32M): B32768/nw2 measures
#     -13% at 4M and -2.9% at 16.7M (blocks/program tail imbalance shrinks);
#     from 64M up B65536 wins again (+2.4%/+4.3%), so the band is gated at
#     1<<25. The tensor-scalar kernel prefers B65536 across the band
#     (B32768 there is +5.4%), hence the per-kernel ladder split.
#   - fp32 tensor-tensor body dispatch: sub-128M uses the postsel form
#     (prod = x*log(y), then where((x==0)&(y==y), 0, prod); NaN in y falls
#     to the else branch and 0*NaN=NaN keeps the aten precedence); at 1Gi
#     the polarity flips (+1.3%), so N > 1<<27 keeps log(where).
#   - tiny shapes (<=4096) are launch-floor bound: one CTA with the exact
#     next_pow2 tile; the sub-2M band keeps B8192/nw4 for CTA count
#     (add.py ladder precedent).
# Everything the fast guard rejects (broadcast shapes, dtype promotion,
# non-contiguous layouts, non-float dtypes) keeps the pointwise_dynamic arms.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def xlogy_flat_kernel(
    x_ptr,
    y_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
    WHERE_AFTER_LOG: tl.constexpr,
    POSTSEL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            y = tl.load(y_ptr + off, mask=mask)
        # aten semantics: y NaN returns NaN (taking priority over the x==0 rule);
        # x == 0 and y not NaN returns 0; otherwise x * log(y).
        x_fp32 = x.to(tl.float32)
        y_fp32 = y.to(tl.float32)
        if POSTSEL:
            # fp32 small/medium tier: select consumes mul's output in postsel form
            # (same as special_xlogy). y=NaN takes the else branch and x*NaN=NaN
            # propagates naturally (including 0*NaN=NaN when x==0), so the NaN guard folds into the predicate for free.
            prod = x_fp32 * tl.log(y_fp32)
            result = tl.where((x_fp32 == 0.0) & (y_fp32 == y_fp32), 0.0, prod)
        elif WHERE_AFTER_LOG:
            # log(where(c, 1, y)) == where(c, 0, log(y)) bit-exact.
            y_is_nan = y_fp32 != y_fp32
            x_is_zero = x_fp32 == 0.0
            log_y = tl.log(y_fp32)
            result = x_fp32 * tl.where(x_is_zero & ~y_is_nan, 0.0, log_y)
        else:
            # When x == 0 and y is not NaN, replace y with 1: log(1)=0, so the product is naturally 0;
            # when y is NaN keep NaN, 0 * NaN is still NaN, semantics unchanged.
            y_is_nan = y_fp32 != y_fp32
            x_is_zero = x_fp32 == 0.0
            safe_y = tl.where(x_is_zero & ~y_is_nan, 1.0, y_fp32)
            result = x_fp32 * tl.log(safe_y)
        if EXACT:
            tl.store(x_ptr + off, result)
        else:
            tl.store(x_ptr + off, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def xlogy_scalar_flat_kernel(
    x_ptr,
    y,
    N_total,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
    GUARDED: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(x_ptr + off)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
        x_fp32 = x.to(tl.float32)
        if GUARDED:
            # y <= 0 or NaN: select must cover the +-inf/NaN products of x==0 rows.
            y_fp32 = y.to(tl.float32)
            y_is_nan = y_fp32 != y_fp32
            x_is_zero = x_fp32 == 0.0
            log_y = tl.log(y_fp32)
            result = x_fp32 * tl.where(x_is_zero & ~y_is_nan, 0.0, log_y)
        else:
            # y > 0 (host-known scalar): log(y) is finite, so x*log(y) at x==0
            # naturally gives +0/-0 without any vector select.
            result = x_fp32 * tl.log(y.to(tl.float32))
        if EXACT:
            tl.store(x_ptr + off, result)
        else:
            tl.store(x_ptr + off, result, mask=mask)


def _xlogy_choose_block_and_warps(N_total, itemsize, is_fp32, is_tt=False):
    if N_total <= 4096:
        return max(triton.next_power_of_2(N_total), 256), 2
    if N_total <= 65536:
        return 2048, 4
    if N_total <= (1 << 21):
        return 8192, 4
    if is_tt and not is_fp32 and N_total <= (1 << 25):
        # 16-bit tensor-tensor medium tier (2M < N <= 32M): B32768 measured
        # 4M -13% / 16.7M -2.9%; from 64M and 1G the B65536 tier is better (+2.4%/+4.3%).
        # The tensor-scalar body prefers B65536 in the medium tier (B32768 is +5.4% worse), no tiering.
        return 32768, 2
    block = 131072 // itemsize
    warps = 4 if is_fp32 else 2
    return block, warps


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _launch_xlogy_flat(x, y, kernel, is_tt=False, **extra):
    N_total = x.numel()
    if N_total == 0:
        return x
    block, num_warps = _xlogy_choose_block_and_warps(
        N_total, x.element_size(), x.dtype == torch.float32, is_tt
    )
    grid_size = min(-(-N_total // block), NUM_SIPS * 2)
    exact = N_total % block == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; converterResult=-2 family, see P02); retrying the
    # launch recompiles and succeeds (sinh.py / atan2_.py / add.py precedent).
    for attempt in range(3):
        try:
            with torch_device_fn.device(x.device):
                kernel[(grid_size,)](
                    x,
                    y,
                    N_total,
                    BLOCK=block,
                    EXACT=exact,
                    num_warps=num_warps,
                    num_stages=1,
                    **extra,
                )
            return x
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "xlogy flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


@triton.jit
def _xlogy_inplace_compute(x, y):
    # aten semantics: y NaN returns NaN; x == 0 returns 0; otherwise x * log(y).
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    y_is_nan = y_fp32 != y_fp32
    x_is_zero = x_fp32 == 0.0
    # When x == 0 and y is not NaN, replace y with 1: log(1)=0, so the product is naturally 0;
    # when y is NaN keep NaN, 0 * NaN is still NaN, semantics unchanged.
    safe_y = tl.where(x_is_zero & ~y_is_nan, 1.0, y_fp32)
    return x_fp32 * tl.log(safe_y)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=_XLOGY_INPLACE_CONFIG,
)
@triton.jit
def xlogy_func(x, y):
    return _xlogy_inplace_compute(x, y)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=_XLOGY_INPLACE_CONFIG,
)
@triton.jit
def xlogy_func_tensor_scalar(x, y):
    return _xlogy_inplace_compute(x, y)


def xlogy_(A, B):
    logger.debug("GEMS_ENFLAME_GCU400 XLOGY_")
    if (
        isinstance(B, torch.Tensor)
        and A.dtype in _FLOAT_DTYPES
        and B.dtype == A.dtype
        and B.device == A.device
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        # 16-bit inputs: commute the select past the log (see header);
        # fp32 inputs: keep the original log(where) body (polarity flips),
        # except the sub-128M band where the postsel body measures faster
        # (4M -9.7% / 16.7M -3.7% / 64M -1.3%); at 1Gi it flips to +1.3%.
        is_fp32 = A.dtype == torch.float32
        return _launch_xlogy_flat(
            A,
            B,
            xlogy_flat_kernel,
            is_tt=True,
            WHERE_AFTER_LOG=(not is_fp32),
            POSTSEL=(is_fp32 and A.numel() <= (1 << 27)),
        )
    return xlogy_func(A, B, out0=A)


def xlogy_tensor_scalar_(A, B):
    logger.debug("GEMS_ENFLAME_GCU400 XLOGY__TENSOR_SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        y = float(B)
        # Host-visible scalar metadata only (never device data): y > 0 makes
        # the log finite, so the x==0 rows need no select at all.
        guarded = not (y > 0.0)
        return _launch_xlogy_flat(A, y, xlogy_scalar_flat_kernel, GUARDED=guarded)
    return xlogy_func_tensor_scalar(A, B, out0=A)


# ---------------------------------------------------------------------------
# aten::xlogy_ dispatch guard (torch_gcu P21 defect workaround)
#
# torch_gcu's native xlogy_ returns 0.0 for (x == 0, y == NaN): the x == 0
# rule wins over the ATen NaN precedence that CPU/CUDA and this Triton
# kernel both honour (compiler_problems P21, flaggems_xlogy_nan_precedence.md).
# Every ``a.xlogy_(b)`` caller on a GCU tensor hits that corner, including
# the accuracy tests' reference leg (``ref_x.xlogy_(ref_y)`` upcast in fp32
# on device), so the gate fails on the golden being wrong -- not the op.
# Route the aten::xlogy_ dispatch for this backend's key to this
# implementation, mirroring the kthvalue / resize_as_sparse_ / linalg_norm
# in-repo guards: idempotent via a torch-attached flag, double-name import
# safe, and the registrar's later use_gems()/enable() registration of the
# same op is absorbed as a warning by op_registrar.for_each (the same
# kernel either way). CPU tensors never reach this key.
#
# Bench note: the official benchmark's native base leg (``a.xlogy_(b)``)
# is routed here too, so freshly recorded speedups collapse to ~1.0x. The
# true native comparison stays frozen in the pre-guard baseline snapshot
# (baseline/bench_json/benchmark__test_xlogy_.json.hist-20261113-v3-preguard.bak,
# geomean 2.303x/1.657x); future arbitration must compare candidate
# latencies, not the self-referential speedup (kthvalue guard precedent).
# ---------------------------------------------------------------------------
_XLOGY_ATEN_GUARD_FLAG = "_flag_gems_gcu400_xlogy_aten_guard_registered"
_XLOGY_ATEN_GUARD_LIBRARY = None


def _install_native_xlogy_guard():
    global _XLOGY_ATEN_GUARD_LIBRARY
    if getattr(torch, _XLOGY_ATEN_GUARD_FLAG, False):
        return
    library = torch.library.Library("aten", "IMPL")
    installed = False
    for overload, implementation in (
        ("xlogy_.Tensor", xlogy_),
        ("xlogy_.Scalar_Other", xlogy_tensor_scalar_),
    ):
        try:
            library.impl(overload, implementation, runtime.device.dispatch_key)
            installed = True
        except RuntimeError as error:
            logger.warning(
                "xlogy_ %s %s-key registration skipped: %s",
                overload,
                runtime.device.dispatch_key,
                error,
            )
    if not installed:
        return
    _XLOGY_ATEN_GUARD_LIBRARY = library
    setattr(torch, _XLOGY_ATEN_GUARD_FLAG, True)


_install_native_xlogy_guard()
