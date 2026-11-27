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
"""Enflame gcu400 fused AdamW: persistent metadata-table single-launch strategy.

On the new stack (flagtree 0.7.0+enflame.gitd8ca7a9e / Triton GCU 4.1.6, 2026-11-18) the
measured attribution merges with the previous round (2026-11-16, commit 24822709f) as follows:

- **Large tensors are wave-serialization bound, not block-shape bound**: with the same shared
  kernel, BLOCK 4096 on a full grid (4096 CTAs) took 273.7us for one f16 tensor (2**24,); a
  persistent grid-stride (48 CTAs, each program stepping by a chunk) gave 149.0us (-46%), and
  sweeping 48/64/96/128/160/192 blocks picked 48 (a single wave). i64 vector offsets, the masked
  predicate and exact mask removal were all identical on the full grid (274.0/274.6/274.5us), so
  the gain is purely from persistence.
- **num_warps (4/8/16) and enable_fp_fusion make zero difference on this kernel** (273.7-273.9us,
  bandwidth-bound), so the backend ignores them and they are no longer tuning dimensions.
- **Multiple tensors are host bound**: many_small_64 (64x257) through the shared implementation
  costs 443us of host time (per-tensor Python validation plus a 524-argument binder at
  ~120us/launch), and many_small_256 reaches 1710-1850us; the official bench (do_bench kernel
  mode) 256MB flush shadow window of ~72-90us can only hide host time below that budget.

So this module replaces the whole path with "one persistent launch":

1. Validation uses C-level bulk primitives (a single ``attrgetter`` pass plus ``map`` set/boolean
   aggregation) to enforce the shared implementation dtype/device/shape/contiguity/state_steps
   contract; anything failing falls back to the shared implementation, which raises as before.
2. Pointers and element counts are packed into one int64 meta table (group-major: 6xNT pointer
   rows + NT count rows), reused through ``flag_gems.utils.foreach._device_meta`` **value-key**
   memoization (rebuilt when pointer values change, zero H2D on steady state); every call
   re-reads all ``data_ptr`` values and caches neither tensor identity nor addresses.
3. The launch goes through ``_foreach_reduction._launch`` cached-launcher replay (the same
   accepted mechanism as ``_foreach_zero``); the 9 scalars such as lr/beta are
   ``do_not_specialize`` runtime arguments, and replay passes their current values, so lr
   scheduling and device-side lr are both correct.
4. The kernel scalar prologue (step read-out and bias-correction polynomial) reuses the shared
   ``_adamw_correction``; the element body is the shared fp32 path rewritten with persistent
   addressing (i32 vectors inside a block plus a scalar i64 block base; reusing the shared body
   verbatim under the persistent form measured 318us vs 149us for this form at f16 (2**24,)).
   The main loop runs whole blocks with a single loop body, and the tail block (if any) is masked
   once after the loop -- inlining both branches per block (whole + masked) pushed register
   pressure up a tier on amsgrad f32 (345 vs 232us).
   The AMSGrad max second-moment store moves after param/moments (A/B -30%: f32 ams1 333->232us,
   f16 ams1 210->167us; numerically identical, but the final memory state on aliased inputs differs).
5. Load balancing across uneven tensors: (NT, 48) fixes 48 programs per tensor and is 2:1 skewed
   under mixed sizes (hetero f16 104.6us); allocating programs by chunk share and writing
   (tensor, start block, in-block stride) per program row into meta (target 192) gives 81.5us
   (f16 ams0, -22%) and is best across every f16/f32 tier (48/96/192 sweep).
   Meta pointer restoration (``int64 -> tl.pointer_type``) follows the ``_foreach_zero``
   precedent and depends on ``TORCH_GCU_ENABLE_INT64_AND_UINT64`` (on by default globally; this
   module reads that environment variable directly and falls back to the shared implementation
   when it is off -- the static ``support_int64`` capability bit does not track the switch, and
   int64 H2D aborts the process with it off).

``_fused_adamw_`` / ``_fused_adamw__tensor_lr`` still export the same function objects from the
shared module; the shared function resolves ``_fused_adamw_run`` by module global name and this
module replaces that global, so ATen dispatch tracing and test monkeypatching are unaffected.
"""

import importlib
import math
import numbers
import operator
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.foreach import _device_meta, tl_dtype

from ._foreach_reduction import _launch

# Public implementation: reused for the kernel math, the numeric path and as
# the fallback entry for every input the fast path does not accept.
_IMPL = importlib.import_module("flag_gems.ops._fused_adamw_")
_ORIGINAL_RUN = _IMPL._fused_adamw_run
_adamw_correction = _IMPL._adamw_correction

# --- TUNING PARAMS START ---
# Launch shape only; the update math is the shared public implementation.
_GCU400_BLOCK_CAP = 4096  # elements per program chunk (public path uses 1024)
_GCU400_PERSISTENT = 48  # programs per tensor axis; one GCU400 wave
_GCU400_PLAN_TARGET = 192  # total programs for the balanced uneven-size grid
_GCU400_MAX_LIST = 65535  # grid.x hardware cap; larger lists fall back
_GCU400_FAST_PATH = True  # False disables the fast path (A/B control)
# --- TUNING PARAMS END ---

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DATA_PTR = torch.Tensor.data_ptr
_NUMEL = torch.Tensor.numel
_IS_CONTIGUOUS = torch.Tensor.is_contiguous
_MAIN_DESC = operator.attrgetter("dtype", "device")
_SHAPE = operator.attrgetter("shape")
_STEP_DESC = operator.attrgetter("dtype", "device", "layout")
_STREAM_CAPTURING = getattr(torch_device_fn, "is_current_stream_capturing", None)


@triton.jit(
    do_not_specialize=[
        "lr",
        "beta1",
        "beta2",
        "one_minus_beta1",
        "one_minus_beta2",
        "log_beta1",
        "log_beta2",
        "weight_decay",
        "eps",
    ]
)
def _adamw_meta_kernel(
    Meta,
    Lr,
    lr,
    beta1,
    beta2,
    one_minus_beta1,
    one_minus_beta2,
    log_beta1,
    log_beta2,
    weight_decay,
    eps,
    TENSOR_LR: tl.constexpr,
    AMSGRAD: tl.constexpr,
    MAXIMIZE: tl.constexpr,
    HAS_PLANS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NT: tl.constexpr,
    PARAM_DT: tl.constexpr,
):
    """Update every tensor of one list in a single persistent launch.

    ``Meta`` is a read-only int64 table, group-major: six pointer rows of NT
    entries (Param, Grad, ExpAvg, ExpAvgSq, MaxExpAvgSq, Step) followed by NT
    element counts and, when ``HAS_PLANS``, P rows of (tensor, first chunk,
    chunk stride) for the balanced multi-tensor grid.

    Without plans the grid is ``(NT, CY)``: axis 0 selects the tensor, axis 1
    is the persistent chunk range and every program strides chunks by CY.
    With plans the grid is ``(P,)`` and each program reads its assignment, so
    uneven tensor sizes still fill the device evenly (2026-11-18 A/B: the
    (NT, 48) form leaves 2:1 program imbalance on the benchmark's mixed
    list at 104.6us against 81.5us for the balanced plan rows).

    The update body is the public fp32 path with the addressing adapted for
    the persistent loop (reusing the public per-block body verbatim costs
    318us on (2**24,) f16 against 149us here; i64 vector offsets plus masked
    predication only tax the persistent form).  Chunk offsets are one scalar
    int64 product plus an int32 tile vector, and a whole chunk skips the
    mask entirely, so only the tail chunk is masked.  The AMSGrad maximum
    store is issued after the parameter and moment stores: measured -30%
    (f32 ams1 333->232us) against the public order; the stored values are
    identical, and the reorder only matters if a caller aliases
    ``max_exp_avg_sqs`` with a main operand, which the public API does not
    support either.
    """
    if HAS_PLANS:
        slot = Meta + 7 * NT + 3 * tl.program_id(0)
        index = tl.load(slot)
        chunk = tl.load(slot + 1)
        stride = tl.load(slot + 2)
    else:
        index = tl.program_id(0)
        chunk = tl.program_id(1)
        stride = tl.num_programs(1)
    Param = tl.load(Meta + index).to(tl.pointer_type(PARAM_DT))
    Grad = tl.load(Meta + NT + index).to(tl.pointer_type(PARAM_DT))
    ExpAvg = tl.load(Meta + 2 * NT + index).to(tl.pointer_type(PARAM_DT))
    ExpAvgSq = tl.load(Meta + 3 * NT + index).to(tl.pointer_type(PARAM_DT))
    MaxExpAvgSq = tl.load(Meta + 4 * NT + index).to(tl.pointer_type(PARAM_DT))
    Step = tl.load(Meta + 5 * NT + index).to(tl.pointer_type(tl.float32))
    n = tl.load(Meta + 6 * NT + index)
    if TENSOR_LR:
        lr_value = tl.load(Lr).to(tl.float32)
    else:
        lr_value = lr.to(tl.float32)
    b1 = beta1.to(tl.float32)
    b2 = beta2.to(tl.float32)
    c1 = one_minus_beta1.to(tl.float32)
    c2 = one_minus_beta2.to(tl.float32)
    decay = weight_decay.to(tl.float32)
    epsilon = eps.to(tl.float32)
    step = tl.load(Step).to(tl.float32)
    bc1 = _adamw_correction(beta1, log_beta1, step, False)
    bc2 = _adamw_correction(beta2, log_beta2, step, False)
    offs = tl.arange(0, BLOCK_SIZE)
    # Full chunks run branch-free in the persistent loop; the one partial
    # tail chunk (if this program owns it) is handled once after the loop.
    # Duplicating the body per chunk measured 345us against 232us here on
    # (2**24,) f32 amsgrad: the second register-pressure tier is real.
    while chunk.to(tl.int64) * BLOCK_SIZE + BLOCK_SIZE <= n:
        offset = chunk.to(tl.int64) * BLOCK_SIZE
        p = tl.load(Param + offset + offs).to(tl.float32)
        g = tl.load(Grad + offset + offs).to(tl.float32)
        m = tl.load(ExpAvg + offset + offs).to(tl.float32)
        v = tl.load(ExpAvgSq + offset + offs).to(tl.float32)
        if MAXIMIZE:
            g = -g
        p = p * (1.0 - lr_value * decay)
        m = b1 * m + c1 * g
        v = b2 * v + c2 * g * g
        variance = v
        if AMSGRAD:
            maximum = tl.load(MaxExpAvgSq + offset + offs).to(tl.float32)
            variance = tl.where(maximum < v, v, maximum)
        denom = tl.sqrt(variance) / tl.sqrt(bc2) + epsilon
        p = p - (lr_value / bc1) * m / denom
        tl.store(Param + offset + offs, p)
        tl.store(ExpAvg + offset + offs, m)
        tl.store(ExpAvgSq + offset + offs, v)
        if AMSGRAD:
            tl.store(MaxExpAvgSq + offset + offs, variance)
        chunk += stride
    offset = chunk.to(tl.int64) * BLOCK_SIZE
    if offset < n:
        mask = offs < (n - offset).to(tl.int32)
        p = tl.load(Param + offset + offs, mask, other=0).to(tl.float32)
        g = tl.load(Grad + offset + offs, mask, other=0).to(tl.float32)
        m = tl.load(ExpAvg + offset + offs, mask, other=0).to(tl.float32)
        v = tl.load(ExpAvgSq + offset + offs, mask, other=0).to(tl.float32)
        if MAXIMIZE:
            g = -g
        p = p * (1.0 - lr_value * decay)
        m = b1 * m + c1 * g
        v = b2 * v + c2 * g * g
        variance = v
        if AMSGRAD:
            maximum = tl.load(MaxExpAvgSq + offset + offs, mask, other=0).to(tl.float32)
            variance = tl.where(maximum < v, v, maximum)
        denom = tl.sqrt(variance) / tl.sqrt(bc2) + epsilon
        p = p - (lr_value / bc1) * m / denom
        tl.store(Param + offset + offs, p, mask)
        tl.store(ExpAvg + offset + offs, m, mask)
        tl.store(ExpAvgSq + offset + offs, v, mask)
        if AMSGRAD:
            tl.store(MaxExpAvgSq + offset + offs, variance, mask)


def _hyperparameters_ok(lr, beta1, beta2, weight_decay, eps):
    """Public-range validation; illegal values fall back to its errors."""
    for name, value in (
        ("beta1", beta1),
        ("beta2", beta2),
        ("weight_decay", weight_decay),
        ("eps", eps),
    ):
        if not isinstance(value, numbers.Real) or not math.isfinite(value):
            return False
        if value < 0 or (name in ("beta1", "beta2") and value >= 1):
            return False
    if isinstance(lr, torch.Tensor):
        # CPU scalar lr is materialized by the public implementation.
        return lr.device.type != "cpu"
    return isinstance(lr, numbers.Real) and math.isfinite(lr) and lr >= 0


def _fast_plan(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
):
    """Validate a whole list with bulk C-level passes and build the meta row.

    Returns ``(dtype, device, values, numels)`` for a launchable list, or
    ``None`` when any check rejects: the public implementation then produces
    its own diagnostics or handles the input itself.  Nothing launches before
    every tensor has been checked, so an invalid tail never updates earlier
    tensors.
    """
    count = len(params)
    if count == 0 or count > _GCU400_MAX_LIST:
        return None
    if (
        len(grads) != count
        or len(exp_avgs) != count
        or len(exp_avg_sqs) != count
        or len(state_steps) != count
        or len(max_exp_avg_sqs) != (count if amsgrad else 0)
    ):
        return None
    if not _hyperparameters_ok(lr, beta1, beta2, weight_decay, eps):
        return None
    first = params[0]
    if not isinstance(first, torch.Tensor):
        return None
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return None
    device = first.device
    # The pointer/numel table needs device-side int64; torch_gcu only turns
    # it on via TORCH_GCU_ENABLE_INT64_AND_UINT64 (the static capability
    # flag does not track that runtime switch, and an int64 upload without
    # it aborts the process).  Gate on the environment directly so a machine
    # without it keeps the public int64-free pointer-only path.
    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") != "1":
        return None
    if not runtime.device.support_int64:
        return None
    if _STREAM_CAPTURING is not None and _STREAM_CAPTURING():
        # The first call uploads the meta table; graph capture stays on the
        # public pointer-only path.
        return None
    # Bulk passes over every operand: dtype and device must be globally
    # uniform (the kernel casts one element type), while shapes only need to
    # match parameter-wise -- heterogeneous groups are the normal case.
    main = params + grads + exp_avgs + exp_avg_sqs
    if amsgrad:
        main += max_exp_avg_sqs
    try:
        if set(map(_MAIN_DESC, main)) != {(dtype, device)}:
            return None
        shapes = tuple(map(_SHAPE, params))
        groups = (grads, exp_avgs, exp_avg_sqs)
        if amsgrad:
            groups += (max_exp_avg_sqs,)
        for group in groups:
            if tuple(map(_SHAPE, group)) != shapes:
                return None
    except AttributeError:
        # A non-tensor entry: the public path raises the real diagnostic.
        return None
    if not all(map(_IS_CONTIGUOUS, main)):
        return None
    try:
        steps_ok = set(map(_STEP_DESC, state_steps)) == {
            (torch.float32, device, torch.strided)
        } and set(map(_NUMEL, state_steps)) == {1}
    except AttributeError:
        return None
    if not steps_ok:
        return None
    if isinstance(lr, torch.Tensor):
        if (
            lr.dtype != torch.float32
            or lr.device != device
            or lr.layout != torch.strided
            or lr.numel() != 1
        ):
            return None
    numels = list(map(_NUMEL, params))
    if all(numels):
        # No filtering; the amsgrad-off kernel never reads the fifth pointer
        # row, so it aliases the second-moment row instead of re-reading it.
        rows = [
            list(map(_DATA_PTR, params)),
            list(map(_DATA_PTR, grads)),
            list(map(_DATA_PTR, exp_avgs)),
            list(map(_DATA_PTR, exp_avg_sqs)),
        ]
        rows.append(list(map(_DATA_PTR, max_exp_avg_sqs)) if amsgrad else rows[3])
        rows.append(list(map(_DATA_PTR, state_steps)))
        values = [pointer for row in rows for pointer in row]
        values.extend(numels)
    else:
        active = [index for index, numel in enumerate(numels) if numel]
        if not active:
            return None
        pick = operator.itemgetter(*active) if len(active) > 1 else None
        rows = []
        for group in (
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs if amsgrad else exp_avg_sqs,
            state_steps,
        ):
            pointers = list(map(_DATA_PTR, group))
            rows.append(pick(pointers) if pick else (pointers[active[0]],))
        values = [value for row in rows for value in row]
        values.extend(numels[index] for index in active)
        numels = [numels[index] for index in active]
    return dtype, device, values, numels


def _gcu400_run(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale,
    found_inf,
):
    """GCU400 launch strategy: one persistent meta-table launch per list."""
    plan = None
    if _GCU400_FAST_PATH and grad_scale is None and found_inf is None:
        plan = _fast_plan(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            amsgrad,
        )
    if plan is None:
        return _ORIGINAL_RUN(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=amsgrad,
            maximize=maximize,
            grad_scale=grad_scale,
            found_inf=found_inf,
        )
    dtype, device, values, numels = plan
    tensor_lr = isinstance(lr, torch.Tensor)
    scalars = (
        0.0 if tensor_lr else float(lr),
        float(beta1),
        float(beta2),
        1.0 - beta1,
        1.0 - beta2,
        math.log(beta1) if beta1 else -math.inf,
        math.log(beta2) if beta2 else -math.inf,
        float(weight_decay),
        float(eps),
    )
    count = len(numels)
    max_numel = max(numels)
    block_size = min(_GCU400_BLOCK_CAP, max(128, 1 << (max_numel - 1).bit_length()))
    chunk_counts = [-(-numel // block_size) for numel in numels]
    has_plans = min(chunk_counts) != max(chunk_counts)
    if has_plans:
        # Uneven tensor sizes: give every tensor programs in proportion to
        # its chunk count so one wave covers the whole list evenly, and let
        # each program stride its own tensor.  Oversized plan tables fall
        # back to the uniform per-tensor grid.
        total_chunks = sum(chunk_counts)
        rows = []
        for tensor, chunks in enumerate(chunk_counts):
            programs = max(
                1, min(chunks, round(_GCU400_PLAN_TARGET * chunks / total_chunks))
            )
            for slot in range(programs):
                rows.extend((tensor, slot, programs))
        if len(rows) <= 3 * _GCU400_MAX_LIST:
            values.extend(rows)
            grid = (len(rows) // 3,)
        else:
            has_plans = False
    if not has_plans:
        grid = (count, min(_GCU400_PERSISTENT, max(chunk_counts)))
    meta = _device_meta(values, device)
    constexprs = (
        tensor_lr,
        amsgrad,
        maximize,
        has_plans,
        block_size,
        count,
        tl_dtype(dtype),
    )
    _launch(
        _adamw_meta_kernel,
        grid,
        constexprs,
        (meta, lr if tensor_lr else None, *scalars, *constexprs),
        (meta, lr) if tensor_lr else (meta,),
        device,
    )
    return None


# The public functions resolve ``_fused_adamw_run`` through module globals;
# the replacement installs the GCU400 strategy while ATen dispatch tracing
# and the tests' monkeypatch of that global keep working.
_IMPL._fused_adamw_run = _gcu400_run

_fused_adamw_ = _IMPL._fused_adamw_
_fused_adamw__tensor_lr = _IMPL._fused_adamw__tensor_lr

__all__ = ["_fused_adamw_", "_fused_adamw__tensor_lr"]
