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

"""GPU-resident masked BF16 rows to E4M3 plus FP32 per-row scales.

The caller supplies the original numeric JIT helpers as constexpr callbacks.
This keeps the validated numeric provider unchanged without a framework or
Humming dependency here. Inputs are [E, capacity, K]; FP8 and scale outputs
retain physical expert/capacity row offsets. Counts are int32[E] on device.
Only valid rows are read/written. No allocation, host count read, environment
access, cache, synchronization, or format conversion occurs in this operator.
"""

import torch
import triton
import triton.language as tl

MASKED_ROW_QUANT_ABI = 1


@triton.jit
def _masked_row_quant(
    X,
    Q,
    S,
    COUNTS,
    E: tl.constexpr,
    CAP: tl.constexpr,
    K: tl.constexpr,
    WORKERS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK: tl.constexpr,
    CALC_SCALE: tl.constexpr,
    FINALIZE_SCALE: tl.constexpr,
    QUANT_TENSOR: tl.constexpr,
):
    worker = tl.program_id(0)
    experts = tl.arange(0, BLOCK_E)
    counts = tl.load(COUNTS + experts, experts < E, other=0)
    tl.device_assert(
        tl.sum(((counts < 0) | (counts > CAP)).to(tl.int32), 0) == 0,
        "Invalid masked row count",
    )
    total = tl.sum(counts, 0)
    if worker < total:
        prefix = tl.cumsum(counts, 0)
        start = prefix - counts
        cols = tl.arange(0, BLOCK)
        for packed in tl.range(worker, total, WORKERS):
            match = (packed >= start) & (packed < prefix) & (experts < E)
            pid = tl.sum(tl.where(match, experts * CAP + packed - start, 0), 0)
            x = tl.load(X + pid * K + cols, mask=cols < K, other=0.0).to(tl.float32)
            scale, stored = FINALIZE_SCALE(
                CALC_SCALE(x, "float8e4m3"), "float32", 1.0, 1.0
            )
            q = QUANT_TENSOR(x * (1 / scale), "float8e4m3")
            tl.store(Q + pid * K + cols, q, mask=cols < K)
            tl.store(S + pid, stored)


@triton.jit
def _quant_compact(
    X,
    Q,
    S,
    COUNTS,
    CAP: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    CALC_SCALE: tl.constexpr,
    FINALIZE_SCALE: tl.constexpr,
    QUANT_TENSOR: tl.constexpr,
):
    expert = tl.program_id(0)
    slot = tl.program_id(1)
    count = tl.load(COUNTS + expert)
    tl.device_assert((count >= 0) & (count <= CAP), "Invalid native masked_m")
    cols = tl.arange(0, BLOCK)
    for row in tl.range(slot, count, SPLITS):
        pid = expert * CAP + row
        x = tl.load(X + pid * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        scale, stored = FINALIZE_SCALE(CALC_SCALE(x, "float8e4m3"), "float32", 1.0, 1.0)
        inv_scale = 1 / scale
        q = QUANT_TENSOR(x * inv_scale, "float8e4m3")
        tl.store(Q + pid * K + cols, q, mask=cols < K)
        tl.store(S + pid, stored)


def masked_row_quant(
    inputs,
    quanted,
    scales,
    counts,
    *,
    calc_scale,
    finalize_scale,
    quant_tensor,
    workers=132,
    num_warps=4,
    row_splits=None,
):
    """Write and return ``(quanted, scales)`` with caller-pinned numeric helpers.

    The callbacks must implement E4M3 row-scale calculation, FP32 scale storage
    and E4M3 conversion. The integration layer owns numeric-provider identity
    and architecture/policy qualification. Count values must be in [0, CAP].
    """
    if inputs.ndim != 3:
        raise ValueError("Expected contiguous [E, capacity, K] BF16 input")
    e, cap, k = inputs.shape
    if (
        not 1 <= e <= 128
        or cap < 1
        or not 1 <= k <= 8192
        or e * cap * k > (1 << 31) - 1
    ):
        raise ValueError(
            "Require E=1..128, capacity>0, K=1..8192 and E*capacity*K<=2**31-1"
        )
    if (
        inputs.dtype != torch.bfloat16
        or quanted.dtype != torch.float8_e4m3fn
        or scales.dtype != torch.float32
    ):
        raise TypeError("Expected BF16 input, E4M3 output and FP32 scales")
    if tuple(quanted.shape) != (e * cap, k) or tuple(scales.shape) != (
        e * cap,
        1,
    ):
        raise ValueError("Expected physical [E*capacity,K] q and [E*capacity,1] scales")
    if counts.dtype != torch.int32 or tuple(counts.shape) != (e,):
        raise TypeError("Expected device int32[E] counts")
    tensors = (inputs, quanted, scales, counts)
    if inputs.device.type != "cuda" or any(
        t.device != inputs.device or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("Expected contiguous tensors on one CUDA device")
    intervals = []
    for t in tensors:
        lo = t.data_ptr()
        hi = lo + t.numel() * t.element_size()
        if any(max(lo, a) < min(hi, b) for a, b in intervals):
            raise ValueError("Live input/output/count buffers overlap")
        intervals.append((lo, hi))
    if (
        type(workers) is not int
        or workers < 1
        or workers > 4096
        or num_warps not in (4, 8)
    ):
        raise ValueError("Invalid explicit launch geometry")
    if any(
        not isinstance(f, triton.runtime.jit.JITFunction)
        for f in (calc_scale, finalize_scale, quant_tensor)
    ):
        raise TypeError("Numeric callbacks must be pinned Triton JIT functions")
    if row_splits is not None:
        if row_splits not in (1, 2, 4, 8, 16, 32):
            raise ValueError("Invalid compact row split count")
        _quant_compact[(e, row_splits)](
            inputs,
            quanted,
            scales,
            counts,
            CAP=cap,
            K=k,
            SPLITS=row_splits,
            BLOCK=triton.next_power_of_2(k),
            CALC_SCALE=calc_scale,
            FINALIZE_SCALE=finalize_scale,
            QUANT_TENSOR=quant_tensor,
            num_warps=num_warps,
        )
        return quanted, scales
    _masked_row_quant[(workers,)](
        inputs,
        quanted,
        scales,
        counts,
        E=e,
        CAP=cap,
        K=k,
        WORKERS=workers,
        BLOCK_E=triton.next_power_of_2(e),
        BLOCK=triton.next_power_of_2(k),
        CALC_SCALE=calc_scale,
        FINALIZE_SCALE=finalize_scale,
        QUANT_TENSOR=quant_tensor,
        num_warps=num_warps,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return quanted, scales
