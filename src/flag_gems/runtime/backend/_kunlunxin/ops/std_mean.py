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
import sys

import triton
import triton.language as tl

from flag_gems.ops.std_mean import _stable_mean_from_shift
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
_GENERIC_MODULE_NAME = "flag_gems.ops.std_mean"


@libentry()
@triton.jit
def _std_mean_global_map_kernel(
    inp,
    scratch,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Build mergeable shifted central moments for a flat real tensor."""
    pid = ext.program_id(0)
    num_programs = tl.num_programs(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    lane_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    shift = tl.load(inp + pid * BLOCK_SIZE).to(compute_dtype)
    grid_stride = num_programs * BLOCK_SIZE

    delta_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    # XPU: keep the count per lane. A loop-carried `count += tl.sum(...)` is
    # mis-compiled here and also corrupts the `m2` reduce further down.
    count_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        values = tl.load(inp + offsets, mask=mask, other=shift).to(compute_dtype)
        delta = values - shift
        delta_sum += tl.where(mask, delta, 0.0)
        value_sum += tl.where(mask, values, 0.0)
        count_lanes += mask.to(tl.float32)
    count = tl.sum(count_lanes)

    delta_total = tl.sum(delta_sum)
    mean = _stable_mean_from_shift(shift, delta_total, tl.sum(value_sum), count)
    squared_deviation_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        values = tl.load(inp + offsets, mask=mask, other=mean).to(compute_dtype)
        deviation = values - mean
        squared_deviation_sum += tl.where(mask, deviation * deviation, 0.0)
    m2 = tl.sum(squared_deviation_sum)
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    tl.store(scratch + pid, mean)
    tl.store(scratch + num_programs + pid, m2)
    tl.store(scratch + 2 * num_programs + pid, count)


@libentry()
@triton.jit
def _std_mean_global_map_complex_kernel(
    inp,
    scratch,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Build mergeable moments for a flat complex tensor."""
    pid = ext.program_id(0)
    num_programs = tl.num_programs(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    lane_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    first = 2 * pid * BLOCK_SIZE
    shift_real = tl.load(inp + first).to(compute_dtype)
    shift_imag = tl.load(inp + first + 1).to(compute_dtype)
    grid_stride = num_programs * BLOCK_SIZE

    delta_real_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    delta_imag_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    real_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    imag_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    # XPU: same loop-carried `tl.sum` defect as the real kernel above.
    count_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        storage_offsets = 2 * offsets
        real = tl.load(inp + storage_offsets, mask=mask, other=shift_real).to(
            compute_dtype
        )
        imag = tl.load(inp + storage_offsets + 1, mask=mask, other=shift_imag).to(
            compute_dtype
        )
        delta_real = real - shift_real
        delta_imag = imag - shift_imag
        delta_real_sum += tl.where(mask, delta_real, 0.0)
        delta_imag_sum += tl.where(mask, delta_imag, 0.0)
        real_sum += tl.where(mask, real, 0.0)
        imag_sum += tl.where(mask, imag, 0.0)
        count_lanes += mask.to(tl.float32)
    count = tl.sum(count_lanes)

    delta_real_total = tl.sum(delta_real_sum)
    delta_imag_total = tl.sum(delta_imag_sum)
    mean_real = _stable_mean_from_shift(
        shift_real, delta_real_total, tl.sum(real_sum), count
    )
    mean_imag = _stable_mean_from_shift(
        shift_imag, delta_imag_total, tl.sum(imag_sum), count
    )
    real_m2_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    imag_m2_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        mask = offsets < N
        storage_offsets = 2 * offsets
        real = tl.load(inp + storage_offsets, mask=mask, other=mean_real).to(
            compute_dtype
        )
        imag = tl.load(inp + storage_offsets + 1, mask=mask, other=mean_imag).to(
            compute_dtype
        )
        real_deviation = real - mean_real
        imag_deviation = imag - mean_imag
        real_m2_sum += tl.where(mask, real_deviation * real_deviation, 0.0)
        imag_m2_sum += tl.where(mask, imag_deviation * imag_deviation, 0.0)
    real_m2 = tl.sum(real_m2_sum)
    imag_m2 = tl.sum(imag_m2_sum)
    real_m2 = tl.where(real_m2 < 0.0, 0.0, real_m2)
    imag_m2 = tl.where(imag_m2 < 0.0, 0.0, imag_m2)
    tl.store(scratch + pid, mean_real)
    tl.store(scratch + num_programs + pid, mean_imag)
    tl.store(scratch + 2 * num_programs + pid, real_m2)
    tl.store(scratch + 3 * num_programs + pid, imag_m2)
    tl.store(scratch + 4 * num_programs + pid, count)


def _patch_generic_map_kernels():
    generic_module = sys.modules.get(_GENERIC_MODULE_NAME)
    if generic_module is None:  # pragma: no cover - import-order safety net
        try:
            import importlib

            generic_module = importlib.import_module(_GENERIC_MODULE_NAME)
        except ImportError:
            return
    generic_module._std_mean_global_map_kernel = _std_mean_global_map_kernel
    generic_module._std_mean_global_map_complex_kernel = (
        _std_mean_global_map_complex_kernel
    )


_patch_generic_map_kernels()


__all__ = [
    "_std_mean_global_map_kernel",
    "_std_mean_global_map_complex_kernel",
]
