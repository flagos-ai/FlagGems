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

import torch
import triton
import triton.language as tl

from flag_gems.ops.std_mean import (
    _GLOBAL_SINGLE_CTA_LIMIT,
    _float64_as_int32_bits,
    _has_names,
    _normalize_dims,
    _output_shape,
    _real_dtype,
    _std_mean_global_finish_kernel,
    _std_mean_impl,
    _supported_dtypes,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024
_MAX_GLOBAL_PROGRAMS = 512


@libentry()
@triton.jit
def _std_mean_global_map_kernel_clamped(
    inp,
    scratch,
    N,
    partial_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Build mergeable shifted central moments with clamped, unmasked loads.

    The bounds test is folded into an arithmetic weight instead of selecting on
    the load result, so a fully out-of-range tail block contributes exactly
    ``0`` to every accumulator.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    compute_dtype = tl.float64 if inp.type.element_ty == tl.float64 else tl.float32
    lane_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    last = N - 1
    shift = tl.load(inp + pid * BLOCK_SIZE).to(compute_dtype)
    grid_stride = num_programs * BLOCK_SIZE

    delta_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    count_vec = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        weight = (offsets < N).to(compute_dtype)
        values = tl.load(inp + tl.minimum(offsets, last)).to(compute_dtype)
        delta_sum += (values - shift) * weight
        value_sum += values * weight
        count_vec += weight

    count = tl.sum(count_vec)
    shifted_mean = shift + tl.sum(delta_sum) / count
    direct_mean = tl.sum(value_sum) / count
    use_shifted = tl.abs(shifted_mean) * 8.0 >= tl.abs(shift)
    mean = tl.where(use_shifted, shifted_mean, direct_mean)

    squared_deviation_sum = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for start in range(0, N, grid_stride):
        offsets = start + lane_offsets
        weight = (offsets < N).to(compute_dtype)
        values = tl.load(inp + tl.minimum(offsets, last)).to(compute_dtype)
        deviation = values - mean
        squared_deviation_sum += deviation * deviation * weight
    m2 = tl.sum(squared_deviation_sum)
    m2 = tl.where(m2 < 0.0, 0.0, m2)
    tl.store(scratch + pid, mean)
    tl.store(scratch + partial_stride + pid, m2)
    tl.store(scratch + 2 * partial_stride + pid, count)


def _use_global_path(inp, dim, correction):
    if _has_names(inp) or inp.is_complex():
        return False
    if inp.dtype not in _supported_dtypes() or inp.ndim == 0:
        return False
    dims = _normalize_dims(dim, inp.ndim)
    if len(dims) != inp.ndim:
        return False
    n = math.prod(inp.shape[index] for index in dims)
    if n <= _GLOBAL_SINGLE_CTA_LIMIT or float(n) - float(correction) <= 0:
        return False
    return True


def _global_std_mean(inp, dims, n, correction, keepdim):
    output_shape = _output_shape(tuple(inp.shape), dims, keepdim)
    std_dtype = _real_dtype(inp.dtype)
    use_fp64_denominator = std_dtype == torch.float64
    denominator = float(n) - float(correction)
    denominator_low_bits, denominator_high_bits = (
        _float64_as_int32_bits(denominator) if use_fp64_denominator else (0, 0)
    )

    out_std = torch.empty(output_shape, dtype=std_dtype, device=inp.device)
    out_mean = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    num_programs = min(triton.cdiv(n, _BLOCK_SIZE), _MAX_GLOBAL_PROGRAMS)
    partial_dtype = torch.float64 if inp.dtype == torch.float64 else torch.float32
    # Pad the partial slots to a power of two and zero the padding so the merge
    # kernel always reads a full, in-range tile.
    partial_stride = triton.next_power_of_2(num_programs)
    scratch = torch.zeros((partial_stride * 3,), dtype=partial_dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        _std_mean_global_map_kernel_clamped[(num_programs,)](
            inp.contiguous(),
            scratch,
            n,
            partial_stride,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        _std_mean_global_finish_kernel[(1,)](
            scratch,
            out_std,
            out_mean,
            n,
            denominator,
            denominator_low_bits,
            denominator_high_bits,
            USE_FP64_DENOMINATOR=use_fp64_denominator,
            NUM_PARTIALS=partial_stride,
            BLOCK_SIZE=partial_stride,
        )
    return out_std, out_mean


def std_mean_correction(inp, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN STD_MEAN_CORRECTION")
    effective_correction = 1.0 if correction is None else float(correction)
    if _use_global_path(inp, dim, effective_correction):
        dims = _normalize_dims(dim, inp.ndim)
        n = math.prod(inp.shape[index] for index in dims)
        return _global_std_mean(inp, dims, n, effective_correction, keepdim)
    return _std_mean_impl(inp, dim, correction, keepdim)
