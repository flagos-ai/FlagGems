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

import torch
import triton
import triton.language as tl

from flag_gems.ops.feature_dropout import generate_feature_mask_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

_PHASE_A_BLOCK = 1024
_B2_MAX_TILE_AREA = 1024
_MAX_GRID_DIM = 2**31 - 1
_MAX_PHILOX_CHANNELS = 2**32 - 1
_MAX_INT64_OFFSET = 2**63 - 1


@triton.jit
def apply_feature_mask_inplace_kernel(
    X,
    MASK,
    numel,
    C,
    spatial_size,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < numel

    channel_spatial_size = C * spatial_size
    n_idx = offset // channel_spatial_size
    c_idx = (offset % channel_spatial_size) // spatial_size
    mask_idx = n_idx * C + c_idx

    x = tl.load(X + offset, mask=mask, other=0.0)
    m = tl.load(MASK + mask_idx, mask=mask, other=0.0)
    tl.store(X + offset, x * m, mask=mask)


@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def packed_feature_dropout_inplace_kernel(
    X,
    total_channels,
    spatial_size,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK_CH: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_ch = tl.program_id(0).to(tl.int64)
    pid_s = tl.program_id(1).to(tl.int64)

    channel_offsets = pid_ch * BLOCK_CH + tl.arange(0, BLOCK_CH).to(tl.int64)
    spatial_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
    channel_mask = channel_offsets < total_channels
    spatial_mask = spatial_offsets < spatial_size

    channel_ids = channel_offsets
    flat_offsets = channel_ids[:, None] * spatial_size + spatial_offsets[None, :]
    valid = channel_mask[:, None] & spatial_mask[None, :]

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0 = c0 + channel_ids.to(tl.uint32)
    zero = c0 * 0
    c1 = c1 + zero
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, zero, zero)
    keep = uint_to_uniform_float(r0) > p
    multiplier = tl.where(keep, scale, 0.0)

    x = tl.load(X + flat_offsets, mask=valid, other=0.0)
    tl.store(X + flat_offsets, x * multiplier[:, None], mask=valid)


def _rounded_scale(input, scale, spatial_size):
    # Match MThreads FP16/BF16 feature-dropout scale rounding for spatial inputs.
    if spatial_size >= 2 and input.dtype in (torch.float16, torch.bfloat16):
        return torch.tensor(scale, dtype=input.dtype).item()
    return scale


def _feature_dropout_phase_a_contiguous(input, p, N, C, spatial_size, numel):
    device = input.device

    scale = _rounded_scale(input, 1.0 / (1.0 - p), spatial_size)
    mask = torch.empty(N, C, device=device, dtype=torch.float32)
    BLOCK_N = min(triton.next_power_of_2(N), 64)
    BLOCK_C = min(triton.next_power_of_2(C), 64)
    grid_mask = (triton.cdiv(N, BLOCK_N), triton.cdiv(C, BLOCK_C))

    increment = triton.cdiv(N * C, 4) * 4
    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        generate_feature_mask_kernel[grid_mask](
            mask, N, C, p, scale, philox_seed, philox_offset, BLOCK_N, BLOCK_C
        )

        grid_apply = (triton.cdiv(numel, _PHASE_A_BLOCK),)
        apply_feature_mask_inplace_kernel[grid_apply](
            input, mask, numel, C, spatial_size, _PHASE_A_BLOCK
        )

    return input


def _feature_dropout_phase_a(input, p, train=True):
    if not train or p == 0:
        return input
    if p == 1:
        input.zero_()
        return input
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    assert 0.0 < p < 1.0, "p must be in (0, 1)"
    numel = input.numel()
    if numel == 0:
        return input

    if not input.is_contiguous():
        from flag_gems.ops.feature_dropout import feature_dropout_

        return feature_dropout_(input, p, train)

    N = input.shape[0]
    C = input.shape[1]
    spatial_size = numel // (N * C)
    return _feature_dropout_phase_a_contiguous(input, p, N, C, spatial_size, numel)


def _phase_b2_config(total_channels, spatial_size):
    # These bins come from the tile search. They keep BLOCK_CH*BLOCK_S within
    # the register-pressure budget while increasing work per program with S.
    if spatial_size == 1:
        return min(128, max(1, triton.next_power_of_2(total_channels))), 1
    if spatial_size <= 4:
        block_s = triton.next_power_of_2(spatial_size)
        return min(64, max(1, 1024 // block_s)), block_s
    if spatial_size <= 32:
        block_s = min(triton.next_power_of_2(spatial_size), 32)
        return min(32, max(1, _B2_MAX_TILE_AREA // block_s)), block_s
    if spatial_size <= 128:
        return 8, 128
    if spatial_size <= 512:
        return 2, 512
    return 1, 1024


def _phase_b2_index_safe(total_channels, spatial_size, numel, block_ch, block_s):
    if total_channels > _MAX_PHILOX_CHANNELS:
        return False
    if numel > _MAX_INT64_OFFSET:
        return False

    grid_ch = (total_channels + block_ch - 1) // block_ch
    grid_s = (spatial_size + block_s - 1) // block_s
    return 0 < grid_ch <= _MAX_GRID_DIM and 0 < grid_s <= _MAX_GRID_DIM


def _phase_b2_coarse_fallback(total_channels, spatial_size, numel, block_ch, block_s):
    if spatial_size <= 1:
        return False
    if total_channels > _MAX_PHILOX_CHANNELS or numel > _MAX_INT64_OFFSET:
        return True

    if total_channels < _PHASE_A_BLOCK * block_ch:
        return False

    # With NC >= 1024 * BLOCK_CH, B2 has more programs than the Phase A
    # apply pass unless it covers full 1024-element tiles exactly. This is
    # algebraically equivalent to the program-count comparison below without
    # constructing either grid on the fallback path.
    return block_ch * block_s != _PHASE_A_BLOCK or spatial_size % block_s != 0


def _phase_b2_is_profitable(total_channels, spatial_size, numel, block_ch, block_s):
    if block_ch <= 0 or block_s <= 0 or block_ch * block_s > _B2_MAX_TILE_AREA:
        return False
    if not _phase_b2_index_safe(total_channels, spatial_size, numel, block_ch, block_s):
        return False

    phase_a_programs = (numel + _PHASE_A_BLOCK - 1) // _PHASE_A_BLOCK
    phase_b2_programs = ((total_channels + block_ch - 1) // block_ch) * (
        (spatial_size + block_s - 1) // block_s
    )
    # When B2 launches more programs than the Phase A apply pass and the
    # channel population can amortize a full Phase A block, keep the proven
    # two-kernel path. This is a structural work-per-program gate.
    return not (
        spatial_size > 1
        and phase_b2_programs > phase_a_programs
        and total_channels >= _PHASE_A_BLOCK * block_ch
    )


def _feature_dropout_phase_b2(input, p, train=True):
    if not train or p == 0:
        return input
    if p == 1:
        input.zero_()
        return input
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    assert 0.0 < p < 1.0, "p must be in (0, 1)"
    numel = input.numel()
    if numel == 0:
        return input

    if not input.is_contiguous():
        return _feature_dropout_phase_a(input, p, train)

    N = input.shape[0]
    C = input.shape[1]
    total_channels = N * C
    spatial_size = numel // total_channels

    block_ch, block_s = _phase_b2_config(total_channels, spatial_size)
    if _phase_b2_coarse_fallback(
        total_channels, spatial_size, numel, block_ch, block_s
    ):
        return _feature_dropout_phase_a_contiguous(input, p, N, C, spatial_size, numel)

    if not _phase_b2_is_profitable(
        total_channels, spatial_size, numel, block_ch, block_s
    ):
        return _feature_dropout_phase_a_contiguous(input, p, N, C, spatial_size, numel)

    scale = _rounded_scale(input, 1.0 / (1.0 - p), spatial_size)
    increment = triton.cdiv(total_channels, 4) * 4
    grid = (triton.cdiv(total_channels, block_ch), triton.cdiv(spatial_size, block_s))

    with torch_device_fn.device(input.device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        packed_feature_dropout_inplace_kernel[grid](
            input,
            total_channels,
            spatial_size,
            p,
            scale,
            philox_seed,
            philox_offset,
            BLOCK_CH=block_ch,
            BLOCK_S=block_s,
        )

    return input


feature_dropout_ = _feature_dropout_phase_b2
feature_dropout_.__name__ = "feature_dropout_"
