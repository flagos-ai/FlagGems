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

"""Public ARM W4A8-G128 packing and Triton execution helpers."""

from __future__ import annotations

import platform

import torch

from .q4_kernels import (
    _pack_lhs_qai8dxp_asym_panel4_kernel,
    _q4_fused_decode_asym_g128_kai_sdot_kernel,
    _q4_prefill_asym_g128_i8mm_kernel,
)

_GROUP_SIZE = 128


def pack_rhs_qsi4c128p(
    quantized: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Pack signed INT4 G128 weights into the FlagGems N4/K128 ABI."""
    if quantized.dtype != torch.int8 or quantized.ndim != 2:
        raise ValueError("quantized weight must be an INT8 [N,K] tensor")
    if quantized.device.type != "cpu" or scales.device != quantized.device:
        raise ValueError("W4A8-G128 packing supports CPU tensors only")

    n, k = quantized.shape
    groups = k // _GROUP_SIZE
    if n <= 0 or k <= 0 or n % 4 or k % _GROUP_SIZE:
        raise ValueError("W4A8-G128 requires N%4=0 and K%128=0")
    if scales.shape != (n, groups) or not scales.is_floating_point():
        raise ValueError("invalid W4A8-G128 scale shape or dtype")
    if not torch.isfinite(scales).all():
        raise ValueError("W4A8-G128 scales must be finite")
    if int(quantized.min()) < -8 or int(quantized.max()) > 7:
        raise ValueError("W4 values must be in [-8,7]")

    tile_stride = groups * 264 + 16
    packed = torch.empty(
        (n // 4, tile_stride), dtype=torch.uint8, device=quantized.device
    )
    packed_groups = packed[:, : groups * 264].reshape(n // 4, groups, 264)
    scales_bf16 = scales.to(torch.bfloat16)
    packed_groups[:, :, :8].view(torch.bfloat16).copy_(
        scales_bf16.reshape(n // 4, 4, groups).permute(0, 2, 1).contiguous()
    )

    groups32 = k // 32
    grouped32 = quantized.reshape(n, groups32, 32)
    low = grouped32[:, :, :16].reshape(n, groups32, 2, 8).to(torch.int16) & 15
    high = grouped32[:, :, 16:].reshape(n, groups32, 2, 8).to(torch.int16) & 15
    packed32 = (
        (low | (high << 4))
        .to(torch.uint8)
        .reshape(n // 4, 4, groups32, 2, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(n // 4, groups32, 64)
    )
    packed_groups[:, :, 8:264].copy_(
        packed32.reshape(n // 4, groups, 4, 64).reshape(n // 4, groups, 256)
    )

    scaled_sums = (
        quantized.reshape(n, groups, _GROUP_SIZE).to(torch.int16).sum(dim=-1) * 16
    )
    weighted_sums = (scaled_sums.to(torch.float32) * scales_bf16.to(torch.float32)).sum(
        dim=1
    )
    packed[:, groups * 264 :].view(torch.float32).copy_(
        weighted_sums.reshape(n // 4, 4)
    )
    return packed.reshape(-1).contiguous()


def _validate_linear_inputs(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    n: int,
    k: int,
) -> None:
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("W4A8-G128 requires an AArch64 host")
    if input.device.type != "cpu" or packed_weight.device.type != "cpu":
        raise ValueError("W4A8-G128 supports CPU tensors only")
    if input.dtype != torch.bfloat16 or not input.is_contiguous():
        raise ValueError("W4A8-G128 requires contiguous BF16 input")
    if packed_weight.dtype != torch.uint8 or not packed_weight.is_contiguous():
        raise ValueError("W4A8-G128 requires a contiguous UINT8 packed weight")
    if input.ndim == 0 or input.numel() == 0 or input.shape[-1] != k:
        raise ValueError("invalid W4A8-G128 input shape")
    if n <= 0 or k <= 0 or n % 4 or k % _GROUP_SIZE:
        raise ValueError("W4A8-G128 requires N%4=0 and K%128=0")
    expected = (n // 4) * ((k // _GROUP_SIZE) * 264 + 16)
    if packed_weight.numel() != expected:
        raise ValueError("invalid W4A8-G128 packed byte count")


def w4a8_g128_linear(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Execute a packed W4A8-G128 linear layer with ARM Triton kernels."""
    _validate_linear_inputs(input, packed_weight, n, k)
    original_shape = tuple(input.shape)
    rows = input.numel() // k
    input_2d = input.view(rows, k)

    if rows < 4:
        partitions = min(max(1, torch.get_num_threads()), n // 4)
        scratch_bytes = rows * partitions * (8 + k)
        output_bytes = rows * n * torch.bfloat16.itemsize
        storage = torch.empty(
            (scratch_bytes + output_bytes) // torch.bfloat16.itemsize,
            dtype=torch.bfloat16,
            device=input.device,
        )
        output = storage.as_strided((rows, n), (n, 1), scratch_bytes // 2)
        _q4_fused_decode_asym_g128_kai_sdot_kernel[(rows, partitions)](
            input_2d,
            storage,
            packed_weight,
            scratch_bytes,
            k,
            K=k,
            N=n,
        )
    else:
        padded_rows = 4 * ((rows + 3) // 4)
        lhs_panel_stride = 32 + 4 * k
        lhs = torch.empty(
            (padded_rows // 4) * lhs_panel_stride,
            dtype=torch.uint8,
            device=input.device,
        )
        output_padded = torch.empty(
            (padded_rows, n), dtype=torch.bfloat16, device=input.device
        )
        _pack_lhs_qai8dxp_asym_panel4_kernel[(padded_rows // 4,)](
            input_2d,
            lhs,
            rows,
            k,
            K=k,
        )
        _q4_prefill_asym_g128_i8mm_kernel[(padded_rows // 4, n // 4)](
            lhs,
            packed_weight,
            output_padded,
            N=n,
            K=k,
        )
        output = output_padded[:rows]

    return output.view(*original_shape[:-1], n)


__all__ = ["pack_rhs_qsi4c128p", "w4a8_g128_linear"]
