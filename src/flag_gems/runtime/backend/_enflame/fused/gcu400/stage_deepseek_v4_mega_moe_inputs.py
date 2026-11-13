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

import os
import sys

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


def _gcu_real_int64_storage():
    # torch_gcu narrows device int64 tensors to 4-byte physical slots unless
    # the env switch is set to exactly "1" or "true" (P17 int64 narrow
    # family); any other value keeps the narrowed layout.
    return os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") in ("1", "true")


def _int32_slot_view(tensor):
    """Zero-copy int32 view of a GCU int64 tensor's 4-byte physical slots."""
    words = torch.empty(0, dtype=torch.int32, device=tensor.device)
    return words.set_(
        tensor.untyped_storage(),
        int(tensor.storage_offset()),
        tuple(int(x) for x in tensor.shape),
        tuple(int(x) for x in tensor.stride()),
    )


@triton.jit
def _stage_deepseek_v4_mega_moe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    topk_ids,
    topk_weights,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    IDX_TY: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=1)
    amax = tl.maximum(amax, 1.0e-4)

    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
    scaled = tl.reshape(scaled, [BLOCK_K])
    fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
        fp8,
        mask=k_mask,
    )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(IDX_TY)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def stage_deepseek_v4_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
            "a multiple of 128."
        )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    # GCU physical layout: torch_gcu narrows int64 tensors to 4-byte slots
    # while they still report int64 ("GCU not support Long use Int replace"),
    # so a Triton store at the reported int64 width strides past the real
    # slots and leaves interleaved garbage. Write through a zero-copy int32
    # slot view instead (element offset == slot offset); with the int64 env
    # switch on (real 8-byte storage) the tensor is passed through untouched.
    if topk_idx_out.dtype == torch.int64:
        if _gcu_real_int64_storage():
            kernel_idx_out = topk_idx_out
            idx_ty = tl.int64
        else:
            kernel_idx_out = _int32_slot_view(topk_idx_out)
            idx_ty = tl.int32
    elif topk_idx_out.dtype == torch.int32:
        kernel_idx_out = topk_idx_out
        idx_ty = tl.int32
    else:
        raise NotImplementedError(
            "DeepSeek V4 MegaMoE input staging requires an int32 or int64 "
            "topk_idx_out tensor on this backend."
        )

    block_k = 128
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(top_k)
    with torch_device_fn.device(hidden_states.device):
        _stage_deepseek_v4_mega_moe_inputs_kernel[grid](
            hidden_states,
            x_fp8,
            x_sf,
            topk_ids,
            topk_weights,
            kernel_idx_out,
            topk_weights_out,
            hidden_states.stride(0),
            hidden_states.stride(1),
            x_fp8.stride(0),
            x_fp8.stride(1),
            x_sf.stride(0),
            x_sf.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            kernel_idx_out.stride(0),
            kernel_idx_out.stride(1),
            topk_weights_out.stride(0),
            topk_weights_out.stride(1),
            hidden_size,
            top_k,
            BLOCK_K=block_k,
            GROUP_K=32,
            BLOCK_TOPK=block_topk,
            IDX_TY=idx_ty,
            num_warps=4,
        )


_stage_deepseek_v4_mega_moe_inputs = stage_deepseek_v4_mega_moe_inputs


# ---------------------------------------------------------------------------
# Generic-module rebinding (test / bench / direct-import routing)
#
# SpecOpRegistrar only replaces the top-level ``flag_gems`` entry.
# ``from flag_gems.fused.stage_deepseek_v4_mega_moe_inputs import
# stage_deepseek_v4_mega_moe_inputs`` (used by the accuracy test and the
# benchmark) keeps resolving to the generic implementation, which makes this
# vendor copy invisible to them. Rebind the generic module's public function
# to this copy, mirroring the mkldnn_rnn_layer / std_mean / bin_topk
# precedents. The generic module is always loaded first
# (``flag_gems/__init__`` imports it before registering the backend), so the
# lookups below find it; when absent the rebinding is a no-op and behaviour
# is unchanged. The vendor module can execute twice under two module names
# (the arch package is imported both directly and through the backend path);
# every execution rebinds to its own function object, and the identity guard
# only prevents pathological self-rebinding.
_generic_stage_module = sys.modules.get(
    "flag_gems.fused.stage_deepseek_v4_mega_moe_inputs"
)
if _generic_stage_module is not None and _generic_stage_module is not sys.modules.get(
    __name__
):
    _generic_stage_module.stage_deepseek_v4_mega_moe_inputs = (
        stage_deepseek_v4_mega_moe_inputs
    )
    _generic_stage_module._stage_deepseek_v4_mega_moe_inputs = (
        _stage_deepseek_v4_mega_moe_inputs
    )

_generic_fused_package = sys.modules.get("flag_gems.fused")
if _generic_fused_package is not None:
    _generic_fused_package.stage_deepseek_v4_mega_moe_inputs = (
        stage_deepseek_v4_mega_moe_inputs
    )
