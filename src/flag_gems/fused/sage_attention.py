# Copyright 2026, The FlagOS Contributors.
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
#

"""Dense inference SageAttention with explicit quantization semantics.

BF16 inputs, D128, equal Q/K/V heads, NHD or HND with unit last stride.
Q/K use per-thread INT8 scales and K is centered in BF16. PV uses BF16
probabilities/values and FP32 accumulation. This is a distinct contract from
the optional Hopper INT8-QK/FP8-PV implementation; precision never changes
as a fallback. Inputs are not modified. No mask, causal mode, LSE or backward.
"""

import logging
import math

import torch

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def validate_sage_inputs(
    query, key, value, tensor_layout, sm_scale, is_causal, return_lse
):
    if tensor_layout not in ("NHD", "HND"):
        raise ValueError("tensor_layout must be NHD or HND")
    if is_causal or return_lse:
        raise NotImplementedError("SageAttention supports dense noncausal output only")
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("SageAttention requires four-dimensional tensors")
    if query.device.type not in ("cuda", "musa"):
        raise NotImplementedError("SageAttention requires a CUDA or MUSA device")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("SageAttention requires BF16 or FP16 inputs")
    if any(x.device != query.device or x.dtype != query.dtype for x in (key, value)):
        raise ValueError("Q/K/V must share dtype and device")
    if any(x.stride(-1) != 1 for x in (query, key, value)):
        raise ValueError("Q/K/V must have unit last-dimension stride")
    if any(x.numel() > 2**31 - 1 for x in (query, key, value)):
        raise ValueError(
            "SageAttention currently supports at most INT32_MAX elements per input"
        )
    if any(
        sum((size - 1) * stride for size, stride in zip(x.shape, x.stride())) >= 2**31
        for x in (query, key, value)
    ):
        raise ValueError("SageAttention currently requires INT32-safe input strides")
    if any(x.requires_grad for x in (query, key, value)) and torch.is_grad_enabled():
        raise NotImplementedError("SageAttention is inference-only")
    head_axis, token_axis = (2, 1) if tensor_layout == "NHD" else (1, 2)
    if query.shape[-1] != 128 or key.shape[-1] != 128:
        raise ValueError("SageAttention currently supports head dimension128")
    if (
        query.shape[0] != key.shape[0]
        or query.shape[head_axis] != key.shape[head_axis]
        or key.shape != value.shape
    ):
        raise ValueError("SageAttention requires equal heads and matching K/V")
    if (
        min(
            query.shape[0],
            query.shape[head_axis],
            query.shape[token_axis],
            key.shape[token_axis],
        )
        <= 0
    ):
        raise ValueError("SageAttention does not support empty dimensions")
    if (
        max(
            ((x.shape[token_axis] + 127) // 128)
            * 128
            * x.shape[0]
            * x.shape[head_axis]
            * 128
            for x in (query, key, value)
        )
        >= 2**31
    ):
        raise ValueError("SageAttention currently requires INT32-safe padded operands")
    scale = 128**-0.5 if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("sm_scale must be finite and positive")
    return scale


def run_bf16_sage(query, key, value, *, tensor_layout, scale, core, ordered_pv=False):
    # Imports are deliberately below capability selection: the baseline does
    # not import a TLE/Gluon/SM90 module, including during dependency scanning.
    from .sage_attention_quantization import quantize_qk

    with torch_device_fn.device(query.device):
        z = quantize_qk(query, key, value, tensor_layout=tensor_layout)
        batch, heads, nq, dimension = z[0].shape
        output = torch.empty(
            (batch, nq, heads, dimension), dtype=query.dtype, device=query.device
        )
        use_ordered = ordered_pv and z[2].shape[2] <= 128
        bm = (
            16
            if use_ordered
            else (128 if query.device.type == "musa" and nq >= 2048 else 64)
        )
        warps = 16 if bm == 128 and query.device.type == "musa" else 8
        core(
            z,
            output,
            sm_scale=scale,
            bm=bm,
            warps=warps,
            stages=1,
            sum_mode=1,
            precise_div=True,
            ordered_pv=use_ordered,
        )
        return output.permute(0, 2, 1, 3) if tensor_layout == "HND" else output


def sage_attention(
    query,
    key,
    value,
    *,
    tensor_layout="NHD",
    sm_scale=None,
    is_causal=False,
    return_lse=False,
    pv_dtype="bf16",
    implementation="auto",
):
    """INT8 QK/BF16 PV baseline using ordinary upstream Triton.

    ``implementation`` selects baseline, TLE or SM90 without changing ``pv_dtype``.
    Vendor overrides are resolved by FlagGems' existing SpecOpRegistrar. The
    generic implementation accepts auto/baseline and BF16 PV only. FP8 PV must
    be explicitly requested on a backend providing that numerical contract.
    """
    logger.debug("GEMS SAGE_ATTENTION_BASELINE")
    scale = validate_sage_inputs(
        query, key, value, tensor_layout, sm_scale, is_causal, return_lse
    )
    if pv_dtype != "bf16" or query.dtype != torch.bfloat16:
        raise NotImplementedError(
            "The portable baseline requires BF16 inputs and BF16 PV"
        )
    if implementation not in ("auto", "baseline"):
        raise NotImplementedError(
            "The generic SageAttention implementation is baseline only"
        )
    from .sage_attention_baseline import core

    return run_bf16_sage(
        query,
        key,
        value,
        tensor_layout=tensor_layout,
        scale=scale,
        core=core,
        ordered_pv=query.device.type == "musa",
    )
