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

"""Optional SM90 INT8 QK / FP8 PV SageAttention, selected before execution."""

import importlib
import logging

import torch

logger = logging.getLogger(__name__)


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
    logger.debug("GEMS_NVIDIA SAGE_ATTENTION")
    from flag_gems.fused.sage_attention import sage_attention as baseline
    from flag_gems.fused.sage_attention import validate_sage_inputs

    if pv_dtype == "bf16":
        return baseline(
            query,
            key,
            value,
            tensor_layout=tensor_layout,
            sm_scale=sm_scale,
            is_causal=is_causal,
            return_lse=return_lse,
            pv_dtype=pv_dtype,
            implementation=implementation,
        )
    scale = validate_sage_inputs(
        query, key, value, tensor_layout, sm_scale, is_causal, return_lse
    )
    if pv_dtype != "fp8" or implementation not in ("auto", "sm90"):
        raise NotImplementedError(
            "Hopper FP8 PV requires implementation auto/sm90 and pv_dtype fp8"
        )
    if query.device.type != "cuda" or torch.cuda.get_device_capability(
        query.device
    ) != (9, 0):
        raise NotImplementedError("FP8 PV SageAttention requires CUDA SM90")
    try:
        tle = importlib.import_module("triton.experimental.tle.language")
    except ImportError as error:
        raise NotImplementedError("SM90 SageAttention requires FlagTree TLE") from error
    if not all(
        hasattr(getattr(tle, "gpu", None), name)
        for name in ("alloc", "copy", "wgmma", "wgmma_wait")
    ):
        raise NotImplementedError("The selected compiler lacks required TLE GPU APIs")
    # All checks precede allocation/quantization. No execution error is caught.
    from .sage_attention_impl import sage_attention as sm90

    with torch.cuda.device(query.device):
        return sm90(
            query,
            key,
            value,
            tensor_layout=tensor_layout,
            sm_scale=scale,
            is_causal=False,
            return_lse=False,
        )
