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

"""MTT S5000 TLE INT8 QK/BF16 PV; baseline remains independently usable."""

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
    logger.debug("GEMS_MTHREADS SAGE_ATTENTION")
    from flag_gems.fused.sage_attention import run_bf16_sage
    from flag_gems.fused.sage_attention import sage_attention as baseline
    from flag_gems.fused.sage_attention import validate_sage_inputs

    if implementation == "baseline":
        return baseline(
            query,
            key,
            value,
            tensor_layout=tensor_layout,
            sm_scale=sm_scale,
            is_causal=is_causal,
            return_lse=return_lse,
            pv_dtype=pv_dtype,
            implementation="baseline",
        )
    scale = validate_sage_inputs(
        query, key, value, tensor_layout, sm_scale, is_causal, return_lse
    )
    if (
        implementation not in ("auto", "tle")
        or pv_dtype != "bf16"
        or query.dtype != torch.bfloat16
    ):
        raise NotImplementedError(
            "S5000 TLE SageAttention requires BF16 inputs/PV and implementation auto/tle"
        )
    if query.device.type != "musa":
        raise NotImplementedError("The S5000 TLE implementation requires MUSA")
    try:
        tle = importlib.import_module("triton.experimental.tle.language")
    except ImportError as error:
        if implementation == "auto":
            return baseline(
                query,
                key,
                value,
                tensor_layout=tensor_layout,
                sm_scale=scale,
                implementation="baseline",
            )
        raise NotImplementedError(
            "S5000 TLE SageAttention requires FlagTree TLE"
        ) from error
    if not all(
        hasattr(getattr(tle, "gpu", None), name)
        for name in ("alloc", "copy", "wgmma", "wgmma_wait")
    ):
        raise NotImplementedError("The compiler lacks the required TLE matrix APIs")
    from ._sage_attention_tle import core

    return run_bf16_sage(
        query,
        key,
        value,
        tensor_layout=tensor_layout,
        scale=scale,
        core=core,
        ordered_pv=True,
    )
