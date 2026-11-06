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

"""SM90 FlagTree/TLE SageAttention: native INT8 QK / FP8 PV arithmetic.

Validated deployment candidate, separate from the installed H3 dispatch.
auto keeps the original pipeline for one K tile and uses KV single buffers
with a 168-register cap for longer sequences. No Gluon or native Sage calls.
"""

import math

import torch

from .core import core as _experimental_core
from .quantization import quantize_sage_sm90


def core(z, output, sm_scale=None, *, mode="auto"):
    """Kernel-only entry; z is the tuple from quantize_sage_sm90."""
    if mode == "auto":
        mode = "control" if z[2].shape[2] <= 128 else "kv1"
    if mode not in ("control", "k1", "kv1"):
        raise ValueError("mode must be auto, control, k1, or kv1")
    return _experimental_core(
        z,
        output,
        sm_scale=sm_scale,
        k_single=mode != "control",
        v_single=mode == "kv1",
        maxnreg=168 if mode != "control" else None,
    )


def sage_attention(
    query,
    key,
    value,
    *,
    tensor_layout="NHD",
    sm_scale=None,
    is_causal=False,
    return_lse=False,
    mode="auto",
):
    """Dense noncausal D128 attention with native INT8 QK/FP8 PV arithmetic."""
    if is_causal or return_lse:
        raise NotImplementedError("Only dense noncausal output is supported")
    scale = 128**-0.5 if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise NotImplementedError("This kernel requires a finite positive scale")
    z = quantize_sage_sm90(query, key, value, tensor_layout=tensor_layout)
    b, h, nq, d = z[0].shape
    output = torch.empty((b, nq, h, d), device=query.device, dtype=query.dtype)
    core(z, output, scale, mode=mode)
    return output.permute(0, 2, 1, 3) if tensor_layout == "HND" else output
