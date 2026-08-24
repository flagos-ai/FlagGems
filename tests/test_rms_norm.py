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

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.ops.rms_norm_w8a16_fp8 import rms_norm_w8a16_fp8

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_GROUP_SIZE = 128


def _cuda_fp8_e4m3fn_available():
    if FP8_DTYPE is None or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _quantize_fp8_weight(weight, group_size=FP8_GROUP_SIZE):
    fp8_info = torch.finfo(FP8_DTYPE)
    grouped_weight = weight.float().reshape(-1, group_size)
    scale = (grouped_weight.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(
        min=1e-8
    )
    weight_fp8 = (
        (grouped_weight / scale)
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
        .reshape_as(weight)
        .contiguous()
    )
    return weight_fp8, scale.squeeze(-1).to(weight.dtype).contiguous()


@pytest.mark.rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, layer_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.tensor(
        np_weight, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    eps = 1e-5

    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(weight)

    def _torch_rms_norm(x, weight, eps):
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        hidden_states = upcast_x * torch.rsqrt(variance + eps).to(torch.float32)
        hidden_states = hidden_states.to(x.dtype)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)
    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    res_grad = torch.tensor(
        np_grad, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_grad = utils.to_reference(res_grad)

    res_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), res_grad)
    ref_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight), ref_grad
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    utils.gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N)


@pytest.mark.rms_norm_w8a16_fp8
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4096),
        (128, 4096),
        (512, 4096),
        (64, 8192),
        (1, 16384),
        (1, 32768),
    ],
)
@pytest.mark.skipif(
    not _cuda_fp8_e4m3fn_available(),
    reason="RMSNorm W8A16 FP8 requires CUDA sm90+ float8_e4m3fn support",
)
def test_rms_norm_w8a16_fp8(shape):
    dtype = torch.bfloat16
    m, n = shape
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, (m, n)).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, (n,)).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device)
    weight = torch.tensor(np_weight, dtype=dtype, device=flag_gems.device)
    weight_fp8, weight_scale = _quantize_fp8_weight(weight)
    dequant_weight = (
        (
            weight_fp8.float().reshape(-1, FP8_GROUP_SIZE)
            * weight_scale.float().unsqueeze(-1)
        )
        .reshape_as(weight)
        .to(dtype)
    )

    eps = 1e-5
    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(dequant_weight)
    ref_out = torch.nn.functional.rms_norm(ref_inp, (n,), ref_weight, eps=eps)
    res_out = rms_norm_w8a16_fp8(
        inp,
        (n,),
        weight_fp8,
        weight_scale,
        eps=eps,
        group_size=FP8_GROUP_SIZE,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
