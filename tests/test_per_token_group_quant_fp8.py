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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# On Enflame GCU the accuracy reference of this operator cannot be evaluated at
# all, so the WHOLE file is skipped for that vendor (see ENFLAME_SKIP_REASON for
# the mechanism and for the two ways to unblock it). Nothing in the reference or
# in the operator is changed here: only the enflame collection is skipped, and
# benchmark/test_per_token_group_quant_fp8.py still times the FlagGems kernel.
ENFLAME_SKIP_REASON = (
    "ENFLAME-SKIP-ALL(accuracy): the reference of this NoCPU-labelled operator "
    "cannot run on the GCU backend, so every case in this file is skipped. "
    "(1) Because the operator is labelled NoCPU in conf/operators.yaml, "
    "tools/run_tests.py never appends '--ref cpu', so the reference below "
    "(native_per_token_group_quant_fp8) always executes on the device. "
    "(2) Its first statement, x_.abs().max(dim=-1, keepdim=True), lowers to "
    "topsatenMax with an int64 index output; bf16 is not an accepted input "
    "dtype there ('int64 idx requires gcu400+ with int64/fp32/int32 data "
    "type'), and the topsaten error surface is an uncaught C++ abort rather "
    "than a Python exception, so the whole pytest process dies instead of "
    "failing a single case (reproduced on GCU400 with "
    "TORCH_GCU_ENABLE_INT64_AND_UINT64=1). (3) FP8_QUANT_SHAPES['DTYPES'] is "
    "[bfloat16], so no other dtype can route around it. Moving the reference "
    "to the CPU or rewriting it would both be forbidden ways of bypassing the "
    "reference, hence this skip. Unblock by either supporting bf16 input with "
    "an int64 index output in topsatenMax, or letting the harness pass "
    "'--ref cpu' for NoCPU operators whose reference is plain PyTorch."
)

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "enflame", reason=ENFLAME_SKIP_REASON
)


def native_per_token_group_quant_fp8(
    x, group_size, eps=1e-10, dtype=None, scale_ue8m0=False
):
    if dtype is None:
        dtype = flag_gems.SUPPORTED_FP8_DTYPE

    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax * torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=x.device)
    if scale_ue8m0:
        min_val = torch.tensor(1e-10, dtype=x_s.dtype, device=x_s.device)
        x_s = torch.exp2(torch.ceil(torch.log2(torch.maximum(x_s.abs(), min_val))))
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s


@pytest.mark.per_token_group_quant_fp8
@pytest.mark.parametrize("seed", utils.FP8_QUANT_SHAPES["SEEDS"])
@pytest.mark.parametrize("group_size", utils.FP8_QUANT_SHAPES["GROUP_SIZE"])
@pytest.mark.parametrize("dtype", utils.FP8_QUANT_SHAPES["DTYPES"])
@pytest.mark.parametrize("d", utils.FP8_QUANT_SHAPES["D"])
@pytest.mark.parametrize("num_tokens", utils.FP8_QUANT_SHAPES["NUM_TOKENS"])
@pytest.mark.parametrize("scale_ue8m0", [True, False])
def test_per_token_group_quant_fp8(num_tokens, d, dtype, group_size, seed, scale_ue8m0):
    torch.manual_seed(seed)

    x = torch.rand(num_tokens, d, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    ref_out, ref_scale = native_per_token_group_quant_fp8(
        ref_x, group_size, scale_ue8m0=scale_ue8m0
    )
    out, scale = flag_gems.per_token_group_quant_fp8(
        x, group_size, scale_ue8m0=scale_ue8m0
    )

    utils.gems_assert_close(scale, ref_scale, dtype=torch.float32)

    out_fp32 = utils.to_cpu(out, ref_out).to(torch.float32)
    ref_out_fp32 = ref_out.to(torch.float32)

    assert torch.allclose(out_fp32, ref_out_fp32, rtol=0.15)
