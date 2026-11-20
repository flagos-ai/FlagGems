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

pytestmark = [
    pytest.mark.inplace_partial_rotary_mul,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    pytest.importorskip("vllm_ascend.vllm_ascend_C")
    return torch.ops._C_ascend.inplace_partial_rotary_mul.default


def _inputs(shape, dtype, device):
    x = torch.randn(shape, device=device, dtype=dtype)
    a = torch.randn((shape[0], 1, 1, 64), device=device)
    return (x, a.cos(), a.sin(), "interleave", [shape[-1] - 64, shape[-1]]), {}


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 64, 512), (1024, 1, 1, 512), (1024, 1, 64, 128), (1024, 1, 64, 512)],
)
def test_inplace_partial_rotary_mul_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    ref_args = list(args)
    ref_args[0] = args[0].clone()
    expected = reference(*ref_args, **kwargs)
    actual = flag_gems.inplace_partial_rotary_mul(*args, **kwargs)
    torch.testing.assert_close(args[0], ref_args[0])
    torch.testing.assert_close(actual, expected)
