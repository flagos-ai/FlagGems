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

import sys

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.npu_transpose_quant_batchmatmul,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.ops.npu.npu_transpose_quant_batchmatmul.default


def _inputs(shape, dtype, device):
    m, b, k, n = shape

    def values(dims):
        return torch.randn(dims, device=device).to(torch.float8_e4m3fn)

    def scales(dims):
        return torch.randint(124, 129, dims, dtype=torch.uint8, device=device).view(
            torch.float8_e8m0fnu
        )

    return (values((m, b, k)), values((b, k, n)), torch.bfloat16), dict(
        x1_scale=scales((m, b, k // 64, 2)),
        x2_scale=scales((b, k // 64, n, 2)),
        group_sizes=(0, 0, 32),
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
    )


@pytest.mark.parametrize("shape", [(1, 8, 4096, 1024), (1024, 8, 4096, 1024)])
@pytest.mark.parametrize("implementation", ["auto", "triton"])
def test_npu_transpose_quant_batchmatmul_native(shape, implementation, monkeypatch):
    if implementation == "triton":
        op_module = sys.modules[flag_gems.npu_transpose_quant_batchmatmul.__module__]
        monkeypatch.setattr(op_module, "launch_mx_cube", lambda *args: False)
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.npu_transpose_quant_batchmatmul(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
