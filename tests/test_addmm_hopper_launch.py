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

"""Public addmm_out: fixed IEEE arithmetic and pre-execution selection guards."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.addmm


@pytest.mark.parametrize(
    "m,n,k", [(414, 5376, 32), (8598, 96, 5376), (75938, 32, 5376)]
)
def test_hopper_fixed_addmm(m, n, k):
    if flag_gems.vendor_name != "nvidia" or torch.cuda.get_device_capability() != (
        9,
        0,
    ):
        pytest.skip("Measured launch configurations target SM90")
    from flag_gems.ops.addmm import addmm_kernel
    from flag_gems.runtime.backend._nvidia.hopper.ops.addmm import _configuration

    x = torch.randn((m, k), device="cuda", dtype=torch.float32)
    w = torch.randn((n, k), device="cuda", dtype=torch.float32).t()
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    out, expected = torch.empty((m, n), device="cuda"), torch.empty(
        (m, n), device="cuda"
    )
    assert _configuration(bias, x, w, out, 1, 1) is not None
    assert _configuration(bias, x, w, out, 0, 1) is None
    assert _configuration(bias, x, w, None, 1, 1) is None
    # Independent launch of the original kernel with a different M/N tile.
    # K32/IEEE reduction grouping is the unchanged numerical contract.
    import triton

    kernel = addmm_kernel
    while not isinstance(kernel, triton.runtime.jit.JITFunction):
        kernel = kernel.fn
    kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        x,
        w,
        bias,
        expected,
        1,
        1,
        m,
        n,
        k,
        k,
        1,
        1,
        k,
        0,
        1,
        n,
        1,
        BIAS_IS_VECTOR=True,
        BIAS_IS_SCALAR=False,
        HAS_K=True,
        IS_FP64=False,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        num_warps=4,
        num_stages=2,
    )
    assert flag_gems.addmm_out(bias, x, w, out=out) is out
    assert torch.equal(out, expected)
    # CPU float64 independently checks the mathematical expression as well.
    ref = torch.addmm(bias.cpu().double(), x.cpu().double(), w.cpu().double()).float()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
